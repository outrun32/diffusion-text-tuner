"""CPU-safe contracts for training metric plotting seams."""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import pytest


def write_metrics_csv(path: Path, rows: list[dict[str, float]]) -> Path:
    columns = ["step", "loss", "reward", "grad_norm", "lr", "elapsed_s"]
    lines = [",".join(columns)]
    for row in rows:
        lines.append(",".join(str(row[column]) for column in columns))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def sample_rows(count: int = 10) -> list[dict[str, float]]:
    return [
        {
            "step": float(index + 1),
            "loss": 1.0 - (index * 0.01),
            "reward": 0.10 + (index * 0.05),
            "grad_norm": 0.5 + (index * 0.1),
            "lr": 0.0001,
            "elapsed_s": 2.0 + index,
        }
        for index in range(count)
    ]


def test_load_metrics_reads_expected_columns(tmp_path: Path) -> None:
    from src.plotting.training_metrics import TrainingMetrics, load_metrics

    csv_path = write_metrics_csv(tmp_path / "metrics.csv", sample_rows(count=2))

    metrics = load_metrics(csv_path)

    assert isinstance(metrics, TrainingMetrics)
    assert metrics.step == [1.0, 2.0]
    assert metrics.loss == [1.0, 0.99]
    assert metrics.reward == [0.1, 0.15000000000000002]
    assert metrics.grad_norm == [0.5, 0.6]
    assert metrics.lr == [0.0001, 0.0001]
    assert metrics.elapsed_s == [2.0, 3.0]


def test_smooth_returns_moving_average_without_matplotlib() -> None:
    sys.modules.pop("matplotlib", None)
    sys.modules.pop("matplotlib.pyplot", None)

    from src.plotting.training_metrics import smooth

    assert smooth([1, 2, 3, 4, 5], window=3).tolist() == [2.0, 3.0, 4.0]
    assert "matplotlib" not in sys.modules



def test_summarize_metrics_returns_reward_and_grad_norm_stats() -> None:
    from src.plotting.training_metrics import TrainingMetrics, summarize_metrics

    metrics = TrainingMetrics(
        step=[1.0, 2.0, 3.0],
        loss=[1.0, 0.9, 0.8],
        reward=[0.2, 0.3, 0.5],
        grad_norm=[0.5, 0.7, 1.0],
        lr=[0.001, 0.001, 0.001],
        elapsed_s=[1.0, 2.0, 3.0],
    )

    summary = summarize_metrics(metrics)

    assert summary == {
        "steps": 3,
        "reward_start": pytest.approx(0.2),
        "reward_end": pytest.approx(0.5),
        "reward_max": pytest.approx(0.5),
        "reward_mean": pytest.approx(1.0 / 3.0),
        "grad_norm_mean": pytest.approx(2.2 / 3.0),
        "grad_norm_max": pytest.approx(1.0),
        "reward_trend_delta": pytest.approx(0.3),
    }


def test_importing_training_metrics_does_not_import_matplotlib() -> None:
    sys.modules.pop("src.plotting.training_metrics", None)
    sys.modules.pop("matplotlib", None)
    sys.modules.pop("matplotlib.pyplot", None)

    importlib.import_module("src.plotting.training_metrics")

    assert "matplotlib" not in sys.modules
    assert "matplotlib.pyplot" not in sys.modules


def test_plot_training_metrics_uses_lazy_matplotlib_and_preserves_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from src.plotting.training_metrics import plot_training_metrics

    csv_path = write_metrics_csv(tmp_path / "metrics.csv", sample_rows(count=10))
    fake_pyplot = FakePyplot()
    fake_matplotlib = types.ModuleType("matplotlib")
    fake_matplotlib.__path__ = []
    fake_matplotlib.use = fake_pyplot.use
    fake_matplotlib.pyplot = fake_pyplot
    monkeypatch.setitem(sys.modules, "matplotlib", fake_matplotlib)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", fake_pyplot)

    out_path = plot_training_metrics(csv_path, tmp_path / "plots")

    captured = capsys.readouterr().out
    assert out_path == tmp_path / "plots" / "training_curves.png"
    assert fake_pyplot.backend == "Agg"
    assert fake_pyplot.saved_path == out_path
    assert "Charts saved to" in captured
    assert "--- Summary ---" in captured
    assert "Steps: 10" in captured
    assert "Reward: start=0.1000, end=0.5500" in captured
    assert "Grad norm: mean=0.9500, max=1.4000" in captured
    assert "Reward trend: first-10-avg=0.3250, last-10-avg=0.3250, delta=+0.0000" in captured


def test_plot_metrics_main_delegates_to_importable_plotter(monkeypatch: pytest.MonkeyPatch) -> None:
    import scripts.plot_metrics as plot_metrics

    calls: list[tuple[str, str | None]] = []

    def fake_plot_training_metrics(metrics_csv: str, output_dir: str | None = None) -> Path:
        calls.append((metrics_csv, output_dir))
        return Path(output_dir or ".") / "training_curves.png"

    monkeypatch.setattr(plot_metrics, "plot_training_metrics", fake_plot_training_metrics)

    assert plot_metrics.main(["runs/example/metrics.csv", "--output-dir", "plots"]) == 0
    assert calls == [("runs/example/metrics.csv", "plots")]
    assert plot_metrics.plot("runs/compat/metrics.csv", "compat-plots") == Path(
        "compat-plots/training_curves.png"
    )
    assert calls[-1] == ("runs/compat/metrics.csv", "compat-plots")
    assert plot_metrics.load_metrics.__module__ == "src.plotting.training_metrics"
    assert plot_metrics.smooth.__module__ == "src.plotting.training_metrics"


class FakeAxis:
    def plot(self, *args: object, **kwargs: object) -> None:
        pass

    def set_xlabel(self, value: str) -> None:
        pass

    def set_ylabel(self, value: str) -> None:
        pass

    def set_title(self, value: str) -> None:
        pass

    def legend(self) -> None:
        pass

    def grid(self, *args: object, **kwargs: object) -> None:
        pass


class FakeFigure:
    def __init__(self, pyplot: FakePyplot) -> None:
        self._pyplot = pyplot

    def suptitle(self, *args: object, **kwargs: object) -> None:
        pass

    def savefig(self, out_path: str | Path, *args: object, **kwargs: object) -> None:
        self._pyplot.saved_path = Path(out_path)


class FakeAxes:
    def __init__(self) -> None:
        self._axes = [[FakeAxis(), FakeAxis()], [FakeAxis(), FakeAxis()]]

    def __getitem__(self, item: tuple[int, int]) -> FakeAxis:
        row, column = item
        return self._axes[row][column]


class FakePyplot(types.ModuleType):
    def __init__(self) -> None:
        super().__init__("matplotlib.pyplot")
        self.backend: str | None = None
        self.saved_path: Path | None = None

    def use(self, backend: str) -> None:
        self.backend = backend

    def subplots(self, *args: object, **kwargs: object) -> tuple[FakeFigure, FakeAxes]:
        return FakeFigure(self), FakeAxes()

    def tight_layout(self) -> None:
        pass

    def close(self, fig: FakeFigure) -> None:
        pass
