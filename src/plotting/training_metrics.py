"""Importable training metric loading, summary, and plotting helpers."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

METRIC_COLUMNS = ("step", "loss", "reward", "grad_norm", "lr", "elapsed_s")


@dataclass(frozen=True)
class TrainingMetrics:
    """Typed in-memory shape for ReFL training metric CSV columns."""

    step: list[float]
    loss: list[float]
    reward: list[float]
    grad_norm: list[float]
    lr: list[float]
    elapsed_s: list[float]


def load_metrics(csv_path: str | Path) -> TrainingMetrics:
    """Load metric columns from a training metrics CSV.

    The loader intentionally requires the current explicit CSV schema and lets
    missing columns or malformed numeric values fail instead of substituting
    hidden defaults.
    """

    data: dict[str, list[float]] = {column: [] for column in METRIC_COLUMNS}
    with Path(csv_path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            for column in METRIC_COLUMNS:
                data[column].append(float(row[column]))

    return TrainingMetrics(
        step=data["step"],
        loss=data["loss"],
        reward=data["reward"],
        grad_norm=data["grad_norm"],
        lr=data["lr"],
        elapsed_s=data["elapsed_s"],
    )


def smooth(values: list[float], window: int = 5) -> np.ndarray[Any, np.dtype[np.float64]]:
    """Return a moving average for ``values`` using the current script semantics."""

    if len(values) < window:
        return np.array(values)
    kernel = np.ones(window) / window
    return np.round(np.convolve(values, kernel, mode="valid"), decimals=12)


def summarize_metrics(metrics: TrainingMetrics) -> dict[str, float | int]:
    """Return deterministic summary fields used by plotting and tests."""

    rewards = metrics.reward
    grad_norms = metrics.grad_norm
    return {
        "steps": len(rewards),
        "reward_start": rewards[0],
        "reward_end": rewards[-1],
        "reward_max": max(rewards),
        "reward_mean": float(np.mean(rewards)),
        "grad_norm_mean": float(np.mean(grad_norms)),
        "grad_norm_max": max(grad_norms),
        "reward_trend_delta": rewards[-1] - rewards[0],
    }


def plot_training_metrics(metrics_csv: str | Path, output_dir: str | Path | None = None) -> Path:
    """Render the current ReFL training metric chart and print summary statistics."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = load_metrics(metrics_csv)
    metrics_path = Path(metrics_csv)
    resolved_output_dir = Path(output_dir) if output_dir is not None else metrics_path.parent

    steps = data.step
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("ReFL Training Metrics", fontsize=16, fontweight="bold")

    ax = axes[0, 0]
    ax.plot(steps, data.reward, alpha=0.3, color="tab:blue", label="raw")
    if len(steps) >= 5:
        smoothed_reward = smooth(data.reward)
        ax.plot(
            steps[len(steps) - len(smoothed_reward) :],
            smoothed_reward,
            color="tab:blue",
            linewidth=2,
            label="smooth(5)",
        )
    ax.set_xlabel("Step")
    ax.set_ylabel("Reward (P(yes))")
    ax.set_title("VLM Reward")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(steps, data.loss, alpha=0.3, color="tab:red", label="raw")
    if len(steps) >= 5:
        smoothed_loss = smooth(data.loss)
        ax.plot(
            steps[len(steps) - len(smoothed_loss) :],
            smoothed_loss,
            color="tab:red",
            linewidth=2,
            label="smooth(5)",
        )
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(steps, data.grad_norm, color="tab:green", alpha=0.5)
    if len(steps) >= 5:
        smoothed_grad_norm = smooth(data.grad_norm)
        ax.plot(
            steps[len(steps) - len(smoothed_grad_norm) :],
            smoothed_grad_norm,
            color="tab:green",
            linewidth=2,
        )
    ax.set_xlabel("Step")
    ax.set_ylabel("Gradient Norm")
    ax.set_title("Gradient Norm")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    if len(steps) > 1:
        speed = [
            step / max(elapsed_s, 0.1)
            for step, elapsed_s in zip(steps, data.elapsed_s, strict=True)
        ]
        ax.plot(steps, speed, color="tab:purple", alpha=0.5)
        if len(speed) >= 5:
            smoothed_speed = smooth(speed)
            ax.plot(
                steps[len(steps) - len(smoothed_speed) :],
                smoothed_speed,
                color="tab:purple",
                linewidth=2,
            )
    ax.set_xlabel("Step")
    ax.set_ylabel("Steps/sec")
    ax.set_title("Training Speed")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = resolved_output_dir / "training_curves.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Charts saved to {out_path}")

    _print_summary(data)
    return out_path


def _print_summary(metrics: TrainingMetrics) -> None:
    rewards = metrics.reward
    summary = summarize_metrics(metrics)
    print("\n--- Summary ---")
    print(f"Steps: {summary['steps']}")
    print(
        f"Reward: start={summary['reward_start']:.4f}, end={summary['reward_end']:.4f}, "
        f"max={summary['reward_max']:.4f}, mean={summary['reward_mean']:.4f}"
    )
    print(
        f"Grad norm: mean={summary['grad_norm_mean']:.4f}, "
        f"max={summary['grad_norm_max']:.4f}"
    )
    if len(rewards) >= 10:
        first_10 = np.mean(rewards[:10])
        last_10 = np.mean(rewards[-10:])
        print(
            f"Reward trend: first-10-avg={first_10:.4f}, last-10-avg={last_10:.4f}, "
            f"delta={last_10 - first_10:+.4f}"
        )
