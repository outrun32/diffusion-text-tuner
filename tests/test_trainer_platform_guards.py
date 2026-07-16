"""Training entrypoints must reject unsupported hosts before side effects."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.runtime import capabilities
from src.training.config import DPOConfig, MaskedSFTConfig, ReflConfig, SFTConfig


@pytest.mark.parametrize(
    ("module_name", "config"),
    [
        ("src.training.sft_trainer", SFTConfig()),
        ("src.training.dpo_trainer", DPOConfig()),
        ("src.training.masked_sft_trainer", MaskedSFTConfig()),
        ("src.training.refl_trainer", ReflConfig()),
    ],
)
def test_trainers_fail_before_creating_outputs(monkeypatch, tmp_path, module_name, config):
    monkeypatch.setattr(
        capabilities,
        "check_stage_support",
        lambda *_args, **_kwargs: SimpleNamespace(ok=False, errors=("unsupported-host",)),
    )
    config.output_dir = str(tmp_path / "must-not-exist")
    module = __import__(module_name, fromlist=["train"])

    with pytest.raises(RuntimeError, match="unsupported-host"):
        module.train(config)

    assert not (tmp_path / "must-not-exist").exists()


@pytest.mark.parametrize(
    ("module_name", "config", "dataset_name", "model_loader_name"),
    [
        ("src.training.sft_trainer", SFTConfig(), "SFTDataset", "load_transformer"),
        ("src.training.dpo_trainer", DPOConfig(), "DPODataset", "load_models"),
    ],
)
@pytest.mark.parametrize(
    ("dataset_size", "message"),
    [(0, "selected dataset is empty"), (3, "drop_last=True would yield no batches")],
)
def test_sft_and_dpo_reject_non_batchable_selection_before_model_loading(
    monkeypatch,
    tmp_path,
    module_name,
    config,
    dataset_name,
    model_loader_name,
    dataset_size,
    message,
):
    monkeypatch.setattr(
        capabilities,
        "check_stage_support",
        lambda *_args, **_kwargs: SimpleNamespace(ok=True, errors=()),
    )
    module = __import__(module_name, fromlist=["train"])

    class StubDataset:
        def __init__(self, **_kwargs):
            pass

        def __len__(self):
            return dataset_size

    monkeypatch.setattr(module, dataset_name, StubDataset)
    monkeypatch.setattr(
        module,
        model_loader_name,
        lambda *_args, **_kwargs: pytest.fail("model loading must not start"),
    )
    config.batch_size = 4
    config.output_dir = str(tmp_path / "must-not-exist")

    with pytest.raises(ValueError, match=message):
        module.train(config)

    assert not (tmp_path / "must-not-exist").exists()
