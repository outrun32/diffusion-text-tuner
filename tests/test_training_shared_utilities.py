from __future__ import annotations

import importlib
import sys
from copy import deepcopy
from pathlib import Path

import torch

from src.training.checkpointing import checkpoint_dir, should_save_checkpoint
from src.training.dpo_objective import compute_sigma as dpo_compute_sigma
from src.training.runtime import training_run_inputs, training_run_outputs
from src.training.sampling import normalize_eval_suite_items, should_sample_step
from src.training.schedulers import compute_sigma


HEAVY_OPTIONAL_MODULES = {
    "accelerate",
    "diffusers",
    "paddleocr",
    "peft",
    "synthtiger",
    "transformers",
}


def test_shared_training_modules_are_import_safe() -> None:
    before = set(sys.modules)

    for module_name in (
        "src.training.sampling",
        "src.training.checkpointing",
        "src.training.schedulers",
        "src.training.runtime",
    ):
        importlib.import_module(module_name)

    newly_imported_heavy_modules = (set(sys.modules) - before) & HEAVY_OPTIONAL_MODULES
    assert newly_imported_heavy_modules == set()


def test_should_sample_step_uses_positive_interval_boundaries() -> None:
    assert should_sample_step(0, 1) is False
    assert should_sample_step(10, 0) is False
    assert should_sample_step(10, -5) is False
    assert should_sample_step(10, 5) is True
    assert should_sample_step(11, 5) is False


def test_should_save_checkpoint_matches_sampling_interval_semantics() -> None:
    assert should_save_checkpoint(0, 1) is False
    assert should_save_checkpoint(10, 0) is False
    assert should_save_checkpoint(10, -5) is False
    assert should_save_checkpoint(10, 5) is True
    assert should_save_checkpoint(11, 5) is False


def test_checkpoint_dir_formats_step_under_output_checkpoints() -> None:
    assert str(checkpoint_dir("outputs/sft", 42)) == "outputs/sft/checkpoints/step_000042"


def test_normalize_eval_suite_items_is_deterministic_and_immutable() -> None:
    items = [
        {
            "prompt": "Render Ж",
            "target_text": "Ж",
            "name": "rare_letter",
            "seed": 1234,
            "resolution": 512,
        },
        {"prompt": "Render ё", "target": "ё"},
    ]
    original = deepcopy(items)

    normalized = normalize_eval_suite_items(items, limit=2)

    assert normalized == [
        {
            "name": "rare_letter",
            "prompt": "Render Ж",
            "resolution": 512,
            "seed": 1234,
            "target_text": "Ж",
        },
        {"name": "item_01", "prompt": "Render ё", "target_text": "ё"},
    ]
    assert items == original
    assert normalized[0] is not items[0]


def test_normalize_eval_suite_items_honors_limit() -> None:
    items = [
        {"prompt": "one", "target_text": "1"},
        {"prompt": "two", "target_text": "2"},
    ]

    assert normalize_eval_suite_items(items, limit=1) == [
        {"name": "item_00", "prompt": "one", "target_text": "1"}
    ]


def test_scheduler_compute_sigma_reexports_dpo_behavior() -> None:
    timesteps = torch.tensor([0, 250, 500, 999])

    assert torch.equal(compute_sigma(timesteps, shift=3.0), dpo_compute_sigma(timesteps, shift=3.0))


def test_training_run_metadata_helpers_return_sorted_path_metadata() -> None:
    snapshot = {
        "output_dir": "outputs/sft",
        "text_embeds_dir": "outputs/generated/text_embeds",
        "ignored": "not a manifest path",
        "latents_dir": "outputs/generated/latents",
        "scores_csv": "outputs/generated/scores.csv",
        "selected_samples_path": "outputs/selection/selected.jsonl",
        "preference_pairs_path": None,
        "data_dir": "data/synth_cyrillic/masked_sft",
        "eval_suite_path": "configs/eval_suite.json",
    }
    original = deepcopy(snapshot)

    assert list(training_run_inputs(snapshot)) == [
        "data_dir",
        "eval_suite_path",
        "latents_dir",
        "scores_csv",
        "selected_samples_path",
        "text_embeds_dir",
    ]
    assert training_run_inputs(snapshot) == {
        "data_dir": "data/synth_cyrillic/masked_sft",
        "eval_suite_path": "configs/eval_suite.json",
        "latents_dir": "outputs/generated/latents",
        "scores_csv": "outputs/generated/scores.csv",
        "selected_samples_path": "outputs/selection/selected.jsonl",
        "text_embeds_dir": "outputs/generated/text_embeds",
    }
    assert training_run_outputs(snapshot) == {"output_dir": "outputs/sft"}
    assert snapshot == original


def test_training_comparability_docs_describe_shared_trainer_seams() -> None:
    docs = Path("docs/training_comparability.md").read_text(encoding="utf-8")

    for expected in (
        "Shared trainer seams",
        "Adding a trainer variant",
        "src.training.sampling",
        "src.training.checkpointing",
        "src.training.schedulers",
        "src.training.runtime",
        "Do not add unrelated sampling, checkpointing, scheduler, or runtime code directly to large trainer modules.",
        "load_stage_config",
        "create/compare run manifests",
        "run CPU-safe tests",
    ):
        assert expected in docs
