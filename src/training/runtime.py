"""Import-safe runtime metadata helpers for training manifests."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


_TRAINING_INPUT_KEYS = frozenset(
    {
        "data_dir",
        "eval_suite_path",
        "latents_dir",
        "preference_pairs_path",
        "scores_csv",
        "selected_samples_path",
        "text_embeds_dir",
    }
)
_TRAINING_OUTPUT_KEYS = frozenset({"output_dir"})


def training_run_inputs(config_snapshot: Mapping[str, Any]) -> dict[str, Any]:
    """Return sorted input metadata keys recognized from a config snapshot."""

    return _sorted_present_fields(config_snapshot, _TRAINING_INPUT_KEYS)


def training_run_outputs(config_snapshot: Mapping[str, Any]) -> dict[str, Any]:
    """Return sorted output metadata keys recognized from a config snapshot."""

    return _sorted_present_fields(config_snapshot, _TRAINING_OUTPUT_KEYS)


def _sorted_present_fields(config_snapshot: Mapping[str, Any], keys: frozenset[str]) -> dict[str, Any]:
    return {
        key: config_snapshot[key]
        for key in sorted(keys)
        if key in config_snapshot and config_snapshot[key] is not None
    }
