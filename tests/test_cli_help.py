"""Smoke tests for supported command wrappers without model loading."""

from __future__ import annotations

import subprocess
import sys

import pytest

SUPPORTED_MODULES = (
    "src.prompt_pipeline.generate",
    "scripts.generate_images",
    "scripts.score_images",
    "scripts.preflight_runtime",
    "scripts.run_manifest",
    "scripts.final_benchmark",
    "scripts.run_heldout_evaluation",
    "scripts.aggregate_heldout_scores",
    "scripts.merge_score_shards",
    "scripts.build_thesis_outputs",
    "scripts.build_evidence_manifest",
    "scripts.validate_prompt_dataset",
    "scripts.download_dataset",
    "scripts.precompute_text_embeddings",
    "scripts.prepare_history_cleanup",
    "scripts.compare_run_manifests",
    "scripts.check_training_comparability",
    "scripts.compare_training_runs",
    "scripts.synth.build_dataset",
)


@pytest.mark.parametrize("module_name", SUPPORTED_MODULES)
def test_supported_module_help_exits_cleanly(module_name):
    result = subprocess.run(
        [sys.executable, "-m", module_name, "--help"],
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )

    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.casefold()
