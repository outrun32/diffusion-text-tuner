from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _read_repo_file(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_command_catalog_publishes_exact_cpu_safe_and_explicit_runtime_commands() -> None:
    docs = _read_repo_file("docs/commands.md")

    required_strings = [
        "## Reward and evaluation validity",
        "CPU-safe evaluation verification",
        "uv run pytest tests/test_evaluation_command_docs.py tests/test_evaluation_reward_interface.py tests/test_heldout_evaluation_harness.py tests/test_evaluation_slices_gold.py tests/test_evaluation_scoring_outputs.py tests/test_reward_diagnostics.py tests/test_thesis_outputs.py -q",
        "uv run python -m scripts.run_heldout_evaluation --config <heldout-config.json> --output-plan runs/evaluation/heldout-001/plan.json --markdown-summary runs/evaluation/heldout-001/plan.md",
        "uv run python -m scripts.score_images --images_dir outputs/generated/images --text_embeds_dir outputs/generated/text_embeds --output_csv outputs/generated/scores.csv --scorer both --ocr_device cpu --product_formula product --manifest_path runs/scoring/manifest.json --source_manifest runs/generation/manifest.json --source_manifest runs/scoring/manifest.json",
        "phase6-score-file/v1",
        ".schema.json",
        "uv run python -c \"from src.runtime.artifacts import validate_artifacts; report = validate_artifacts('evaluation_scores', {'scores_csv': 'outputs/generated/scores.csv'}); raise SystemExit(0 if report.ok else 1)\"",
        "uv run python -m scripts.analyze_reward_diagnostics --scores runs/eval/baseline/scores.csv --gold tests/fixtures/evaluation/gold_diagnostic.jsonl --output-report runs/eval/baseline/reward_diagnostics.json --markdown-summary runs/eval/baseline/reward_diagnostics.md --positive-threshold 0.80 --negative-threshold 0.50",
        "uv run python -c \"from src.evaluation.gold_benchmark import evaluate_gold_predictions; report = evaluate_gold_predictions('tests/fixtures/evaluation/gold_diagnostic.jsonl', []); raise SystemExit(0 if report['missing_prediction_count'] >= 0 else 1)\"",
        "uv run python -m scripts.build_thesis_outputs --config <reviewed-evidence-config.json> --output-bundle outputs/thesis/eval_bundle/bundle.json --markdown-summary outputs/thesis/eval_bundle/bundle.md",
        "GPU/model/OCR jobs remain explicit",
        "Generated score files, diagnostics, contact sheets, thesis bundles, plots, images, tensors",
        "checkpoints, logs, and run outputs remain runtime artifacts",
    ]

    missing = [value for value in required_strings if value not in docs]
    assert missing == []


def test_readme_summarizes_reward_and_links_detailed_evaluation_docs() -> None:
    readme = _read_repo_file("README.md")

    required_strings = [
        "docs/reward_evaluation.md",
        "Multiplying them makes a candidate score highly only when both checks agree",
        "## Quick start",
        "docs/commands.md",
    ]

    missing = [value for value in required_strings if value not in readme]
    assert missing == []


def test_phase6_docs_cross_link_related_guides_and_runtime_artifact_boundaries() -> None:
    doc_paths = [
        "docs/reward_evaluation.md",
        "docs/evaluation_harness.md",
        "docs/evaluation_diagnostics.md",
        "docs/thesis_outputs.md",
    ]
    combined = "\n".join(_read_repo_file(path) for path in doc_paths)

    required_strings = [
        "docs/reward_evaluation.md",
        "docs/evaluation_harness.md",
        "docs/evaluation_diagnostics.md",
        "docs/thesis_outputs.md",
        "generated score files",
        "diagnostic reports",
        "contact sheets",
        "thesis bundles",
        "plots",
        "images",
        "tensors",
        "checkpoints",
        "logs",
        "runtime artifacts",
    ]

    missing = [value for value in required_strings if value not in combined]
    assert missing == []


def test_makefile_exposes_phase6_aliases_without_executing_expensive_jobs() -> None:
    result = subprocess.run(
        [
            "make",
            "-n",
            "phase6-heldout-plan",
            "phase6-score-validation",
            "phase6-reward-diagnostics",
            "phase6-gold-diagnostics",
            "phase6-thesis-outputs",
            "phase6-evaluation-tests",
            "HELDOUT_EVAL_CONFIG=configs/experiments/evaluation/heldout_product_vs_baseline.json",
            "HELDOUT_EVAL_PLAN=runs/evaluation/heldout-001/plan.json",
            "HELDOUT_EVAL_PLAN_MD=runs/evaluation/heldout-001/plan.md",
            "EVAL_SCORES_CSV=outputs/generated/scores.csv",
            "REWARD_DIAGNOSTIC_REPORT=runs/eval/baseline/reward_diagnostics.json",
            "REWARD_DIAGNOSTIC_MD=runs/eval/baseline/reward_diagnostics.md",
            "THESIS_OUTPUT_CONFIG=configs/thesis/eval_bundle.json",
            "THESIS_OUTPUT_BUNDLE=outputs/thesis/eval_bundle/bundle.json",
            "THESIS_OUTPUT_MD=outputs/thesis/eval_bundle/bundle.md",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    stdout = result.stdout
    assert "python -m scripts.run_heldout_evaluation" in stdout
    assert "--config configs/experiments/evaluation/heldout_product_vs_baseline.json" in stdout
    assert "from src.runtime.artifacts import validate_artifacts" in stdout
    assert "'scores_csv': 'outputs/generated/scores.csv'" in stdout
    assert "python scripts/analyze_reward_diagnostics.py" in stdout
    assert "--gold tests/fixtures/evaluation/gold_diagnostic.jsonl" in stdout
    assert "from src.evaluation.gold_benchmark import evaluate_gold_predictions" in stdout
    assert "python scripts/build_thesis_outputs.py" in stdout
    assert "--config configs/thesis/eval_bundle.json" in stdout
    assert "uv run pytest tests/test_evaluation_command_docs.py" in stdout
    assert "FLUX" not in stdout
    assert "PaddleOCR" not in stdout
