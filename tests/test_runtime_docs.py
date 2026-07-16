from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_repo_file(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_runtime_command_docs_publish_contract_surface() -> None:
    docs = read_repo_file("docs/commands.md")

    required_strings = [
        "## Runtime contracts",
        "docs/runtime_contracts.md",
        "configs/experiments",
        "scripts.preflight_runtime",
        "scripts.run_manifest",
        "runs/<run_id>/manifest.json",
        "generated artifacts",
        "non-committable",
        "uv run python -m scripts.run_manifest init --stage sft --config configs/sft.json",
        "uv run python -m scripts.run_manifest inspect runs/<run_id>/manifest.json",
        "uv run python -m scripts.run_manifest note runs/<run_id>/manifest.json",
        "uv run python -m scripts.run_manifest metrics runs/<run_id>/manifest.json",
        "uv run python -m scripts.preflight_runtime --stage generate",
        "uv run python -m scripts.preflight_runtime --stage score",
        "uv run python -m scripts.preflight_runtime --stage sft --config configs/sft.json",
        "uv run python -m scripts.preflight_runtime --stage dpo --config configs/dpo.json",
        "uv run python -m scripts.preflight_runtime --stage masked-sft --config configs/masked_sft.json",
        "uv run python -m scripts.preflight_runtime --stage synthetic",
        "uv run python -m scripts.preflight_runtime --stage evaluation",
    ]

    missing = [value for value in required_strings if value not in docs]
    assert missing == []


def test_makefile_exposes_cpu_safe_runtime_aliases() -> None:
    makefile = read_repo_file("Makefile")

    required_targets = [
        "preflight-sft:",
        "preflight-dpo:",
        "preflight-masked-sft:",
        "preflight-generate:",
        "preflight-score:",
        "manifest-init-sft:",
        "manifest-inspect:",
    ]
    required_commands = [
        "python -m scripts.preflight_runtime --stage sft --config configs/sft.json --json",
        "python -m scripts.preflight_runtime --stage dpo --config configs/dpo.json --json",
        "python -m scripts.preflight_runtime --stage masked-sft --config configs/masked_sft.json --json",
        "python -m scripts.run_manifest init --stage sft --config configs/sft.json",
        "python -m scripts.run_manifest inspect runs/<run_id>/manifest.json",
    ]

    missing = [value for value in [*required_targets, *required_commands] if value not in makefile]
    assert missing == []


def test_readme_links_runtime_contracts_before_expensive_work() -> None:
    readme = read_repo_file("README.md")
    readme_lower = readme.casefold()

    required_strings = [
        "docs/runtime_contracts.md",
        "configs/README.md",
        "preflight",
        "manifest",
        "before long-running GPU/model work",
        "generated artifacts",
        "outputs/",
        "runs/",
    ]

    missing = [value for value in required_strings if value.casefold() not in readme_lower]
    assert missing == []


def test_readme_cuda_clean_run_materializes_selection_before_sft() -> None:
    readme = read_repo_file("README.md")
    required_strings = [
        "uv run python -m scripts.download_dataset",
        "--revision ecd8b2da9820b35afc65e2d56eaf37a662c37976",
        "uv run python -m scripts.generate_images",
        "uv run python -m scripts.score_images",
        "--product_formula thesis",
        "uv run python -m scripts.materialize_training_data",
        "--output outputs/generated/selected_samples.jsonl",
        "uv run accelerate launch --config_file configs/accelerate/single_gpu.yaml",
        "--config configs/experiments/sft/sft_product_rerun_v2.json",
        "LEFT_MANIFEST=",
        "RIGHT_MANIFEST=",
    ]
    missing = [value for value in required_strings if value not in readme]
    assert missing == []

    ordered_commands = [
        "uv run python -m scripts.download_dataset",
        "uv run python -m scripts.generate_images",
        "uv run python -m scripts.score_images",
        "uv run python -m scripts.materialize_training_data",
        "uv run accelerate launch --config_file configs/accelerate/single_gpu.yaml",
    ]
    positions = [readme.index(command) for command in ordered_commands]
    assert positions == sorted(positions)
