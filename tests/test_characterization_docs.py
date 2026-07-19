from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _read_repo_file(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_docs_publish_cpu_safe_characterization_command_surface() -> None:
    docs = _read_repo_file("docs/commands.md")
    readme = _read_repo_file("README.md")

    required_strings = [
        "## CPU-safe characterization tests",
        "CPU-safe characterization",
        "uv run pytest tests/test_characterization_config_artifacts.py",
        "uv run pytest tests/test_training_dataset_contracts.py",
        "uv run pytest tests/test_training_objective_math.py",
        "uv run pytest tests/test_prompt_generation_determinism.py",
        "uv run pytest tests/test_reward_wrapper_contracts.py",
        "config/artifact characterization",
        "dataset and collator characterization",
        "objective math and DPO characterization",
        "prompt determinism characterization",
        "reward wrapper fake characterization",
        "make characterization-test",
        "make characterization-runtime",
        "make characterization-datasets",
        "make characterization-objectives",
        "make characterization-prompts",
        "make characterization-rewards",
    ]

    missing_from_docs = [value for value in required_strings if value not in docs]
    assert missing_from_docs == []

    readme_required = [
        "## Quick start",
        "make check",
        "docs/commands.md",
        "docs/runtime_contracts.md",
    ]
    missing_from_readme = [value for value in readme_required if value not in readme]
    assert missing_from_readme == []
    assert "## CPU-safe quality gates" not in readme
    assert "make characterization-test" not in readme


def test_makefile_exposes_focused_characterization_aliases() -> None:
    makefile = _read_repo_file("Makefile")

    required_strings = [
        "characterization-test:",
        "characterization-runtime:",
        "characterization-datasets:",
        "characterization-objectives:",
        "characterization-prompts:",
        "characterization-rewards:",
        "uv run pytest tests/test_characterization_config_artifacts.py tests/test_training_dataset_contracts.py tests/test_training_objective_math.py tests/test_prompt_generation_determinism.py tests/test_reward_wrapper_contracts.py tests/test_characterization_docs.py",
        "uv run pytest tests/test_characterization_config_artifacts.py",
        "uv run pytest tests/test_training_dataset_contracts.py",
        "uv run pytest tests/test_training_objective_math.py",
        "uv run pytest tests/test_prompt_generation_determinism.py",
        "uv run pytest tests/test_reward_wrapper_contracts.py",
    ]

    missing = [value for value in required_strings if value not in makefile]
    assert missing == []


def test_default_pytest_boundary_excludes_heavy_diagnostics() -> None:
    docs = _read_repo_file("docs/commands.md")
    runtime_docs = _read_repo_file("docs/runtime_contracts.md")
    readme = _read_repo_file("README.md")
    combined_docs = " ".join("\n".join([docs, runtime_docs, readme]).split())

    required_strings = [
        "default pytest does not load CUDA, FLUX, Qwen, PaddleOCR, vLLM, MLX, or SynthTIGER",
        "optional slow/GPU/OCR/model/integration/manual diagnostics",
        "pytest -m slow",
        "pytest -m gpu",
        "pytest -m ocr",
        "pytest -m model",
        "pytest -m integration",
        "pytest -m manual",
        "tmp_path",
        'torch.load(..., map_location="cpu", weights_only=True)',
        "generated artifacts and private prompts remain out of git",
    ]

    missing = [value for value in required_strings if value not in combined_docs]
    assert missing == []
