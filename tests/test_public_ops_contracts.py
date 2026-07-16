"""Public release checks for cluster, dependency, and container surfaces."""

from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_root_training_configs_pin_current_base_model_commit():
    source = json.loads(_read("reports/final/current_model_sources.json"))
    expected = source["models"]["base_diffusion"]["revision"]

    assert re.fullmatch(r"[0-9a-f]{40}", expected)
    for relative in ("configs/sft.json", "configs/dpo.json"):
        config = json.loads(_read(relative))
        assert config["model_revision"] == expected


def test_cluster_cache_and_jobs_use_immutable_revisions_offline():
    source = json.loads(_read("reports/final/current_model_sources.json"))
    setup = _read("scripts/cluster/setup_env.sh")

    for model in source["models"].values():
        assert model["model_id"] in setup
        assert model["revision"] in setup
    assert "snapshot_download" in setup
    assert "revision=revision" in setup

    for relative in (
        "scripts/cluster/generate_images.sbatch",
        "scripts/cluster/score_images.sbatch",
        "scripts/cluster/sft.sbatch",
        "scripts/cluster/dpo.sbatch",
    ):
        launcher = _read(relative)
        assert "HF_HUB_OFFLINE=1" in launcher
        assert "TRANSFORMERS_OFFLINE=1" in launcher
        assert "40-character" in launcher


def test_reward_extra_installs_torchvision_pair():
    payload = tomllib.loads(_read("pyproject.toml"))
    reward = payload["project"]["optional-dependencies"]["reward"]

    assert any(requirement.startswith("torch>=") for requirement in reward)
    assert any(requirement.startswith("torchvision>=") for requirement in reward)


def test_vllm_backend_has_a_pinned_conflicting_linux_profile() -> None:
    payload = tomllib.loads(_read("pyproject.toml"))
    vllm = payload["project"]["optional-dependencies"]["vllm"]
    conflicts = payload["tool"]["uv"]["conflicts"]

    assert vllm == ["vllm==0.25.1; sys_platform == 'linux'"]
    conflict_sets = {
        frozenset((item.get("extra") or f"group:{item['group']}") for item in conflict)
        for conflict in conflicts
    }
    assert frozenset({"vllm", "group:dev"}) in conflict_sets
    assert frozenset({"vllm", "gpu"}) in conflict_sets
    assert frozenset({"vllm", "reward"}) in conflict_sets
    assert 'name = "vllm"' in _read("uv.lock")


def test_docker_context_excludes_private_and_binary_artifacts_but_keeps_evidence():
    dockerignore = _read(".dockerignore")
    required_exclusions = {
        ".env.*",
        ".vscode",
        ".copilot",
        "**/*handoff*",
        "data/prompts*.jsonl",
        "**/*.safetensors",
        "**/*.parquet",
        "**/*.db",
        "**/*.png",
        "**/*.jpg",
        "**/*.webp",
        "**/*.pdf",
        "**/*.pptx",
    }
    required_exceptions = {
        "!docs/project-page/assets/*.webp",
        "!reports/final/evidence_manifest.json",
        "!reports/final/historical_selection_bias.json",
        "!tests/fixtures/evaluation/gold_diagnostic.jsonl",
    }

    lines = {line.strip() for line in dockerignore.splitlines() if line.strip()}
    assert required_exclusions <= lines
    assert required_exceptions <= lines

    dockerfile = _read("Dockerfile")
    assert "test ! -e data/prompts_simple.jsonl" in dockerfile
    assert "test ! -e experiments/assets/bad_text.png" in dockerfile
    assert "test -f reports/final/evidence_manifest.json" in dockerfile
    assert "test -f docs/project-page/assets/product_bias.webp" in dockerfile
