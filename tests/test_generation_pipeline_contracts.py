from __future__ import annotations

import importlib
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest


def _write_prompts(path: Path) -> Path:
    records = [
        {"prompt": "render letter А", "target_text": "А"},
        {"prompt": "render word мир", "target_text": "мир"},
        {"prompt": "render word текст", "target_text": "текст"},
        {"prompt": "render phrase добрый день", "target_text": "добрый день"},
    ]
    path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )
    return path


def test_generation_config_preserves_cli_defaults_and_slice_fields(tmp_path: Path) -> None:
    from src.generation.pipeline import GenerationConfig

    prompts_path = tmp_path / "prompts.jsonl"
    config = GenerationConfig(prompts=prompts_path, start_idx=1, end_idx=3)

    assert config.prompts == prompts_path
    assert config.output_dir == Path("outputs/generated")
    assert config.model_id == "black-forest-labs/FLUX.2-klein-base-4B"
    assert config.lora_path is None
    assert config.versions_per_prompt == 5
    assert config.batch_size == 1
    assert config.num_inference_steps == 50
    assert config.guidance_scale == 4.0
    assert config.resolution == 512
    assert config.seed == 42
    assert config.device == "cuda"
    assert config.start_idx == 1
    assert config.end_idx == 3
    assert config.save_latents is True
    assert config.save_png is True


def test_load_prompt_records_slices_and_preserves_prompt_fields(tmp_path: Path) -> None:
    from src.generation.pipeline import load_prompt_records

    prompts_path = _write_prompts(tmp_path / "prompts.jsonl")

    records = load_prompt_records(prompts_path, start_idx=1, end_idx=3)

    assert records == [
        {"prompt": "render word мир", "target_text": "мир"},
        {"prompt": "render word текст", "target_text": "текст"},
    ]


def test_resolve_generation_paths_is_deterministic_and_path_object_based() -> None:
    from src.generation.pipeline import resolve_generation_paths

    paths = resolve_generation_paths(Path("outputs/generated"))

    assert paths.output_dir == Path("outputs/generated")
    assert paths.latents_dir == Path("outputs/generated/latents")
    assert paths.text_embeds_dir == Path("outputs/generated/text_embeds")
    assert paths.images_dir == Path("outputs/generated/images")
    assert paths.manifest_path == Path("outputs/generated/manifest.json")


def test_plan_generation_seed_matches_existing_script_formula() -> None:
    from src.generation.pipeline import plan_generation_seed

    assert (
        plan_generation_seed(
            seed=42,
            prompt_index=3,
            versions_per_prompt=5,
            version=2,
        )
        == 59
    )


def test_pipeline_import_does_not_load_model_or_cuda_stacks() -> None:
    forbidden_modules = {
        "diffusers",
        "torchvision",
        "src.training.flux2_utils",
    }
    for module_name in ["src.generation.pipeline", *forbidden_modules]:
        sys.modules.pop(module_name, None)

    importlib.import_module("src.generation.pipeline")

    assert forbidden_modules.isdisjoint(sys.modules)


def test_generate_images_cli_builds_config_and_delegates(monkeypatch, tmp_path: Path) -> None:
    import scripts.generate_images as generate_images
    from src.generation.pipeline import GenerationConfig

    captured: dict[str, GenerationConfig] = {}

    def fake_run_generation(config: GenerationConfig) -> None:
        captured["config"] = config

    monkeypatch.setattr(generate_images, "run_generation", fake_run_generation)

    prompts_path = _write_prompts(tmp_path / "prompts.jsonl")
    result = generate_images.main(
        [
            "--prompts",
            str(prompts_path),
            "--output_dir",
            str(tmp_path / "generated"),
            "--lora_path",
            "outputs/lora",
            "--versions_per_prompt",
            "2",
            "--batch_size",
            "1",
            "--num_inference_steps",
            "12",
            "--guidance_scale",
            "3.5",
            "--resolution",
            "256",
            "--seed",
            "7",
            "--start_idx",
            "1",
            "--end_idx",
            "3",
        ]
    )

    assert result == 0
    assert captured["config"] == GenerationConfig(
        prompts=prompts_path,
        output_dir=tmp_path / "generated",
        lora_path="outputs/lora",
        versions_per_prompt=2,
        batch_size=1,
        num_inference_steps=12,
        guidance_scale=3.5,
        resolution=256,
        seed=7,
        start_idx=1,
        end_idx=3,
    )


def test_generate_images_cli_defaults_match_existing_behavior(monkeypatch, tmp_path: Path) -> None:
    import scripts.generate_images as generate_images
    from src.generation.pipeline import GenerationConfig

    captured: dict[str, GenerationConfig] = {}

    def fake_run_generation(config: GenerationConfig) -> None:
        captured["config"] = config

    monkeypatch.setattr(generate_images, "run_generation", fake_run_generation)

    prompts_path = _write_prompts(tmp_path / "prompts.jsonl")
    result = generate_images.main(["--prompts", str(prompts_path)])

    assert result == 0
    assert captured["config"] == GenerationConfig(prompts=prompts_path)


def test_generation_config_rejects_noop_batching(tmp_path: Path) -> None:
    from src.generation.pipeline import GenerationConfig

    with pytest.raises(ValueError, match="batching is not implemented"):
        GenerationConfig(prompts=tmp_path / "prompts.jsonl", batch_size=2)


def test_generation_manifest_is_created_before_resume_and_never_rewritten(
    tmp_path: Path,
) -> None:
    from src.generation.pipeline import (
        GenerationConfig,
        ensure_generation_resume_contract,
        load_prompt_records,
        resolve_generation_paths,
    )

    prompts = _write_prompts(tmp_path / "prompts.jsonl")
    config = GenerationConfig(
        prompts=prompts,
        output_dir=tmp_path / "generated",
        model_revision="immutable-model-revision",
        start_idx=1,
        end_idx=3,
        versions_per_prompt=2,
        num_inference_steps=12,
        guidance_scale=3.5,
        resolution=256,
        seed=17,
    )
    paths = resolve_generation_paths(config.output_dir)
    records = load_prompt_records(prompts, start_idx=1, end_idx=3)

    manifest_path = ensure_generation_resume_contract(config, paths, records)
    initial_bytes = manifest_path.read_bytes()
    initial_inode = manifest_path.stat().st_ino

    paths.text_embeds_dir.mkdir(parents=True)
    (paths.text_embeds_dir / "000001.pt").write_bytes(b"partial-embedding")
    resumed_path = ensure_generation_resume_contract(config, paths, records)

    assert resumed_path == manifest_path
    assert manifest_path.read_bytes() == initial_bytes
    assert manifest_path.stat().st_ino == initial_inode
    manifest = json.loads(initial_bytes)
    assert manifest["schema_version"] == "generation-manifest/v4"
    contract = manifest["contract"]
    assert contract["model_revision"] == "immutable-model-revision"
    assert contract["start_idx"] == 1
    assert contract["end_idx"] == 3
    assert contract["artifact_layout"] == {
        "images": "images/{prompt_id}/v{version}.png",
        "latents": "latents/{prompt_id}/v{version}.pt",
        "prompt_id": "{global_index:06d}",
        "text_embeddings": "text_embeds/{prompt_id}.pt",
        "version_start": 0,
        "version_stop_exclusive": 2,
    }
    assert len(contract["prompts_sha256"]) == 64
    assert len(contract["selected_records_sha256"]) == 64
    assert len(manifest["contract_sha256"]) == 64
    assert manifest["completion"]["status"] == "planned"
    assert len(manifest["completion"]["completion_sha256"]) == 64


@pytest.mark.parametrize(
    "changes",
    [
        {"model_id": "different/model"},
        {"model_revision": "revision-b"},
        {"lora_path": "adapters/different"},
        {"versions_per_prompt": 3},
        {"num_inference_steps": 13},
        {"guidance_scale": 2.0},
        {"resolution": 768},
        {"seed": 18},
        {"device": "cpu"},
        {"start_idx": 0},
        {"end_idx": 4},
        {"save_png": False},
        {"save_latents": False},
    ],
    ids=[
        "model-id",
        "model-revision",
        "lora-locator",
        "versions",
        "steps",
        "guidance",
        "resolution",
        "seed",
        "device",
        "slice-start",
        "slice-end",
        "png-layout",
        "latent-layout",
    ],
)
def test_generation_resume_rejects_contract_drift_without_overwriting_manifest(
    tmp_path: Path,
    changes: dict[str, object],
) -> None:
    from src.generation.pipeline import (
        GenerationConfig,
        GenerationResumeError,
        ensure_generation_resume_contract,
        load_prompt_records,
        resolve_generation_paths,
    )

    prompts = _write_prompts(tmp_path / "prompts.jsonl")
    config = GenerationConfig(
        prompts=prompts,
        output_dir=tmp_path / "generated",
        model_revision="revision-a",
        lora_path="adapters/original",
        versions_per_prompt=2,
        num_inference_steps=12,
        guidance_scale=3.5,
        resolution=256,
        seed=17,
        start_idx=1,
        end_idx=3,
        run_manifest_path="runs/original/manifest.json",
    )
    paths = resolve_generation_paths(config.output_dir)
    records = load_prompt_records(prompts, start_idx=config.start_idx, end_idx=config.end_idx)
    manifest_path = ensure_generation_resume_contract(config, paths, records)
    original_manifest = manifest_path.read_bytes()

    changed = replace(config, **changes)
    changed_records = load_prompt_records(
        prompts,
        start_idx=changed.start_idx,
        end_idx=changed.end_idx,
    )
    with pytest.raises(GenerationResumeError, match="resume contract mismatch"):
        ensure_generation_resume_contract(changed, paths, changed_records)

    assert manifest_path.read_bytes() == original_manifest


def test_generation_shards_coexist_and_resubmit_with_new_run_lineage(tmp_path: Path) -> None:
    from src.generation.pipeline import (
        GenerationConfig,
        begin_generation_attempt,
        ensure_generation_resume_contract,
        load_prompt_records,
        resolve_generation_paths,
        validate_generation_manifest,
    )

    prompts = _write_prompts(tmp_path / "prompts.jsonl")
    output_dir = tmp_path / "generated"
    paths = resolve_generation_paths(output_dir)
    shard_0 = GenerationConfig(
        prompts=prompts,
        output_dir=output_dir,
        model_revision="a" * 40,
        versions_per_prompt=1,
        start_idx=0,
        end_idx=2,
        shard_index=0,
        shard_count=2,
        manifest_path=tmp_path / "runs" / "generation-shard-0.json",
        run_manifest_path="runs/attempt-1/manifest.json",
    )
    shard_1 = replace(
        shard_0,
        start_idx=2,
        end_idx=4,
        shard_index=1,
        manifest_path=tmp_path / "runs" / "generation-shard-1.json",
        run_manifest_path="runs/attempt-2/manifest.json",
    )

    path_0 = ensure_generation_resume_contract(
        shard_0,
        paths,
        load_prompt_records(prompts, start_idx=0, end_idx=2),
    )
    path_1 = ensure_generation_resume_contract(
        shard_1,
        paths,
        load_prompt_records(prompts, start_idx=2, end_idx=4),
    )

    assert path_0 != path_1
    assert validate_generation_manifest(path_0)["contract"]["slice_index"] == 0
    assert validate_generation_manifest(path_1)["contract"]["slice_index"] == 1

    resubmitted = replace(shard_0, run_manifest_path="runs/attempt-3/manifest.json")
    assert (
        ensure_generation_resume_contract(
            resubmitted,
            paths,
            load_prompt_records(prompts, start_idx=0, end_idx=2),
        )
        == path_0
    )
    begin_generation_attempt(path_0, run_manifest_path=resubmitted.run_manifest_path)
    completion = validate_generation_manifest(path_0)["completion"]
    assert completion["run_manifest_path"] == "runs/attempt-3/manifest.json"
    assert completion["attempt"] == 1


def test_generation_completion_requires_full_hashed_artifact_coverage(tmp_path: Path) -> None:
    from src.generation.pipeline import (
        GenerationConfig,
        GenerationResumeError,
        _contract_artifact_paths,
        begin_generation_attempt,
        complete_generation_attempt,
        ensure_generation_resume_contract,
        load_prompt_records,
        resolve_generation_paths,
        validate_generation_manifest,
    )

    prompts = _write_prompts(tmp_path / "prompts.jsonl")
    config = GenerationConfig(
        prompts=prompts,
        output_dir=tmp_path / "generated",
        model_revision="a" * 40,
        versions_per_prompt=1,
        start_idx=0,
        end_idx=1,
        run_manifest_path="runs/generation/manifest.json",
    )
    paths = resolve_generation_paths(config.output_dir)
    records = load_prompt_records(prompts, start_idx=0, end_idx=1)
    manifest_path = ensure_generation_resume_contract(config, paths, records)
    begin_generation_attempt(manifest_path, run_manifest_path=config.run_manifest_path)

    with pytest.raises(GenerationResumeError, match="artifact coverage is incomplete"):
        complete_generation_attempt(
            manifest_path,
            generated={"text_embeddings": 1, "images": 1, "latents": 1},
            skipped={"text_embeddings": 0, "images": 0, "latents": 0},
        )
    assert validate_generation_manifest(manifest_path)["completion"]["status"] == "in-progress"

    contract = validate_generation_manifest(manifest_path)["contract"]
    for artifact_paths in _contract_artifact_paths(contract).values():
        for artifact_path in artifact_paths:
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            artifact_path.write_bytes(f"fixture:{artifact_path.name}".encode())

    completed = complete_generation_attempt(
        manifest_path,
        generated={"text_embeddings": 1, "images": 1, "latents": 1},
        skipped={"text_embeddings": 0, "images": 0, "latents": 0},
    )
    assert completed["completion"]["status"] == "complete"
    assert (
        validate_generation_manifest(
            manifest_path,
            require_complete=True,
            verify_artifacts=True,
        )["completion"]["discovered"]
        == contract["expected"]
    )

    prompts.write_text(
        prompts.read_text(encoding="utf-8").replace("render letter А", "render letter Б"),
        encoding="utf-8",
    )
    with pytest.raises(GenerationResumeError, match="prompt source SHA-256 mismatch"):
        validate_generation_manifest(manifest_path, require_complete=True, verify_artifacts=True)


def test_generation_resume_rejects_prompt_content_drift(tmp_path: Path) -> None:
    from src.generation.pipeline import (
        GenerationConfig,
        GenerationResumeError,
        ensure_generation_resume_contract,
        load_prompt_records,
        resolve_generation_paths,
    )

    prompts = _write_prompts(tmp_path / "prompts.jsonl")
    config = GenerationConfig(prompts=prompts, output_dir=tmp_path / "generated")
    paths = resolve_generation_paths(config.output_dir)
    records = load_prompt_records(prompts)
    manifest_path = ensure_generation_resume_contract(config, paths, records)
    original_manifest = manifest_path.read_bytes()

    prompts.write_text(
        prompts.read_text(encoding="utf-8").replace("render word мир", "render word МИР"),
        encoding="utf-8",
    )
    with pytest.raises(GenerationResumeError, match="prompt source SHA-256 mismatch"):
        ensure_generation_resume_contract(config, paths, load_prompt_records(prompts))

    assert manifest_path.read_bytes() == original_manifest


def test_generation_resume_hashes_local_lora_contents(tmp_path: Path) -> None:
    from src.generation.pipeline import (
        GenerationConfig,
        GenerationResumeError,
        ensure_generation_resume_contract,
        load_prompt_records,
        resolve_generation_paths,
    )

    prompts = _write_prompts(tmp_path / "prompts.jsonl")
    lora = tmp_path / "adapter"
    lora.mkdir()
    weights = lora / "adapter.safetensors"
    weights.write_bytes(b"original-weights")
    config = GenerationConfig(
        prompts=prompts,
        output_dir=tmp_path / "generated",
        lora_path=str(lora),
    )
    paths = resolve_generation_paths(config.output_dir)
    records = load_prompt_records(prompts)
    manifest_path = ensure_generation_resume_contract(config, paths, records)
    original_manifest = manifest_path.read_bytes()

    weights.write_bytes(b"changed-weights")
    with pytest.raises(GenerationResumeError, match=r"lora\.sha256"):
        ensure_generation_resume_contract(config, paths, records)

    assert manifest_path.read_bytes() == original_manifest


def test_generation_resume_rejects_artifacts_without_manifest(tmp_path: Path) -> None:
    from src.generation.pipeline import (
        GenerationConfig,
        GenerationResumeError,
        ensure_generation_resume_contract,
        load_prompt_records,
        resolve_generation_paths,
    )

    prompts = _write_prompts(tmp_path / "prompts.jsonl")
    config = GenerationConfig(prompts=prompts, output_dir=tmp_path / "generated")
    paths = resolve_generation_paths(config.output_dir)
    stale_image = paths.images_dir / "000000" / "v0.png"
    stale_image.parent.mkdir(parents=True)
    stale_image.write_bytes(b"stale")

    with pytest.raises(GenerationResumeError, match="immutable manifest is missing"):
        ensure_generation_resume_contract(config, paths, load_prompt_records(prompts))

    assert stale_image.read_bytes() == b"stale"
    assert not paths.manifest_path.exists()


def test_generation_resume_rejects_undeclared_artifact_layout(tmp_path: Path) -> None:
    from src.generation.pipeline import (
        GenerationConfig,
        GenerationResumeError,
        ensure_generation_resume_contract,
        load_prompt_records,
        resolve_generation_paths,
    )

    prompts = _write_prompts(tmp_path / "prompts.jsonl")
    config = GenerationConfig(
        prompts=prompts,
        output_dir=tmp_path / "generated",
        start_idx=1,
        end_idx=3,
        versions_per_prompt=2,
    )
    paths = resolve_generation_paths(config.output_dir)
    records = load_prompt_records(prompts, start_idx=1, end_idx=3)
    manifest_path = ensure_generation_resume_contract(config, paths, records)
    original_manifest = manifest_path.read_bytes()
    stale_image = paths.images_dir / "000000" / "v9.png"
    stale_image.parent.mkdir(parents=True)
    stale_image.write_bytes(b"wrong-slice-and-version")

    with pytest.raises(GenerationResumeError, match="undeclared (directory|artifact)"):
        ensure_generation_resume_contract(config, paths, records)

    assert manifest_path.read_bytes() == original_manifest
    assert stale_image.read_bytes() == b"wrong-slice-and-version"


def test_run_generation_rejects_stale_resume_before_loading_model_stack(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from src.generation.pipeline import (
        GenerationConfig,
        GenerationResumeError,
        ensure_generation_resume_contract,
        load_prompt_records,
        resolve_generation_paths,
        run_generation,
    )
    from src.runtime import capabilities

    prompts = _write_prompts(tmp_path / "prompts.jsonl")
    config = GenerationConfig(
        prompts=prompts,
        output_dir=tmp_path / "generated",
        model_revision="revision-a",
        seed=17,
    )
    paths = resolve_generation_paths(config.output_dir)
    records = load_prompt_records(prompts)
    manifest_path = ensure_generation_resume_contract(config, paths, records)
    original_manifest = manifest_path.read_bytes()
    monkeypatch.setattr(
        capabilities,
        "check_stage_support",
        lambda *_args, **_kwargs: SimpleNamespace(ok=True, errors=()),
    )
    for module_name in ("diffusers", "src.training.flux2_utils"):
        sys.modules.pop(module_name, None)

    with pytest.raises(GenerationResumeError, match=r"seed: expected 18, found 17"):
        run_generation(replace(config, seed=18))

    assert manifest_path.read_bytes() == original_manifest
    assert "diffusers" not in sys.modules
    assert "src.training.flux2_utils" not in sys.modules


def test_generate_images_script_imports_pipeline_run_generation() -> None:
    script = Path("scripts/generate_images.py").read_text(encoding="utf-8")

    assert "from src.generation.pipeline import" in script
    assert "run_generation" in script
