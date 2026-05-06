from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path


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
    assert config.batch_size == 4
    assert config.num_inference_steps == 50
    assert config.guidance_scale == 4.0
    assert config.resolution == 512
    assert config.seed == 42
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

    assert plan_generation_seed(
        seed=42,
        prompt_index=3,
        versions_per_prompt=5,
        version=2,
    ) == 59


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


def test_generate_images_script_imports_pipeline_run_generation() -> None:
    script = Path("scripts/generate_images.py").read_text(encoding="utf-8")

    assert "from src.generation.pipeline import" in script
    assert "run_generation" in script
