import json
import re
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from src.data_quality.curriculum import (
    CurriculumConfigError,
    load_prompt_generation_config,
)
from src.prompt_pipeline import generate


def _write_config(path: Path, **overrides) -> Path:
    payload = {
        "schema_version": "prompt-generation/v1",
        "mode": "unit-test",
        "seed": 123,
        "output_path": "data/prompts/unit_test.jsonl",
        "generation": {
            "n": 24,
            "no_llm": True,
            "model": "Qwen/Qwen3.5-4B",
            "backend": "transformers",
            "batch_size": 1,
            "temperature": 0.7,
            "expand_scenes": 0,
        },
        "curriculum_stages": [
            {"name": "single_letters", "family": "single_letters", "weight": 1},
            {"name": "short_words", "family": "short_words", "weight": 1},
            {"name": "phrases", "family": "phrases", "weight": 1},
            {"name": "digits", "family": "digits", "weight": 1},
            {"name": "punctuation", "family": "punctuation", "weight": 1},
            {"name": "mixed_case", "family": "mixed_case", "weight": 1},
            {"name": "multiline", "family": "multiline", "weight": 1},
            {"name": "style_heavy", "family": "style", "weight": 1},
            {"name": "scene_heavy", "family": "scene", "weight": 1},
        ],
        "validation_thresholds": {
            "min_rare_char_coverage": 0.05,
            "max_duplicate_rate": 0.1,
        },
    }
    payload.update(overrides)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_loads_frozen_prompt_generation_config_without_heavy_imports(tmp_path):
    before = set(sys.modules)
    config = load_prompt_generation_config(_write_config(tmp_path / "prompt.json"))
    after = set(sys.modules)

    assert config.mode == "unit-test"
    assert config.seed == 123
    assert config.generation.n == 24
    assert config.output_path == Path("data/prompts/unit_test.jsonl")
    assert config.curriculum_stages[0].name == "single_letters"
    assert config.allocate_stage_samples() == {
        "single_letters": 3,
        "short_words": 3,
        "phrases": 3,
        "digits": 3,
        "punctuation": 2,
        "mixed_case": 2,
        "multiline": 2,
        "style_heavy": 3,
        "scene_heavy": 3,
    }

    heavy_modules = {"diffusers", "transformers", "paddleocr", "vllm", "mlx_lm", "synthtiger"}
    assert not heavy_modules & (after - before)
    with pytest.raises(FrozenInstanceError):
        config.seed = 99


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("curriculum_stages", [{"name": "bad", "family": "short_words", "weight": 0}], "weight"),
        ("curriculum_stages", [{"name": " ", "family": "short_words", "weight": 1}], "name"),
        (
            "curriculum_stages",
            [{"name": "bad", "family": "short_words", "weight": 1, "scripts": ["emoji"]}],
            "script",
        ),
        (
            "curriculum_stages",
            [{"name": "bad", "family": "single_letters", "weight": 1, "scripts": ["mixed"]}],
            "cannot use mixed",
        ),
        ("generation", {"n": -1, "no_llm": True}, "n"),
        ("output_path", "~/private/prompts.jsonl", "output_path"),
    ],
)
def test_rejects_invalid_curriculum_config_values(tmp_path, field, value, message):
    path = _write_config(tmp_path / "invalid.json", **{field: value})

    with pytest.raises(CurriculumConfigError, match=message):
        load_prompt_generation_config(path)


def test_exposes_required_data_01_stage_families(tmp_path):
    config = load_prompt_generation_config(_write_config(tmp_path / "prompt.json"))

    assert config.stage_families() == {
        "single_letters",
        "short_words",
        "phrases",
        "digits",
        "punctuation",
        "mixed_case",
        "multiline",
        "style",
        "scene",
    }
    generator_settings = config.to_generator_settings()
    assert generator_settings["content_type_weights"]["poster"] > 0
    assert generator_settings["tier_weights"][1] > 0
    assert generator_settings["case_weights"]["upper"] > 0


def test_committed_simple_full_and_curriculum_configs_load():
    configs = {
        "simple": load_prompt_generation_config("configs/prompts/simple.json"),
        "full": load_prompt_generation_config("configs/prompts/full.json"),
        "curriculum": load_prompt_generation_config("configs/prompts/curriculum.json"),
    }

    assert configs["simple"].mode == "simple"
    assert configs["simple"].generation.no_llm is True
    assert configs["simple"].output_path == Path("data/prompts/simple.jsonl")
    assert {stage.family for stage in configs["simple"].curriculum_stages} >= {
        "single_letters",
        "short_words",
    }

    full = configs["full"]
    assert full.mode == "full"
    assert full.generation.model == "Qwen/Qwen3.5-4B"
    assert full.generation.backend == "transformers"
    assert full.generation.no_llm is False
    assert {stage.family for stage in full.curriculum_stages} >= {"phrases", "style", "scene"}

    curriculum = configs["curriculum"]
    assert curriculum.mode == "curriculum"
    assert curriculum.allocate_stage_samples()["single_letters"] > 0
    assert curriculum.stage_families() == {
        "single_letters",
        "short_words",
        "phrases",
        "digits",
        "punctuation",
        "mixed_case",
        "multiline",
        "style",
        "scene",
    }


def test_committed_configs_include_required_contract_fields():
    for path in sorted(Path("configs/prompts").glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["schema_version"] == "prompt-generation/v1"
        assert payload["mode"] in {"simple", "full", "curriculum"}
        assert isinstance(payload["seed"], int)
        assert payload["output_path"].startswith("data/prompts/")
        assert "generation" in payload
        assert "curriculum_stages" in payload and payload["curriculum_stages"]
        assert "validation_thresholds" in payload


def test_generate_cli_uses_config_values_without_llm_import(monkeypatch, tmp_path):
    output_path = Path("data/prompts/cli_config_test.jsonl")
    config_path = _write_config(
        tmp_path / "cli_config.json",
        mode="cli-config-test",
        output_path=str(output_path),
        generation={
            "n": 7,
            "no_llm": True,
            "model": "Qwen/Qwen3.5-4B",
            "backend": "transformers",
            "batch_size": 3,
            "temperature": 0.2,
            "expand_scenes": 0,
        },
    )
    captured = {}

    def fake_generate_dataset(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(generate, "generate_dataset", fake_generate_dataset)
    before = set(sys.modules)

    assert generate.main(["--config", str(config_path)]) == 0

    assert captured["n"] == 7
    assert captured["output_path"] == str(output_path)
    assert captured["seed"] == 123
    assert captured["batch_size"] == 3
    assert captured["prompt_config"].mode == "cli-config-test"
    assert "src.prompt_pipeline.llm_client" not in (set(sys.modules) - before)


def test_generate_cli_preserves_flag_only_defaults(monkeypatch, tmp_path):
    captured = {}

    def fake_generate_dataset(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(generate, "generate_dataset", fake_generate_dataset)
    output = tmp_path / "legacy.jsonl"

    assert generate.main(["--n", "5", "--output", str(output), "--seed", "77", "--no-llm"]) == 0

    assert captured["n"] == 5
    assert captured["output_path"] == str(output)
    assert captured["seed"] == 77
    assert captured["batch_size"] == 1
    assert captured["prompt_config"] is None


def test_invalid_config_path_fails_before_heavy_llm_import(capsys):
    before = set(sys.modules)

    status = generate.main(["--config", "configs/prompts/missing.json"])
    captured = capsys.readouterr()

    assert status == 2
    assert "config" in captured.err
    assert "configs/prompts/missing.json" in captured.err
    assert "src.prompt_pipeline.llm_client" not in (set(sys.modules) - before)


def test_generate_dataset_tags_records_with_config_stage_provenance(tmp_path):
    config = load_prompt_generation_config("configs/prompts/simple.json")
    output = tmp_path / "prompts.jsonl"

    generate.generate_dataset(
        n=4,
        output_path=str(output),
        llm=None,
        seed=config.seed,
        batch_size=1,
        prompt_config=config,
    )

    records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 4
    assert {record["prompt_mode"] for record in records} == {"simple"}
    assert {record["curriculum_stage"] for record in records} <= {"single_letters", "short_words"}
    assert all(
        record["curriculum_family"] in {"single_letters", "short_words"} for record in records
    )


def test_single_letter_curriculum_never_uses_numeric_dedup_suffixes(tmp_path):
    config_path = _write_config(
        tmp_path / "single_letters.json",
        generation={
            "n": 100,
            "no_llm": True,
            "model": "unused",
            "backend": "transformers",
            "batch_size": 1,
            "temperature": 0.0,
            "expand_scenes": 0,
        },
        curriculum_stages=[
            {
                "name": "letters",
                "family": "single_letters",
                "weight": 1,
                "scripts": ["cyrillic"],
                "content_types": ["typography"],
                "tiers": [1],
                "cases": ["upper", "lower"],
                "languages": ["ru"],
            }
        ],
    )
    config = load_prompt_generation_config(config_path)
    output = tmp_path / "letters.jsonl"

    generate.generate_dataset(
        n=100,
        output_path=str(output),
        llm=None,
        seed=config.seed,
        prompt_config=config,
    )

    records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 100
    assert all(re.fullmatch(r"[А-ЯЁа-яё]", record["target_text"]) for record in records)
    assert {record["target_script"] for record in records} == {"cyrillic"}


@pytest.mark.parametrize(
    ("script", "target_pattern"),
    [
        ("latin", r"(?=.*[A-Za-z])[^А-ЯЁа-яё]+"),
        ("mixed", r"(?=.*[A-Za-z])(?=.*[А-ЯЁа-яё]).+"),
    ],
)
def test_curriculum_honors_configured_target_script(
    tmp_path,
    script,
    target_pattern,
):
    config_path = _write_config(
        tmp_path / f"{script}.json",
        generation={
            "n": 20,
            "no_llm": True,
            "model": "unused",
            "backend": "transformers",
            "batch_size": 1,
            "temperature": 0.0,
            "expand_scenes": 0,
        },
        curriculum_stages=[
            {
                "name": f"{script}_words",
                "family": "short_words",
                "weight": 1,
                "scripts": [script],
                "content_types": ["typography"],
                "tiers": [2],
                "cases": ["lower"],
                "languages": ["ru"],
            }
        ],
    )
    config = load_prompt_generation_config(config_path)
    output = tmp_path / f"{script}.jsonl"

    generate.generate_dataset(
        n=20,
        output_path=str(output),
        llm=None,
        seed=config.seed,
        prompt_config=config,
    )

    records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert all(record["target_script"] == script for record in records)
    assert all(re.fullmatch(target_pattern, record["target_text"]) for record in records)
