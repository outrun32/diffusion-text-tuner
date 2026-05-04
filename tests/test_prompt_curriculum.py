import json
import sys
from pathlib import Path

import pytest


from src.data_quality.curriculum import (
    CurriculumConfigError,
    load_prompt_generation_config,
)


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
    with pytest.raises(Exception):
        config.seed = 99


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("curriculum_stages", [{"name": "bad", "family": "short_words", "weight": 0}], "weight"),
        ("curriculum_stages", [{"name": " ", "family": "short_words", "weight": 1}], "name"),
        ("curriculum_stages", [{"name": "bad", "family": "short_words", "weight": 1, "scripts": ["emoji"]}], "script"),
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
