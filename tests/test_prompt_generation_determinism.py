import json
import random
import sys
from pathlib import Path

from src.data_quality.curriculum import load_prompt_generation_config
from src.prompt_pipeline import generate

CONFIG_PATHS = (
    Path("configs/prompts/simple.json"),
    Path("configs/prompts/full.json"),
    Path("configs/prompts/curriculum.json"),
)


class FakeTextGenerator:
    def __init__(self, *args, **kwargs):
        self.words = ["альфа", "бета", "гамма"]
        self.coverage: dict[str, int] = {}
        self._tier_calls = 0
        self._case_calls = 0

    def sample_tier(self, content_type):
        self._tier_calls += 1
        return 1 + (self._tier_calls % 3)

    def sample_case(self):
        cases = ("upper", "title", "lower", "mixed")
        value = cases[self._case_calls % len(cases)]
        self._case_calls += 1
        return value

    def generate_text(self, tier, case):
        text = f"текст {tier}"
        if case == "upper":
            return text.upper()
        if case == "title":
            return text.title()
        return text

    def generate_text_fallback(self, tier, case):
        return self.generate_text(tier, case)

    def get_must_include_words(self, tier):
        return [f"слово{tier}"]

    def generate_number_text(self):
        return "42"

    def update_coverage(self, text):
        for character in text.lower():
            self.coverage[character] = self.coverage.get(character, 0) + 1

    def coverage_report(self):
        return dict(self.coverage)


class FakeStyleGenerator:
    def __init__(self, *args, **kwargs):
        pass

    def sample(self, content_type):
        return {"font": f"{content_type}-font", "color": "black"}


class FakeScenePool:
    def __init__(self, *args, **kwargs):
        pass

    def __len__(self):
        return 1

    def sample(self, content_type, lang):
        return {"id": f"{content_type}-{lang}", "ru": "сцена", "en": "scene"}


class FakeAssembler:
    def __init__(self, *args, **kwargs):
        pass

    def assemble(self, target_text, scene, style, content_type, lang):
        return f"{lang}:{content_type}:{target_text}:{scene['id']}:{style['font']}"


def _plan_signature(plan):
    keys = ("stage_name", "stage_family", "content_type", "tier", "case", "lang")
    return [tuple(item[key] for key in keys) for item in plan]


def _build_plan(config_path: str, seed: int, n: int | None = None):
    config = load_prompt_generation_config(config_path)
    return generate._build_generation_plan(
        n=config.generation.n if n is None else n,
        rng=random.Random(seed),
        text_gen=FakeTextGenerator(),
        prompt_config=config,
    )


def _write_config(path: Path, *, output_path: Path, n: int = 8) -> Path:
    payload = {
        "schema_version": "prompt-generation/v1",
        "mode": "determinism-test",
        "seed": 2026,
        "output_path": str(output_path),
        "generation": {
            "n": n,
            "no_llm": True,
            "model": "Qwen/Qwen3.5-4B",
            "backend": "transformers",
            "batch_size": 2,
            "temperature": 0.7,
            "expand_scenes": 0,
        },
        "curriculum_stages": [
            {
                "name": "letters",
                "family": "single_letters",
                "sample_count": 2,
                "content_types": ["typography"],
                "tiers": [1],
                "cases": ["upper"],
                "languages": ["ru"],
            },
            {
                "name": "digits",
                "family": "digits",
                "sample_count": 2,
                "content_types": ["poster"],
                "tiers": [2],
                "cases": ["title"],
                "languages": ["ru"],
            },
            {
                "name": "multiline",
                "family": "multiline",
                "weight": 1,
                "content_types": ["book_cover"],
                "tiers": [4],
                "cases": ["lower"],
                "languages": ["ru"],
            },
        ],
        "validation_thresholds": {"require_stage_provenance": True},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _patch_lightweight_generators(monkeypatch):
    monkeypatch.setattr(generate, "TextGenerator", FakeTextGenerator)
    monkeypatch.setattr(generate, "StyleGenerator", FakeStyleGenerator)
    monkeypatch.setattr(generate, "ScenePool", FakeScenePool)
    monkeypatch.setattr(generate, "Assembler", FakeAssembler)


def test_committed_prompt_configs_have_stable_stage_allocations_and_metadata():
    snapshots = {}
    for path in CONFIG_PATHS:
        first = load_prompt_generation_config(path)
        second = load_prompt_generation_config(path)

        assert first.stage_names() == second.stage_names()
        assert first.stage_families() == second.stage_families()
        assert first.allocate_stage_samples() == second.allocate_stage_samples()
        assert sum(first.allocate_stage_samples().values()) == first.generation.n
        snapshots[first.mode] = {
            stage.name: (stage.family, first.allocate_stage_samples()[stage.name])
            for stage in first.curriculum_stages
        }

    assert snapshots["simple"] == {
        "single_letters": ("single_letters", 25),
        "short_words": ("short_words", 75),
    }
    assert snapshots["full"] == {
        "broad_letters_words": ("short_words", 5455),
        "broad_phrases": ("phrases", 13636),
        "style_distribution": ("style", 5455),
        "scene_distribution": ("scene", 5454),
    }
    assert snapshots["curriculum"]["single_letters"] == ("single_letters", 857)
    assert snapshots["curriculum"]["scene_heavy"] == ("scene", 1714)


def test_build_generation_plan_is_seed_stable_and_seed_sensitive():
    same_seed_a = _build_plan("configs/prompts/curriculum.json", seed=31415, n=36)
    same_seed_b = _build_plan("configs/prompts/curriculum.json", seed=31415, n=36)
    different_seed = _build_plan("configs/prompts/curriculum.json", seed=27182, n=36)

    assert _plan_signature(same_seed_a) == _plan_signature(same_seed_b)
    assert _plan_signature(same_seed_a) != _plan_signature(different_seed)


def test_config_plan_items_are_tagged_and_fallback_items_are_explicit():
    config = load_prompt_generation_config("configs/prompts/simple.json")
    plan = generate._build_generation_plan(
        n=config.generation.n + 3,
        rng=random.Random(config.seed),
        text_gen=FakeTextGenerator(),
        prompt_config=config,
    )

    configured = plan[: config.generation.n]
    fallback = plan[config.generation.n :]

    assert all(item["stage_name"] in config.stage_names() for item in configured)
    assert all(item["stage_family"] in config.stage_families() for item in configured)
    assert [item["stage_name"] for item in fallback] == ["unallocated"] * 3
    assert [item["stage_family"] for item in fallback] == ["fallback"] * 3


def test_legacy_generation_plan_marks_metadata_as_unconfigured():
    legacy_plan = generate._build_generation_plan(
        n=4,
        rng=random.Random(42),
        text_gen=FakeTextGenerator(),
        prompt_config=None,
    )

    assert [item["stage_name"] for item in legacy_plan] == [None, None, None, None]
    assert [item["stage_family"] for item in legacy_plan] == [None, None, None, None]


def test_stage_family_text_policies_are_seeded_and_predictable():
    text_gen = FakeTextGenerator()
    single_lower = generate._apply_stage_text_policy(
        "ignored",
        {"stage_family": "single_letters", "case": "lower"},
        text_gen,
        random.Random(10),
    )
    single_upper = generate._apply_stage_text_policy(
        "ignored",
        {"stage_family": "single_letters", "case": "upper"},
        text_gen,
        random.Random(10),
    )

    assert single_lower == "в"
    assert single_upper == "В"
    assert (
        generate._apply_stage_text_policy(
            "цена", {"stage_family": "digits"}, text_gen, random.Random(1)
        )
        == "цена 42"
    )
    assert (
        generate._apply_stage_text_policy(
            "привет", {"stage_family": "punctuation"}, text_gen, random.Random(4)
        )
        == "привет?"
    )
    assert (
        generate._apply_stage_text_policy(
            "абвг", {"stage_family": "mixed_case"}, text_gen, random.Random(1)
        )
        == "АбВг"
    )
    assert (
        generate._apply_stage_text_policy(
            "первая вторая третья", {"stage_family": "multiline"}, text_gen, random.Random(1)
        )
        == "первая\nвторая третья"
    )


def test_generate_dataset_with_config_writes_deterministic_provenance_records(
    monkeypatch, tmp_path
):
    _patch_lightweight_generators(monkeypatch)
    output_path = tmp_path / "prompts.jsonl"
    config_path = _write_config(
        tmp_path / "config.json", output_path=Path("data/prompts/unit.jsonl")
    )
    config = load_prompt_generation_config(config_path)

    generate.generate_dataset(
        n=config.generation.n,
        output_path=str(output_path),
        llm=None,
        seed=config.seed,
        batch_size=config.generation.batch_size,
        prompt_config=config,
    )
    first = output_path.read_text(encoding="utf-8").splitlines()
    generate.generate_dataset(
        n=config.generation.n,
        output_path=str(output_path),
        llm=None,
        seed=config.seed,
        batch_size=config.generation.batch_size,
        prompt_config=config,
    )
    second = output_path.read_text(encoding="utf-8").splitlines()

    assert first == second
    records = [json.loads(line) for line in first]
    assert {record["prompt_mode"] for record in records} == {"determinism-test"}
    assert {record["curriculum_stage"] for record in records} == {"letters", "digits", "multiline"}
    assert {record["curriculum_family"] for record in records} == {
        "single_letters",
        "digits",
        "multiline",
    }


def test_no_llm_cli_path_avoids_heavy_backend_imports(monkeypatch, tmp_path):
    output_path = Path("data/prompts/no_llm_test.jsonl")
    config_path = _write_config(tmp_path / "config.json", output_path=output_path, n=3)
    captured = {}

    def fake_generate_dataset(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(generate, "generate_dataset", fake_generate_dataset)
    before = set(sys.modules)

    assert generate.main(["--config", str(config_path), "--no-llm"]) == 0

    newly_imported = set(sys.modules) - before
    assert captured["llm"] is None
    assert captured["prompt_config"].mode == "determinism-test"
    assert "src.prompt_pipeline.llm_client" not in newly_imported
    assert not {"vllm", "mlx_lm", "diffusers", "transformers", "torch"} & newly_imported
