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
