"""Prompt curriculum configuration contracts.

This module is intentionally CPU-safe and dependency-light. It validates local JSON
contracts for prompt generation without importing model, OCR, CUDA, or synthesis
libraries.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.prompt_pipeline.config import CASE_WEIGHTS, CONTENT_TYPES, TIER_WEIGHTS


SCHEMA_VERSION = "prompt-generation/v1"
ALLOWED_BACKENDS = frozenset({"transformers", "mlx", "vllm"})
ALLOWED_SCRIPTS = frozenset({"cyrillic", "latin", "digits", "punctuation", "mixed"})
ALLOWED_STAGE_FAMILIES = frozenset(
    {
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
)
MAX_SAMPLE_COUNT = 1_000_000
MAX_BATCH_SIZE = 1024


class CurriculumConfigError(ValueError):
    """Raised when a prompt curriculum config is malformed or unsafe."""


@dataclass(frozen=True)
class GenerationSettings:
    """Prompt generation runtime settings parsed from config JSON."""

    n: int
    no_llm: bool = False
    model: str = "Qwen/Qwen3.5-4B"
    backend: str = "transformers"
    batch_size: int = 1
    temperature: float = 0.7
    expand_scenes: int = 0


@dataclass(frozen=True)
class CurriculumStage:
    """Named curriculum stage for text-rendering prompt generation."""

    name: str
    family: str
    weight: float = 1.0
    sample_count: int | None = None
    scripts: tuple[str, ...] = ("cyrillic",)
    content_types: tuple[str, ...] = field(default_factory=tuple)
    tiers: tuple[int, ...] = field(default_factory=tuple)
    cases: tuple[str, ...] = field(default_factory=tuple)
    languages: tuple[str, ...] = ("ru",)


@dataclass(frozen=True)
class PromptGenerationConfig:
    """Frozen prompt-generation config contract loaded from JSON."""

    schema_version: str
    mode: str
    seed: int
    output_path: Path
    generation: GenerationSettings
    curriculum_stages: tuple[CurriculumStage, ...]
    validation_thresholds: dict[str, Any] = field(default_factory=dict)
    source_path: Path | None = None

    def stage_names(self) -> tuple[str, ...]:
        return tuple(stage.name for stage in self.curriculum_stages)

    def stage_families(self) -> set[str]:
        return {stage.family for stage in self.curriculum_stages}

    def allocate_stage_samples(self) -> dict[str, int]:
        """Allocate sample counts deterministically across named stages."""

        explicit = {stage.name: stage.sample_count for stage in self.curriculum_stages}
        explicit_total = sum(count for count in explicit.values() if count is not None)
        remaining = self.generation.n - explicit_total
        if remaining < 0:
            raise CurriculumConfigError("curriculum_stages sample_count exceeds generation.n")

        weighted_stages = [stage for stage in self.curriculum_stages if explicit[stage.name] is None]
        allocations: dict[str, int] = {
            name: int(count) for name, count in explicit.items() if count is not None
        }
        if not weighted_stages:
            return {stage.name: allocations.get(stage.name, 0) for stage in self.curriculum_stages}

        total_weight = sum(stage.weight for stage in weighted_stages)
        raw = [(stage, remaining * stage.weight / total_weight) for stage in weighted_stages]
        for stage, value in raw:
            allocations[stage.name] = int(value)

        remainder = remaining - sum(allocations[stage.name] for stage in weighted_stages)
        ranked = sorted(
            raw,
            key=lambda item: (
                -(item[1] - int(item[1])),
                _allocation_priority(item[0].family),
                item[0].name,
            ),
        )
        for stage, _value in ranked[:remainder]:
            allocations[stage.name] += 1

        return {stage.name: allocations.get(stage.name, 0) for stage in self.curriculum_stages}

    def expanded_stages(self) -> tuple[tuple[CurriculumStage, int], ...]:
        allocations = self.allocate_stage_samples()
        return tuple((stage, allocations[stage.name]) for stage in self.curriculum_stages)

    def to_generator_settings(self) -> dict[str, Any]:
        """Expose values that map onto existing prompt generator constants."""

        return {
            "content_type_weights": dict(CONTENT_TYPES),
            "tier_weights": dict(TIER_WEIGHTS),
            "case_weights": dict(CASE_WEIGHTS),
            "mode": self.mode,
            "stage_allocations": self.allocate_stage_samples(),
        }


def load_prompt_generation_config(path: str | Path) -> PromptGenerationConfig:
    """Load and validate a prompt generation config from JSON."""

    config_path = Path(path)
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise CurriculumConfigError(f"config path not readable: {config_path}") from exc
    except json.JSONDecodeError as exc:
        raise CurriculumConfigError(f"config path has invalid JSON: {config_path}") from exc

    if not isinstance(payload, dict):
        raise CurriculumConfigError("config root must be a JSON object")

    config = _parse_prompt_generation_config(payload, source_path=config_path)
    return config


def _parse_prompt_generation_config(
    payload: dict[str, Any], *, source_path: Path | None = None
) -> PromptGenerationConfig:
    schema_version = _required_str(payload, "schema_version")
    if schema_version != SCHEMA_VERSION:
        raise CurriculumConfigError(f"schema_version must be {SCHEMA_VERSION}")

    mode = _required_str(payload, "mode")
    seed = _required_int(payload, "seed", minimum=0)
    output_path = _validate_output_path(_required_str(payload, "output_path"))
    generation = _parse_generation(payload.get("generation"))
    stages = _parse_stages(payload.get("curriculum_stages"))
    thresholds = payload.get("validation_thresholds", {})
    if not isinstance(thresholds, dict):
        raise CurriculumConfigError("validation_thresholds must be an object")

    return PromptGenerationConfig(
        schema_version=schema_version,
        mode=mode,
        seed=seed,
        output_path=output_path,
        generation=generation,
        curriculum_stages=tuple(stages),
        validation_thresholds=dict(thresholds),
        source_path=source_path,
    )


def _parse_generation(raw: Any) -> GenerationSettings:
    if not isinstance(raw, dict):
        raise CurriculumConfigError("generation must be an object")
    n = _required_int(raw, "n", minimum=0, maximum=MAX_SAMPLE_COUNT)
    no_llm = bool(raw.get("no_llm", False))
    model = _str_value(raw.get("model", "Qwen/Qwen3.5-4B"), "generation.model")
    backend = _str_value(raw.get("backend", "transformers"), "generation.backend")
    if backend not in ALLOWED_BACKENDS:
        raise CurriculumConfigError("generation.backend must be one of transformers, mlx, vllm")
    batch_size = _int_value(raw.get("batch_size", 1), "generation.batch_size", minimum=1, maximum=MAX_BATCH_SIZE)
    temperature = _float_value(raw.get("temperature", 0.7), "generation.temperature", minimum=0.0)
    expand_scenes = _int_value(raw.get("expand_scenes", 0), "generation.expand_scenes", minimum=0)
    return GenerationSettings(
        n=n,
        no_llm=no_llm,
        model=model,
        backend=backend,
        batch_size=batch_size,
        temperature=temperature,
        expand_scenes=expand_scenes,
    )


def _parse_stages(raw: Any) -> list[CurriculumStage]:
    if not isinstance(raw, list) or not raw:
        raise CurriculumConfigError("curriculum_stages must be a non-empty list")
    stages = [_parse_stage(item, index) for index, item in enumerate(raw)]
    names = [stage.name for stage in stages]
    duplicates = {name for name in names if names.count(name) > 1}
    if duplicates:
        raise CurriculumConfigError(f"curriculum_stages duplicate name: {sorted(duplicates)[0]}")
    return stages


def _parse_stage(raw: Any, index: int) -> CurriculumStage:
    if not isinstance(raw, dict):
        raise CurriculumConfigError(f"curriculum_stages[{index}] must be an object")
    name = _required_str(raw, "name", context=f"curriculum_stages[{index}]")
    family = _required_str(raw, "family", context=f"curriculum_stages[{index}]")
    if family not in ALLOWED_STAGE_FAMILIES:
        raise CurriculumConfigError(f"curriculum_stages[{index}].family unsupported: {family}")
    weight = _float_value(raw.get("weight", 1.0), f"curriculum_stages[{index}].weight", minimum=0.0)
    if weight <= 0:
        raise CurriculumConfigError(f"curriculum_stages[{index}].weight must be > 0")
    sample_count = raw.get("sample_count")
    parsed_sample_count = None
    if sample_count is not None:
        parsed_sample_count = _int_value(
            sample_count,
            f"curriculum_stages[{index}].sample_count",
            minimum=0,
            maximum=MAX_SAMPLE_COUNT,
        )

    scripts = _tuple_of_strings(raw.get("scripts", ["cyrillic"]), f"curriculum_stages[{index}].scripts")
    unsupported_scripts = set(scripts) - ALLOWED_SCRIPTS
    if unsupported_scripts:
        raise CurriculumConfigError(
            f"curriculum_stages[{index}].script unsupported: {sorted(unsupported_scripts)[0]}"
        )

    content_types = _tuple_of_strings(raw.get("content_types", []), f"curriculum_stages[{index}].content_types")
    unsupported_content_types = set(content_types) - set(CONTENT_TYPES)
    if unsupported_content_types:
        raise CurriculumConfigError(
            f"curriculum_stages[{index}].content_types unsupported: "
            f"{sorted(unsupported_content_types)[0]}"
        )
    tiers = tuple(_int_value(value, f"curriculum_stages[{index}].tiers", minimum=1, maximum=5) for value in raw.get("tiers", []))
    cases = _tuple_of_strings(raw.get("cases", []), f"curriculum_stages[{index}].cases")
    unsupported_cases = set(cases) - set(CASE_WEIGHTS)
    if unsupported_cases:
        raise CurriculumConfigError(
            f"curriculum_stages[{index}].cases unsupported: {sorted(unsupported_cases)[0]}"
        )
    languages = _tuple_of_strings(raw.get("languages", ["ru"]), f"curriculum_stages[{index}].languages")
    unsupported_languages = set(languages) - {"ru", "en"}
    if unsupported_languages:
        raise CurriculumConfigError(
            f"curriculum_stages[{index}].languages unsupported: {sorted(unsupported_languages)[0]}"
        )
    return CurriculumStage(
        name=name,
        family=family,
        weight=weight,
        sample_count=parsed_sample_count,
        scripts=scripts,
        content_types=content_types,
        tiers=tiers,
        cases=cases,
        languages=languages,
    )


def _validate_output_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute() or raw_path.startswith("~"):
        raise CurriculumConfigError("output_path must be repository-relative")
    if ".." in path.parts:
        raise CurriculumConfigError("output_path must not contain traversal")
    if path.parts and path.parts[0] not in {"data", "outputs", "runs"}:
        raise CurriculumConfigError("output_path must live under data/, outputs/, or runs/")
    return path


def _required_str(raw: dict[str, Any], key: str, *, context: str = "config") -> str:
    if key not in raw:
        raise CurriculumConfigError(f"{context}.{key} is required")
    return _str_value(raw[key], f"{context}.{key}")


def _str_value(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CurriculumConfigError(f"{field_name} must be a non-empty string")
    return value.strip()


def _required_int(
    raw: dict[str, Any], key: str, *, minimum: int | None = None, maximum: int | None = None
) -> int:
    if key not in raw:
        raise CurriculumConfigError(f"{key} is required")
    return _int_value(raw[key], key, minimum=minimum, maximum=maximum)


def _int_value(
    value: Any, field_name: str, *, minimum: int | None = None, maximum: int | None = None
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CurriculumConfigError(f"{field_name} must be an integer")
    if minimum is not None and value < minimum:
        raise CurriculumConfigError(f"{field_name} must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise CurriculumConfigError(f"{field_name} must be <= {maximum}")
    return value


def _float_value(value: Any, field_name: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CurriculumConfigError(f"{field_name} must be a number")
    parsed = float(value)
    if minimum is not None and parsed < minimum:
        raise CurriculumConfigError(f"{field_name} must be >= {minimum}")
    return parsed


def _tuple_of_strings(value: Any, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise CurriculumConfigError(f"{field_name} must be a list")
    parsed = tuple(_str_value(item, field_name) for item in value)
    return parsed


def _allocation_priority(family: str) -> int:
    priorities = {
        "single_letters": 0,
        "short_words": 1,
        "phrases": 2,
        "digits": 3,
        "style": 4,
        "scene": 5,
        "punctuation": 6,
        "mixed_case": 7,
        "multiline": 8,
    }
    return priorities.get(family, 99)
