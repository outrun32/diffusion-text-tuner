"""CPU-safe runtime config loading and validation helpers."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, ValidationInfo, field_validator

from src.training.config import (
    DPOConfig,
    LoraConfig,
    MaskedSFTConfig,
    MultiRankLoraConfig,
    SFTConfig,
)

SCHEMA_VERSION = "runtime-config/v1"
SUPPORTED_MODEL_ID = "black-forest-labs/FLUX.2-klein-base-4B"
ALLOWED_RELATIVE_ROOTS = frozenset({"configs", "data", "outputs", "runs"})
ALLOWED_MIXED_PRECISION = frozenset({"no", "fp16", "bf16"})
REPO_ROOT = Path(__file__).resolve().parents[2]


class RuntimeConfigError(ValueError):
    """Raised when a runtime config cannot be safely loaded."""


def validate_path_policy(
    value: str | None,
    *,
    field_name: str,
    config_path: str | Path,
    allow_environment_input: bool = False,
) -> str | None:
    """Validate committed config paths without touching the filesystem.

    Relative paths under canonical runtime roots are allowed. Traversal, home expansion, and
    absolute paths outside the repository are rejected unless the caller explicitly identifies the
    field as an environment-provided input.
    """

    if value is None:
        return None
    if allow_environment_input:
        return value

    path = Path(value)
    config_path = Path(config_path)
    if value.startswith("~"):
        raise _runtime_error(config_path, field_name, "home-directory paths are not allowed")
    if ".." in path.parts:
        raise _runtime_error(config_path, field_name, "path traversal is not allowed")
    if path.is_absolute():
        try:
            path.resolve().relative_to(REPO_ROOT)
        except ValueError as exc:
            raise _runtime_error(
                config_path,
                field_name,
                "absolute paths must stay inside the repository",
            ) from exc
        return value
    if not path.parts or path.parts[0] not in ALLOWED_RELATIVE_ROOTS:
        raise _runtime_error(
            config_path,
            field_name,
            "relative paths must start with configs/, data/, outputs/, or runs/",
        )
    return value


def load_stage_config(stage: str, path: str | Path) -> SFTConfig | DPOConfig | MaskedSFTConfig:
    """Load and validate a stage JSON config, returning existing trainer dataclasses."""

    config_path = Path(path)
    data = _read_config_json(config_path)
    model_type = _stage_model(stage, config_path)
    try:
        validated = model_type.model_validate(data, context={"config_path": config_path})
    except ValidationError as exc:
        raise _validation_error(config_path, exc) from exc
    return validated.to_dataclass()


def resolve_config_snapshot(config: SFTConfig | DPOConfig | MaskedSFTConfig) -> dict[str, Any]:
    """Return a sorted, JSON-serializable immutable snapshot without mutating the dataclass."""

    if not is_dataclass(config):
        raise TypeError("config must be one of the training config dataclasses")

    stage = _stage_from_dataclass(config)
    snapshot = {
        "schema_version": SCHEMA_VERSION,
        "stage": stage,
        **asdict(config),
    }
    return _sort_mapping(snapshot)


class _StrictRuntimeModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    @field_validator("model_id", mode="after", check_fields=False)
    @classmethod
    def _validate_model_id(cls, value: str) -> str:
        if value != SUPPORTED_MODEL_ID:
            raise ValueError("unsupported model ID; expected configured FLUX.2 Klein base model")
        return value

    @field_validator("mixed_precision", mode="after", check_fields=False)
    @classmethod
    def _validate_mixed_precision(cls, value: str) -> str:
        if value not in ALLOWED_MIXED_PRECISION:
            raise ValueError("mixed precision must be one of no, fp16, or bf16")
        return value

    @field_validator(
        "latents_dir",
        "text_embeds_dir",
        "scores_csv",
        "selected_samples_path",
        "preference_pairs_path",
        "output_dir",
        "resume_lora_path",
        "sft_lora_path",
        "data_dir",
        "eval_suite_path",
        mode="after",
        check_fields=False,
    )
    @classmethod
    def _validate_path_fields(cls, value: str | None, info: ValidationInfo) -> str | None:
        context = info.context if isinstance(info.context, dict) else {}
        config_path = context.get("config_path", "<unknown config>")
        return validate_path_policy(
            value,
            field_name=info.field_name or "path",
            config_path=config_path,
        )


class _LoraModel(_StrictRuntimeModel):
    r: int = Field(..., gt=0)
    lora_alpha: int = Field(..., gt=0)
    target_modules: list[str] = Field(..., min_length=1)

    def to_dataclass(self) -> LoraConfig:
        return LoraConfig(**self.model_dump())


class _MultiRankLoraModel(_StrictRuntimeModel):
    attn_r: int = Field(..., ge=0)
    attn_alpha: int = Field(..., ge=0)
    ffn_r: int = Field(0, ge=0)
    ffn_alpha: int = Field(0, ge=0)
    joint_attn_r: int = Field(..., ge=0)
    joint_attn_alpha: int = Field(..., ge=0)
    dropout: float = Field(0.0, ge=0.0, le=1.0)
    attn_modules: list[str] = Field(
        default_factory=lambda: ["to_q", "to_k", "to_v", "to_out.0"], min_length=1
    )
    ffn_modules: list[str] = Field(
        default_factory=lambda: [
            "ff.linear_in",
            "ff.linear_out",
            "ff_context.linear_in",
            "ff_context.linear_out",
        ]
    )
    joint_attn_modules: list[str] = Field(
        default_factory=lambda: ["add_q_proj", "add_k_proj", "add_v_proj"], min_length=1
    )

    def to_dataclass(self) -> MultiRankLoraConfig:
        return MultiRankLoraConfig(**self.model_dump())


class _SFTModel(_StrictRuntimeModel):
    model_id: str
    latents_dir: str
    text_embeds_dir: str
    scores_csv: str
    score_threshold: float = Field(0.3, ge=0.0, le=1.0)
    selection_mode: Literal[
        "hard_positive", "score_weighted", "threshold", "top_k_per_prompt"
    ] = "threshold"
    selected_samples_path: str | None = None
    score_column: str = "score"
    hard_negative_threshold: float = Field(0.2, ge=0.0, le=1.0)
    sample_weighting: Literal["score_normalized", "uniform"] = "uniform"
    num_training_steps: int = Field(..., gt=0)
    batch_size: int = Field(..., gt=0)
    gradient_accumulation_steps: int = Field(..., gt=0)
    lr: float = Field(..., gt=0.0)
    weight_decay: float = Field(0.0, ge=0.0)
    max_grad_norm: float = Field(1.0, gt=0.0)
    warmup_steps: int = Field(0, ge=0)
    seed: int
    resume_lora_path: str | None = None
    resume_step: int = Field(0, ge=0)
    num_train_timesteps: int = Field(..., gt=0)
    shift: float = Field(..., gt=0.0)
    resolution: int = Field(..., gt=0)
    lora: _LoraModel
    sample_prompt: str = SFTConfig.sample_prompt
    sample_target_text: str = SFTConfig.sample_target_text
    sample_interval: int = Field(200, ge=0)
    eval_suite_path: str | None = None
    eval_suite_n_per_step: int = Field(0, ge=0)
    num_inference_steps: int = Field(28, gt=0)
    log_interval: int = Field(..., gt=0)
    save_interval: int = Field(..., gt=0)
    output_dir: str
    experiment_name: str
    gradient_checkpointing: bool = True
    mixed_precision: str

    def to_dataclass(self) -> SFTConfig:
        payload = self.model_dump()
        payload["lora"] = self.lora.to_dataclass()
        return SFTConfig(**payload)


class _DPOModel(_StrictRuntimeModel):
    model_id: str
    sft_lora_path: str | None = None
    latents_dir: str
    text_embeds_dir: str
    scores_csv: str
    score_threshold: float = Field(0.5, ge=0.0, le=1.0)
    score_diff_min: float = Field(0.1, gt=0.0, le=1.0)
    pair_construction_mode: Literal[
        "all_separated_pairs", "ambiguity_filtered", "best_vs_worst", "margin_weighted"
    ] = "best_vs_worst"
    preference_pairs_path: str | None = None
    score_column: str = "score"
    ambiguity_margin: float = Field(0.0, ge=0.0, le=1.0)
    pair_weighting: Literal["margin_normalized", "uniform"] = "uniform"
    num_training_steps: int = Field(..., gt=0)
    batch_size: int = Field(..., gt=0)
    gradient_accumulation_steps: int = Field(..., gt=0)
    lr: float = Field(..., gt=0.0)
    weight_decay: float = Field(0.0, ge=0.0)
    max_grad_norm: float = Field(1.0, gt=0.0)
    warmup_steps: int = Field(0, ge=0)
    seed: int
    beta: float = Field(..., gt=0.0)
    num_train_timesteps: int = Field(..., gt=0)
    shift: float = Field(..., gt=0.0)
    resolution: int = Field(..., gt=0)
    lora: _LoraModel
    sample_prompt: str = DPOConfig.sample_prompt
    sample_target_text: str = DPOConfig.sample_target_text
    sample_interval: int = Field(200, ge=0)
    eval_suite_path: str | None = None
    eval_suite_n_per_step: int = Field(0, ge=0)
    num_inference_steps: int = Field(28, gt=0)
    log_interval: int = Field(..., gt=0)
    save_interval: int = Field(..., gt=0)
    output_dir: str
    experiment_name: str
    gradient_checkpointing: bool = True
    mixed_precision: str

    def to_dataclass(self) -> DPOConfig:
        payload = self.model_dump()
        payload["lora"] = self.lora.to_dataclass()
        return DPOConfig(**payload)


class _MaskedSFTModel(_StrictRuntimeModel):
    model_id: str
    data_dir: str
    val_n_samples: int = Field(200, ge=0)
    num_training_steps: int = Field(..., gt=0)
    batch_size: int = Field(..., gt=0)
    gradient_accumulation_steps: int = Field(..., gt=0)
    lr: float = Field(..., gt=0.0)
    lr_min: float = Field(1e-6, ge=0.0)
    lr_schedule: Literal["cosine", "constant"] = "cosine"
    weight_decay: float = Field(0.0, ge=0.0)
    max_grad_norm: float = Field(1.0, gt=0.0)
    warmup_steps: int = Field(0, ge=0)
    seed: int
    resume_lora_path: str | None = None
    resume_step: int = Field(0, ge=0)
    num_train_timesteps: int = Field(..., gt=0)
    shift: float = Field(..., gt=0.0)
    masked_lambda: float = Field(..., ge=0.0, le=1.0)
    resolution: int = Field(..., gt=0)
    lora: _MultiRankLoraModel
    sample_prompt: str = MaskedSFTConfig.sample_prompt
    sample_target_text: str = MaskedSFTConfig.sample_target_text
    sample_interval: int = Field(0, ge=0)
    num_inference_steps: int = Field(28, gt=0)
    validation_interval: int = Field(250, gt=0)
    val_t_anchors: list[int] = Field(default_factory=lambda: [100, 300, 500, 700, 900])
    eval_suite_path: str | None = None
    eval_suite_n_per_step: int = Field(4, gt=0)
    log_interval: int = Field(..., gt=0)
    save_interval: int = Field(..., gt=0)
    output_dir: str
    experiment_name: str
    progress_bar_mininterval: float = Field(30.0, gt=0.0)
    gradient_checkpointing: bool = True
    mixed_precision: str

    def to_dataclass(self) -> MaskedSFTConfig:
        payload = self.model_dump()
        payload["lora"] = self.lora.to_dataclass()
        return MaskedSFTConfig(**payload)


def _read_config_json(config_path: Path) -> dict[str, Any]:
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except json.JSONDecodeError as exc:
        raise _runtime_error(config_path, "json", "malformed JSON") from exc
    except OSError as exc:
        raise _runtime_error(config_path, "path", "could not read config file") from exc

    if not isinstance(data, dict):
        raise _runtime_error(config_path, "json", "top-level JSON value must be an object")
    return data


def _stage_model(stage: str, config_path: Path) -> type[_SFTModel | _DPOModel | _MaskedSFTModel]:
    if stage == "sft":
        return _SFTModel
    if stage == "dpo":
        return _DPOModel
    if stage == "masked_sft":
        return _MaskedSFTModel
    raise _runtime_error(config_path, "stage", "stage must be one of sft, dpo, or masked_sft")


def _stage_from_dataclass(config: SFTConfig | DPOConfig | MaskedSFTConfig) -> str:
    if isinstance(config, SFTConfig):
        return "sft"
    if isinstance(config, DPOConfig):
        return "dpo"
    if isinstance(config, MaskedSFTConfig):
        return "masked_sft"
    raise TypeError("unsupported config dataclass")


def _validation_error(config_path: Path, exc: ValidationError) -> RuntimeConfigError:
    field_messages = []
    for error in exc.errors(include_url=False, include_context=False, include_input=False):
        loc = ".".join(str(part) for part in error.get("loc", ())) or "config"
        msg = error.get("msg", "invalid value")
        field_messages.append(f"{loc}: {msg}")
    details = "; ".join(field_messages) or "invalid config"
    return _runtime_error(config_path, "validation", details)


def _runtime_error(config_path: Path, field_name: str, reason: str) -> RuntimeConfigError:
    return RuntimeConfigError(f"{config_path}: {field_name}: {reason}")


def _sort_mapping(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _sort_mapping(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_sort_mapping(item) for item in value]
    if isinstance(value, tuple):
        return [_sort_mapping(item) for item in value]
    return value
