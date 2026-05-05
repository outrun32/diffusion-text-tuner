"""Training configuration."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class LoraConfig:
    r: int = 64
    lora_alpha: int = 64
    target_modules: list[str] = field(default_factory=lambda: [
        "to_k", "to_q", "to_v", "to_out.0",
    ])


# ── SFT config ──────────────────────────────────────────────────────────────


@dataclass
class SFTConfig:
    # Model
    model_id: str = "black-forest-labs/FLUX.2-klein-base-4B"

    # Data — pre-encoded latents + text embeddings
    latents_dir: str = "outputs/latents"          # {id}/v{version}.pt
    text_embeds_dir: str = "outputs/text_embeds"   # {id}.pt
    scores_csv: str = "outputs/scores.csv"         # id,version,score,target_text
    score_threshold: float = 0.3                   # min reward to include in SFT
    selection_mode: str = "threshold"
    selected_samples_path: str | None = None
    score_column: str = "score"
    hard_negative_threshold: float = 0.2
    sample_weighting: str = "uniform"

    # Training
    num_training_steps: int = 1000
    batch_size: int = 8
    gradient_accumulation_steps: int = 1
    lr: float = 2e-5
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    warmup_steps: int = 100
    seed: int = 42

    # Flow-matching
    num_train_timesteps: int = 1000
    shift: float = 3.0

    # Resolution (used for latent shape validation)
    resolution: int = 512

    # LoRA
    lora: LoraConfig = field(default_factory=LoraConfig)

    # Logging & saving
    log_interval: int = 10
    save_interval: int = 200
    output_dir: str = "outputs/sft"
    experiment_name: str = "sft_v1"

    # Hardware
    gradient_checkpointing: bool = True
    mixed_precision: str = "bf16"


# ── DPO config ──────────────────────────────────────────────────────────────


@dataclass
class DPOConfig:
    # Model
    model_id: str = "black-forest-labs/FLUX.2-klein-base-4B"
    sft_lora_path: str | None = None  # Path to SFT LoRA checkpoint to init from

    # Data
    latents_dir: str = "outputs/latents"
    text_embeds_dir: str = "outputs/text_embeds"
    scores_csv: str = "outputs/scores.csv"
    score_threshold: float = 0.5     # winner must be above this
    score_diff_min: float = 0.1      # min score diff between winner and loser
    pair_construction_mode: str = "best_vs_worst"
    preference_pairs_path: str | None = None
    score_column: str = "score"
    ambiguity_margin: float = 0.0
    pair_weighting: str = "uniform"

    # Training
    num_training_steps: int = 1000
    batch_size: int = 4
    gradient_accumulation_steps: int = 1
    lr: float = 1e-4
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    warmup_steps: int = 100
    seed: int = 42

    # DPO
    beta: float = 5000.0             # DPO beta (time-dependent scaling applied)

    # Flow-matching
    num_train_timesteps: int = 1000
    shift: float = 3.0

    resolution: int = 512

    # LoRA
    lora: LoraConfig = field(default_factory=LoraConfig)

    # Logging & saving
    log_interval: int = 10
    save_interval: int = 100
    output_dir: str = "outputs/dpo"
    experiment_name: str = "dpo_v1"

    # Hardware
    gradient_checkpointing: bool = True
    mixed_precision: str = "bf16"


@dataclass
class ReflConfig:
    # Model
    model_id: str = "black-forest-labs/FLUX.2-klein-base-4B"
    vlm_model_id: str = "Qwen/Qwen3.5-9B"
    is_distilled: bool = False  # True for klein-4B, False for klein-4B-Base

    # Data
    prompts_path: str = "data/prompts_llm.jsonl"
    text_embeds_dir: str = "outputs/text_embeds"
    num_samples: int | None = None  # None = use all

    # Training
    num_training_steps: int = 1000
    batch_size: int = 1
    gradient_accumulation_steps: int = 1
    lr: float = 2e-5
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    warmup_steps: int = 50
    seed: int = 42

    # ReFL specifics — adapted per model variant
    # Distilled: 4 steps, guidance=1.0; Base: 50 steps, guidance=4.0
    num_inference_steps: int = 50
    guidance_scale: float = 4.0
    steps_with_grad_min: int = 40
    steps_with_grad_max: int = 45
    resolution: int = 512  # Train at 512 to save VRAM, eval at 1024

    # Reward weights
    alpha_vlm: float = 1.0
    beta_siglip: float = 0.0  # Off by default, enable later
    gamma_hps: float = 0.0

    # LoRA
    lora: LoraConfig = field(default_factory=LoraConfig)

    # Eval samples — fixed prompt/target pairs for visual monitoring across training.
    # Copy from training data or write your own; no dataset indices needed.
    eval_prompts: list[dict] = field(default_factory=lambda: [
        {
            "prompt": "Концертный постер с абстрактными световыми лучами и дымкой на тёмном фоне. Небольшой текст 'ОТЪЕЗДОМ СЮРПРИЗЫ', оранжевым курсивным антиквенным шрифтом, с градиентом",
            "target_text": "ОТЪЕЗДОМ СЮРПРИЗЫ",
        },
        {
            "prompt": "An advertising poster with a background of bright geometric shapes and gradients with clearly readable text 'стэном чёрная', italic serif font, deep purple color, engraved",
            "target_text": "стэном чёрная",
        },
        {
            "prompt": "Тёмный фон с тонкими светящимися линиями и частицами. Надпись 'Превращаем ненависть в эстетику', цвет: orange, шрифт: calligraphic, с обводкой. Высокое качество",
            "target_text": "Превращаем ненависть в эстетику",
        },
        {
            "prompt": "A gradient background from dark blue to purple with tiny stars. Medium text 'Свобода радиостанции для мятежников' in red decorative font, watercolor style",
            "target_text": "Свобода радиостанции для мятежников",
        },
    ])

    # Logging & saving
    log_interval: int = 1
    save_interval: int = 100
    output_dir: str = "outputs/refl"
    experiment_name: str = "refl_v1"

    # Hardware
    gradient_checkpointing: bool = True
    mixed_precision: str = "bf16"
