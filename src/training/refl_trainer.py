"""
ReFL (Reward Feedback Learning) trainer for FLUX.2 Klein.

Trains LoRA adapters on the FLUX2Klein DiT transformer by:
1. Running the full denoising loop (no grad) up to a random late step
2. Running one step WITH grad, then jumping to x₀
3. Decoding to pixels and computing differentiable reward
4. Backprop reward → LoRA weights
"""

import csv
import gc
import json
import logging
import os
import random
import time
from pathlib import Path

import torch
from peft import LoraConfig as PeftLoraConfig
from peft import get_peft_model
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .config import ReflConfig
from .flux2_utils import (
    decode_latents,
    pack_latents,
    prepare_latent_ids,
    prepare_text_ids,
)
from .rewards import QwenYesProbReward

logger = logging.getLogger(__name__)


# ── Dataset ─────────────────────────────────────────────────────────────────


class TextEmbedDataset(Dataset):
    """Loads precomputed text embeddings + target text for training."""

    def __init__(self, embeds_dir: str, num_samples: int | None = None):
        self.embeds_dir = Path(embeds_dir)
        self.files = sorted(self.embeds_dir.glob("*.pt"))
        if num_samples is not None:
            self.files = self.files[:num_samples]
        logger.info(f"Dataset: {len(self.files)} samples from {embeds_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = torch.load(self.files[idx], map_location="cpu", weights_only=True)
        return {
            "prompt_embeds": data["prompt_embeds"],  # (seq_len, dim)
            "target_text": data["target_text"],
            "prompt": data["prompt"],
        }


# ── Scheduler with step_to_zero ─────────────────────────────────────────────


class FlowMatchScheduler:
    """
    Flow matching Euler scheduler with step_to_zero for ReFL.
    Matches the reference repo and FLUX2Klein's FlowMatchEulerDiscreteScheduler
    with shift=3.0.
    """

    def __init__(self, num_train_timesteps: int = 1000, shift: float = 3.0):
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self.timesteps = None
        self.sigmas = None
        self._step_index = None

        # Compute sigma_max / sigma_min from the shifted training schedule
        import numpy as np

        train_ts = np.linspace(1, num_train_timesteps, num_train_timesteps, dtype=np.float32)[
            ::-1
        ].copy()
        train_sigmas = train_ts / num_train_timesteps
        train_sigmas = shift * train_sigmas / (1 + (shift - 1) * train_sigmas)
        self.sigma_max = float(train_sigmas[0])  # 1.0
        self.sigma_min = float(train_sigmas[-1])  # ~0.003

    def set_timesteps(self, num_inference_steps: int, device=None):
        import numpy as np

        self.num_inference_steps = num_inference_steps
        # Linearly space in timestep domain, then shift to get sigmas
        timesteps = np.linspace(
            self.sigma_max * self.num_train_timesteps,
            self.sigma_min * self.num_train_timesteps,
            num_inference_steps,
        )
        sigmas = timesteps / self.num_train_timesteps
        sigmas = self.shift * sigmas / (1 + (self.shift - 1) * sigmas)
        sigmas = torch.from_numpy(sigmas).to(dtype=torch.float32, device=device)
        self.sigmas = torch.cat([sigmas, torch.zeros(1, device=device)])
        self.timesteps = sigmas * self.num_train_timesteps
        self._step_index = 0

    def step(self, model_output: torch.Tensor, sample: torch.Tensor) -> torch.Tensor:
        """Euler step: x_{t-1} = x_t + (sigma_next - sigma) * v_pred"""
        sigma = self.sigmas[self._step_index]
        sigma_next = self.sigmas[self._step_index + 1]
        sample = sample.float()
        prev = sample + (sigma_next - sigma) * model_output
        self._step_index += 1
        return prev.to(model_output.dtype)

    def step_to_zero(self, model_output: torch.Tensor, sample: torch.Tensor) -> torch.Tensor:
        """Jump directly to x₀: x_0 = x_t + (0 - sigma_t) * v_pred"""
        sigma = self.sigmas[self._step_index]
        sample = sample.float()
        x0 = sample + (0 - sigma) * model_output
        self._step_index += 1
        return x0.to(model_output.dtype)

    def reset(self):
        self._step_index = 0


# ── ReFL denoising loop ────────────────────────────────────────────────────


def refl_denoise(
    latents: torch.Tensor,
    latent_ids: torch.Tensor,
    prompt_embeds: torch.Tensor,
    text_ids: torch.Tensor,
    transformer,
    scheduler: FlowMatchScheduler,
    num_inference_steps: int,
    guidance_scale: float,
    step_with_grad: int,
    negative_prompt_embeds: torch.Tensor | None = None,
    negative_text_ids: torch.Tensor | None = None,
):
    """
    ReFL denoising: run all steps without grad except step_with_grad,
    then jump to x₀.

    Returns: packed latents at x₀ prediction, and the latent_ids.
    """
    do_cfg = guidance_scale > 1.0 and negative_prompt_embeds is not None
    device = latents.device

    scheduler.set_timesteps(num_inference_steps, device=device)
    scheduler.reset()

    for i in range(num_inference_steps):
        timestep = scheduler.timesteps[i]
        timestep_input = (timestep / 1000).expand(latents.shape[0]).to(latents.dtype)

        if i < step_with_grad:
            # No grad for early steps
            with torch.no_grad():
                noise_pred = transformer(
                    hidden_states=latents.to(transformer.dtype),
                    timestep=timestep_input,
                    guidance=None,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_ids,
                    return_dict=False,
                )[0]
                noise_pred = noise_pred[:, : latents.size(1)]

                if do_cfg:
                    neg_pred = transformer(
                        hidden_states=latents.to(transformer.dtype),
                        timestep=timestep_input,
                        guidance=None,
                        encoder_hidden_states=negative_prompt_embeds,
                        txt_ids=negative_text_ids,
                        img_ids=latent_ids,
                        return_dict=False,
                    )[0]
                    neg_pred = neg_pred[:, : latents.size(1)]
                    noise_pred = neg_pred + guidance_scale * (noise_pred - neg_pred)

                latents = scheduler.step(noise_pred, latents)
        else:
            # WITH grad — then jump to x₀
            noise_pred = transformer(
                hidden_states=latents.to(transformer.dtype),
                timestep=timestep_input,
                guidance=None,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                img_ids=latent_ids,
                return_dict=False,
            )[0]
            noise_pred = noise_pred[:, : latents.size(1)]

            if do_cfg:
                with torch.no_grad():
                    neg_pred = transformer(
                        hidden_states=latents.to(transformer.dtype),
                        timestep=timestep_input,
                        guidance=None,
                        encoder_hidden_states=negative_prompt_embeds,
                        txt_ids=negative_text_ids,
                        img_ids=latent_ids,
                        return_dict=False,
                    )[0]
                    neg_pred = neg_pred[:, : latents.size(1)]
                noise_pred = neg_pred + guidance_scale * (noise_pred - neg_pred)

            latents = scheduler.step_to_zero(noise_pred, latents)
            break  # We're at x₀ now

    return latents


# ── Main training function ──────────────────────────────────────────────────


def setup_logging(output_dir: str, experiment_name: str):
    log_dir = os.path.join(output_dir, experiment_name, "logs")
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, f"train_{time.strftime('%Y%m%d_%H%M%S')}.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )
    logger.info(f"Logging to {log_file}")
    return log_file


def train(cfg: ReflConfig):
    from src.runtime.capabilities import check_stage_support

    support = check_stage_support("refl", mixed_precision=cfg.mixed_precision)
    if not support.ok:
        raise RuntimeError("; ".join(support.errors))

    setup_logging(cfg.output_dir, cfg.experiment_name)

    logger.info("=" * 60)
    logger.info("ReFL Training — FLUX.2 Klein + Qwen3.5 VLM Reward")
    logger.info("=" * 60)

    device = "cuda"
    dtype = torch.bfloat16

    # ── 1. Seed ──
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)

    # ── 2. Load dataset ──
    logger.info(f"Loading dataset from {cfg.text_embeds_dir}")
    dataset = TextEmbedDataset(cfg.text_embeds_dir, num_samples=cfg.num_samples)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )

    # ── 3. Load FLUX2Klein (transformer + VAE only, no text encoder) ──
    logger.info(f"Loading FLUX2Klein model: {cfg.model_id}")
    from diffusers import Flux2KleinPipeline

    pipe = Flux2KleinPipeline.from_pretrained(
        cfg.model_id,
        revision=cfg.model_revision,
        torch_dtype=dtype,
    )
    transformer = pipe.transformer
    vae = pipe.vae.to(device, dtype=dtype)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False

    # Encode all prompts that need the text encoder before freeing it
    pipe.text_encoder.to(device)

    # Precompute negative (empty) prompt embeddings for CFG
    if cfg.guidance_scale > 1.0:
        logger.info("Precomputing negative prompt embeddings for CFG...")
        neg_prompt_embeds, neg_text_ids = pipe.encode_prompt(prompt="", device=device)
        neg_prompt_embeds = neg_prompt_embeds.detach().to(dtype)
        neg_text_ids = neg_text_ids.detach()
    else:
        neg_prompt_embeds = None
        neg_text_ids = None

    # Fixed eval prompts — encode from cfg.eval_prompts using the text encoder
    logger.info(f"Encoding {len(cfg.eval_prompts)} eval prompts...")
    eval_prompt_embeds_list = []
    eval_text_ids_list = []
    for ep in cfg.eval_prompts:
        emb, _ = pipe.encode_prompt(prompt=ep["prompt"], device=device)
        emb = emb.detach().to(dtype)
        eval_prompt_embeds_list.append(emb)
        eval_text_ids_list.append(prepare_text_ids(emb).to(device))
    EVAL_PROMPTS = cfg.eval_prompts

    # Free text encoder (huge — 8B params)
    pipe.text_encoder.to("cpu")
    del pipe.text_encoder, pipe.tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    logger.info("Text encoder freed from memory")

    # ── 4. Apply LoRA ──
    lora_config = PeftLoraConfig(
        r=cfg.lora.r,
        lora_alpha=cfg.lora.lora_alpha,
        target_modules=cfg.lora.target_modules,
        lora_dropout=0.0,
    )
    transformer = get_peft_model(transformer, lora_config)
    transformer.to(device)
    transformer.print_trainable_parameters()

    if cfg.gradient_checkpointing:
        # Enable gradient checkpointing on the underlying model
        base = transformer.get_base_model()
        if hasattr(base, "enable_gradient_checkpointing"):
            base.enable_gradient_checkpointing()

        # Ensure inputs require grad so checkpointing works
        def _make_inputs_require_grad(module, input, output):
            if isinstance(output, tuple):
                for o in output:
                    if isinstance(o, torch.Tensor) and o.is_floating_point():
                        o.requires_grad_(True)
            elif isinstance(output, torch.Tensor) and output.is_floating_point():
                output.requires_grad_(True)

        transformer.get_base_model().register_forward_hook(_make_inputs_require_grad)
        logger.info("Gradient checkpointing enabled")

    transformer.train()

    # ── 5. Load VLM reward model ──
    logger.info("Loading VLM reward model...")
    vlm_reward = QwenYesProbReward(
        model_id=cfg.vlm_model_id,
        device=device,
        revision=cfg.vlm_model_revision,
    )

    # ── 6. Optimizer & scheduler ──
    optimizer = torch.optim.AdamW(
        [p for p in transformer.parameters() if p.requires_grad],
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        betas=(0.9, 0.99),
    )
    from transformers.optimization import get_constant_schedule_with_warmup

    lr_scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=cfg.warmup_steps)

    # ── 7. Scheduler for denoising ──
    noise_scheduler = FlowMatchScheduler()

    # ── 8. Compute latent dimensions ──
    vae_scale_factor = 8
    patch_factor = 2
    latent_height = cfg.resolution // vae_scale_factor // patch_factor
    latent_width = cfg.resolution // vae_scale_factor // patch_factor
    num_latent_channels = 128  # 32 * 4 after patchify
    logger.info(
        f"Latent shape: ({cfg.batch_size}, {num_latent_channels}, {latent_height}, {latent_width})"
    )
    logger.info(
        f"Packed shape: ({cfg.batch_size}, {latent_height * latent_width}, {num_latent_channels})"
    )

    # ── 9. Training loop ──
    save_dir = os.path.join(cfg.output_dir, cfg.experiment_name)
    os.makedirs(save_dir, exist_ok=True)

    # Save config
    import dataclasses

    config_path = os.path.join(save_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(dataclasses.asdict(cfg), f, indent=2)
    logger.info(f"Config saved to {config_path}")

    # Metrics CSV
    metrics_path = os.path.join(save_dir, "metrics.csv")
    metrics_file = open(metrics_path, "w", newline="")
    metrics_writer = csv.writer(metrics_file)
    metrics_writer.writerow(["step", "loss", "reward", "grad_norm", "lr", "grad_step", "elapsed_s"])
    metrics_file.flush()

    # Sample images directory
    samples_dir = os.path.join(save_dir, "samples")
    os.makedirs(samples_dir, exist_ok=True)

    # Fixed noise per eval prompt (different seed per prompt for variety)
    eval_fixed_noises = []
    eval_fixed_latent_ids_list = []
    eval_fixed_latents_list = []
    for i in range(len(EVAL_PROMPTS)):
        gen = torch.Generator(device=device).manual_seed(42 + i)
        noise = torch.randn(
            1,
            num_latent_channels,
            latent_height,
            latent_width,
            generator=gen,
            device=device,
            dtype=dtype,
        )
        eval_fixed_noises.append(noise)
        eval_fixed_latent_ids_list.append(prepare_latent_ids(noise).to(device))
        eval_fixed_latents_list.append(pack_latents(noise))

    for ep in EVAL_PROMPTS:
        logger.info(f"  Eval prompt: {ep['prompt'][:80]}  target: {ep['target_text']}")
    with open(os.path.join(samples_dir, "eval_prompts.txt"), "w") as f:
        for i, ep in enumerate(EVAL_PROMPTS):
            f.write(f"[{i}] prompt: {ep['prompt']}\n    target: {ep['target_text']}\n")

    global_step = 0
    total_reward = 0.0
    start_time = time.time()

    logger.info(f"Starting training for {cfg.num_training_steps} steps")
    logger.info(f"  Batch size: {cfg.batch_size}")
    logger.info(f"  Gradient accumulation: {cfg.gradient_accumulation_steps}")
    logger.info(f"  Effective batch size: {cfg.batch_size * cfg.gradient_accumulation_steps}")
    logger.info(f"  Learning rate: {cfg.lr}")
    logger.info(f"  LoRA rank: {cfg.lora.r}")
    logger.info(f"  Resolution: {cfg.resolution}")
    logger.info(f"  Inference steps: {cfg.num_inference_steps}")
    logger.info(f"  Guidance scale: {cfg.guidance_scale}")
    logger.info(f"  Grad steps range: [{cfg.steps_with_grad_min}, {cfg.steps_with_grad_max})")

    accum_steps = cfg.gradient_accumulation_steps
    optimizer.zero_grad()

    while global_step < cfg.num_training_steps:
        for batch in dataloader:
            if global_step >= cfg.num_training_steps:
                break

            prompt_embeds = batch["prompt_embeds"].to(device, dtype=dtype)
            target_texts = batch["target_text"]

            # Prepare text IDs
            text_ids = prepare_text_ids(prompt_embeds).to(device)

            # Initialize random noise in patchified space
            latents_spatial = torch.randn(
                cfg.batch_size,
                num_latent_channels,
                latent_height,
                latent_width,
                device=device,
                dtype=dtype,
            )
            latent_ids = prepare_latent_ids(latents_spatial).to(device)
            latents = pack_latents(latents_spatial)  # (B, H*W, C)

            # Pick random step for gradient
            step_with_grad = random.randint(cfg.steps_with_grad_min, cfg.steps_with_grad_max - 1)

            # ReFL denoising loop
            x0_packed = refl_denoise(
                latents=latents,
                latent_ids=latent_ids,
                prompt_embeds=prompt_embeds,
                text_ids=text_ids,
                transformer=transformer,
                scheduler=noise_scheduler,
                num_inference_steps=cfg.num_inference_steps,
                guidance_scale=cfg.guidance_scale,
                step_with_grad=step_with_grad,
                negative_prompt_embeds=neg_prompt_embeds,
                negative_text_ids=neg_text_ids,
            )

            # Decode to pixels
            images = decode_latents(x0_packed, latent_ids, vae)  # (B, 3, H, W) in [0, 1]

            # Compute reward
            reward = vlm_reward.score_batch(images, target_texts)  # (B,)
            mean_reward = reward.mean()

            # Loss = negative reward (maximize reward), scale by accumulation
            loss = -cfg.alpha_vlm * mean_reward / accum_steps

            # Backward
            if torch.isnan(loss):
                logger.warning(f"Step {global_step}: NaN loss detected, skipping")
                global_step += 1
                continue

            loss.backward()

            # Track the unscaled loss/reward for logging
            accum_loss = loss.item() * accum_steps  # unscaled
            accum_reward = mean_reward.item()

            # Step optimizer every accum_steps micro-batches
            micro_step = global_step % accum_steps
            if micro_step == accum_steps - 1:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [p for p in transformer.parameters() if p.requires_grad],
                    cfg.max_grad_norm,
                )
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            else:
                grad_norm = torch.tensor(0.0)

            # Logging
            total_reward += accum_reward
            global_step += 1

            if global_step % cfg.log_interval == 0:
                avg_reward = total_reward / cfg.log_interval
                elapsed = time.time() - start_time
                steps_per_sec = global_step / elapsed
                eta = (cfg.num_training_steps - global_step) / max(steps_per_sec, 1e-6)
                current_lr = lr_scheduler.get_last_lr()[0]

                logger.info(
                    f"step={global_step:5d} | "
                    f"loss={accum_loss:.4f} | "
                    f"reward={avg_reward:.4f} | "
                    f"grad_step={step_with_grad} | "
                    f"grad_norm={grad_norm:.4f} | "
                    f"lr={current_lr:.2e} | "
                    f"speed={steps_per_sec:.2f} it/s | "
                    f"eta={eta / 60:.1f}min"
                )

                # Write to CSV
                metrics_writer.writerow(
                    [
                        global_step,
                        f"{accum_loss:.6f}",
                        f"{avg_reward:.6f}",
                        f"{grad_norm:.6f}",
                        f"{current_lr:.2e}",
                        step_with_grad,
                        f"{elapsed:.1f}",
                    ]
                )
                metrics_file.flush()

                total_reward = 0.0

            # Save sample images periodically (every 10 steps)
            if global_step % 10 == 0 or global_step == 1:
                with torch.no_grad():
                    for eval_idx, ep in enumerate(EVAL_PROMPTS):
                        sample_scheduler = FlowMatchScheduler()
                        sample_scheduler.set_timesteps(cfg.num_inference_steps, device=device)
                        sample_scheduler.reset()
                        sample_latents = eval_fixed_latents_list[eval_idx].clone()
                        s_embeds = eval_prompt_embeds_list[eval_idx]
                        s_text_ids = eval_text_ids_list[eval_idx]
                        s_latent_ids = eval_fixed_latent_ids_list[eval_idx]
                        for si in range(cfg.num_inference_steps):
                            ts = sample_scheduler.timesteps[si]
                            ts_input = (ts / 1000).expand(1).to(dtype)
                            pred = transformer(
                                hidden_states=sample_latents.to(transformer.dtype),
                                timestep=ts_input,
                                guidance=None,
                                encoder_hidden_states=s_embeds,
                                txt_ids=s_text_ids,
                                img_ids=s_latent_ids,
                                return_dict=False,
                            )[0]
                            pred = pred[:, : sample_latents.size(1)]
                            if cfg.guidance_scale > 1.0 and neg_prompt_embeds is not None:
                                neg_pred = transformer(
                                    hidden_states=sample_latents.to(transformer.dtype),
                                    timestep=ts_input,
                                    guidance=None,
                                    encoder_hidden_states=neg_prompt_embeds,
                                    txt_ids=neg_text_ids,
                                    img_ids=s_latent_ids,
                                    return_dict=False,
                                )[0]
                                neg_pred = neg_pred[:, : sample_latents.size(1)]
                                pred = neg_pred + cfg.guidance_scale * (pred - neg_pred)
                            sample_latents = sample_scheduler.step(pred, sample_latents)
                        sample_images = decode_latents(sample_latents, s_latent_ids, vae)
                        sample_pil = Image.fromarray(
                            (sample_images[0].cpu().clamp(0, 1) * 255)
                            .byte()
                            .permute(1, 2, 0)
                            .numpy()
                        )
                        tag = ep["target_text"].replace(" ", "_").lower()
                        sample_pil.save(
                            os.path.join(samples_dir, f"step_{global_step:05d}_{tag}.png")
                        )

            # Save checkpoint
            if global_step % cfg.save_interval == 0:
                ckpt_dir = os.path.join(save_dir, f"checkpoint-{global_step}")
                os.makedirs(ckpt_dir, exist_ok=True)
                transformer.save_pretrained(ckpt_dir)
                torch.save(
                    {
                        "optimizer": optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                        "global_step": global_step,
                    },
                    os.path.join(ckpt_dir, "training_state.pt"),
                )
                logger.info(f"Checkpoint saved: {ckpt_dir}")

    # Final save
    final_dir = os.path.join(save_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    transformer.save_pretrained(final_dir)
    metrics_file.close()
    logger.info(f"Training complete. Final model saved to {final_dir}")
    logger.info(f"Metrics saved to {metrics_path}")
    logger.info(f"Total time: {(time.time() - start_time) / 60:.1f} minutes")


# ── CLI entry point ─────────────────────────────────────────────────────────


def main():
    import argparse

    parser = argparse.ArgumentParser(description="ReFL trainer for FLUX.2 Klein")
    parser.add_argument("--config", type=str, default=None, help="JSON config file")
    parser.add_argument("--model-id", type=str, default=None)
    parser.add_argument("--model-revision", type=str, default=None)
    parser.add_argument("--vlm-model-id", type=str, default=None)
    parser.add_argument("--vlm-model-revision", type=str, default=None)
    parser.add_argument("--prompts-path", type=str, default=None)
    parser.add_argument("--text-embeds-dir", type=str, default=None)
    parser.add_argument("--num-training-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--resolution", type=int, default=None)
    parser.add_argument("--num-inference-steps", type=int, default=None)
    parser.add_argument("--guidance-scale", type=float, default=None)
    parser.add_argument("--lora-rank", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--save-interval", type=int, default=None)
    parser.add_argument("--log-interval", type=int, default=None)
    args = parser.parse_args()

    cfg = ReflConfig()

    if args.config:
        with open(args.config) as f:
            overrides = json.load(f)
        for k, v in overrides.items():
            if not hasattr(cfg, k):
                continue
            if k == "lora" and isinstance(v, dict):
                from .config import LoraConfig

                cfg.lora = LoraConfig(**v)
            elif k == "eval_prompts" and isinstance(v, list):
                cfg.eval_prompts = v
            else:
                setattr(cfg, k, v)

    # CLI overrides
    for key in [
        "model_id",
        "model_revision",
        "vlm_model_id",
        "vlm_model_revision",
        "prompts_path",
        "text_embeds_dir",
        "num_training_steps",
        "batch_size",
        "lr",
        "resolution",
        "num_inference_steps",
        "guidance_scale",
        "output_dir",
        "experiment_name",
        "num_samples",
        "save_interval",
        "log_interval",
    ]:
        val = getattr(args, key.replace("-", "_"), None)
        if val is not None:
            setattr(cfg, key, val)

    if args.lora_rank is not None:
        cfg.lora.r = args.lora_rank
        cfg.lora.lora_alpha = args.lora_rank

    train(cfg)


if __name__ == "__main__":
    main()
