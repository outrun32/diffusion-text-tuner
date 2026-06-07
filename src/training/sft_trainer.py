"""
SFT (Supervised Fine-Tuning) trainer for FLUX.2 Klein.

Standard flow-matching MSE loss on cherry-picked high-reward samples.
Uses HuggingFace Accelerate for single/multi-GPU + SLURM support.

Usage:
  # Single GPU (local 5090)
  accelerate launch --config_file configs/accelerate/single_gpu.yaml \
    -m src.training.sft_trainer --config configs/sft.json

  # Multi-GPU cluster
  accelerate launch --config_file configs/accelerate/multi_gpu.yaml \
    -m src.training.sft_trainer --config configs/sft.json
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed
from peft import (
    LoraConfig as PeftLoraConfig,
    get_peft_model,
    load_peft_weights,
    set_peft_model_state_dict,
)
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import SFTConfig, LoraConfig
from .dataset import SFTDataset, sft_collate_fn
from .flux2_utils import (
    decode_latents,
    pack_latents,
    prepare_latent_ids,
    prepare_text_ids,
    bn_normalize,
    patchify_latents,
)
from .refl_trainer import FlowMatchScheduler
from src.runtime import config_io

logger = logging.getLogger(__name__)


# ── Flow-matching schedule helpers ──────────────────────────────────────────


def compute_sigma(t: torch.Tensor, shift: float = 3.0) -> torch.Tensor:
    """Compute shifted sigma from timestep t ∈ [0, 1000).

    sigma = shift * (t/1000) / (1 + (shift - 1) * (t/1000))
    """
    t_norm = t.float() / 1000.0
    sigma = shift * t_norm / (1.0 + (shift - 1.0) * t_norm)
    return sigma


# ── Model loading ───────────────────────────────────────────────────────────


def load_transformer(model_id: str, lora_cfg: LoraConfig, device, dtype):
    """Load FLUX transformer with LoRA, return (model, vae)."""
    from diffusers import Flux2KleinPipeline

    logger.info("Loading pipeline: %s", model_id)
    pipe = Flux2KleinPipeline.from_pretrained(model_id, torch_dtype=dtype)

    transformer = pipe.transformer
    vae = pipe.vae.to(device, dtype=dtype)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False

    # Apply LoRA
    peft_config = PeftLoraConfig(
        r=lora_cfg.r,
        lora_alpha=lora_cfg.lora_alpha,
        target_modules=lora_cfg.target_modules,
        lora_dropout=0.0,
    )
    transformer = get_peft_model(transformer, peft_config)
    trainable = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
    total = sum(p.numel() for p in transformer.parameters())
    logger.info("LoRA: %d trainable / %d total (%.2f%%)", trainable, total, 100 * trainable / total)

    # Free text encoder (not needed — embeddings are precomputed)
    del pipe.text_encoder, pipe.tokenizer
    torch.cuda.empty_cache()

    return transformer, vae


# ── Sampling ────────────────────────────────────────────────────────────────


def _sample_items(cfg) -> list[dict]:
    if cfg.sample_interval <= 0:
        return []
    if cfg.eval_suite_path:
        suite_path = Path(cfg.eval_suite_path)
        payload = json.loads(suite_path.read_text(encoding="utf-8"))
        items = payload.get("items", [])
        limit = cfg.eval_suite_n_per_step or len(items)
        return items[:limit]
    if not cfg.sample_prompt:
        return []
    return [
        {
            "name": "sample",
            "prompt": cfg.sample_prompt,
            "target_text": cfg.sample_target_text,
            "seed": 0,
            "resolution": cfg.resolution,
        }
    ]


def setup_sample_states(cfg, device, dtype):
    """Pre-encode sample prompts and prepare fixed noise for consistent sampling."""
    items = _sample_items(cfg)
    if not items:
        return []

    from .flux2_utils import precompute_text_embeddings
    import tempfile, shutil

    # Write a temp JSONL for precompute_text_embeddings
    tmpdir = tempfile.mkdtemp()
    try:
        prompts_file = os.path.join(tmpdir, "prompts.jsonl")
        with open(prompts_file, "w", encoding="utf-8") as f:
            for index, item in enumerate(items):
                json.dump(
                    {
                        "id": item.get("name", f"sample_{index:02d}"),
                        "prompt": item["prompt"],
                        "target_text": item.get("target_text", ""),
                    },
                    f,
                    ensure_ascii=False,
                )
                f.write("\n")

        embeds_dir = os.path.join(tmpdir, "embeds")
        precompute_text_embeddings(
            prompts_path=prompts_file,
            output_dir=embeds_dir,
            model_id=cfg.model_id,
            device=str(device),
        )
        embed_data = [
            torch.load(
                os.path.join(embeds_dir, f"{index:06d}.pt"),
                map_location=device,
                weights_only=True,
            )
            for index in range(len(items))
        ]
    finally:
        shutil.rmtree(tmpdir)

    sample_states = []
    for index, (item, embed) in enumerate(zip(items, embed_data, strict=True)):
        prompt_embeds = embed["prompt_embeds"].unsqueeze(0).to(dtype)  # (1, L, D)
        text_ids = prepare_text_ids(prompt_embeds).to(device)
        resolution = int(item.get("resolution", cfg.resolution))
        h = resolution // 8 // 2
        w = h
        fixed_noise = torch.randn(
            (1, 128, h, w),
            device=device,
            dtype=dtype,
            generator=torch.Generator(device).manual_seed(int(item.get("seed", index))),
        )
        sample_states.append(
            {
                "name": item.get("name", f"sample_{index:02d}"),
                "prompt_embeds": prompt_embeds,
                "text_ids": text_ids,
                "fixed_noise": pack_latents(fixed_noise),
                "latent_ids": prepare_latent_ids(fixed_noise).to(device),
            }
        )
    return sample_states


def setup_sample_state(cfg, device, dtype):
    states = setup_sample_states(cfg, device, dtype)
    return states[0] if states else None


@torch.no_grad()
def generate_sample(transformer, vae, sample_state, cfg, device, dtype):
    """Run full denoising loop and decode to PIL image."""
    if sample_state is None:
        return None

    transformer.eval()

    scheduler = FlowMatchScheduler(num_train_timesteps=cfg.num_train_timesteps, shift=cfg.shift)
    scheduler.set_timesteps(cfg.num_inference_steps, device=device)
    scheduler.reset()

    latents = sample_state["fixed_noise"].clone()
    prompt_embeds = sample_state["prompt_embeds"]
    text_ids = sample_state["text_ids"]
    latent_ids = sample_state["latent_ids"]

    for i in range(cfg.num_inference_steps):
        ts = scheduler.timesteps[i]
        ts_input = (ts / 1000).expand(1).to(dtype)
        pred = transformer(
            hidden_states=latents.to(transformer.dtype),
            timestep=ts_input,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=latent_ids,
            return_dict=False,
        )[0]
        pred = pred[:, :latents.size(1)]
        latents = scheduler.step(pred, latents)

    latents = latents.to(vae.dtype)
    images = decode_latents(latents, latent_ids, vae)
    pil_img = Image.fromarray(
        (images[0].cpu().clamp(0, 1) * 255).byte().permute(1, 2, 0).numpy()
    )

    transformer.train()
    return pil_img


def save_sample(transformer, vae, sample_state, cfg, device, dtype, samples_dir, step_name: str):
    """Generate and save one fixed-noise sample if sampling is enabled."""
    if sample_state is None or samples_dir is None:
        return
    pil_img = generate_sample(transformer, vae, sample_state, cfg, device, dtype)
    if pil_img is None:
        return
    path = os.path.join(samples_dir, f"{step_name}.png")
    pil_img.save(path)
    logger.info("Saved sample: %s", path)


def save_samples(transformer, vae, sample_states, cfg, device, dtype, samples_dir, step_name: str):
    """Generate and save fixed-noise samples for one checkpoint step."""
    if not sample_states or samples_dir is None:
        return
    if len(sample_states) == 1:
        save_sample(transformer, vae, sample_states[0], cfg, device, dtype, samples_dir, step_name)
        return
    step_dir = os.path.join(samples_dir, step_name)
    os.makedirs(step_dir, exist_ok=True)
    for state in sample_states:
        pil_img = generate_sample(transformer, vae, state, cfg, device, dtype)
        if pil_img is None:
            continue
        path = os.path.join(step_dir, f"{state['name']}.png")
        pil_img.save(path)
        logger.info("Saved sample: %s", path)


# ── Training ────────────────────────────────────────────────────────────────


def train(cfg: SFTConfig):
    accelerator = Accelerator(
        mixed_precision=cfg.mixed_precision,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=cfg.output_dir,
    )
    set_seed(cfg.seed)

    if accelerator.is_main_process:
        os.makedirs(cfg.output_dir, exist_ok=True)
        os.makedirs(os.path.join(cfg.output_dir, "checkpoints"), exist_ok=True)

    dtype = torch.bfloat16 if cfg.mixed_precision == "bf16" else torch.float32
    device = accelerator.device

    # ── Load model ──
    transformer, vae = load_transformer(cfg.model_id, cfg.lora, device, dtype)
    if cfg.resume_lora_path:
        logger.info("Loading LoRA weights from checkpoint: %s", cfg.resume_lora_path)
        peft_state = load_peft_weights(cfg.resume_lora_path, device="cpu")
        set_peft_model_state_dict(transformer, peft_state, adapter_name="default")
        if cfg.resume_step > 0:
            logger.warning(
                "Resuming weights from step %d without optimizer or dataloader state; "
                "optimizer/scheduler will be re-initialized.",
                cfg.resume_step,
            )

    # ── Setup sampling ──
    sample_states = []
    samples_dir = None
    if accelerator.is_main_process and cfg.sample_interval > 0:
        samples_dir = os.path.join(cfg.output_dir, "samples")
        os.makedirs(samples_dir, exist_ok=True)
        sample_states = setup_sample_states(cfg, device, dtype)
        logger.info("Sampling enabled: every %d steps → %s", cfg.sample_interval, samples_dir)

    if cfg.gradient_checkpointing:
        base = transformer.get_base_model()
        if hasattr(base, "enable_gradient_checkpointing"):
            base.enable_gradient_checkpointing()

            def _make_inputs_require_grad(module, input, output):
                if isinstance(output, tuple):
                    for o in output:
                        if isinstance(o, torch.Tensor) and o.is_floating_point():
                            o.requires_grad_(True)
                elif isinstance(output, torch.Tensor) and output.is_floating_point():
                    output.requires_grad_(True)

            base.register_forward_hook(_make_inputs_require_grad)

    # ── Dataset ──
    dataset = SFTDataset(
        latents_dir=cfg.latents_dir,
        text_embeds_dir=cfg.text_embeds_dir,
        scores_csv=cfg.scores_csv,
        score_threshold=cfg.score_threshold,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=sft_collate_fn,
        drop_last=True,
    )

    # ── Optimizer ──
    trainable_params = [p for p in transformer.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        betas=(0.9, 0.95),
        eps=1e-8,
    )

    from transformers import get_constant_schedule_with_warmup
    lr_scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=cfg.warmup_steps)

    # ── Prepare with accelerate ──
    transformer, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, dataloader, lr_scheduler
    )

    if accelerator.is_main_process and sample_states:
        unwrapped = accelerator.unwrap_model(transformer)
        save_samples(
            unwrapped,
            vae,
            sample_states,
            cfg,
            device,
            dtype,
            samples_dir,
            "step_000000",
        )

    # ── Latent geometry ──
    vae_scale_factor = 8
    patch_factor = 2
    latent_h = cfg.resolution // vae_scale_factor // patch_factor
    latent_w = latent_h

    # ── Training loop ──
    global_step = 0
    epoch = 0
    t_start = time.time()

    # CSV logger
    csv_path = os.path.join(cfg.output_dir, "metrics.csv")
    csv_fields = ["step", "loss", "grad_norm", "lr", "elapsed_s"]
    if accelerator.is_main_process:
        with open(csv_path, "w", newline="") as f:
            csv.DictWriter(f, csv_fields).writeheader()

    accelerator.init_trackers(cfg.experiment_name)

    progress_bar = tqdm(
        total=cfg.num_training_steps,
        disable=not accelerator.is_local_main_process,
        desc="SFT Training",
    )

    transformer.train()

    while global_step < cfg.num_training_steps:
        epoch += 1
        for batch in dataloader:
            if global_step >= cfg.num_training_steps:
                break

            with accelerator.accumulate(transformer):
                # batch["latent"]: (B, C, H, W) — patchified, BN-normalized
                x0 = batch["latent"].to(dtype)
                prompt_embeds = batch["prompt_embeds"].to(dtype)

                B = x0.shape[0]

                # Prepare position IDs
                latent_ids = prepare_latent_ids(x0).to(device)
                text_ids = prepare_text_ids(prompt_embeds).to(device)

                # Pack latents to sequence format: (B, C, H, W) → (B, H*W, C)
                x0_packed = pack_latents(x0)

                # Sample random timesteps
                t = torch.randint(0, cfg.num_train_timesteps, (B,), device=device)
                sigma = compute_sigma(t, shift=cfg.shift)  # (B,)

                # Create noisy latents: x_t = (1 - σ) * x₀ + σ * ε
                noise = torch.randn_like(x0_packed)
                sigma_bc = sigma.view(B, 1, 1)
                x_t = (1.0 - sigma_bc) * x0_packed + sigma_bc * noise

                # Velocity target: v = ε - x₀
                velocity_target = noise - x0_packed

                # Forward pass
                timestep = t.float() / 1000.0  # normalize to [0, 1]
                noise_pred = transformer(
                    hidden_states=x_t,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_ids,
                    return_dict=False,
                )[0]
                noise_pred = noise_pred[:, :x0_packed.shape[1]]

                # Flow-matching MSE loss
                loss = F.mse_loss(noise_pred.float(), velocity_target.float())

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(trainable_params, cfg.max_grad_norm)
                    grad_norm = grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm
                else:
                    grad_norm = 0.0

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                progress_bar.update(1)

                # Logging
                if global_step % cfg.log_interval == 0 and accelerator.is_main_process:
                    elapsed = time.time() - t_start
                    row = {
                        "step": global_step,
                        "loss": f"{loss.item():.6f}",
                        "grad_norm": f"{grad_norm:.4f}",
                        "lr": f"{lr_scheduler.get_last_lr()[0]:.2e}",
                        "elapsed_s": f"{elapsed:.1f}",
                    }
                    with open(csv_path, "a", newline="") as f:
                        csv.DictWriter(f, csv_fields).writerow(row)

                    accelerator.log({
                        "loss": loss.item(),
                        "grad_norm": grad_norm,
                        "lr": lr_scheduler.get_last_lr()[0],
                    }, step=global_step)

                    logger.info(
                        "step %d | loss %.4f | grad_norm %.3f | lr %.2e",
                        global_step, loss.item(), grad_norm,
                        lr_scheduler.get_last_lr()[0],
                    )

                # Checkpointing + sampling
                if global_step % cfg.save_interval == 0:
                    ckpt_dir = os.path.join(cfg.output_dir, "checkpoints", f"step_{global_step:06d}")
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        unwrapped = accelerator.unwrap_model(transformer)
                        unwrapped.save_pretrained(ckpt_dir)
                        logger.info("Saved checkpoint: %s", ckpt_dir)

                if global_step % cfg.sample_interval == 0 and cfg.sample_interval > 0:
                    if accelerator.is_main_process and sample_states:
                        unwrapped = accelerator.unwrap_model(transformer)
                        save_samples(
                            unwrapped,
                            vae,
                            sample_states,
                            cfg,
                            device,
                            dtype,
                            samples_dir,
                            f"step_{global_step:06d}",
                        )

    accelerator.end_training()
    progress_bar.close()

    # Final save + sample
    final_dir = os.path.join(cfg.output_dir, "checkpoints", "final")
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(transformer)
        unwrapped.save_pretrained(final_dir)
        logger.info("Training complete. Final checkpoint: %s", final_dir)
        if sample_states:
            save_samples(
                unwrapped,
                vae,
                sample_states,
                cfg,
                device,
                dtype,
                samples_dir,
                "step_final",
            )


# ── CLI ─────────────────────────────────────────────────────────────────────


def load_config(path: str) -> SFTConfig:
    return config_io.load_stage_config("sft", path)


def main():
    parser = argparse.ArgumentParser(description="SFT training for FLUX.2 Klein")
    parser.add_argument("--config", type=str, required=True, help="Path to JSON config")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = load_config(args.config)
    train(cfg)


if __name__ == "__main__":
    main()
