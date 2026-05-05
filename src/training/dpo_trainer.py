"""
DPO (Direct Preference Optimization) trainer for FLUX.2 Klein.

Sigmoid DPO loss with time-dependent beta on preference pairs (winner, loser).
Uses HuggingFace Accelerate for single/multi-GPU + SLURM support.

Usage:
  accelerate launch --config_file configs/accelerate/single_gpu.yaml \
    -m src.training.dpo_trainer --config configs/dpo.json
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import logging
import math
import os
import time

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from peft import LoraConfig as PeftLoraConfig, get_peft_model, PeftModel
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import DPOConfig, LoraConfig
from .dataset import DPODataset, dpo_collate_fn
from .dpo_objective import compute_dpo_objective, compute_sigma, time_dependent_beta
from .flux2_utils import pack_latents, prepare_latent_ids, prepare_text_ids
from src.runtime import config_io

logger = logging.getLogger(__name__)


# ── Model loading ───────────────────────────────────────────────────────────


def load_models(cfg: DPOConfig, device, dtype):
    """Load policy + frozen reference transformers (with optional SFT LoRA init)."""
    from diffusers import Flux2KleinPipeline

    logger.info("Loading pipeline: %s", cfg.model_id)
    pipe = Flux2KleinPipeline.from_pretrained(cfg.model_id, torch_dtype=dtype)

    base_transformer = pipe.transformer

    # Apply LoRA to policy
    peft_config = PeftLoraConfig(
        r=cfg.lora.r,
        lora_alpha=cfg.lora.lora_alpha,
        target_modules=cfg.lora.target_modules,
        lora_dropout=0.0,
    )
    policy = get_peft_model(base_transformer, peft_config)

    # If SFT LoRA checkpoint provided, load its weights as initialization
    if cfg.sft_lora_path:
        logger.info("Loading SFT LoRA weights from: %s", cfg.sft_lora_path)
        policy.load_adapter(cfg.sft_lora_path, adapter_name="default")

    trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    total = sum(p.numel() for p in policy.parameters())
    logger.info("Policy LoRA: %d trainable / %d total (%.2f%%)", trainable, total, 100 * trainable / total)

    # Reference model: deep copy of policy → freeze everything
    # We copy the full LoRA model so reference has the same SFT init
    ref_model = copy.deepcopy(policy)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    vae = pipe.vae.to(device, dtype=dtype)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False

    del pipe.text_encoder, pipe.tokenizer
    torch.cuda.empty_cache()

    return policy, ref_model, vae


# ── DPO loss ────────────────────────────────────────────────────────────────


def compute_dpo_loss(
    policy: torch.nn.Module,
    ref_model: torch.nn.Module,
    winner_packed: torch.Tensor,     # (B, S, C) packed x0
    loser_packed: torch.Tensor,
    prompt_embeds: torch.Tensor,     # (B, L, D)
    text_ids: torch.Tensor,
    latent_ids: torch.Tensor,
    t: torch.Tensor,                 # (B,) timesteps
    noise: torch.Tensor,             # (B, S, C) shared noise
    beta_conf: float,
    shift: float,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, dict]:
    """Compute DPO sigmoid loss with time-dependent beta and shared noise."""

    B = winner_packed.shape[0]
    sigma = compute_sigma(t, shift=shift)
    sigma_bc = sigma.view(B, 1, 1)
    timestep = t.float() / 1000.0

    # Noisy latents with shared noise
    w_noisy = (1.0 - sigma_bc) * winner_packed + sigma_bc * noise
    l_noisy = (1.0 - sigma_bc) * loser_packed + sigma_bc * noise

    # Velocity targets: v = ε - x₀
    w_target = noise - winner_packed
    l_target = noise - loser_packed

    # Policy predictions
    w_policy_pred = policy(
        hidden_states=w_noisy, timestep=timestep,
        encoder_hidden_states=prompt_embeds, txt_ids=text_ids, img_ids=latent_ids,
        return_dict=False,
    )[0][:, :winner_packed.shape[1]]

    l_policy_pred = policy(
        hidden_states=l_noisy, timestep=timestep,
        encoder_hidden_states=prompt_embeds, txt_ids=text_ids, img_ids=latent_ids,
        return_dict=False,
    )[0][:, :loser_packed.shape[1]]

    # Reference predictions (no grad)
    with torch.no_grad():
        w_ref_pred = ref_model(
            hidden_states=w_noisy, timestep=timestep,
            encoder_hidden_states=prompt_embeds, txt_ids=text_ids, img_ids=latent_ids,
            return_dict=False,
        )[0][:, :winner_packed.shape[1]]

        l_ref_pred = ref_model(
            hidden_states=l_noisy, timestep=timestep,
            encoder_hidden_states=prompt_embeds, txt_ids=text_ids, img_ids=latent_ids,
            return_dict=False,
        )[0][:, :loser_packed.shape[1]]

    # Per-sample MSE losses (mean over seq, dim)
    w_policy_loss = (w_policy_pred.float() - w_target.float()).pow(2).mean(dim=(1, 2))
    l_policy_loss = (l_policy_pred.float() - l_target.float()).pow(2).mean(dim=(1, 2))
    w_ref_loss = (w_ref_pred.float() - w_target.float()).pow(2).mean(dim=(1, 2))
    l_ref_loss = (l_ref_pred.float() - l_target.float()).pow(2).mean(dim=(1, 2))

    loss, objective_metrics = compute_dpo_objective(
        w_policy_loss=w_policy_loss,
        l_policy_loss=l_policy_loss,
        w_ref_loss=w_ref_loss,
        l_ref_loss=l_ref_loss,
        t=t,
        beta_conf=beta_conf,
        shift=shift,
    )

    metrics = {
        "reward_margin": objective_metrics["reward_margin"].item(),
        "accuracy": objective_metrics["accuracy"].item(),
        "w_policy_loss": objective_metrics["w_policy_loss"].item(),
        "l_policy_loss": objective_metrics["l_policy_loss"].item(),
    }

    return loss, metrics


# ── Training ────────────────────────────────────────────────────────────────


def train(cfg: DPOConfig):
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

    # ── Load models ──
    policy, ref_model, vae = load_models(cfg, device, dtype)

    if cfg.gradient_checkpointing:
        base = policy.get_base_model()
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
    dataset = DPODataset(
        latents_dir=cfg.latents_dir,
        text_embeds_dir=cfg.text_embeds_dir,
        scores_csv=cfg.scores_csv,
        score_threshold=cfg.score_threshold,
        score_diff_min=cfg.score_diff_min,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=dpo_collate_fn,
        drop_last=True,
    )

    # ── Optimizer ──
    trainable_params = [p for p in policy.parameters() if p.requires_grad]
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
    policy, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        policy, optimizer, dataloader, lr_scheduler
    )

    # Move reference model to device (not wrapped by accelerate — frozen)
    ref_model = ref_model.to(device)

    # ── Training loop ──
    global_step = 0
    epoch = 0
    t_start = time.time()

    csv_path = os.path.join(cfg.output_dir, "metrics.csv")
    csv_fields = ["step", "loss", "grad_norm", "lr", "reward_margin", "accuracy", "elapsed_s"]
    if accelerator.is_main_process:
        with open(csv_path, "w", newline="") as f:
            csv.DictWriter(f, csv_fields).writeheader()

    accelerator.init_trackers(cfg.experiment_name)

    progress_bar = tqdm(
        total=cfg.num_training_steps,
        disable=not accelerator.is_local_main_process,
        desc="DPO Training",
    )

    policy.train()

    while global_step < cfg.num_training_steps:
        epoch += 1
        for batch in dataloader:
            if global_step >= cfg.num_training_steps:
                break

            with accelerator.accumulate(policy):
                w_x0 = batch["winner_latent"].to(dtype)
                l_x0 = batch["loser_latent"].to(dtype)
                prompt_embeds = batch["prompt_embeds"].to(dtype)

                B = w_x0.shape[0]

                latent_ids = prepare_latent_ids(w_x0).to(device)
                text_ids = prepare_text_ids(prompt_embeds).to(device)

                w_packed = pack_latents(w_x0)
                l_packed = pack_latents(l_x0)

                # Shared noise and timesteps
                t = torch.randint(0, cfg.num_train_timesteps, (B,), device=device)
                noise = torch.randn_like(w_packed)

                loss, metrics = compute_dpo_loss(
                    policy=policy,
                    ref_model=ref_model,
                    winner_packed=w_packed,
                    loser_packed=l_packed,
                    prompt_embeds=prompt_embeds,
                    text_ids=text_ids,
                    latent_ids=latent_ids,
                    t=t,
                    noise=noise,
                    beta_conf=cfg.beta,
                    shift=cfg.shift,
                    dtype=dtype,
                )

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

                if global_step % cfg.log_interval == 0 and accelerator.is_main_process:
                    elapsed = time.time() - t_start
                    row = {
                        "step": global_step,
                        "loss": f"{loss.item():.6f}",
                        "grad_norm": f"{grad_norm:.4f}",
                        "lr": f"{lr_scheduler.get_last_lr()[0]:.2e}",
                        "reward_margin": f"{metrics['reward_margin']:.4f}",
                        "accuracy": f"{metrics['accuracy']:.3f}",
                        "elapsed_s": f"{elapsed:.1f}",
                    }
                    with open(csv_path, "a", newline="") as f:
                        csv.DictWriter(f, csv_fields).writerow(row)

                    accelerator.log({
                        "loss": loss.item(),
                        "grad_norm": grad_norm,
                        "lr": lr_scheduler.get_last_lr()[0],
                        "reward_margin": metrics["reward_margin"],
                        "accuracy": metrics["accuracy"],
                    }, step=global_step)

                    logger.info(
                        "step %d | loss %.4f | margin %.4f | acc %.3f | grad %.3f",
                        global_step, loss.item(), metrics["reward_margin"],
                        metrics["accuracy"], grad_norm,
                    )

                if global_step % cfg.save_interval == 0:
                    ckpt_dir = os.path.join(cfg.output_dir, "checkpoints", f"step_{global_step:06d}")
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        unwrapped = accelerator.unwrap_model(policy)
                        unwrapped.save_pretrained(ckpt_dir)
                        logger.info("Saved checkpoint: %s", ckpt_dir)

    accelerator.end_training()
    progress_bar.close()

    final_dir = os.path.join(cfg.output_dir, "checkpoints", "final")
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(policy)
        unwrapped.save_pretrained(final_dir)
        logger.info("Training complete. Final checkpoint: %s", final_dir)


# ── CLI ─────────────────────────────────────────────────────────────────────


def load_config(path: str) -> DPOConfig:
    return config_io.load_stage_config("dpo", path)


def main():
    parser = argparse.ArgumentParser(description="DPO training for FLUX.2 Klein")
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
