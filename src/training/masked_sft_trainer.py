"""Masked SFT trainer for FLUX.2 Klein.

Same as `sft_trainer` but uses a region-weighted flow-matching loss driven by a
per-pixel text mask aligned to the latent grid. See `losses.masked_flow_matching_loss`.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from peft import LoraConfig as PeftLoraConfig
from peft import get_peft_model, load_peft_weights, set_peft_model_state_dict
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.runtime import config_io

from .checkpointing import checkpoint_dir, should_save_checkpoint
from .config import MaskedSFTConfig, MultiRankLoraConfig
from .dataset import MaskedSFTDataset, ResolutionBucketSampler, masked_sft_collate_fn
from .flux2_utils import decode_latents, pack_latents, prepare_latent_ids, prepare_text_ids
from .losses import masked_flow_matching_loss
from .refl_trainer import FlowMatchScheduler
from .sampling import should_sample_step
from .schedulers import compute_sigma

logger = logging.getLogger(__name__)


# ── Model loading with multi-rank LoRA ─────────────────────────────────────


def load_transformer_multi_rank(
    model_id: str,
    model_revision: str | None,
    lora_cfg: MultiRankLoraConfig,
    device,
    dtype,
):
    """Load FLUX transformer with three LoRA groups (attn / ffn / joint-attn).

    Returns (transformer_with_lora, vae). The text encoder is freed.
    """
    from diffusers import Flux2KleinPipeline

    logger.info("Loading pipeline: %s", model_id)
    pipe = Flux2KleinPipeline.from_pretrained(
        model_id,
        revision=model_revision,
        torch_dtype=dtype,
    )

    transformer = pipe.transformer
    vae = pipe.vae.to(device, dtype=dtype)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False

    # Build merged target_modules list and per-group rank/alpha overrides via
    # PEFT's `rank_pattern` / `alpha_pattern` (regex matched against full module
    # paths). Rank 0 disables a group. We validate suffix matches before PEFT so
    # a typo cannot silently turn an ablation into a different experiment.
    module_names = [
        name for name, module in transformer.named_modules() if hasattr(module, "weight")
    ]

    def _matched(suffixes: list[str]) -> list[str]:
        return [name for name in module_names if any(name.endswith(s) for s in suffixes)]

    groups = [
        ("attn", lora_cfg.attn_r, lora_cfg.attn_alpha, list(lora_cfg.attn_modules)),
        ("ffn", lora_cfg.ffn_r, lora_cfg.ffn_alpha, list(lora_cfg.ffn_modules)),
        (
            "joint",
            lora_cfg.joint_attn_r,
            lora_cfg.joint_attn_alpha,
            list(lora_cfg.joint_attn_modules),
        ),
    ]
    active_groups = []
    for group_name, rank, alpha, suffixes in groups:
        if rank <= 0:
            logger.info("LoRA group %s disabled", group_name)
            continue
        matches = _matched(suffixes)
        if not matches:
            raise ValueError(f"LoRA group {group_name} matched no modules: {suffixes}")
        logger.info(
            "LoRA group %s: r=%d alpha=%d modules=%d", group_name, rank, alpha, len(matches)
        )
        active_groups.append((group_name, rank, alpha, suffixes))
    if not active_groups:
        raise ValueError("at least one LoRA group must have rank > 0")

    all_modules = []
    for _, _, _, suffixes in active_groups:
        all_modules.extend(suffixes)
    all_modules = sorted(set(all_modules))

    def _suffix_regex(suffixes: list[str]) -> str:
        # Match parameter paths ending in any of the given suffixes.
        # Escape dots (the only meta char we use).
        escaped = [s.replace(".", r"\.") for s in suffixes]
        return r".*(?:" + "|".join(escaped) + r")$"

    base_name, base_r, base_alpha, _ = active_groups[0]
    rank_pattern = {
        _suffix_regex(suffixes): rank
        for group_name, rank, _alpha, suffixes in active_groups
        if group_name != base_name
    }
    alpha_pattern = {
        _suffix_regex(suffixes): alpha
        for group_name, _rank, alpha, suffixes in active_groups
        if group_name != base_name
    }

    peft_config = PeftLoraConfig(
        r=base_r,
        lora_alpha=base_alpha,
        target_modules=all_modules,
        lora_dropout=lora_cfg.dropout,
        rank_pattern=rank_pattern,
        alpha_pattern=alpha_pattern,
    )
    transformer = get_peft_model(transformer, peft_config)

    trainable = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
    total = sum(p.numel() for p in transformer.parameters())
    logger.info(
        "LoRA: attn r=%d, ffn r=%d, joint r=%d → %d trainable / %d total (%.3f%%)",
        lora_cfg.attn_r,
        lora_cfg.ffn_r,
        lora_cfg.joint_attn_r,
        trainable,
        total,
        100 * trainable / total,
    )

    del pipe.text_encoder, pipe.tokenizer
    torch.cuda.empty_cache()
    return transformer, vae


# ── Cosine-with-warmup LR schedule ─────────────────────────────────────────


def make_lr_scheduler(cfg: MaskedSFTConfig, optimizer):
    import math

    from torch.optim.lr_scheduler import LambdaLR

    base_lr = cfg.lr
    min_ratio = cfg.lr_min / base_lr if base_lr > 0 else 0.0
    total = max(1, cfg.num_training_steps)
    warmup = max(0, cfg.warmup_steps)

    if cfg.lr_schedule == "constant":

        def lr_lambda(step: int) -> float:
            if step < warmup:
                return float(step) / max(1, warmup)
            return 1.0
    elif cfg.lr_schedule == "cosine":

        def lr_lambda(step: int) -> float:
            if step < warmup:
                return float(step) / max(1, warmup)
            progress = (step - warmup) / max(1, total - warmup)
            progress = min(1.0, max(0.0, progress))
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_ratio + (1.0 - min_ratio) * cosine
    else:
        raise ValueError(f"unknown lr_schedule: {cfg.lr_schedule}")

    last_epoch = cfg.resume_step - 1 if cfg.resume_step > 0 else -1
    return LambdaLR(optimizer, lr_lambda=lr_lambda, last_epoch=last_epoch)


# ── Eval suite (multi-prompt sampler) ──────────────────────────────────────


def _load_eval_suite(cfg: MaskedSFTConfig, device, dtype):
    """Load eval suite JSON and pre-encode every prompt + reserve fixed noise.

    JSON schema:
        {"items": [{"name": "rare_letters", "prompt": "...", "resolution": 512}, ...]}

    Returns a list of dicts each with prompt_embeds, text_ids, fixed_noise (packed),
    latent_ids, name, resolution.
    """
    if not cfg.eval_suite_path:
        return []

    suite_path = Path(cfg.eval_suite_path)
    if not suite_path.is_file():
        logger.warning("eval_suite_path %s not found; skipping eval suite", suite_path)
        return []

    payload = json.loads(suite_path.read_text(encoding="utf-8"))
    items = payload.get("items", [])
    if not items:
        return []

    # Encode prompts via Qwen3 once at startup.
    import shutil
    import tempfile

    tmp = tempfile.mkdtemp()
    try:
        prompts_file = os.path.join(tmp, "prompts.jsonl")
        with open(prompts_file, "w", encoding="utf-8") as f:
            for it in items:
                f.write(json.dumps({"prompt": it["prompt"]}, ensure_ascii=False) + "\n")
        embeds_dir = os.path.join(tmp, "embeds")
        from .flux2_utils import precompute_text_embeddings

        precompute_text_embeddings(
            prompts_path=prompts_file,
            output_dir=embeds_dir,
            model_id=cfg.model_id,
            model_revision=cfg.model_revision,
            device=str(device),
        )

        out = []
        for i, it in enumerate(items):
            embed = torch.load(
                os.path.join(embeds_dir, f"{i:06d}.pt"),
                map_location=device,
                weights_only=True,
            )
            prompt_embeds = embed["prompt_embeds"].unsqueeze(0).to(dtype)  # (1,L,D)
            text_ids = prepare_text_ids(prompt_embeds).to(device)
            res = int(it.get("resolution", cfg.resolution))
            h = res // 8 // 2
            w = h
            gen = torch.Generator(device).manual_seed(int(it.get("seed", 1234 + i)))
            fixed_noise = torch.randn(
                (1, 128, h, w),
                device=device,
                dtype=dtype,
                generator=gen,
            )
            latent_ids = prepare_latent_ids(fixed_noise).to(device)
            out.append(
                {
                    "name": it.get("name", f"item_{i:02d}"),
                    "resolution": res,
                    "prompt": it["prompt"],
                    "prompt_embeds": prompt_embeds,
                    "text_ids": text_ids,
                    "fixed_noise": pack_latents(fixed_noise),
                    "latent_ids": latent_ids,
                }
            )
    finally:
        shutil.rmtree(tmp)

    logger.info("Eval suite: %d prompts loaded", len(out))
    return out


@torch.no_grad()
def _generate_for_item(transformer, vae, item, cfg: MaskedSFTConfig, device, dtype):
    transformer.eval()
    scheduler = FlowMatchScheduler(num_train_timesteps=cfg.num_train_timesteps, shift=cfg.shift)
    scheduler.set_timesteps(cfg.num_inference_steps, device=device)
    scheduler.reset()

    latents = item["fixed_noise"].clone()
    for i in range(cfg.num_inference_steps):
        ts = scheduler.timesteps[i]
        ts_input = (ts / 1000).expand(1).to(dtype)
        pred = transformer(
            hidden_states=latents.to(transformer.dtype),
            timestep=ts_input,
            encoder_hidden_states=item["prompt_embeds"],
            txt_ids=item["text_ids"],
            img_ids=item["latent_ids"],
            return_dict=False,
        )[0]
        pred = pred[:, : latents.size(1)]
        latents = scheduler.step(pred, latents)

    latents = latents.to(vae.dtype)
    images = decode_latents(latents, item["latent_ids"], vae)
    pil = Image.fromarray((images[0].cpu().clamp(0, 1) * 255).byte().permute(1, 2, 0).numpy())
    transformer.train()
    return pil


# ── Validation loss over fixed t-anchors ───────────────────────────────────


@torch.no_grad()
def _run_validation(
    transformer,
    val_loader,
    cfg: MaskedSFTConfig,
    device,
    dtype,
):
    """Compute mean masked/global loss across val batches at fixed t-anchors."""
    transformer.eval()
    n_anchors = len(cfg.val_t_anchors)
    sums = {"loss": 0.0, "masked": 0.0, "global": 0.0}
    count = 0

    for batch in val_loader:
        x0 = batch["latent"].to(device, dtype=dtype)
        prompt_embeds = batch["prompt_embeds"].to(device, dtype=dtype)
        mask_lat = batch["mask_lat"].to(device)
        B, _, H, W = x0.shape

        latent_ids = prepare_latent_ids(x0).to(device)
        text_ids = prepare_text_ids(prompt_embeds).to(device)
        x0_packed = pack_latents(x0)
        mask_seq = mask_lat.reshape(B, H * W)

        # Seed noise per-batch deterministically (reuse for all anchors).
        gen = torch.Generator(device).manual_seed(20240611 + count)
        noise = torch.randn(x0_packed.shape, device=device, dtype=dtype, generator=gen)
        velocity_target = noise - x0_packed

        for t_anchor in cfg.val_t_anchors:
            t = torch.full((B,), int(t_anchor), device=device, dtype=torch.long)
            sigma = compute_sigma(t, shift=cfg.shift)
            sigma_bc = sigma.view(B, 1, 1)
            x_t = (1.0 - sigma_bc) * x0_packed + sigma_bc * noise
            timestep = t.float() / 1000.0
            noise_pred = transformer(
                hidden_states=x_t,
                timestep=timestep,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                img_ids=latent_ids,
                return_dict=False,
            )[0]
            noise_pred = noise_pred[:, : x0_packed.shape[1]]
            loss, parts = masked_flow_matching_loss(
                noise_pred,
                velocity_target,
                mask_seq,
                lam=cfg.masked_lambda,
            )
            sums["loss"] += loss.item()
            sums["masked"] += parts["masked"].item()
            sums["global"] += parts["global"].item()
        count += 1

    transformer.train()
    if count == 0:
        return None
    denom = count * n_anchors
    return {k: v / denom for k, v in sums.items()}


def train(cfg: MaskedSFTConfig):
    from src.runtime.capabilities import check_stage_support

    support = check_stage_support("masked-sft", mixed_precision=cfg.mixed_precision)
    if not support.ok:
        raise RuntimeError("; ".join(support.errors))

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
    transformer, vae = load_transformer_multi_rank(
        cfg.model_id,
        cfg.model_revision,
        cfg.lora,
        device,
        dtype,
    )
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

    # ── Setup eval suite (multi-prompt sampler) ──
    eval_suite: list[dict] = []
    samples_dir = None
    if accelerator.is_main_process:
        samples_dir = os.path.join(cfg.output_dir, "samples")
        os.makedirs(samples_dir, exist_ok=True)
        eval_suite = _load_eval_suite(cfg, device, dtype)

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

    # ── Dataset (split off the last val_n_samples for held-out validation) ──
    full_dataset = MaskedSFTDataset(data_dir=cfg.data_dir)
    n_val = min(cfg.val_n_samples, max(0, len(full_dataset) - cfg.batch_size))
    if n_val > 0:
        val_ids = full_dataset.sample_ids[-n_val:]
        train_ids = full_dataset.sample_ids[:-n_val]
        full_dataset.sample_ids = train_ids
        dataset = full_dataset
        val_dataset = MaskedSFTDataset(data_dir=cfg.data_dir)
        val_dataset.sample_ids = val_ids
        val_bucket_sampler = ResolutionBucketSampler(
            val_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            drop_last=False,
            seed=cfg.seed,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_sampler=val_bucket_sampler,
            num_workers=2,
            pin_memory=True,
            collate_fn=masked_sft_collate_fn,
        )
        logger.info("Train/val split: %d / %d", len(dataset), len(val_dataset))
    else:
        dataset = full_dataset
        val_loader = None
        logger.info("No val split (dataset too small); skipping validation loss")

    bucket_sampler = ResolutionBucketSampler(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
        seed=cfg.seed,
    )
    dataloader = DataLoader(
        dataset,
        batch_sampler=bucket_sampler,
        num_workers=4,
        pin_memory=True,
        collate_fn=masked_sft_collate_fn,
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
    for group in optimizer.param_groups:
        group.setdefault("initial_lr", group["lr"])

    lr_scheduler = make_lr_scheduler(cfg, optimizer)

    # ── Prepare with accelerate ──
    transformer, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, dataloader, lr_scheduler
    )

    # ── Training loop ──
    if cfg.resume_step >= cfg.num_training_steps:
        raise ValueError(
            f"resume_step={cfg.resume_step} must be smaller than "
            f"num_training_steps={cfg.num_training_steps}"
        )

    global_step = cfg.resume_step
    epoch = cfg.resume_step // len(bucket_sampler)
    t_start = time.time()

    csv_path = os.path.join(cfg.output_dir, "metrics.csv")
    csv_fields = ["step", "loss", "loss_masked", "loss_global", "grad_norm", "lr", "elapsed_s"]
    append_metrics = cfg.resume_step > 0 and os.path.exists(csv_path)
    if accelerator.is_main_process and not append_metrics:
        with open(csv_path, "w", newline="") as f:
            csv.DictWriter(f, csv_fields).writeheader()

    accelerator.init_trackers(cfg.experiment_name)

    progress_bar = tqdm(
        total=cfg.num_training_steps,
        initial=global_step,
        disable=not accelerator.is_local_main_process or not sys.stderr.isatty(),
        desc="Masked SFT Training",
        mininterval=cfg.progress_bar_mininterval,
    )

    transformer.train()

    while global_step < cfg.num_training_steps:
        epoch += 1
        bucket_sampler.set_epoch(epoch)
        for batch in dataloader:
            if global_step >= cfg.num_training_steps:
                break

            with accelerator.accumulate(transformer):
                # batch["latent"]:   (B, C, H, W) — patchified, BN-normalized
                # batch["mask_lat"]: (B, H, W)   — float in [0, 1] on the latent grid
                x0 = batch["latent"].to(dtype)
                prompt_embeds = batch["prompt_embeds"].to(dtype)
                mask_lat = batch["mask_lat"].to(device)

                B, _, H, W = x0.shape

                # Validate mask grid matches latent grid.
                if mask_lat.shape[-2:] != (H, W):
                    raise ValueError(
                        f"mask_lat spatial shape {tuple(mask_lat.shape[-2:])} "
                        f"!= latent spatial shape {(H, W)}"
                    )

                # Position IDs + packing.
                latent_ids = prepare_latent_ids(x0).to(device)
                text_ids = prepare_text_ids(prompt_embeds).to(device)
                x0_packed = pack_latents(x0)  # (B, S, C)
                mask_seq = mask_lat.reshape(B, H * W)  # (B, S)

                # Sample timesteps and noisy latents.
                t = torch.randint(0, cfg.num_train_timesteps, (B,), device=device)
                sigma = compute_sigma(t, shift=cfg.shift)
                noise = torch.randn_like(x0_packed)
                sigma_bc = sigma.view(B, 1, 1)
                x_t = (1.0 - sigma_bc) * x0_packed + sigma_bc * noise
                velocity_target = noise - x0_packed

                # Forward pass.
                timestep = t.float() / 1000.0
                noise_pred = transformer(
                    hidden_states=x_t,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_ids,
                    return_dict=False,
                )[0]
                noise_pred = noise_pred[:, : x0_packed.shape[1]]

                # Region-weighted flow-matching loss.
                loss, loss_parts = masked_flow_matching_loss(
                    noise_pred,
                    velocity_target,
                    mask_seq,
                    lam=cfg.masked_lambda,
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
                        "loss_masked": f"{loss_parts['masked'].item():.6f}",
                        "loss_global": f"{loss_parts['global'].item():.6f}",
                        "grad_norm": f"{grad_norm:.4f}",
                        "lr": f"{lr_scheduler.get_last_lr()[0]:.2e}",
                        "elapsed_s": f"{elapsed:.1f}",
                    }
                    with open(csv_path, "a", newline="") as f:
                        csv.DictWriter(f, csv_fields).writerow(row)

                    accelerator.log(
                        {
                            "loss": loss.item(),
                            "loss_masked": loss_parts["masked"].item(),
                            "loss_global": loss_parts["global"].item(),
                            "grad_norm": grad_norm,
                            "lr": lr_scheduler.get_last_lr()[0],
                        },
                        step=global_step,
                    )

                    logger.info(
                        "step %d | loss %.4f (masked %.4f, global %.4f) | grad_norm %.3f | lr %.2e",
                        global_step,
                        loss.item(),
                        loss_parts["masked"].item(),
                        loss_parts["global"].item(),
                        grad_norm,
                        lr_scheduler.get_last_lr()[0],
                    )

                if should_save_checkpoint(global_step, cfg.save_interval):
                    ckpt_dir = checkpoint_dir(cfg.output_dir, global_step)
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        unwrapped = accelerator.unwrap_model(transformer)
                        unwrapped.save_pretrained(ckpt_dir)
                        logger.info("Saved checkpoint: %s", ckpt_dir)

                if should_sample_step(global_step, cfg.validation_interval):
                    # ── Validation loss across fixed t-anchors ──
                    if val_loader is not None:
                        unwrapped = accelerator.unwrap_model(transformer)
                        val_metrics = _run_validation(unwrapped, val_loader, cfg, device, dtype)
                        if val_metrics is not None and accelerator.is_main_process:
                            val_csv = os.path.join(cfg.output_dir, "val_metrics.csv")
                            new_file = not os.path.exists(val_csv)
                            with open(val_csv, "a", newline="") as f:
                                w = csv.DictWriter(
                                    f,
                                    ["step", "val_loss", "val_loss_masked", "val_loss_global"],
                                )
                                if new_file:
                                    w.writeheader()
                                w.writerow(
                                    {
                                        "step": global_step,
                                        "val_loss": f"{val_metrics['loss']:.6f}",
                                        "val_loss_masked": f"{val_metrics['masked']:.6f}",
                                        "val_loss_global": f"{val_metrics['global']:.6f}",
                                    }
                                )
                            accelerator.log(
                                {f"val/{k}": v for k, v in val_metrics.items()},
                                step=global_step,
                            )
                            logger.info(
                                "step %d | val loss %.4f (masked %.4f, global %.4f)",
                                global_step,
                                val_metrics["loss"],
                                val_metrics["masked"],
                                val_metrics["global"],
                            )

                    # ── Eval suite samples (rotate through items) ──
                    if accelerator.is_main_process and eval_suite:
                        unwrapped = accelerator.unwrap_model(transformer)
                        step_dir = os.path.join(samples_dir, f"step_{global_step:06d}")
                        os.makedirs(step_dir, exist_ok=True)
                        n = min(cfg.eval_suite_n_per_step, len(eval_suite))
                        offset = (global_step // max(1, cfg.validation_interval)) % len(eval_suite)
                        for i in range(n):
                            item = eval_suite[(offset + i) % len(eval_suite)]
                            try:
                                pil = _generate_for_item(unwrapped, vae, item, cfg, device, dtype)
                                pil.save(os.path.join(step_dir, f"{item['name']}.png"))
                            except Exception as e:
                                logger.warning("eval suite item %s failed: %s", item["name"], e)
                        logger.info("Saved %d eval samples → %s", n, step_dir)

    accelerator.end_training()
    progress_bar.close()

    final_dir = os.path.join(cfg.output_dir, "checkpoints", "final")
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(transformer)
        unwrapped.save_pretrained(final_dir)
        logger.info("Training complete. Final checkpoint: %s", final_dir)
        if eval_suite:
            step_dir = os.path.join(samples_dir, "step_final")
            os.makedirs(step_dir, exist_ok=True)
            for item in eval_suite:
                try:
                    pil = _generate_for_item(unwrapped, vae, item, cfg, device, dtype)
                    pil.save(os.path.join(step_dir, f"{item['name']}.png"))
                except Exception as e:
                    logger.warning("final eval suite item %s failed: %s", item["name"], e)
            logger.info("Saved final eval samples → %s", step_dir)


# ── CLI ─────────────────────────────────────────────────────────────────────


def load_config(path: str) -> MaskedSFTConfig:
    return config_io.load_stage_config("masked_sft", path)


def main():
    parser = argparse.ArgumentParser(description="Masked-SFT training for FLUX.2 Klein")
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
