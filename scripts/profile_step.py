"""Profile a single ReFL training step to find bottlenecks."""
import sys, os, time
sys.path.insert(0, "src")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import random
import json
from pathlib import Path
from training.config import ReflConfig
from training.flux2_utils import (
    decode_latents, pack_latents, prepare_latent_ids, prepare_text_ids,
)
from training.refl_trainer import FlowMatchScheduler, refl_denoise
from training.rewards import QwenYesProbReward

cfg = ReflConfig()

device = "cuda"
dtype = torch.bfloat16
random.seed(42)
torch.manual_seed(42)

# ── Load models ──
print("Loading models...")
from diffusers import Flux2KleinPipeline
from peft import LoraConfig as PeftLoraConfig, get_peft_model

pipe = Flux2KleinPipeline.from_pretrained(cfg.model_id, torch_dtype=dtype)
transformer = pipe.transformer
vae = pipe.vae.to(device, dtype=dtype)
vae.eval()
for p in vae.parameters():
    p.requires_grad = False

# Neg embeds
pipe.text_encoder.to(device)
neg_embeds, neg_text_ids = pipe.encode_prompt(prompt="", device=device)
neg_embeds = neg_embeds.detach().to(dtype)
neg_text_ids = neg_text_ids.detach()
pipe.text_encoder.to("cpu")
del pipe.text_encoder, pipe.tokenizer

import gc; gc.collect(); torch.cuda.empty_cache()

# LoRA
lora_config = PeftLoraConfig(r=64, lora_alpha=64, target_modules=["to_k","to_q","to_v","to_out.0"], lora_dropout=0.0)
transformer = get_peft_model(transformer, lora_config)
transformer.to(device)

base = transformer.get_base_model()
if hasattr(base, "enable_gradient_checkpointing"):
    base.enable_gradient_checkpointing()
def _hook(module, input, output):
    if isinstance(output, tuple):
        for o in output:
            if isinstance(o, torch.Tensor) and o.is_floating_point():
                o.requires_grad_(True)
    elif isinstance(output, torch.Tensor) and output.is_floating_point():
        output.requires_grad_(True)
base.register_forward_hook(_hook)
transformer.train()

# VLM reward
vlm_reward = QwenYesProbReward(model_id=cfg.vlm_model_id, device=device)

# Optimizer
from transformers.optimization import get_constant_schedule_with_warmup
optimizer = torch.optim.AdamW([p for p in transformer.parameters() if p.requires_grad], lr=cfg.lr)
lr_scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=0)
scheduler = FlowMatchScheduler()

# Latent dims
latent_h = cfg.resolution // 16
latent_w = cfg.resolution // 16
n_ch = 128

# Load one batch from dataset
from training.refl_trainer import TextEmbedDataset
ds = TextEmbedDataset(cfg.text_embeds_dir)
data = ds[0]
prompt_embeds = data["prompt_embeds"].unsqueeze(0).to(device, dtype=dtype)
target_text = data["target_text"]
text_ids = prepare_text_ids(prompt_embeds).to(device)

print("\n=== Profiling 3 full training steps ===\n")

for step_i in range(3):
    timings = {}
    torch.cuda.synchronize()
    
    # --- Noise init ---
    t0 = time.perf_counter()
    latents_spatial = torch.randn(1, n_ch, latent_h, latent_w, device=device, dtype=dtype)
    latent_ids = prepare_latent_ids(latents_spatial).to(device)
    latents = pack_latents(latents_spatial)
    step_with_grad = random.randint(cfg.steps_with_grad_min, cfg.steps_with_grad_max - 1)
    torch.cuda.synchronize()
    timings["noise_init"] = time.perf_counter() - t0

    # --- Forward (ReFL denoise) ---
    t0 = time.perf_counter()
    optimizer.zero_grad()
    x0 = refl_denoise(
        latents=latents, latent_ids=latent_ids,
        prompt_embeds=prompt_embeds, text_ids=text_ids,
        transformer=transformer, scheduler=scheduler,
        num_inference_steps=cfg.num_inference_steps,
        guidance_scale=cfg.guidance_scale,
        step_with_grad=step_with_grad,
        negative_prompt_embeds=neg_embeds,
        negative_text_ids=neg_text_ids,
    )
    torch.cuda.synchronize()
    timings["denoise_loop"] = time.perf_counter() - t0

    # --- VAE decode ---
    t0 = time.perf_counter()
    images = decode_latents(x0, latent_ids, vae)
    torch.cuda.synchronize()
    timings["vae_decode"] = time.perf_counter() - t0

    # --- VLM reward ---
    t0 = time.perf_counter()
    reward = vlm_reward.score_batch(images, [target_text])
    mean_reward = reward.mean()
    loss = -mean_reward
    torch.cuda.synchronize()
    timings["vlm_reward"] = time.perf_counter() - t0

    # --- Backward ---
    t0 = time.perf_counter()
    loss.backward()
    torch.cuda.synchronize()
    timings["backward"] = time.perf_counter() - t0

    # --- Optimizer step ---
    t0 = time.perf_counter()
    torch.nn.utils.clip_grad_norm_([p for p in transformer.parameters() if p.requires_grad], 1.0)
    optimizer.step()
    lr_scheduler.step()
    torch.cuda.synchronize()
    timings["optim_step"] = time.perf_counter() - t0

    total = sum(timings.values())
    print(f"Step {step_i} (grad_step={step_with_grad}, reward={mean_reward.item():.4f}):")
    for name, t in timings.items():
        pct = t / total * 100
        print(f"  {name:20s}: {t:6.2f}s  ({pct:5.1f}%)")
    print(f"  {'TOTAL':20s}: {total:6.2f}s")
    print()

# Breakdown of denoise: no-grad steps vs grad step
print("=== Denoise breakdown (step 3) ===\n")
latents_spatial = torch.randn(1, n_ch, latent_h, latent_w, device=device, dtype=dtype)
latent_ids = prepare_latent_ids(latents_spatial).to(device)
latents = pack_latents(latents_spatial)
step_with_grad = 45

scheduler.set_timesteps(cfg.num_inference_steps, device=device)
scheduler.reset()

nograd_time = 0
grad_time = 0
cfg_time = 0

for i in range(cfg.num_inference_steps):
    timestep = scheduler.timesteps[i]
    ts_input = (timestep / 1000).expand(1).to(dtype)
    torch.cuda.synchronize()

    if i < step_with_grad:
        t0 = time.perf_counter()
        with torch.no_grad():
            pred = transformer(hidden_states=latents.to(transformer.dtype), timestep=ts_input,
                             guidance=None, encoder_hidden_states=prompt_embeds,
                             txt_ids=text_ids, img_ids=latent_ids, return_dict=False)[0]
            pred = pred[:, :latents.size(1)]
        torch.cuda.synchronize()
        nograd_time += time.perf_counter() - t0

        t0 = time.perf_counter()
        with torch.no_grad():
            neg = transformer(hidden_states=latents.to(transformer.dtype), timestep=ts_input,
                            guidance=None, encoder_hidden_states=neg_embeds,
                            txt_ids=neg_text_ids, img_ids=latent_ids, return_dict=False)[0]
            neg = neg[:, :latents.size(1)]
            pred = neg + cfg.guidance_scale * (pred - neg)
        torch.cuda.synchronize()
        cfg_time += time.perf_counter() - t0

        latents = scheduler.step(pred, latents)
    else:
        t0 = time.perf_counter()
        pred = transformer(hidden_states=latents.to(transformer.dtype), timestep=ts_input,
                         guidance=None, encoder_hidden_states=prompt_embeds,
                         txt_ids=text_ids, img_ids=latent_ids, return_dict=False)[0]
        pred = pred[:, :latents.size(1)]
        torch.cuda.synchronize()
        grad_time += time.perf_counter() - t0

        t0 = time.perf_counter()
        with torch.no_grad():
            neg = transformer(hidden_states=latents.to(transformer.dtype), timestep=ts_input,
                            guidance=None, encoder_hidden_states=neg_embeds,
                            txt_ids=neg_text_ids, img_ids=latent_ids, return_dict=False)[0]
            neg = neg[:, :latents.size(1)]
            pred = neg + cfg.guidance_scale * (pred - neg)
        torch.cuda.synchronize()
        cfg_time += time.perf_counter() - t0

        latents = scheduler.step_to_zero(pred, latents)
        break

total_denoise = nograd_time + grad_time + cfg_time
print(f"No-grad forward   ({step_with_grad} steps):  {nograd_time:.2f}s  ({nograd_time/total_denoise*100:.1f}%)")
print(f"Grad forward      (1 step):           {grad_time:.2f}s  ({grad_time/total_denoise*100:.1f}%)")
print(f"CFG neg forward   ({step_with_grad+1} steps): {cfg_time:.2f}s  ({cfg_time/total_denoise*100:.1f}%)")
print(f"Total denoise:                        {total_denoise:.2f}s")
print(f"\nCFG doubles the forward passes — {cfg_time/(nograd_time+grad_time)*100:.0f}% overhead")
