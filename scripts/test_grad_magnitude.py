"""Test gradient magnitude at different grad_steps."""
import torch, gc, math

from src.training.flux2_utils import decode_latents, pack_latents, prepare_latent_ids, prepare_text_ids
from src.training.refl_trainer import FlowMatchScheduler, refl_denoise

# Load model
from diffusers import Flux2KleinPipeline
from peft import LoraConfig, get_peft_model

pipe = Flux2KleinPipeline.from_pretrained("black-forest-labs/FLUX.2-klein-base-4B", torch_dtype=torch.bfloat16)
transformer = pipe.transformer
vae = pipe.vae.to("cuda", dtype=torch.bfloat16)
vae.eval()
for p in vae.parameters():
    p.requires_grad = False

lora_cfg = LoraConfig(r=64, lora_alpha=64, target_modules=["to_k","to_q","to_v","to_out.0"])
transformer = get_peft_model(transformer, lora_cfg)
transformer.to("cuda")
transformer.train()

def _hook(module, input, output):
    if isinstance(output, tuple):
        for o in output:
            if isinstance(o, torch.Tensor) and o.is_floating_point():
                o.requires_grad_(True)
    elif isinstance(output, torch.Tensor) and output.is_floating_point():
        output.requires_grad_(True)
transformer.get_base_model().register_forward_hook(_hook)

# Use negative prompt embeddings
pipe.text_encoder.to("cuda")
neg_pe, neg_ti = pipe.encode_prompt(prompt="", device="cuda")
neg_pe = neg_pe.detach().to(torch.bfloat16)
neg_ti = neg_ti.detach()
pipe.text_encoder.to("cpu")
gc.collect(); torch.cuda.empty_cache()

data = torch.load("outputs/text_embeds/000000.pt", weights_only=True)
prompt_embeds = data["prompt_embeds"].unsqueeze(0).to("cuda", dtype=torch.bfloat16)
text_ids = prepare_text_ids(prompt_embeds).to("cuda")

# Fixed noise
gen = torch.Generator(device="cuda").manual_seed(42)
latents_sp = torch.randn(1, 128, 32, 32, generator=gen, device="cuda", dtype=torch.bfloat16)
lat_ids = prepare_latent_ids(latents_sp).to("cuda")

print("Testing gradient magnitude at different grad_steps (50 inference steps):")
print(f"{'grad_step':>10} | {'x0_grad':>12} | {'img_grad':>12} | {'lora_grad':>12} | {'img_range':>12}")
print("-" * 70)

for step_with_grad in [5, 10, 20, 30, 40, 45, 49]:
    optimizer = torch.optim.AdamW([p for p in transformer.parameters() if p.requires_grad], lr=1e-4)
    optimizer.zero_grad()
    
    latents = pack_latents(latents_sp.clone())
    scheduler = FlowMatchScheduler()
    
    x0 = refl_denoise(
        latents=latents, latent_ids=lat_ids,
        prompt_embeds=prompt_embeds, text_ids=text_ids,
        transformer=transformer, scheduler=scheduler,
        num_inference_steps=50, guidance_scale=4.0,
        step_with_grad=step_with_grad,
        negative_prompt_embeds=neg_pe, negative_text_ids=neg_ti,
    )
    
    imgs = decode_latents(x0, lat_ids, vae)
    
    # Use a simple proxy loss (mean pixel value) to test grad magnitude
    # (avoids needing VLM which OOMs)
    loss = -imgs.mean()
    loss.backward()
    
    x0_grad = x0.grad.norm().item() if x0.grad is not None else 0
    img_grad = imgs.grad.norm().item() if imgs.grad is not None else 0
    
    lora_grad_norm = 0.0
    for name, p in transformer.named_parameters():
        if p.requires_grad and p.grad is not None:
            lora_grad_norm += p.grad.norm().item() ** 2
    lora_grad_norm = math.sqrt(lora_grad_norm)
    
    img_min = imgs.min().item()
    img_max = imgs.max().item()
    
    print(f"{step_with_grad:>10} | {x0_grad:>12.6f} | {img_grad:>12.6f} | {lora_grad_norm:>12.6f} | [{img_min:.3f}, {img_max:.3f}]")
    
    optimizer.zero_grad()
    gc.collect(); torch.cuda.empty_cache()

# Also test: what does the scheduler sigma look like at each step?
print()
print("Scheduler sigmas:")
scheduler2 = FlowMatchScheduler()
scheduler2.set_timesteps(50, device="cuda")
for i in [0, 5, 10, 20, 30, 40, 45, 49]:
    print(f"  step {i}: sigma={scheduler2.sigmas[i].item():.6f}, sigma_next={scheduler2.sigmas[i+1].item():.6f}, diff={scheduler2.sigmas[i+1].item() - scheduler2.sigmas[i].item():.6f}")
