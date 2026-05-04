"""Manual gradient-flow diagnostic for the ReFL pipeline."""


def main() -> int:
    """Run the CUDA/model gradient-flow diagnostic."""
    import gc
    import math

    import torch
    from diffusers import Flux2KleinPipeline
    from peft import LoraConfig, get_peft_model

    from src.training.flux2_utils import (
        decode_latents,
        pack_latents,
        prepare_latent_ids,
        prepare_text_ids,
    )
    from src.training.refl_trainer import FlowMatchScheduler, refl_denoise
    from src.training.rewards import QwenYesProbReward

    print("=" * 60)
    print("TEST 1: Gradient flow through VLM reward")
    print("=" * 60)

    img = torch.rand(3, 512, 512, device="cuda", requires_grad=True)
    reward_model = QwenYesProbReward(device="cuda")
    score = reward_model.score_single(img, "hello")
    print(f"Score: {score.item():.6f}")
    print(f"Score requires_grad: {score.requires_grad}")
    print(f"Score grad_fn: {score.grad_fn}")

    score.backward()
    print(f"img.grad is None: {img.grad is None}")
    if img.grad is not None:
        print(f"img.grad norm: {img.grad.norm().item():.8f}")
        print(f"img.grad max: {img.grad.abs().max().item():.8f}")
        print("PASS: Gradient flows through VLM reward")
    else:
        print("FAIL: No gradient flows through VLM reward!")

    del img, score
    gc.collect()
    torch.cuda.empty_cache()

    print()
    print("=" * 60)
    print("TEST 2: Gradient flow through decode_latents")
    print("=" * 60)

    pipe = Flux2KleinPipeline.from_pretrained(
        "black-forest-labs/FLUX.2-klein-base-4B",
        torch_dtype=torch.bfloat16,
    )
    vae = pipe.vae.to("cuda", dtype=torch.bfloat16)
    vae.eval()
    for parameter in vae.parameters():
        parameter.requires_grad = False

    # Simulate what the trainer does: packed latents -> decode
    latents_spatial = torch.randn(
        1,
        128,
        32,
        32,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    latent_ids = prepare_latent_ids(latents_spatial).to("cuda")
    packed = pack_latents(latents_spatial)
    print(f"packed requires_grad: {packed.requires_grad}")

    images = decode_latents(packed, latent_ids, vae)
    print(f"images shape: {images.shape}")
    print(f"images requires_grad: {images.requires_grad}")
    print(f"images grad_fn: {images.grad_fn}")

    loss = images.mean()
    loss.backward()
    print(f"latents_spatial.grad is None: {latents_spatial.grad is None}")
    if latents_spatial.grad is not None:
        print(f"latents_spatial.grad norm: {latents_spatial.grad.norm().item():.8f}")
        print("PASS: Gradient flows through VAE decode")
    else:
        print("FAIL: No gradient flows through VAE decode!")

    del pipe, vae, latents_spatial, packed, images
    gc.collect()
    torch.cuda.empty_cache()

    print()
    print("=" * 60)
    print("TEST 3: Gradient flow through full ReFL step")
    print("=" * 60)

    pipe2 = Flux2KleinPipeline.from_pretrained(
        "black-forest-labs/FLUX.2-klein-base-4B",
        torch_dtype=torch.bfloat16,
    )
    transformer = pipe2.transformer
    vae2 = pipe2.vae.to("cuda", dtype=torch.bfloat16)
    vae2.eval()

    lora_cfg = LoraConfig(
        r=64,
        lora_alpha=64,
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    transformer = get_peft_model(transformer, lora_cfg)
    transformer.to("cuda")
    transformer.train()

    # Manual gradient hook
    def _hook(module, inputs, output):
        del module, inputs
        if isinstance(output, tuple):
            for item in output:
                if isinstance(item, torch.Tensor) and item.is_floating_point():
                    item.requires_grad_(True)
        elif isinstance(output, torch.Tensor) and output.is_floating_point():
            output.requires_grad_(True)

    transformer.get_base_model().register_forward_hook(_hook)

    # Load a sample embed
    data = torch.load("outputs/text_embeds/000000.pt", weights_only=True)
    prompt_embeds = data["prompt_embeds"].unsqueeze(0).to("cuda", dtype=torch.bfloat16)
    text_ids = prepare_text_ids(prompt_embeds).to("cuda")
    target_text = data["target_text"]

    latents_sp = torch.randn(1, 128, 32, 32, device="cuda", dtype=torch.bfloat16)
    lat_ids = prepare_latent_ids(latents_sp).to("cuda")
    latents = pack_latents(latents_sp)

    scheduler = FlowMatchScheduler()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in transformer.parameters() if parameter.requires_grad],
        lr=1e-4,
    )
    optimizer.zero_grad()

    x0 = refl_denoise(
        latents=latents,
        latent_ids=lat_ids,
        prompt_embeds=prompt_embeds,
        text_ids=text_ids,
        transformer=transformer,
        scheduler=scheduler,
        num_inference_steps=50,
        guidance_scale=4.0,
        step_with_grad=49,
        negative_prompt_embeds=None,
        negative_text_ids=None,
    )
    print(f"x0 requires_grad: {x0.requires_grad}")
    print(f"x0 grad_fn: {x0.grad_fn}")

    imgs = decode_latents(x0, lat_ids, vae2)
    print(f"imgs requires_grad: {imgs.requires_grad}")

    # Test reward
    reward_model2 = QwenYesProbReward(device="cuda")
    reward = reward_model2.score_single(imgs[0], target_text)
    print(f"Reward: {reward.item():.6f}")
    print(f"Reward requires_grad: {reward.requires_grad}")
    print(f"Reward grad_fn: {reward.grad_fn}")

    loss = -reward
    loss.backward()

    lora_grad_norm = 0.0
    lora_param_count = 0
    for name, parameter in transformer.named_parameters():
        if parameter.requires_grad and parameter.grad is not None:
            lora_grad_norm += parameter.grad.norm().item() ** 2
            lora_param_count += 1
        elif parameter.requires_grad and parameter.grad is None:
            print(f"  WARNING: {name} has no gradient!")
    lora_grad_norm = math.sqrt(lora_grad_norm)
    print(f"LoRA params with grad: {lora_param_count}")
    print(f"LoRA grad norm: {lora_grad_norm:.8f}")

    if lora_grad_norm > 0:
        print("PASS: Gradients reach LoRA parameters")
    else:
        print("FAIL: No gradients reach LoRA parameters!")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
