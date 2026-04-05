# FLUX.2 Klein 4B — API Reference

## Pipeline
- from diffusers import Flux2KleinPipeline
- Text encoder: Qwen3ForCausalLM (Qwen2TokenizerFast)
- Components: transformer, vae, text_encoder, tokenizer, scheduler (FlowMatchEulerDiscreteScheduler)
- model_cpu_offload_seq = text_encoder->transformer->vae

## Models
- Distilled (black-forest-labs/FLUX.2-klein-4B): is_distilled=True, 4 steps, guidance_scale=1.0
- Base (black-forest-labs/FLUX.2-klein-4B-Base): is_distilled=False, 50 steps, guidance_scale=4.0 — USE FOR LORA TRAINING
- 9B variants also exist (klein-9B, klein-base-9B)

## Inference
Example:
pipe = Flux2KleinPipeline.from_pretrained("black-forest-labs/FLUX.2-klein-4B", torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload()
image = pipe("prompt", num_inference_steps=4, guidance_scale=1.0).images[0]

## Key Details
- Default resolution: 1024x1024 (sample_size=128, vae_scale_factor=8, patchify 2x)
- CFG only for non-distilled (guidance_scale > 1.0 ignored for distilled)
- LoRA support via Flux2LoraLoaderMixin
- VAE: AutoencoderKLFlux2 with batch norm
- Install: pip install git+https://github.com/huggingface/diffusers.git
