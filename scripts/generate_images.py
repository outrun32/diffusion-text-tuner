"""
Generate multiple image versions per prompt, saving latents and PNGs.

This is the first step of the SFT+DPO data pipeline:
  1. generate_images.py  — this file
  2. score_images.py     — score each image with VLM
  3. SFT/DPO training   — train on curated data

Usage:
  python -m scripts.generate_images \
    --prompts data/prompts_simple.jsonl \
    --output_dir outputs/generated \
    --model_id black-forest-labs/FLUX.2-klein-base-4B \
    --versions_per_prompt 5 \
    --batch_size 4
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Generate images for SFT/DPO pipeline")
    parser.add_argument("--prompts", type=str, required=True, help="JSONL file with prompts")
    parser.add_argument("--output_dir", type=str, default="outputs/generated")
    parser.add_argument("--model_id", type=str, default="black-forest-labs/FLUX.2-klein-base-4B")
    parser.add_argument("--lora_path", type=str, default=None, help="Optional LoRA checkpoint to load")
    parser.add_argument("--versions_per_prompt", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Images per batch (limited by VRAM)")
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=4.0)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start_idx", type=int, default=0, help="Resume from prompt index")
    parser.add_argument("--end_idx", type=int, default=None, help="Stop at prompt index")
    parser.add_argument("--save_latents", action="store_true", default=True,
                        help="Save VAE-encoded latents for training")
    parser.add_argument("--save_png", action="store_true", default=True,
                        help="Save decoded PNGs for scoring")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # Directories
    latents_dir = os.path.join(args.output_dir, "latents")
    text_embeds_dir = os.path.join(args.output_dir, "text_embeds")
    images_dir = os.path.join(args.output_dir, "images")
    os.makedirs(latents_dir, exist_ok=True)
    os.makedirs(text_embeds_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)

    # Load prompts
    records = []
    with open(args.prompts, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    records = records[args.start_idx : args.end_idx]
    logger.info("Loaded %d prompts (indices %d–%d)", len(records), args.start_idx,
                args.start_idx + len(records) - 1)

    # Load pipeline
    from diffusers import Flux2KleinPipeline

    logger.info("Loading pipeline: %s", args.model_id)
    pipe = Flux2KleinPipeline.from_pretrained(args.model_id, torch_dtype=torch.bfloat16)
    pipe = pipe.to("cuda")

    if args.lora_path:
        logger.info("Loading LoRA: %s", args.lora_path)
        pipe.load_lora_weights(args.lora_path)

    # Import VAE encode utilities
    from src.training.flux2_utils import encode_image, patchify_latents, bn_normalize

    vae = pipe.vae

    # Process prompts
    for rec_idx, record in enumerate(tqdm(records, desc="Generating")):
        global_idx = args.start_idx + rec_idx
        prompt_id = f"{global_idx:06d}"
        prompt = record["prompt"]
        target_text = record.get("target_text", "")

        prompt_latent_dir = os.path.join(latents_dir, prompt_id)
        prompt_image_dir = os.path.join(images_dir, prompt_id)
        os.makedirs(prompt_latent_dir, exist_ok=True)
        os.makedirs(prompt_image_dir, exist_ok=True)

        # Pre-compute text embeddings once per prompt
        embed_path = os.path.join(text_embeds_dir, f"{prompt_id}.pt")
        if not os.path.exists(embed_path):
            with torch.no_grad():
                prompt_embeds = pipe._get_qwen3_prompt_embeds(
                    text_encoder=pipe.text_encoder,
                    tokenizer=pipe.tokenizer,
                    prompt=[prompt],
                    device="cuda",
                    max_sequence_length=512,
                    hidden_states_layers=(9, 18, 27),
                )
            torch.save({
                "prompt_embeds": prompt_embeds[0].cpu(),
                "target_text": target_text,
                "prompt": prompt,
            }, embed_path)

        # Generate versions
        for ver in range(args.versions_per_prompt):
            latent_path = os.path.join(prompt_latent_dir, f"v{ver}.pt")
            image_path = os.path.join(prompt_image_dir, f"v{ver}.png")

            # Skip if already generated
            if os.path.exists(latent_path) and os.path.exists(image_path):
                continue

            seed = args.seed + global_idx * args.versions_per_prompt + ver
            generator = torch.Generator(device="cuda").manual_seed(seed)

            with torch.no_grad():
                output = pipe(
                    prompt=prompt,
                    height=args.resolution,
                    width=args.resolution,
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.guidance_scale,
                    generator=generator,
                    output_type="pil",
                )

            pil_image = output.images[0]

            # Save PNG for VLM scoring
            if args.save_png:
                pil_image.save(image_path)

            # Encode to latent and save for training
            if args.save_latents:
                import torchvision.transforms.functional as TF
                img_tensor = TF.to_tensor(pil_image).unsqueeze(0).to("cuda", dtype=torch.bfloat16)
                latent = encode_image(img_tensor, vae)  # (1, C*4, H', W')
                torch.save({"latent": latent[0].cpu()}, latent_path)

    # Save manifest
    manifest_path = os.path.join(args.output_dir, "manifest.json")
    manifest = {
        "model_id": args.model_id,
        "lora_path": args.lora_path,
        "versions_per_prompt": args.versions_per_prompt,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "resolution": args.resolution,
        "num_prompts": len(records),
        "start_idx": args.start_idx,
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info("Done. Generated %d × %d = %d images → %s",
                len(records), args.versions_per_prompt,
                len(records) * args.versions_per_prompt, args.output_dir)


if __name__ == "__main__":
    main()
