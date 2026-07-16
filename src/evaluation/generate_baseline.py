"""
Baseline image generation with FLUX.2 Klein 4B.

Reads prompts from a JSONL dataset (each line has {"prompt": ..., "target_text": ...})
and generates images using the distilled (4-step) or base (50-step) variant.
Saves images + a metadata JSONL for downstream reward evaluation.

Usage:
    python -m src.evaluation.generate_baseline \
        --prompts data/prompts_llm.jsonl \
        --output-dir outputs/baseline \
        --model black-forest-labs/FLUX.2-klein-4B \
        --num-samples 500 \
        --batch-size 1 \
        --seed 42
"""

import argparse
import json
import random
import time
from pathlib import Path

import torch
from diffusers import Flux2KleinPipeline
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser(description="Generate baseline images with FLUX.2 Klein")
    p.add_argument(
        "--prompts",
        type=str,
        required=True,
        help="Path to JSONL file with prompts (fields: prompt, target_text)",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="outputs/baseline",
        help="Directory to save images and metadata",
    )
    p.add_argument(
        "--model",
        type=str,
        default="black-forest-labs/FLUX.2-klein-4B",
        help="HuggingFace model ID or local path",
    )
    p.add_argument(
        "--num-samples", type=int, default=500, help="Number of images to generate (0 = all)"
    )
    p.add_argument(
        "--batch-size", type=int, default=1, help="Images per batch (>1 needs more VRAM)"
    )
    p.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Override num_inference_steps (default: 4 distilled, 50 base)",
    )
    p.add_argument(
        "--guidance-scale",
        type=float,
        default=None,
        help="Override guidance_scale (default: 1.0 distilled, 4.0 base)",
    )
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--offload", action="store_true", default=True, help="Enable model CPU offload to save VRAM"
    )
    p.add_argument("--no-offload", dest="offload", action="store_false")
    p.add_argument(
        "--start-idx", type=int, default=0, help="Resume from this index (skip already generated)"
    )
    return p.parse_args()


def load_prompts(path: str, num_samples: int, seed: int):
    """Load prompts from JSONL, optionally sample a subset."""
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    if num_samples > 0 and num_samples < len(records):
        rng = random.Random(seed)
        records = rng.sample(records, num_samples)

    return records


def main():
    args = parse_args()

    # --- Create output directory ---
    out_dir = Path(args.output_dir)
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "metadata.jsonl"

    # --- Load prompts ---
    print(f"Loading prompts from {args.prompts} ...")
    records = load_prompts(args.prompts, args.num_samples, args.seed)
    print(f"  Total prompts: {len(records)}")

    # --- Load pipeline ---
    print(f"Loading pipeline: {args.model} ...")
    pipe = Flux2KleinPipeline.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
    )

    is_distilled = getattr(pipe.config, "is_distilled", False)
    print(f"  is_distilled={is_distilled}")

    if args.offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")

    # Determine inference params
    num_steps = args.steps
    if num_steps is None:
        num_steps = 4 if is_distilled else 50

    guidance = args.guidance_scale
    if guidance is None:
        guidance = 1.0 if is_distilled else 4.0

    print(f"  num_inference_steps={num_steps}, guidance_scale={guidance}")
    print(f"  resolution={args.width}x{args.height}")

    # --- Generate images ---
    generator = torch.Generator(device="cpu").manual_seed(args.seed)

    # Open metadata file in append mode for resumability
    meta_file = open(meta_path, "a", encoding="utf-8")

    total = len(records)
    start = args.start_idx
    bs = args.batch_size

    t0 = time.time()
    try:
        for i in tqdm(range(start, total, bs), desc="Generating", unit="batch"):
            batch = records[i : i + bs]
            prompts = [r["prompt"] for r in batch]
            results = pipe(
                prompt=prompts,
                height=args.height,
                width=args.width,
                num_inference_steps=num_steps,
                guidance_scale=guidance,
                num_images_per_prompt=1,
                generator=generator,
            )

            for j, (image, record) in enumerate(zip(results.images, batch, strict=True)):
                idx = i + j
                fname = f"{idx:06d}.png"
                fpath = img_dir / fname
                image.save(fpath)

                meta = {
                    "idx": idx,
                    "image": str(fpath),
                    "prompt": record["prompt"],
                    "target_text": record.get("target_text", ""),
                    "content_type": record.get("content_type", ""),
                    "tier": record.get("tier", ""),
                    "model": args.model,
                    "steps": num_steps,
                    "guidance_scale": guidance,
                    "seed": args.seed,
                }
                meta_file.write(json.dumps(meta, ensure_ascii=False) + "\n")

            meta_file.flush()
    finally:
        meta_file.close()

    elapsed = time.time() - t0
    generated = total - start
    print(
        f"\nDone. Generated {generated} images in {elapsed:.1f}s "
        f"({elapsed / max(generated, 1):.2f}s/image)"
    )
    print(f"Images saved to: {img_dir}")
    print(f"Metadata saved to: {meta_path}")


if __name__ == "__main__":
    main()
