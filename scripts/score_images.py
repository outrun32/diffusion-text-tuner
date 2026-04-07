"""
Score generated images using VLM (Qwen3.5-9B) reward model.

Reads PNGs from the image generation output, computes P(yes) scores,
and writes a scores CSV for the SFT/DPO pipeline.

Usage:
  python -m scripts.score_images \
    --images_dir outputs/generated/images \
    --text_embeds_dir outputs/generated/text_embeds \
    --output_csv outputs/generated/scores.csv \
    --vlm_model_id Qwen/Qwen3.5-9B
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Score generated images with VLM")
    parser.add_argument("--images_dir", type=str, required=True,
                        help="Dir with {prompt_id}/v{ver}.png images")
    parser.add_argument("--text_embeds_dir", type=str, required=True,
                        help="Dir with {prompt_id}.pt (contains target_text)")
    parser.add_argument("--output_csv", type=str, default="outputs/generated/scores.csv")
    parser.add_argument("--vlm_model_id", type=str, default="Qwen/Qwen3.5-9B")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="VLM scoring batch size (1 is safest for memory)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip already-scored entries if CSV exists")
    parser.add_argument("--shard_idx", type=int, default=0,
                        help="Shard index for parallel scoring (0-based)")
    parser.add_argument("--num_shards", type=int, default=1,
                        help="Total number of shards for parallel scoring")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    images_dir = Path(args.images_dir)
    text_embeds_dir = Path(args.text_embeds_dir)

    # Collect all (prompt_id, version, image_path, target_text) tuples
    tasks = []
    for prompt_dir in sorted(images_dir.iterdir()):
        if not prompt_dir.is_dir():
            continue
        prompt_id = prompt_dir.name

        # Load target_text from text embedding file
        embed_path = text_embeds_dir / f"{prompt_id}.pt"
        if not embed_path.exists():
            logger.warning("No text embedding for %s, skipping", prompt_id)
            continue
        embed_data = torch.load(embed_path, map_location="cpu", weights_only=True)
        target_text = embed_data.get("target_text", "")
        if not target_text:
            logger.warning("No target_text in %s, skipping", prompt_id)
            continue

        for img_file in sorted(prompt_dir.glob("v*.png")):
            version = int(img_file.stem[1:])  # "v3" -> 3
            tasks.append({
                "id": prompt_id,
                "version": version,
                "image_path": str(img_file),
                "target_text": target_text,
            })

    logger.info("Found %d images to score", len(tasks))

    # Shard for parallel scoring (each GPU gets a disjoint slice)
    if args.num_shards > 1:
        tasks = [t for i, t in enumerate(tasks) if i % args.num_shards == args.shard_idx]
        logger.info("Shard %d/%d: %d images", args.shard_idx, args.num_shards, len(tasks))
        # Per-shard output CSV to avoid write conflicts
        base, ext = os.path.splitext(args.output_csv)
        output_csv = f"{base}_shard{args.shard_idx:03d}{ext}"
    else:
        output_csv = args.output_csv

    # Resume: load already scored
    scored_keys = set()
    if args.resume and os.path.exists(output_csv):
        with open(output_csv, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                scored_keys.add((row["id"], int(row["version"])))
        logger.info("Resuming: %d already scored", len(scored_keys))
        tasks = [t for t in tasks if (t["id"], t["version"]) not in scored_keys]
        logger.info("Remaining: %d to score", len(tasks))

    if not tasks:
        logger.info("Nothing to score, exiting.")
        return

    # Load VLM
    from src.training.rewards import QwenYesProbReward
    reward_model = QwenYesProbReward(model_id=args.vlm_model_id, device="cuda")

    # Score
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    write_header = not os.path.exists(output_csv) or not args.resume
    mode = "w" if write_header else "a"

    with open(output_csv, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "version", "score", "target_text"])
        if write_header:
            writer.writeheader()

        for task in tqdm(tasks, desc="Scoring"):
            pil_image = Image.open(task["image_path"]).convert("RGB")
            import torchvision.transforms.functional as TF
            img_tensor = TF.to_tensor(pil_image).to("cuda")

            with torch.no_grad():
                score = reward_model.score_single(img_tensor, task["target_text"])
                score_val = score.item()

            writer.writerow({
                "id": task["id"],
                "version": task["version"],
                "score": f"{score_val:.6f}",
                "target_text": task["target_text"],
            })
            f.flush()

    logger.info("Scoring complete. Results → %s", output_csv)

    # Print summary stats
    scores = []
    with open(output_csv, "r") as f:
        for row in csv.DictReader(f):
            scores.append(float(row["score"]))

    if scores:
        import statistics
        logger.info(
            "Score stats: n=%d, mean=%.4f, median=%.4f, min=%.4f, max=%.4f",
            len(scores), statistics.mean(scores), statistics.median(scores),
            min(scores), max(scores),
        )


if __name__ == "__main__":
    main()
