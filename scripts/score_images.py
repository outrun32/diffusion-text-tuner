"""
Score generated images using VLM or OCR reward model.

Reads PNGs from the image generation output, computes scores,
and writes a scores CSV for the SFT/DPO pipeline.

Scorers:
  vlm  — Qwen3.5-9B P(yes) (default, ~2.5GB VRAM, ~500ms/img)
  ocr  — PaddleOCR v5 CER+Entropy (~50MB VRAM, ~10ms/img, needs paddle)
  both — runs both and writes all columns

Usage:
  python -m scripts.score_images \
    --images_dir outputs/generated/images \
    --text_embeds_dir outputs/generated/text_embeds \
    --output_csv outputs/generated/scores.csv \
    --scorer vlm

  python -m scripts.score_images \
    --scorer ocr --entropy_lambda 1.5 ...

  python -m scripts.score_images \
    --scorer both ...
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
    parser.add_argument("--scorer", type=str, default="vlm", choices=["vlm", "ocr", "both"],
                        help="Scoring method: vlm (Qwen yes-prob), ocr (CER+entropy), both")
    parser.add_argument("--vlm_model_id", type=str, default="Qwen/Qwen3.5-9B")
    parser.add_argument("--entropy_lambda", type=float, default=1.0,
                        help="Lambda for OCR entropy scaling (R = (1-CER)*exp(-λ*H))")
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

    # Load scorer(s)
    vlm_model = None
    ocr_model = None

    if args.scorer in ("vlm", "both"):
        from src.training.rewards import QwenYesProbReward
        vlm_model = QwenYesProbReward(model_id=args.vlm_model_id, device="cuda")

    if args.scorer in ("ocr", "both"):
        from src.training.rewards import OcrCerEntropyReward
        ocr_model = OcrCerEntropyReward(
            lang="ru", device="gpu", entropy_lambda=args.entropy_lambda,
        )

    # CSV fields depend on scorer
    base_fields = ["id", "version", "score", "target_text"]
    extra_fields = []
    if ocr_model:
        extra_fields += [
            "official_conf",
            "cer",
            "entropy",
            "min_p",
            "frac_unc",
            "ocr_detected",
        ]
    if vlm_model and ocr_model:
        extra_fields += ["score_vlm", "score_ocr"]
    all_fields = base_fields + extra_fields

    # Score
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    write_header = not os.path.exists(output_csv) or not args.resume
    mode = "w" if write_header else "a"

    with open(output_csv, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_fields)
        if write_header:
            writer.writeheader()

        for task in tqdm(tasks, desc="Scoring"):
            row = {
                "id": task["id"],
                "version": task["version"],
                "target_text": task["target_text"],
            }

            score_vlm = None
            score_ocr = None

            if vlm_model:
                pil_image = Image.open(task["image_path"]).convert("RGB")
                import torchvision.transforms.functional as TF
                img_tensor = TF.to_tensor(pil_image).to("cuda")
                with torch.no_grad():
                    score_vlm = vlm_model.score_single(
                        img_tensor, task["target_text"]).item()

            if ocr_model:
                ocr_result = ocr_model.score(task["image_path"], task["target_text"])
                score_ocr = ocr_result["reward_ocr"]
                row["official_conf"] = f"{ocr_result['official_conf']:.4f}"
                row["cer"] = f"{ocr_result['cer']:.4f}"
                row["entropy"] = f"{ocr_result['entropy']:.4f}"
                row["min_p"] = f"{ocr_result['min_p']:.4f}"
                row["frac_unc"] = f"{ocr_result['frac_unc']:.4f}"
                row["ocr_detected"] = ocr_result["ocr_detected"]

            # Primary score: VLM if available, else OCR
            if vlm_model and ocr_model:
                row["score"] = f"{score_vlm:.6f}"
                row["score_vlm"] = f"{score_vlm:.6f}"
                row["score_ocr"] = f"{score_ocr:.6f}"
            elif vlm_model:
                row["score"] = f"{score_vlm:.6f}"
            else:
                row["score"] = f"{score_ocr:.6f}"

            writer.writerow(row)
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
