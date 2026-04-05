"""
Reward evaluation for generated images.

Runs two reward signals on baseline images:
    1. Qwen3.5 VLM yes-token probability (CUDA, via transformers)
  2. PaddleOCR v3 character-level accuracy

Reads metadata.jsonl from generate_baseline and appends reward scores.

Usage:
    python -m src.evaluation.evaluate_rewards \
        --metadata outputs/baseline/metadata.jsonl \
        --output outputs/baseline/scores.jsonl \
        --reward qwen_yes_prob \
        --vlm-model Qwen/Qwen3.5-4B \
        --batch-size 4

    python -m src.evaluation.evaluate_rewards \
        --metadata outputs/baseline/metadata.jsonl \
        --output outputs/baseline/scores.jsonl \
        --reward paddleocr
"""

import argparse
import json
import os
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm


# ── Qwen3.5 yes-prob reward (CUDA) ─────────────────────────────────────────

PROMPT_TEMPLATE = (
    "Carefully examine each character in this image one by one. "
    'Does this image contain the text "{target_text}" with every single '
    "character rendered accurately and correctly? "
    "Respond with only 'yes' or 'no'."
)


class QwenYesProbReward:
    """
    Uses a Qwen vision-language model (default: Qwen3.5-4B) to compute P(yes)
    as a differentiable reward for text rendering accuracy.
    """

    def __init__(self, model_id: str = "Qwen/Qwen3.5-4B", device: str = "cuda"):
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.device = device
        print(f"Loading VLM reward model: {model_id} ...")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map=device,
        )
        self.model.eval()

        tokenizer = self.processor.tokenizer
        # Precompute yes/no token IDs
        yes_variants = ["yes", "Yes", "YES", "да", "Да", "ДА"]
        no_variants = ["no", "No", "NO", "нет", "Нет", "НЕТ"]
        self.yes_ids = []
        self.no_ids = []
        for w in yes_variants:
            ids = tokenizer.encode(w, add_special_tokens=False)
            self.yes_ids.append(ids[0])
        for w in no_variants:
            ids = tokenizer.encode(w, add_special_tokens=False)
            self.no_ids.append(ids[0])

        print(f"  yes token IDs: {self.yes_ids}")
        print(f"  no  token IDs: {self.no_ids}")

    def _build_prompt(self, target_text: str) -> str:
        return PROMPT_TEMPLATE.format(target_text=target_text)

    @torch.no_grad()
    def score(self, image_path: str, target_text: str) -> dict:
        """Compute normalized P(yes) for a single image."""
        prompt_text = self._build_prompt(target_text)

        messages = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": (
                        "You are a precise image analysis tool. "
                        "Answer ONLY with a single word: 'Yes' or 'No'. Do not explain."
                    )},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"file://{os.path.abspath(image_path)}"},
                    {"type": "text", "text": prompt_text},
                ],
            },
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        # Process image + text into model inputs
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(
            text=[text], images=[image], return_tensors="pt", padding=True,
        ).to(self.device)

        outputs = self.model(**inputs)
        logits = outputs.logits[0, -1, :]  # last token position

        # Extract yes/no probabilities
        probs = torch.softmax(logits.float(), dim=-1)

        p_yes = sum(probs[tid].item() for tid in self.yes_ids)
        p_no = sum(probs[tid].item() for tid in self.no_ids)
        total = p_yes + p_no
        normalized = p_yes / total if total > 0 else 0.0

        return {
            "reward_qwen_yes_prob": normalized,
            "p_yes_raw": p_yes,
            "p_no_raw": p_no,
        }


# ── PaddleOCR v3 reward ────────────────────────────────────────────────────

class PaddleOCRReward:
    """
    Uses PaddleOCR v3 (cyrillic_PP-OCRv3_mobile_rec) for honest
    character-level OCR accuracy as a reward signal.
    """

    def __init__(self):
        from paddleocr import PaddleOCR

        print("Loading PaddleOCR v3 ...")
        self.ocr = PaddleOCR(
            use_angle_cls=True,
            lang="cyrillic",
            ocr_version="PP-OCRv3",
            show_log=False,
        )

    @staticmethod
    def _char_accuracy(predicted: str, target: str) -> float:
        """Character-level accuracy (order-independent, case-insensitive)."""
        pred = predicted.lower().strip()
        tgt = target.lower().strip()
        if not tgt:
            return 1.0 if not pred else 0.0

        # Count character matches
        from collections import Counter
        pred_chars = Counter(pred)
        tgt_chars = Counter(tgt)

        matches = 0
        for ch, count in tgt_chars.items():
            matches += min(pred_chars.get(ch, 0), count)

        # Penalize extra characters
        total_pred = sum(pred_chars.values())
        total_tgt = sum(tgt_chars.values())
        extra_penalty = max(0, total_pred - total_tgt) / max(total_tgt, 1)

        accuracy = matches / total_tgt
        score = max(0.0, accuracy - 0.5 * extra_penalty)
        return score

    def score(self, image_path: str, target_text: str) -> dict:
        """Run OCR on image and compute char-level accuracy vs target."""
        result = self.ocr.ocr(image_path, cls=True)

        # Concatenate all detected text
        detected_parts = []
        if result and result[0]:
            for line in result[0]:
                text = line[1][0]
                conf = line[1][1]
                detected_parts.append(text)

        detected_full = " ".join(detected_parts)
        accuracy = self._char_accuracy(detected_full, target_text)

        return {
            "reward_paddleocr": accuracy,
            "ocr_detected": detected_full,
            "ocr_num_boxes": len(detected_parts),
        }


# ── Main ────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate rewards on generated images")
    p.add_argument("--metadata", type=str, required=True,
                    help="Path to metadata.jsonl from generate_baseline")
    p.add_argument("--output", type=str, default=None,
                    help="Output scores JSONL (default: <metadata_dir>/scores.jsonl)")
    p.add_argument("--reward", type=str, nargs="+",
                    choices=["qwen_yes_prob", "paddleocr", "all"],
                    default=["all"],
                    help="Which reward(s) to compute")
    p.add_argument("--vlm-model", type=str, default="Qwen/Qwen3.5-4B",
                    help="HuggingFace VLM model ID for yes-prob reward")
    p.add_argument("--start-idx", type=int, default=0,
                    help="Resume from this index")
    return p.parse_args()


def main():
    args = parse_args()

    # Resolve rewards
    rewards_to_run = set(args.reward)
    if "all" in rewards_to_run:
        rewards_to_run = {"qwen_yes_prob", "paddleocr"}

    # Load metadata
    print(f"Loading metadata from {args.metadata} ...")
    records = []
    with open(args.metadata, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"  Total records: {len(records)}")

    # Output path
    out_path = args.output
    if out_path is None:
        out_path = str(Path(args.metadata).parent / "scores.jsonl")

    # Initialize reward models
    scorers = {}
    if "qwen_yes_prob" in rewards_to_run:
        scorers["qwen_yes_prob"] = QwenYesProbReward(
            model_id=args.vlm_model,
            device="cuda",
        )
    if "paddleocr" in rewards_to_run:
        scorers["paddleocr"] = PaddleOCRReward()

    # Score all records
    out_file = open(out_path, "a", encoding="utf-8")
    try:
        for i, record in enumerate(tqdm(records[args.start_idx:],
                                        desc="Scoring", unit="img")):
            image_path = record["image"]
            target_text = record.get("target_text", "")

            if not os.path.exists(image_path):
                print(f"  SKIP: {image_path} not found")
                continue

            scored = {**record}
            for name, scorer in scorers.items():
                result = scorer.score(image_path, target_text)
                scored.update(result)

            out_file.write(json.dumps(scored, ensure_ascii=False) + "\n")

            if (i + 1) % 50 == 0:
                out_file.flush()
    finally:
        out_file.close()

    # Print summary statistics
    print(f"\nScores saved to: {out_path}")
    _print_summary(out_path, rewards_to_run)


def _print_summary(scores_path: str, reward_names: set):
    """Print basic statistics for each reward."""
    import numpy as np

    records = []
    with open(scores_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        return

    print(f"\n{'='*60}")
    print(f"REWARD SUMMARY ({len(records)} images)")
    print(f"{'='*60}")

    if "qwen_yes_prob" in reward_names:
        vals = [r.get("reward_qwen_yes_prob", 0.0) for r in records
                if "reward_qwen_yes_prob" in r]
        if vals:
            arr = np.array(vals)
            print(f"\n  Qwen Yes-Prob:")
            print(f"    mean={arr.mean():.4f}  std={arr.std():.4f}")
            print(f"    min={arr.min():.4f}  max={arr.max():.4f}")
            print(f"    median={np.median(arr):.4f}")
            # Distribution bins
            bins = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.01]
            hist, _ = np.histogram(arr, bins=bins)
            for lo, hi, cnt in zip(bins[:-1], bins[1:], hist):
                pct = 100 * cnt / len(arr)
                bar = "█" * int(pct / 2)
                print(f"    [{lo:.1f}-{hi:.1f}): {cnt:4d} ({pct:5.1f}%) {bar}")

    if "paddleocr" in reward_names:
        vals = [r.get("reward_paddleocr", 0.0) for r in records
                if "reward_paddleocr" in r]
        if vals:
            arr = np.array(vals)
            print(f"\n  PaddleOCR Accuracy:")
            print(f"    mean={arr.mean():.4f}  std={arr.std():.4f}")
            print(f"    min={arr.min():.4f}  max={arr.max():.4f}")
            print(f"    median={np.median(arr):.4f}")
            bins = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.01]
            hist, _ = np.histogram(arr, bins=bins)
            for lo, hi, cnt in zip(bins[:-1], bins[1:], hist):
                pct = 100 * cnt / len(arr)
                bar = "█" * int(pct / 2)
                print(f"    [{lo:.1f}-{hi:.1f}): {cnt:4d} ({pct:5.1f}%) {bar}")


if __name__ == "__main__":
    main()
