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
from typing import Any

import torch
from PIL import Image
from tqdm import tqdm

from src.evaluation.reward_interface import (
    ProductScoreFormula,
    build_score_metadata,
    compute_product_score,
)

PHASE6_JSONL_SCHEMA_VERSION = "phase6-score-jsonl/v1"


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
            enable_thinking=False,
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
                detected_parts.append(text)

        detected_full = " ".join(detected_parts)
        accuracy = self._char_accuracy(detected_full, target_text)

        return {
            "reward_paddleocr": accuracy,
            "ocr_detected": detected_full,
            "ocr_num_boxes": len(detected_parts),
        }


def _normalize_text(value: str) -> str:
    return " ".join(value.casefold().strip().split())


def _character_metrics(detected_text: str, target_text: str) -> dict[str, int | float | bool | str]:
    detected = _normalize_text(detected_text)
    target = _normalize_text(target_text)
    compared_total = max(len(detected), len(target))
    matches = sum(1 for left, right in zip(detected, target, strict=False) if left == right)
    accuracy = 1.0 if compared_total == 0 else matches / compared_total
    exact = detected == target and bool(target)
    if not detected:
        detection_status = "not_detected"
    elif exact:
        detection_status = "detected_exact"
    else:
        detection_status = "detected_mismatch"
    return {
        "detected_text": detected_text,
        "exact_text_match": exact,
        "char_accuracy": accuracy,
        "char_matches": matches,
        "char_total": compared_total,
        "detection_status": detection_status,
    }


def _sample_id_from_record(record: dict[str, Any]) -> str:
    for key in ("sample_id", "id", "prompt_id"):
        value = record.get(key)
        if isinstance(value, str) and value:
            return value
    image_path = str(record.get("image") or record.get("image_path") or "")
    return Path(image_path).stem if image_path else "unknown"


def build_canonical_evaluation_record(
    *,
    source_record: dict[str, Any],
    reward_outputs: dict[str, Any],
    version: int = 0,
    formula: ProductScoreFormula | None = None,
    manifest_path: str = "",
) -> dict[str, Any]:
    """Convert raw evaluation reward outputs into canonical Phase 6 JSONL fields."""

    active_formula = formula or ProductScoreFormula()
    target_text = str(source_record.get("target_text") or "")
    detected_text = str(reward_outputs.get("ocr_detected") or "")
    text_metrics = _character_metrics(detected_text, target_text)
    evidence = {
        "score_vlm": reward_outputs.get("score_vlm", reward_outputs.get("reward_qwen_yes_prob")),
        "score_ocr": reward_outputs.get("score_ocr", reward_outputs.get("reward_paddleocr")),
        "cer": reward_outputs.get("cer"),
        "entropy": reward_outputs.get("entropy"),
        "exact_text_match": text_metrics["exact_text_match"],
    }
    product = compute_product_score(evidence, formula=active_formula)
    sample_id = _sample_id_from_record(source_record)

    return {
        **source_record,
        "schema_version": "reward-result/v1",
        "score_file_schema_version": PHASE6_JSONL_SCHEMA_VERSION,
        "sample_id": sample_id,
        "version": version,
        "target_text": target_text,
        "score": product.score,
        "product_score": product.score,
        "score_vlm": evidence["score_vlm"],
        "score_ocr": evidence["score_ocr"],
        "cer": evidence["cer"],
        "entropy": evidence["entropy"],
        "ocr_detected": detected_text,
        "detection_status": text_metrics["detection_status"],
        "exact_text_match": text_metrics["exact_text_match"],
        "char_accuracy": text_metrics["char_accuracy"],
        "char_matches": text_metrics["char_matches"],
        "char_total": text_metrics["char_total"],
        "missing_components": list(product.missing_components),
        "formula_complete": product.formula_complete,
        "manifest_path": manifest_path,
        "text_metrics": text_metrics,
        "scorer_metadata": {
            "formula_name": active_formula.name,
            "scorer_versions": dict(active_formula.scorer_versions),
        },
        "thresholds": dict(product.threshold_flags),
    }


def write_evaluation_score_metadata(
    output_path: str | Path,
    *,
    formula: ProductScoreFormula | None = None,
    source_manifest_paths: list[str] | tuple[str, ...] = (),
    generated_at: str | None = None,
) -> Path:
    """Write canonical JSONL score sidecar metadata."""

    path = Path(output_path)
    sidecar = path.with_suffix(".schema.json")
    metadata = build_score_metadata(
        formula=formula,
        source_manifest_paths=source_manifest_paths,
        generated_at=generated_at,
    )
    metadata.update(
        {
            "score_file_schema_version": PHASE6_JSONL_SCHEMA_VERSION,
            "required_phase6_fields": [
                "sample_id",
                "version",
                "score",
                "product_score",
                "target_text",
                "score_vlm",
                "score_ocr",
                "cer",
                "entropy",
                "detection_status",
                "exact_text_match",
                "char_accuracy",
                "missing_components",
                "formula_complete",
                "manifest_path",
            ],
        }
    )
    sidecar.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return sidecar


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
    p.add_argument("--vlm-model", type=str, default="Qwen/Qwen3.5-9B",
                    help="HuggingFace VLM model ID for yes-prob reward")
    p.add_argument("--start-idx", type=int, default=0,
                    help="Resume from this index")
    p.add_argument("--manifest-path", type=str, default="",
                    help="Run/evaluation manifest path to link in each canonical record")
    p.add_argument("--source-manifest", action="append", default=[],
                    help="Source manifest path to include in the JSONL schema sidecar")
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
    with open(args.metadata, encoding="utf-8") as f:
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

    scorer_versions = {}
    if "qwen_yes_prob" in rewards_to_run:
        scorer_versions["vlm"] = args.vlm_model
    if "paddleocr" in rewards_to_run:
        scorer_versions["ocr"] = "paddleocr-PP-OCRv3-cyrillic"
    formula = ProductScoreFormula(scorer_versions=scorer_versions)

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

            reward_outputs = {}
            for scorer in scorers.values():
                result = scorer.score(image_path, target_text)
                reward_outputs.update(result)

            scored = build_canonical_evaluation_record(
                source_record=record,
                reward_outputs=reward_outputs,
                version=int(record.get("version", 0) or 0),
                formula=formula,
                manifest_path=args.manifest_path,
            )

            out_file.write(json.dumps(scored, ensure_ascii=False) + "\n")

            if (i + 1) % 50 == 0:
                out_file.flush()
    finally:
        out_file.close()

    # Print summary statistics
    source_manifests = tuple(
        args.source_manifest or ([args.manifest_path] if args.manifest_path else [])
    )
    sidecar = write_evaluation_score_metadata(
        out_path,
        formula=formula,
        source_manifest_paths=source_manifests,
    )
    print(f"\nScores saved to: {out_path}")
    print(f"Score schema metadata saved to: {sidecar}")
    _print_summary(out_path, rewards_to_run)


def _print_summary(scores_path: str, reward_names: set):
    """Print basic statistics for each reward."""
    import numpy as np

    records = []
    with open(scores_path, encoding="utf-8") as f:
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
            print("\n  Qwen Yes-Prob:")
            print(f"    mean={arr.mean():.4f}  std={arr.std():.4f}")
            print(f"    min={arr.min():.4f}  max={arr.max():.4f}")
            print(f"    median={np.median(arr):.4f}")
            # Distribution bins
            bins = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.01]
            hist, _ = np.histogram(arr, bins=bins)
            for lo, hi, cnt in zip(bins[:-1], bins[1:], hist, strict=False):
                pct = 100 * cnt / len(arr)
                bar = "█" * int(pct / 2)
                print(f"    [{lo:.1f}-{hi:.1f}): {cnt:4d} ({pct:5.1f}%) {bar}")

    if "paddleocr" in reward_names:
        vals = [r.get("reward_paddleocr", 0.0) for r in records
                if "reward_paddleocr" in r]
        if vals:
            arr = np.array(vals)
            print("\n  PaddleOCR Accuracy:")
            print(f"    mean={arr.mean():.4f}  std={arr.std():.4f}")
            print(f"    min={arr.min():.4f}  max={arr.max():.4f}")
            print(f"    median={np.median(arr):.4f}")
            bins = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.01]
            hist, _ = np.histogram(arr, bins=bins)
            for lo, hi, cnt in zip(bins[:-1], bins[1:], hist, strict=False):
                pct = 100 * cnt / len(arr)
                bar = "█" * int(pct / 2)
                print(f"    [{lo:.1f}-{hi:.1f}): {cnt:4d} ({pct:5.1f}%) {bar}")


if __name__ == "__main__":
    main()
