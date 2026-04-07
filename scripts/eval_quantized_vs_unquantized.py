"""
Evaluate baseline images with 4-bit quantized Qwen3.5-9B and compare
against existing unquantized scores in outputs/baseline/scores_qwen9b.jsonl.
"""

import json
import torch
from pathlib import Path
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

ROOT = Path("/root/diffusion-text-tuner")
BASELINE_SCORES = ROOT / "outputs/baseline/scores_qwen9b.jsonl"
OUTPUT_FILE = ROOT / "outputs/baseline/scores_qwen9b_4bit.jsonl"

MODEL_ID = "Qwen/Qwen3.5-9B"

SYSTEM_PROMPT = (
    "You are a precise image analysis tool. "
    "Answer ONLY with a single word: 'Yes' or 'No'. Do not explain."
)

PROMPT_TEMPLATE = (
    "Carefully examine each character in this image one by one. "
    'Does this image contain the text "{target_text}" with every single '
    "character rendered accurately and correctly? "
    "Respond with only 'yes' or 'no'."
)


def main():
    # Load existing unquantized scores
    records = []
    with open(BASELINE_SCORES) as f:
        for line in f:
            records.append(json.loads(line))
    print(f"Loaded {len(records)} records from unquantized baseline")

    # Load 4-bit quantized model
    print(f"Loading {MODEL_ID} (4-bit quantized)...")
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        quantization_config=quant_config,
        device_map="cuda",
    )
    model.eval()

    tokenizer = processor.tokenizer
    yes_variants = ["yes", "Yes", "YES", "да", "Да", "ДА"]
    no_variants = ["no", "No", "NO", "нет", "Нет", "НЕТ"]
    yes_ids = [tokenizer.encode(w, add_special_tokens=False)[0] for w in yes_variants]
    no_ids = [tokenizer.encode(w, add_special_tokens=False)[0] for w in no_variants]
    print(f"yes IDs: {yes_ids}, no IDs: {no_ids}")

    results = []
    for i, rec in enumerate(records):
        image_path = ROOT / rec["image"]
        target_text = rec["target_text"]

        img = Image.open(image_path).convert("RGB")
        prompt_text = PROMPT_TEMPLATE.format(target_text=target_text)

        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [
                {"type": "image", "image": "placeholder"},
                {"type": "text", "text": prompt_text},
            ]},
        ]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = processor(text=[text], images=[img], return_tensors="pt", padding=True)
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits[0, -1, :].float()
        probs = torch.softmax(logits, dim=-1)
        p_yes = sum(probs[tid].item() for tid in yes_ids)
        p_no = sum(probs[tid].item() for tid in no_ids)
        total = p_yes + p_no
        reward = p_yes / (total + 1e-8)

        result = {
            "idx": rec["idx"],
            "image": rec["image"],
            "target_text": target_text,
            "reward_4bit": reward,
            "p_yes_4bit": p_yes,
            "p_no_4bit": p_no,
            "reward_unquant": rec["reward_qwen_yes_prob"],
            "p_yes_unquant": rec["p_yes_raw"],
            "p_no_unquant": rec["p_no_raw"],
        }
        results.append(result)

        diff = reward - rec["reward_qwen_yes_prob"]
        print(f"[{i:3d}] 4bit={reward:.4f}  unquant={rec['reward_qwen_yes_prob']:.4f}  diff={diff:+.4f}  | {target_text[:40]}")

    # Save
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Summary stats
    import numpy as np
    r4 = np.array([r["reward_4bit"] for r in results])
    ru = np.array([r["reward_unquant"] for r in results])
    diffs = r4 - ru

    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(f"N samples:           {len(results)}")
    print(f"4-bit mean:          {r4.mean():.4f}  (std={r4.std():.4f})")
    print(f"Unquant mean:        {ru.mean():.4f}  (std={ru.std():.4f})")
    print(f"Mean diff (4b - uq): {diffs.mean():.4f}")
    print(f"Std diff:            {diffs.std():.4f}")
    print(f"Max abs diff:        {np.abs(diffs).max():.4f}")
    print(f"Correlation:         {np.corrcoef(r4, ru)[0,1]:.4f}")
    print(f"Rank correlation:    ", end="")
    from scipy.stats import spearmanr
    corr, pval = spearmanr(r4, ru)
    print(f"{corr:.4f} (p={pval:.2e})")

    # Agreement on binary yes/no (threshold=0.5)
    agree = ((r4 > 0.5) == (ru > 0.5)).sum()
    print(f"Binary agreement:    {agree}/{len(results)} ({100*agree/len(results):.1f}%)")

    print(f"\nSaved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
