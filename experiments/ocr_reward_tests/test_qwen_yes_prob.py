"""
Test Qwen3.5-4B as an offline MLX-VLM OCR reward probe.
Extract P(yes) from the model's logits as a continuous score
for how well text is rendered in an image.

Approach: Use logit_bias to force the model to pick from yes/no tokens,
then read the raw logprobs (before bias) for the normalized reward score.
Also test with a system prompt enforcing direct yes/no answers.
"""

import os
from pathlib import Path
import mlx.core as mx
import numpy as np
from mlx_vlm import load, stream_generate, apply_chat_template
from mlx_vlm.generate import generate_step
from mlx_vlm import prepare_inputs

MODEL_ID = "mlx-community/Qwen3.5-4B-MLX-4bit"

TARGET_TEXT = "ЛУЧШЕЕ КАПУЧИНО 2025!"
PROMPT_TEMPLATE = (
    "Carefully examine each character in this image one by one. "
    'Does this image contain the text "{target_text}" with every single '
    "character rendered accurately and correctly? "
    "Respond with only 'yes' or 'no'."
)

IMAGES = [
    ("bad_text.png", "broken text"),
    ("good_text.jpg", "correct text"),
    ("good_handwritten_text.jpg", "handwritten correct"),
]

BASE_DIR = Path(__file__).resolve().parents[1] / "assets"


def get_yes_no_probs(model, processor, tokenizer, image_path, prompt_text,
                     yes_ids, no_ids):
    """
    Get the first-token logprobs via generate_step directly.
    This gives us the raw log-softmax over the full vocab.
    """
    # Build messages with system prompt enforcing direct answer
    messages = [
        {
            "role": "system",
            "content": "You are a precise image analysis tool. Answer ONLY with a single word: 'Yes' or 'No'. Do not explain."
        },
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt_text},
            ],
        },
    ]

    # Apply chat template with thinking disabled (adds empty <think></think> block)
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    # Prepare inputs (tokenize + process image)
    image_token_index = getattr(model.config, "image_token_index", None)
    inputs = prepare_inputs(
        processor,
        images=[image_path],
        prompts=formatted_prompt,
        image_token_index=image_token_index,
    )

    input_ids = inputs["input_ids"]
    pixel_values = inputs.get("pixel_values", None)
    mask = inputs.get("attention_mask", None)

    # Collect extra kwargs (e.g. image_grid_thw) needed by the model
    extra_kwargs = {
        k: v for k, v in inputs.items()
        if k not in ("input_ids", "pixel_values", "attention_mask")
    }

    # Use generate_step to get the first token + full logprobs
    gen = generate_step(
        input_ids, model, pixel_values, mask,
        max_tokens=1,
        temperature=0.0,
        **extra_kwargs,
    )

    token_id, logprobs_arr = next(gen)

    token_text = tokenizer.decode([token_id.item() if hasattr(token_id, 'item') else token_id])
    token_id_val = token_id.item() if hasattr(token_id, 'item') else token_id

    # Sum probabilities for all yes and no variants
    p_yes = 0.0
    p_yes_details = {}
    for w, tid in yes_ids.items():
        p = mx.exp(logprobs_arr[tid]).item()
        p_yes += p
        p_yes_details[w] = (tid, p)

    p_no = 0.0
    p_no_details = {}
    for w, tid in no_ids.items():
        p = mx.exp(logprobs_arr[tid]).item()
        p_no += p
        p_no_details[w] = (tid, p)

    total = p_yes + p_no
    normalized = p_yes / total if total > 0 else 0.0

    return {
        "token_text": token_text,
        "token_id": token_id_val,
        "p_yes": p_yes,
        "p_no": p_no,
        "normalized": normalized,
        "yes_details": p_yes_details,
        "no_details": p_no_details,
    }


def main():
    print("Loading model...")
    model, processor = load(MODEL_ID)
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor

    # Find token IDs for yes/no variants
    yes_variants = ["yes", "Yes", "YES", "да", "Да", "ДА"]
    no_variants = ["no", "No", "NO", "нет", "Нет", "НЕТ"]

    yes_ids = {}
    no_ids = {}
    for w in yes_variants:
        ids = tokenizer.encode(w, add_special_tokens=False)
        yes_ids[w] = ids[0]
    for w in no_variants:
        ids = tokenizer.encode(w, add_special_tokens=False)
        no_ids[w] = ids[0]

    print(f"Yes token IDs: {yes_ids}")
    print(f"No token IDs:  {no_ids}")

    prompt_text = PROMPT_TEMPLATE.format(target_text=TARGET_TEXT)

    print(f"\n{'='*60}")
    print(f"Qwen3.5-4B Yes-Token Probability Test")
    print(f"{'='*60}\n")
    print(f"Prompt: {prompt_text}\n")

    results = []
    for filename, description in IMAGES:
        image_path = str(BASE_DIR / filename)
        if not os.path.exists(image_path):
            print(f"  SKIP: {image_path} not found\n")
            continue

        print(f"Processing: {filename} ({description})...")
        r = get_yes_no_probs(
            model, processor, tokenizer, image_path, prompt_text,
            yes_ids, no_ids,
        )
        results.append((filename, description, r))

    # Print results
    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}\n")

    for filename, description, r in results:
        print(f"Image: {filename} ({description})")
        print(f"  Generated token: '{r['token_text']}' (id={r['token_id']})")
        print(f"  P(yes) = {r['p_yes']:.6f}")
        print(f"  P(no)  = {r['p_no']:.6f}")
        print(f"  Normalized P(yes) = {r['normalized']:.6f}  <- REWARD SCORE")
        print(f"  --- Per-variant breakdown ---")
        for w, (tid, p) in r['yes_details'].items():
            if p > 1e-8:
                print(f"    P('{w}') [id={tid}] = {p:.8f}")
        for w, (tid, p) in r['no_details'].items():
            if p > 1e-8:
                print(f"    P('{w}') [id={tid}] = {p:.8f}")
        print()

    # Summary
    print(f"{'='*60}")
    print(f"SUMMARY — Does the score discriminate?")
    print(f"{'='*60}")
    for filename, description, r in results:
        bar = "█" * int(r["normalized"] * 50)
        print(f"  {filename:30s}  P(yes)={r['normalized']:.4f}  {bar}")

    if len(results) >= 2:
        bad = results[0][2]["normalized"]
        good = results[1][2]["normalized"]
        print(f"\n  Delta (good - bad) = {good - bad:.4f}")
        if good > bad:
            print(f"  ✓ Good text scores HIGHER than bad text — discrimination works!")
        else:
            print(f"  ✗ Good text does NOT score higher — discrimination failed")

    print("\nDone.")


if __name__ == "__main__":
    main()
