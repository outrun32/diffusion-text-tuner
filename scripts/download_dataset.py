"""Download the prompts dataset from HuggingFace into a local JSONL file.

Usage:
  python scripts/download_dataset.py                          # default output
  python scripts/download_dataset.py --output data/prompts.jsonl
"""

import argparse
import json
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="Outrun32/cyrillic-prompts-15k")
    parser.add_argument("--output", default="data/prompts_simple.jsonl")
    args = parser.parse_args()

    from datasets import load_dataset

    ds = load_dataset(args.repo, split="train")
    print(f"Downloaded {len(ds)} records from {args.repo}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for row in ds:
            # Reconstruct the style dict from flat columns
            style = {}
            for key in ("font", "color", "effect"):
                if row.get(key):
                    style[key] = row.pop(key)
            if row.get("text_size"):
                style["size"] = row.pop("text_size")
            else:
                row.pop("text_size", None)

            # Parse char_coverage back to dict
            cc = row.get("char_coverage", "{}")
            if isinstance(cc, str):
                row["char_coverage"] = json.loads(cc)

            row["style"] = style
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
