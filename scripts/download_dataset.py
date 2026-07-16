"""Download the prompts dataset from HuggingFace into a local JSONL file.

Usage:
  python scripts/download_dataset.py                          # default output
  python scripts/download_dataset.py --output data/prompts.jsonl
"""

import argparse
import hashlib
import json
from pathlib import Path

DEFAULT_DATASET_REVISION = "ecd8b2da9820b35afc65e2d56eaf37a662c37976"
DEFAULT_DATA_FILE = "data/train-00000-of-00001.parquet"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="Outrun32/cyrillic-prompts-15k")
    parser.add_argument("--output", default="data/prompts_simple.jsonl")
    parser.add_argument("--revision", default=DEFAULT_DATASET_REVISION)
    parser.add_argument("--split", default="train")
    parser.add_argument("--data-file", default=DEFAULT_DATA_FILE)
    parser.add_argument(
        "--manifest",
        default=None,
        help="Download manifest path (default: <output>.manifest.json)",
    )
    args = parser.parse_args()

    from datasets import load_dataset
    from huggingface_hub import HfApi, hf_hub_download

    info = HfApi().dataset_info(args.repo, revision=args.revision)
    resolved_revision = info.sha
    source_file = Path(
        hf_hub_download(
            args.repo,
            filename=args.data_file,
            repo_type="dataset",
            revision=resolved_revision,
        )
    )
    builder = "parquet" if source_file.suffix == ".parquet" else "json"
    ds = load_dataset(
        builder,
        data_files={args.split: str(source_file)},
        split=args.split,
    )
    print(f"Downloaded {len(ds)} records from {args.repo}@{resolved_revision}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary_output.open("w", encoding="utf-8") as f:
        for source_row in ds:
            row = dict(source_row)
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
    temporary_output.replace(output_path)

    print(f"Saved to {args.output}")
    manifest_path = (
        Path(args.manifest)
        if args.manifest
        else output_path.with_suffix(output_path.suffix + ".manifest.json")
    )
    manifest = {
        "schema_version": "prompt-dataset-download/v1",
        "repository": args.repo,
        "requested_revision": args.revision,
        "resolved_revision": resolved_revision,
        "split": args.split,
        "source_file": args.data_file,
        "source_sha256": _sha256(source_file),
        "output_path": str(output_path),
        "output_sha256": _sha256(output_path),
        "row_count": len(ds),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_manifest = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    temporary_manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_manifest.replace(manifest_path)
    print(f"Manifest saved to {manifest_path}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
