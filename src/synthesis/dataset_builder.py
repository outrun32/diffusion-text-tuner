"""Build synthetic Cyrillic datasets behind an import-safe module seam.

The public phases mirror the historical ``scripts.synth.build_dataset``
implementation while keeping SynthTIGER, FLUX, CUDA, Pillow, NumPy, and text
encoder imports out of module import time. Callers can reuse pure metadata
phases in CPU-safe tests and invoke GPU/model phases only behind explicit gates.
"""

from __future__ import annotations

import csv
import json
import logging
import shutil
import subprocess
import sys
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass(frozen=True)
class SynthesisBuildConfig:
    """Configuration for synthetic dataset building.

    Defaults intentionally match the historical CLI in
    ``scripts/synth/build_dataset.py`` so existing commands keep the same
    behavior after delegation through ``build_dataset``.
    """

    num: int
    workers: int = 8
    template: Path = Path("scripts/synth/synthtiger_template.py")
    template_name: str = "CyrillicScene"
    config: Path = Path("configs/synth/cyrillic.yaml")
    runner: Path = Path("scripts/synth/run_synthtiger.py")
    raw_dir: Path = Path("data/synth_cyrillic/raw")
    out_masked: Path = Path("data/synth_cyrillic/masked_sft")
    out_anyword: Path = Path("data/synth_cyrillic/anyword_format")
    seed: int = 0
    skip_render: bool = False
    clean: bool = False
    bake_latents: bool = False
    encode_text: bool = False
    model_id: str = "black-forest-labs/FLUX.2-klein-base-4B"
    device: str = "cuda"


def _progress(items: Iterable[T], *, desc: str) -> Iterator[T]:
    try:
        from tqdm import tqdm
    except ImportError:
        yield from items
        return

    yield from tqdm(items, desc=desc)


def render_phase(
    *,
    num: int,
    workers: int,
    template_path: Path,
    template_name: str,
    config_path: Path,
    raw_dir: Path,
    seed: int,
    runner: Path,
) -> None:
    """Invoke the SynthTIGER runner to write raw dataset artifacts."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(runner),
        "-o",
        str(raw_dir),
        "-c",
        str(num),
        "-w",
        str(workers),
        "-s",
        str(seed),
        str(template_path),
        template_name,
        str(config_path),
    ]
    logger.info("Running synthtiger: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def collate_records(raw_dir: Path) -> list[dict]:
    """Read raw metadata and return downstream records with annotations."""
    index_path = raw_dir / "index.jsonl"
    meta_dir = raw_dir / "meta"
    if not index_path.is_file():
        raise FileNotFoundError(f"missing {index_path} — synthtiger render failed?")

    records: list[dict] = []
    with index_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            sample_id = row["id"]
            meta_path = meta_dir / f"{sample_id}.json"
            if not meta_path.is_file():
                logger.warning("missing meta for %s; skipping", sample_id)
                continue
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if not meta.get("annotations"):
                continue
            records.append(
                {
                    "id": sample_id,
                    "resolution": int(meta["resolution"]),
                    "bg": meta.get("bg_path", ""),
                    "annotations": meta["annotations"],
                    "caption_human": meta["label"],
                    "caption_anyword": meta["caption_anyword"],
                }
            )
    logger.info("Collated %d records (with text) from %s", len(records), index_path)
    return records


def _hardlink_or_copy(src: Path, dst: Path) -> None:
    dst.unlink(missing_ok=True)
    try:
        dst.hardlink_to(src)
    except OSError:
        shutil.copyfile(src, dst)


def fan_out(
    records: list[dict],
    raw_dir: Path,
    out_masked: Path,
    out_anyword: Path,
) -> None:
    """Fan raw images and masks into masked-SFT and AnyWord layouts."""
    raw_imgs = raw_dir / "imgs"
    raw_masks = raw_dir / "masks"
    masked_imgs = out_masked / "raw_imgs"
    masked_masks = out_masked / "raw_masks"
    anyword_imgs = out_anyword / "imgs"
    anyword_masks = out_anyword / "masks"
    for directory in (masked_imgs, masked_masks, anyword_imgs, anyword_masks):
        directory.mkdir(parents=True, exist_ok=True)

    for record in _progress(records, desc="link"):
        sample_id = record["id"]
        src_img = raw_imgs / f"{sample_id}.png"
        src_mask = raw_masks / f"{sample_id}.png"
        _hardlink_or_copy(src_img, masked_imgs / f"{sample_id}.png")
        _hardlink_or_copy(src_mask, masked_masks / f"{sample_id}.png")
        _hardlink_or_copy(src_img, anyword_imgs / f"{sample_id}.png")
        _hardlink_or_copy(src_mask, anyword_masks / f"{sample_id}.png")


def write_anyword_json(records: list[dict], out_anyword: Path) -> None:
    """Write the FluxText AnyWord-3M-compatible data.json schema."""
    data_list = []
    for record in records:
        width = height = record["resolution"]
        data_list.append(
            {
                "img_name": f"imgs/{record['id']}.png",
                "width": width,
                "height": height,
                "caption": record["caption_anyword"],
                "wm_score": 0.0,
                "annotations": record["annotations"],
            }
        )
    payload = {"data_root": str(out_anyword.resolve()), "data_list": data_list}
    out_path = out_anyword / "data.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=1))
    logger.info("Wrote %s (%d entries)", out_path, len(data_list))


def write_masked_index(records: list[dict], out_masked: Path) -> Path:
    """Write masked-SFT index.csv and prompts.jsonl files."""
    idx_path = out_masked / "index.csv"
    with idx_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "resolution", "n_words", "text", "caption"])
        for record in records:
            joined = " | ".join(annotation["text"] for annotation in record["annotations"])
            n_words = sum(
                len(annotation["text"].split()) for annotation in record["annotations"]
            )
            writer.writerow(
                [
                    record["id"],
                    record["resolution"],
                    n_words,
                    joined,
                    record["caption_human"],
                ]
            )

    prompts_path = out_masked / "prompts.jsonl"
    with prompts_path.open("w", encoding="utf-8") as handle:
        for record in records:
            payload = {"id": record["id"], "prompt": record["caption_human"]}
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    logger.info("Wrote %s (%d) and %s", idx_path, len(records), prompts_path)
    return prompts_path


def bake_latents_phase(
    records: list[dict],
    out_masked: Path,
    model_id: str,
    device: str,
) -> None:
    """Encode images and masks into latent tensors behind an explicit gate."""
    import numpy as np
    import torch
    from diffusers import Flux2KleinPipeline
    from PIL import Image

    from src.training.flux2_utils import encode_image
    from src.training.losses import mask_to_latent_grid

    masked_imgs = out_masked / "raw_imgs"
    masked_masks = out_masked / "raw_masks"
    latents_dir = out_masked / "latents"
    latents_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading FLUX pipeline VAE: %s", model_id)
    pipe = Flux2KleinPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    vae = pipe.vae.to(device).eval()

    shapes_rows: list[tuple[str, int, int]] = []
    for record in _progress(records, desc="bake-latents"):
        sample_id = record["id"]
        resolution = record["resolution"]
        latent_height = resolution // 16
        latent_width = resolution // 16
        image = Image.open(masked_imgs / f"{sample_id}.png").convert("RGB")
        mask = Image.open(masked_masks / f"{sample_id}.png").convert("L")

        image_tensor = (
            torch.from_numpy(np.asarray(image, dtype="uint8"))
            .permute(2, 0, 1)
            .float()
            .unsqueeze(0)
            / 255.0
        )
        mask_tensor = (
            torch.from_numpy(np.asarray(mask, dtype="uint8"))
            .float()
            .unsqueeze(0)
            .unsqueeze(0)
            / 255.0
        )

        with torch.no_grad():
            input_tensor = image_tensor.to(device, dtype=torch.bfloat16)
            latent = encode_image(input_tensor, vae)
        mask_lat = mask_to_latent_grid(
            mask_tensor.to(device), (latent_height, latent_width)
        )

        if latent.shape[-2:] != (latent_height, latent_width):
            raise RuntimeError(
                f"id={sample_id}: latent shape {tuple(latent.shape)} != "
                f"expected (.,.,{latent_height},{latent_width})"
            )

        torch.save(
            {
                "latent": latent[0].cpu(),
                "mask_lat": mask_lat[0].cpu().float(),
            },
            latents_dir / f"{sample_id}.pt",
        )
        shapes_rows.append((sample_id, latent_height, latent_width))

    del vae, pipe
    torch.cuda.empty_cache()
    logger.info("Wrote %d latents to %s", len(records), latents_dir)

    shapes_path = out_masked / "shapes.csv"
    with shapes_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "H", "W"])
        for sample_id, height, width in shapes_rows:
            writer.writerow([sample_id, height, width])
    logger.info("Wrote %s (%d entries)", shapes_path, len(shapes_rows))


def encode_text_phase(
    prompts_path: Path,
    out_masked: Path,
    model_id: str,
    device: str,
) -> None:
    """Precompute text embeddings behind an explicit gate."""
    import tempfile

    from src.training.flux2_utils import precompute_text_embeddings

    embeds_dir = out_masked / "text_embeds"
    records: list[dict] = []
    with prompts_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    tmp_root = Path(tempfile.mkdtemp(prefix="text_embeds_", dir=str(out_masked)))
    tmp_embeds = tmp_root / "encoded"
    try:
        precompute_text_embeddings(
            prompts_path=str(prompts_path),
            output_dir=str(tmp_embeds),
            model_id=model_id,
            device=device,
        )

        new_embeds = out_masked / "text_embeds.new"
        if new_embeds.exists():
            shutil.rmtree(new_embeds)
        new_embeds.mkdir(parents=True)

        for index, record in enumerate(records):
            src = tmp_embeds / f"{index:06d}.pt"
            if not src.is_file():
                raise FileNotFoundError(f"missing encoded prompt: {src}")
            dst = new_embeds / f"{record['id']}.pt"
            shutil.move(str(src), str(dst))

        old_embeds = out_masked / "text_embeds.old"
        if old_embeds.exists():
            shutil.rmtree(old_embeds)
        if embeds_dir.exists():
            embeds_dir.rename(old_embeds)
        new_embeds.rename(embeds_dir)
        if old_embeds.exists():
            shutil.rmtree(old_embeds)
        logger.info("Wrote %d text embeddings to %s", len(records), embeds_dir)
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


def build_dataset(config: SynthesisBuildConfig) -> int:
    """Run synthetic dataset phases in the historical order."""
    if config.clean and not config.skip_render:
        for path in (config.raw_dir, config.out_masked, config.out_anyword):
            if path.exists():
                logger.info("Removing %s", path)
                shutil.rmtree(path)

    config.out_masked.mkdir(parents=True, exist_ok=True)
    config.out_anyword.mkdir(parents=True, exist_ok=True)

    if not config.skip_render:
        render_phase(
            num=config.num,
            workers=config.workers,
            template_path=config.template,
            template_name=config.template_name,
            config_path=config.config,
            raw_dir=config.raw_dir,
            seed=config.seed,
            runner=config.runner,
        )

    records = collate_records(config.raw_dir)
    fan_out(records, config.raw_dir, config.out_masked, config.out_anyword)
    write_anyword_json(records, config.out_anyword)
    prompts_path = write_masked_index(records, config.out_masked)

    if config.bake_latents:
        bake_latents_phase(records, config.out_masked, config.model_id, config.device)

    if config.encode_text:
        encode_text_phase(prompts_path, config.out_masked, config.model_id, config.device)

    logger.info("Done. %d samples ready.", len(records))
    return 0
