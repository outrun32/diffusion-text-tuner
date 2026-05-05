"""Datasets for SFT and DPO training on pre-encoded FLUX latents."""

from __future__ import annotations

import csv
import logging
from collections import defaultdict
from pathlib import Path

import random

import torch
from torch.utils.data import Dataset, Sampler

logger = logging.getLogger(__name__)


class SFTDataset(Dataset):
    """Loads pre-encoded VAE latents + text embeddings for SFT.

    Expects:
      latents_dir/{prompt_id}/v{version}.pt  — dict with "latent" key (patchified, BN-normed)
      text_embeds_dir/{prompt_id}.pt         — dict with "prompt_embeds" key
      scores_csv: id,version,score,target_text
    """

    def __init__(
        self,
        latents_dir: str,
        text_embeds_dir: str,
        scores_csv: str,
        score_threshold: float = 0.3,
    ):
        self.latents_dir = Path(latents_dir)
        self.text_embeds_dir = Path(text_embeds_dir)

        # Parse scores CSV and filter by threshold
        self.samples: list[dict] = []
        with open(scores_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                score = float(row["score"])
                if score >= score_threshold:
                    self.samples.append({
                        "prompt_id": row["id"],
                        "version": int(row["version"]),
                        "score": score,
                    })

        logger.info(
            "SFT dataset: %d samples above threshold %.2f (from %s)",
            len(self.samples), score_threshold, scores_csv,
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]
        prompt_id = s["prompt_id"]
        version = s["version"]

        # Load latent
        latent_path = self.latents_dir / prompt_id / f"v{version}.pt"
        latent_data = torch.load(latent_path, map_location="cpu", weights_only=True)
        latent = latent_data["latent"]  # (C, H, W) patchified, BN-normalized, packed-ready

        # Load text embedding
        embed_path = self.text_embeds_dir / f"{prompt_id}.pt"
        embed_data = torch.load(embed_path, map_location="cpu", weights_only=True)
        prompt_embeds = embed_data["prompt_embeds"]  # (L, D)

        return {
            "latent": latent,
            "prompt_embeds": prompt_embeds,
            "score": s["score"],
        }


def sft_collate_fn(batch: list[dict]) -> dict:
    """Collate SFT samples, padding text embeddings to max length."""
    latents = torch.stack([b["latent"] for b in batch])
    scores = torch.tensor([b["score"] for b in batch], dtype=torch.float32)

    # Pad prompt_embeds to max sequence length in batch
    embeds = [b["prompt_embeds"] for b in batch]
    max_len = max(e.shape[0] for e in embeds)
    dim = embeds[0].shape[1]
    padded = torch.zeros(len(embeds), max_len, dim, dtype=embeds[0].dtype)
    for i, e in enumerate(embeds):
        padded[i, :e.shape[0]] = e

    return {
        "latent": latents,
        "prompt_embeds": padded,
        "score": scores,
    }


class DPODataset(Dataset):
    """Loads preference pairs (winner, loser) for DPO training.

    For each prompt, selects the best-scoring and worst-scoring versions.
    Filters: best > score_threshold AND (best - worst) > score_diff_min.
    """

    def __init__(
        self,
        latents_dir: str,
        text_embeds_dir: str,
        scores_csv: str,
        score_threshold: float = 0.5,
        score_diff_min: float = 0.1,
    ):
        self.latents_dir = Path(latents_dir)
        self.text_embeds_dir = Path(text_embeds_dir)

        # Group scores by prompt_id
        by_prompt: dict[str, list[dict]] = defaultdict(list)
        with open(scores_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                by_prompt[row["id"]].append({
                    "version": int(row["version"]),
                    "score": float(row["score"]),
                })

        # Create preference pairs
        self.pairs: list[dict] = []
        for prompt_id, versions in by_prompt.items():
            if len(versions) < 2:
                continue
            versions.sort(key=lambda x: x["score"])
            best = versions[-1]
            worst = versions[0]

            if best["score"] < score_threshold:
                continue
            if best["score"] - worst["score"] < score_diff_min:
                continue

            self.pairs.append({
                "prompt_id": prompt_id,
                "winner_version": best["version"],
                "loser_version": worst["version"],
                "winner_score": best["score"],
                "loser_score": worst["score"],
            })

        logger.info(
            "DPO dataset: %d pairs (threshold=%.2f, diff_min=%.2f)",
            len(self.pairs), score_threshold, score_diff_min,
        )

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        p = self.pairs[idx]
        prompt_id = p["prompt_id"]

        # Load winner and loser latents
        w_path = self.latents_dir / prompt_id / f"v{p['winner_version']}.pt"
        l_path = self.latents_dir / prompt_id / f"v{p['loser_version']}.pt"
        w_data = torch.load(w_path, map_location="cpu", weights_only=True)
        l_data = torch.load(l_path, map_location="cpu", weights_only=True)

        # Load text embedding
        embed_path = self.text_embeds_dir / f"{prompt_id}.pt"
        embed_data = torch.load(embed_path, map_location="cpu", weights_only=True)

        return {
            "winner_latent": w_data["latent"],
            "loser_latent": l_data["latent"],
            "prompt_embeds": embed_data["prompt_embeds"],
        }


def dpo_collate_fn(batch: list[dict]) -> dict:
    """Collate DPO pairs, padding text embeddings to max length."""
    w_latents = torch.stack([b["winner_latent"] for b in batch])
    l_latents = torch.stack([b["loser_latent"] for b in batch])

    embeds = [b["prompt_embeds"] for b in batch]
    max_len = max(e.shape[0] for e in embeds)
    dim = embeds[0].shape[1]
    padded = torch.zeros(len(embeds), max_len, dim, dtype=embeds[0].dtype)
    for i, e in enumerate(embeds):
        padded[i, :e.shape[0]] = e

    return {
        "winner_latent": w_latents,
        "loser_latent": l_latents,
        "prompt_embeds": padded,
    }


class MaskedSFTDataset(Dataset):
    """Synthetic-text dataset for masked flow-matching SFT.

    On-disk layout under `data_dir`:
        latents/{sample_id}.pt      dict with "latent" (C, H, W) patchified+BN-normalized
                                    and "mask_lat" (H, W) float in [0, 1] on the latent grid
        text_embeds/{sample_id}.pt  dict with "prompt_embeds" (L, D)

    Sample IDs are taken from `latents/*.pt` filenames; a matching text-embed file
    must exist for each.
    """

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        latents_dir = self.data_dir / "latents"
        embeds_dir = self.data_dir / "text_embeds"
        if not latents_dir.is_dir():
            raise FileNotFoundError(f"Missing latents dir: {latents_dir}")
        if not embeds_dir.is_dir():
            raise FileNotFoundError(f"Missing text_embeds dir: {embeds_dir}")

        sample_ids: list[str] = []
        for p in sorted(latents_dir.glob("*.pt")):
            sid = p.stem
            if not (embeds_dir / f"{sid}.pt").is_file():
                logger.warning("Skipping %s: no matching text embedding", sid)
                continue
            sample_ids.append(sid)

        if not sample_ids:
            raise RuntimeError(f"No samples found under {data_dir}")

        self.sample_ids = sample_ids
        self.latents_dir = latents_dir
        self.embeds_dir = embeds_dir

        # Optional shapes.csv (id,H,W) — used by ResolutionBucketSampler.
        # Falls back to per-sample on-disk inspection if missing.
        self.shapes: dict[str, tuple[int, int]] = {}
        shapes_csv = self.data_dir / "shapes.csv"
        if shapes_csv.is_file():
            with shapes_csv.open("r", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    self.shapes[row["id"]] = (int(row["H"]), int(row["W"]))
            logger.info("Loaded %d latent shapes from %s", len(self.shapes), shapes_csv)

        logger.info("MaskedSFTDataset: %d samples from %s", len(sample_ids), data_dir)

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, idx: int) -> dict:
        sid = self.sample_ids[idx]
        lat_data = torch.load(self.latents_dir / f"{sid}.pt", map_location="cpu", weights_only=True)
        emb_data = torch.load(self.embeds_dir / f"{sid}.pt", map_location="cpu", weights_only=True)
        return {
            "sample_id": sid,
            "latent": lat_data["latent"],          # (C, H, W)
            "mask_lat": lat_data["mask_lat"],      # (H, W) float in [0, 1]
            "prompt_embeds": emb_data["prompt_embeds"],  # (L, D)
        }


def masked_sft_collate_fn(batch: list[dict]) -> dict:
    """Collate masked-SFT samples; pad text embeddings to max sequence length."""
    latents = torch.stack([b["latent"] for b in batch])
    masks = torch.stack([b["mask_lat"] for b in batch])

    embeds = [b["prompt_embeds"] for b in batch]
    max_len = max(e.shape[0] for e in embeds)
    dim = embeds[0].shape[1]
    padded = torch.zeros(len(embeds), max_len, dim, dtype=embeds[0].dtype)
    for i, e in enumerate(embeds):
        padded[i, : e.shape[0]] = e

    return {
        "sample_ids": [b["sample_id"] for b in batch],
        "latent": latents,
        "mask_lat": masks,
        "prompt_embeds": padded,
    }


class ResolutionBucketSampler(Sampler[list[int]]):
    """Yields batches of dataset indices grouped by latent (H, W) shape.

    Reads `dataset.shapes` (populated from `shapes.csv`). If a sample has no
    shape entry, its shape is loaded once from the latent file on disk.

    Behavior:
      - Indices within each bucket are shuffled per epoch (when shuffle=True).
      - Buckets are then concatenated in random order, sliced into batches.
      - If `drop_last=True`, partial bucket tails are dropped.
    """

    def __init__(
        self,
        dataset: MaskedSFTDataset,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = True,
        seed: int = 0,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0

        # Resolve shapes for every sample (lazy fallback to disk).
        buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
        missing: list[str] = []
        for idx, sid in enumerate(dataset.sample_ids):
            shp = dataset.shapes.get(sid)
            if shp is None:
                missing.append(sid)
                d = torch.load(
                    dataset.latents_dir / f"{sid}.pt",
                    map_location="cpu", weights_only=True,
                )
                lat = d["latent"]
                shp = (int(lat.shape[-2]), int(lat.shape[-1]))
                dataset.shapes[sid] = shp
            buckets[shp].append(idx)

        if missing:
            logger.warning(
                "ResolutionBucketSampler: %d samples lacked shape entries; "
                "loaded shapes from disk (slow). Re-run dataset build to emit shapes.csv.",
                len(missing),
            )

        self.buckets = dict(buckets)
        # Pre-compute number of batches per epoch.
        n = 0
        for ids in self.buckets.values():
            if drop_last:
                n += len(ids) // batch_size
            else:
                n += (len(ids) + batch_size - 1) // batch_size
        self._num_batches = n
        logger.info(
            "ResolutionBucketSampler: buckets=%s batches/epoch=%d (batch_size=%d, drop_last=%s)",
            {f"{h}x{w}": len(v) for (h, w), v in self.buckets.items()},
            n, batch_size, drop_last,
        )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self._num_batches

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        all_batches: list[list[int]] = []
        for ids in self.buckets.values():
            ids = list(ids)
            if self.shuffle:
                rng.shuffle(ids)
            for i in range(0, len(ids), self.batch_size):
                chunk = ids[i : i + self.batch_size]
                if self.drop_last and len(chunk) < self.batch_size:
                    continue
                all_batches.append(chunk)
        if self.shuffle:
            rng.shuffle(all_batches)
        yield from all_batches
