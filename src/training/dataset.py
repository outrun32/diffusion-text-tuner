"""Datasets for SFT and DPO training on pre-encoded FLUX latents."""

from __future__ import annotations

import csv
import logging
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import Dataset

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
