"""Datasets for SFT and DPO training on pre-encoded FLUX latents."""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
import random
from collections import defaultdict
from pathlib import Path

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
        selection_mode: str = "threshold",
        selected_samples_path: str | None = None,
        score_column: str = "score",
        hard_negative_threshold: float = 0.2,
        sample_weighting: str = "uniform",
    ):
        self.latents_dir = Path(latents_dir)
        self.text_embeds_dir = Path(text_embeds_dir)

        self.samples = (
            _load_selected_sft_samples(
                Path(selected_samples_path),
                expected_mode=selection_mode,
                scores_csv=Path(scores_csv),
                score_column=score_column,
                score_threshold=score_threshold,
                hard_negative_threshold=hard_negative_threshold,
            )
            if selected_samples_path
            else _load_threshold_sft_samples(
                Path(scores_csv),
                score_threshold=score_threshold,
                score_column=score_column,
            )
        )
        _assign_sample_weights(self.samples, sample_weighting=sample_weighting)

        logger.info(
            "SFT dataset: %d samples (mode=%s, threshold=%.2f, source=%s)",
            len(self.samples),
            selection_mode,
            score_threshold,
            selected_samples_path or scores_csv,
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
            "sample_weight": s["sample_weight"],
        }


def sft_collate_fn(batch: list[dict]) -> dict:
    """Collate SFT samples, padding text embeddings to max length."""
    latents = torch.stack([b["latent"] for b in batch])
    scores = torch.tensor([b["score"] for b in batch], dtype=torch.float32)
    sample_weights = torch.tensor(
        [b.get("sample_weight", 1.0) for b in batch],
        dtype=torch.float32,
    )

    # Pad prompt_embeds to max sequence length in batch
    embeds = [b["prompt_embeds"] for b in batch]
    max_len = max(e.shape[0] for e in embeds)
    dim = embeds[0].shape[1]
    padded = torch.zeros(len(embeds), max_len, dim, dtype=embeds[0].dtype)
    for i, e in enumerate(embeds):
        padded[i, : e.shape[0]] = e

    return {
        "latent": latents,
        "prompt_embeds": padded,
        "score": scores,
        "sample_weight": sample_weights,
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
        pair_construction_mode: str = "best_vs_worst",
        preference_pairs_path: str | None = None,
        score_column: str = "score",
        ambiguity_margin: float = 0.0,
        pair_weighting: str = "uniform",
    ):
        self.latents_dir = Path(latents_dir)
        self.text_embeds_dir = Path(text_embeds_dir)

        self.pairs = (
            _load_materialized_dpo_pairs(
                Path(preference_pairs_path),
                expected_mode=pair_construction_mode,
                scores_csv=Path(scores_csv),
                score_column=score_column,
                score_threshold=score_threshold,
                score_diff_min=score_diff_min,
                ambiguity_margin=ambiguity_margin,
            )
            if preference_pairs_path
            else _load_best_worst_pairs(
                Path(scores_csv),
                score_threshold=score_threshold,
                score_diff_min=score_diff_min,
                score_column=score_column,
            )
        )
        _assign_pair_weights(self.pairs, pair_weighting=pair_weighting)

        logger.info(
            "DPO dataset: %d pairs (mode=%s, threshold=%.2f, diff_min=%.2f)",
            len(self.pairs),
            pair_construction_mode,
            score_threshold,
            score_diff_min,
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
            "pair_weight": p["pair_weight"],
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
        padded[i, : e.shape[0]] = e

    return {
        "winner_latent": w_latents,
        "loser_latent": l_latents,
        "prompt_embeds": padded,
        "pair_weight": torch.tensor(
            [b.get("pair_weight", 1.0) for b in batch],
            dtype=torch.float32,
        ),
    }


def _load_threshold_sft_samples(
    scores_csv: Path,
    *,
    score_threshold: float,
    score_column: str,
) -> list[dict]:
    samples: list[dict] = []
    with scores_csv.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        _require_columns(scores_csv, reader.fieldnames, {"id", "version", score_column})
        for row in reader:
            score = float(row[score_column])
            if score >= score_threshold:
                samples.append(
                    {
                        "prompt_id": row["id"],
                        "version": int(row["version"]),
                        "score": score,
                    }
                )
    return samples


def _load_selected_sft_samples(
    path: Path,
    *,
    expected_mode: str,
    scores_csv: Path,
    score_column: str,
    score_threshold: float,
    hard_negative_threshold: float,
) -> list[dict]:
    samples: list[dict] = []
    expected_source_hash = _sha256_file(scores_csv)
    source_scores = _load_score_index(scores_csv, score_column=score_column)
    seen_samples: set[tuple[str, int]] = set()
    for line_number, row in _read_jsonl(path):
        if row.get("schema_version") != "selected-samples/v1":
            raise ValueError(f"{path}:{line_number}: expected selected-samples/v1")
        if row.get("selection_mode") != expected_mode:
            raise ValueError(
                f"{path}:{line_number}: selection_mode does not match config {expected_mode!r}"
            )
        _require_row_value(path, line_number, row, "score_column", score_column)
        _require_row_value(path, line_number, row, "threshold", score_threshold)
        _require_row_value(
            path,
            line_number,
            row,
            "source_scores_sha256",
            expected_source_hash,
        )
        if expected_mode == "hard_positive":
            _require_row_value(
                path,
                line_number,
                row,
                "hard_negative_threshold",
                hard_negative_threshold,
            )
        prompt_id = str(row["prompt_id"])
        version = int(row["version"])
        sample_key = (prompt_id, version)
        if sample_key in seen_samples:
            raise ValueError(f"{path}:{line_number}: duplicate selected sample: {sample_key}")
        seen_samples.add(sample_key)
        selected_score = float(row["selected_score"])
        if not math.isfinite(selected_score) or selected_score < score_threshold:
            raise ValueError(f"{path}:{line_number}: selected_score violates config threshold")
        _require_source_score(
            path,
            line_number,
            source_scores,
            prompt_id=prompt_id,
            version=version,
            recorded_score=selected_score,
            role="selected sample",
        )
        samples.append(
            {
                "prompt_id": prompt_id,
                "version": version,
                "score": selected_score,
                "sample_weight": row.get("sample_weight"),
            }
        )
    return samples


def _assign_sample_weights(samples: list[dict], *, sample_weighting: str) -> None:
    if sample_weighting == "uniform":
        for sample in samples:
            sample["sample_weight"] = 1.0
        return
    max_score = max((float(sample["score"]) for sample in samples), default=0.0)
    for sample in samples:
        recorded = sample.get("sample_weight")
        sample["sample_weight"] = (
            float(recorded)
            if recorded is not None
            else (float(sample["score"]) / max_score if max_score > 0 else 0.0)
        )
        if not math.isfinite(sample["sample_weight"]) or sample["sample_weight"] < 0:
            raise ValueError("sample weights must be finite and non-negative")


def _load_best_worst_pairs(
    scores_csv: Path,
    *,
    score_threshold: float,
    score_diff_min: float,
    score_column: str,
) -> list[dict]:
    by_prompt: dict[str, list[dict]] = defaultdict(list)
    with scores_csv.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        _require_columns(scores_csv, reader.fieldnames, {"id", "version", score_column})
        for row in reader:
            by_prompt[row["id"]].append(
                {"version": int(row["version"]), "score": float(row[score_column])}
            )

    pairs: list[dict] = []
    for prompt_id, versions in by_prompt.items():
        if len(versions) < 2:
            continue
        versions.sort(key=lambda item: (item["score"], item["version"]))
        best = versions[-1]
        worst = versions[0]
        if best["score"] < score_threshold or best["score"] - worst["score"] < score_diff_min:
            continue
        pairs.append(
            {
                "prompt_id": prompt_id,
                "winner_version": best["version"],
                "loser_version": worst["version"],
                "winner_score": best["score"],
                "loser_score": worst["score"],
            }
        )
    return pairs


def _load_materialized_dpo_pairs(
    path: Path,
    *,
    expected_mode: str,
    scores_csv: Path,
    score_column: str,
    score_threshold: float,
    score_diff_min: float,
    ambiguity_margin: float,
) -> list[dict]:
    pairs: list[dict] = []
    expected_source_hash = _sha256_file(scores_csv)
    source_scores = _load_score_index(scores_csv, score_column=score_column)
    seen_pairs: set[tuple[str, int, int]] = set()
    for line_number, row in _read_jsonl(path):
        if row.get("schema_version") != "preference-pairs/v1":
            raise ValueError(f"{path}:{line_number}: expected preference-pairs/v1")
        if row.get("pair_construction_mode") != expected_mode:
            raise ValueError(
                f"{path}:{line_number}: pair_construction_mode does not match config "
                f"{expected_mode!r}"
            )
        _require_row_value(path, line_number, row, "score_column", score_column)
        _require_row_value(path, line_number, row, "threshold", score_threshold)
        _require_row_value(path, line_number, row, "minimum_margin", score_diff_min)
        _require_row_value(path, line_number, row, "ambiguity_margin", ambiguity_margin)
        _require_row_value(
            path,
            line_number,
            row,
            "source_scores_sha256",
            expected_source_hash,
        )
        prompt_id = str(row["prompt_id"])
        winner_version = int(row["winner_version"])
        loser_version = int(row["loser_version"])
        winner_score = float(row["winner_score"])
        loser_score = float(row["loser_score"])
        pair_key = (prompt_id, winner_version, loser_version)
        if pair_key in seen_pairs:
            raise ValueError(f"{path}:{line_number}: duplicate preference pair: {pair_key}")
        seen_pairs.add(pair_key)
        margin = winner_score - loser_score
        if winner_version == loser_version:
            raise ValueError(f"{path}:{line_number}: winner and loser versions must differ")
        if not all(math.isfinite(value) for value in (winner_score, loser_score, margin)):
            raise ValueError(f"{path}:{line_number}: pair scores must be finite")
        source_winner_score = _require_source_score(
            path,
            line_number,
            source_scores,
            prompt_id=prompt_id,
            version=winner_version,
            recorded_score=winner_score,
            role="winner",
        )
        source_loser_score = _require_source_score(
            path,
            line_number,
            source_scores,
            prompt_id=prompt_id,
            version=loser_version,
            recorded_score=loser_score,
            role="loser",
        )
        if source_winner_score <= source_loser_score:
            raise ValueError(
                f"{path}:{line_number}: source scores do not preserve winner/loser direction"
            )
        if winner_score < score_threshold or margin < score_diff_min or margin <= ambiguity_margin:
            raise ValueError(f"{path}:{line_number}: preference pair violates config thresholds")
        pairs.append(
            {
                "prompt_id": prompt_id,
                "winner_version": winner_version,
                "loser_version": loser_version,
                "winner_score": winner_score,
                "loser_score": loser_score,
                "pair_weight": row.get("pair_weight"),
            }
        )
    return pairs


def _load_score_index(scores_csv: Path, *, score_column: str) -> dict[tuple[str, int], float]:
    scores: dict[tuple[str, int], float] = {}
    with scores_csv.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        _require_columns(scores_csv, reader.fieldnames, {"id", "version", score_column})
        for line_number, row in enumerate(reader, start=2):
            prompt_id = str(row.get("id") or "")
            if not prompt_id:
                raise ValueError(f"{scores_csv}:{line_number}: source prompt id is required")
            try:
                version = int(str(row.get("version") or ""))
            except ValueError as exc:
                raise ValueError(
                    f"{scores_csv}:{line_number}: source version must be an integer"
                ) from exc
            try:
                score = float(str(row.get(score_column) or ""))
            except ValueError as exc:
                raise ValueError(
                    f"{scores_csv}:{line_number}: source {score_column} must be numeric"
                ) from exc
            if not math.isfinite(score):
                raise ValueError(f"{scores_csv}:{line_number}: source score must be finite")
            key = (prompt_id, version)
            if key in scores:
                raise ValueError(f"{scores_csv}:{line_number}: duplicate source score row: {key}")
            scores[key] = score
    return scores


def _require_source_score(
    artifact_path: Path,
    line_number: int,
    source_scores: dict[tuple[str, int], float],
    *,
    prompt_id: str,
    version: int,
    recorded_score: float,
    role: str,
) -> float:
    key = (prompt_id, version)
    if key not in source_scores:
        raise ValueError(
            f"{artifact_path}:{line_number}: {role} {key} is absent from the source score CSV"
        )
    source_score = source_scores[key]
    if recorded_score != source_score:
        raise ValueError(
            f"{artifact_path}:{line_number}: {role} score {recorded_score!r} does not match "
            f"source score {source_score!r} for {key}"
        )
    return source_score


def require_drop_last_batch(
    dataset_size: int,
    batch_size: int,
    *,
    stage: str,
) -> None:
    """Reject datasets that would make a ``drop_last=True`` loader yield no batches."""

    if batch_size < 1:
        raise ValueError(f"{stage}: batch_size must be >= 1")
    if dataset_size == 0:
        raise ValueError(f"{stage}: selected dataset is empty")
    if dataset_size < batch_size:
        raise ValueError(
            f"{stage}: selected dataset has {dataset_size} items, smaller than "
            f"batch_size={batch_size}; drop_last=True would yield no batches"
        )


def _assign_pair_weights(pairs: list[dict], *, pair_weighting: str) -> None:
    if pair_weighting == "uniform":
        for pair in pairs:
            pair["pair_weight"] = 1.0
        return
    margins = [float(pair["winner_score"]) - float(pair["loser_score"]) for pair in pairs]
    max_margin = max(margins, default=0.0)
    for pair, margin in zip(pairs, margins, strict=True):
        recorded = pair.get("pair_weight")
        pair["pair_weight"] = (
            float(recorded)
            if recorded is not None
            else (margin / max_margin if max_margin > 0 else 0.0)
        )
        if not math.isfinite(pair["pair_weight"]) or pair["pair_weight"] < 0:
            raise ValueError("pair weights must be finite and non-negative")


def _read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            yield line_number, row


def _require_columns(path: Path, fieldnames: list[str] | None, required: set[str]) -> None:
    missing = required - set(fieldnames or [])
    if missing:
        raise ValueError(f"{path}: missing required columns: {', '.join(sorted(missing))}")


def _require_row_value(
    path: Path,
    line_number: int,
    row: dict,
    field: str,
    expected: object,
) -> None:
    if row.get(field) != expected:
        raise ValueError(
            f"{path}:{line_number}: {field}={row.get(field)!r} does not match config {expected!r}"
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
            "latent": lat_data["latent"],  # (C, H, W)
            "mask_lat": lat_data["mask_lat"],  # (H, W) float in [0, 1]
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
                    map_location="cpu",
                    weights_only=True,
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
            n,
            batch_size,
            drop_last,
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
