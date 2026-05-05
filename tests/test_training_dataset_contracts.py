from __future__ import annotations

import csv
from pathlib import Path

import torch

from src.training.dataset import DPODataset, SFTDataset, dpo_collate_fn, sft_collate_fn


def _write_scores(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["id", "version", "score", "target_text"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _write_latent(root: Path, prompt_id: str, version: int, latent: torch.Tensor) -> Path:
    path = root / prompt_id / f"v{version}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"latent": latent}, path)
    return path


def _write_text_embed(root: Path, prompt_id: str, prompt_embeds: torch.Tensor) -> Path:
    path = root / f"{prompt_id}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"prompt_embeds": prompt_embeds}, path)
    return path


def test_sft_dataset_loads_score_filtered_tensor_samples(tmp_path: Path) -> None:
    latents_dir = tmp_path / "latents"
    text_embeds_dir = tmp_path / "text_embeds"
    scores_csv = _write_scores(
        tmp_path / "scores.csv",
        [
            {"id": "p1", "version": 1, "score": "0.29", "target_text": "Ёж"},
            {"id": "p1", "version": 2, "score": "0.30", "target_text": "Ёж"},
            {"id": "p2", "version": 1, "score": "0.90", "target_text": "Жук"},
        ],
    )
    _write_latent(latents_dir, "p1", 2, torch.full((2, 3, 4), 1.5))
    _write_latent(latents_dir, "p2", 1, torch.full((2, 3, 4), 2.5))
    _write_text_embed(text_embeds_dir, "p1", torch.arange(6, dtype=torch.float32).reshape(2, 3))
    _write_text_embed(text_embeds_dir, "p2", torch.arange(9, dtype=torch.float32).reshape(3, 3))

    dataset = SFTDataset(str(latents_dir), str(text_embeds_dir), str(scores_csv), score_threshold=0.3)

    assert len(dataset) == 2
    assert [(sample["prompt_id"], sample["version"], sample["score"]) for sample in dataset.samples] == [
        ("p1", 2, 0.3),
        ("p2", 1, 0.9),
    ]
    item = dataset[0]
    assert item["latent"].shape == (2, 3, 4)
    assert item["prompt_embeds"].shape == (2, 3)
    assert item["score"] == 0.3
    assert torch.equal(item["latent"], torch.full((2, 3, 4), 1.5))


def test_sft_collate_stacks_latents_and_pads_prompt_embeddings() -> None:
    batch = [
        {
            "latent": torch.ones(2, 2, 2),
            "prompt_embeds": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            "score": 0.7,
        },
        {
            "latent": torch.full((2, 2, 2), 2.0),
            "prompt_embeds": torch.tensor([[5.0, 6.0]]),
            "score": 0.9,
        },
    ]

    collated = sft_collate_fn(batch)

    assert collated["latent"].shape == (2, 2, 2, 2)
    assert collated["prompt_embeds"].shape == (2, 2, 2)
    assert torch.equal(collated["prompt_embeds"][1, 0], torch.tensor([5.0, 6.0]))
    assert torch.equal(collated["prompt_embeds"][1, 1], torch.zeros(2))
    assert torch.allclose(collated["score"], torch.tensor([0.7, 0.9]))


def test_dpo_dataset_constructs_pairs_when_winner_and_margin_pass(tmp_path: Path) -> None:
    latents_dir = tmp_path / "latents"
    text_embeds_dir = tmp_path / "text_embeds"
    scores_csv = _write_scores(
        tmp_path / "scores.csv",
        [
            {"id": "p1", "version": 1, "score": "0.20", "target_text": "Ёж"},
            {"id": "p1", "version": 2, "score": "0.80", "target_text": "Ёж"},
            {"id": "p2", "version": 1, "score": "0.45", "target_text": "Жук"},
            {"id": "p2", "version": 2, "score": "0.53", "target_text": "Жук"},
            {"id": "p3", "version": 1, "score": "0.10", "target_text": "Цех"},
            {"id": "p3", "version": 2, "score": "0.40", "target_text": "Цех"},
        ],
    )
    for prompt_id, versions in {"p1": [1, 2], "p2": [1, 2], "p3": [1, 2]}.items():
        for version in versions:
            _write_latent(latents_dir, prompt_id, version, torch.full((1, 2, 2), float(version)))
        _write_text_embed(text_embeds_dir, prompt_id, torch.ones(2, 3))

    dataset = DPODataset(
        str(latents_dir),
        str(text_embeds_dir),
        str(scores_csv),
        score_threshold=0.5,
        score_diff_min=0.1,
    )

    assert len(dataset) == 1
    assert dataset.pairs == [
        {
            "prompt_id": "p1",
            "winner_version": 2,
            "loser_version": 1,
            "winner_score": 0.8,
            "loser_score": 0.2,
        }
    ]
    item = dataset[0]
    assert item["winner_latent"].shape == (1, 2, 2)
    assert item["loser_latent"].shape == (1, 2, 2)
    assert item["prompt_embeds"].shape == (2, 3)
    assert torch.equal(item["winner_latent"], torch.full((1, 2, 2), 2.0))


def test_dpo_collate_stacks_pair_latents_and_pads_shared_prompt_embeddings() -> None:
    batch = [
        {
            "winner_latent": torch.ones(1, 2, 2),
            "loser_latent": torch.zeros(1, 2, 2),
            "prompt_embeds": torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
        },
        {
            "winner_latent": torch.full((1, 2, 2), 3.0),
            "loser_latent": torch.full((1, 2, 2), -1.0),
            "prompt_embeds": torch.tensor([[7.0, 8.0]]),
        },
    ]

    collated = dpo_collate_fn(batch)

    assert collated["winner_latent"].shape == (2, 1, 2, 2)
    assert collated["loser_latent"].shape == (2, 1, 2, 2)
    assert collated["prompt_embeds"].shape == (2, 3, 2)
    assert torch.equal(collated["prompt_embeds"][1, 0], torch.tensor([7.0, 8.0]))
    assert torch.equal(collated["prompt_embeds"][1, 1:], torch.zeros(2, 2))
