from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import torch

from src.training.dataset import (
    DPODataset,
    MaskedSFTDataset,
    ResolutionBucketSampler,
    SFTDataset,
    dpo_collate_fn,
    masked_sft_collate_fn,
    require_drop_last_batch,
    sft_collate_fn,
)
from src.training.selection import materialize_dpo_pairs, materialize_sft_samples


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


def _write_masked_sample(
    data_dir: Path,
    sample_id: str,
    latent: torch.Tensor,
    mask_lat: torch.Tensor,
    prompt_embeds: torch.Tensor,
) -> None:
    latents_dir = data_dir / "latents"
    embeds_dir = data_dir / "text_embeds"
    latents_dir.mkdir(parents=True, exist_ok=True)
    embeds_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"latent": latent, "mask_lat": mask_lat}, latents_dir / f"{sample_id}.pt")
    torch.save({"prompt_embeds": prompt_embeds}, embeds_dir / f"{sample_id}.pt")


def _write_shapes(path: Path, rows: list[dict[str, object]]) -> Path:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "H", "W"])
        writer.writeheader()
        writer.writerows(rows)
    return path


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


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
    _write_text_embed(
        text_embeds_dir,
        "p2",
        torch.arange(9, dtype=torch.float32).reshape(3, 3),
    )

    dataset = SFTDataset(
        str(latents_dir),
        str(text_embeds_dir),
        str(scores_csv),
        score_threshold=0.3,
    )

    assert len(dataset) == 2
    selected_samples = [
        (sample["prompt_id"], sample["version"], sample["score"]) for sample in dataset.samples
    ]
    assert selected_samples == [
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
    assert torch.allclose(collated["sample_weight"], torch.ones(2))


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
            "pair_weight": 1.0,
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
    assert torch.allclose(collated["pair_weight"], torch.ones(2))


def test_datasets_consume_materialized_selection_and_pair_weights(tmp_path: Path) -> None:
    scores_csv = _write_scores(
        tmp_path / "scores.csv",
        [
            {"id": "p1", "version": 0, "score": "0.2", "target_text": "Ёж"},
            {"id": "p1", "version": 1, "score": "0.8", "target_text": "Ёж"},
            {"id": "p2", "version": 0, "score": "0.1", "target_text": "Щит"},
            {"id": "p2", "version": 1, "score": "0.5", "target_text": "Щит"},
        ],
    )
    selected_path = tmp_path / "selected.jsonl"
    pairs_path = tmp_path / "pairs.jsonl"
    materialize_sft_samples(
        scores_csv,
        selected_path,
        mode="score_weighted",
        threshold=0.3,
    )
    materialize_dpo_pairs(
        scores_csv,
        pairs_path,
        mode="margin_weighted",
        threshold=0.3,
        margin=0.1,
    )

    sft_dataset = SFTDataset(
        str(tmp_path / "latents"),
        str(tmp_path / "embeds"),
        str(scores_csv),
        selection_mode="score_weighted",
        selected_samples_path=str(selected_path),
        sample_weighting="score_normalized",
    )
    dpo_dataset = DPODataset(
        str(tmp_path / "latents"),
        str(tmp_path / "embeds"),
        str(scores_csv),
        pair_construction_mode="margin_weighted",
        preference_pairs_path=str(pairs_path),
        score_threshold=0.3,
        pair_weighting="margin_normalized",
    )

    assert [(row["prompt_id"], row["version"]) for row in sft_dataset.samples] == [
        ("p1", 1),
        ("p2", 1),
    ]
    assert [row["sample_weight"] for row in sft_dataset.samples] == [1.0, 0.625]
    assert [row["pair_weight"] for row in dpo_dataset.pairs] == pytest.approx([1.0, 2 / 3])


def test_materialized_artifacts_reject_source_drift_and_inverted_pairs(tmp_path: Path) -> None:
    scores_csv = _write_scores(
        tmp_path / "scores.csv",
        [
            {"id": "p1", "version": 0, "score": "0.1", "target_text": "Ёж"},
            {"id": "p1", "version": 1, "score": "0.9", "target_text": "Ёж"},
        ],
    )
    selected_path = tmp_path / "selected.jsonl"
    pairs_path = tmp_path / "pairs.jsonl"
    materialize_sft_samples(scores_csv, selected_path, threshold=0.3)
    materialize_dpo_pairs(scores_csv, pairs_path, threshold=0.5, margin=0.1)

    selected_rows = _read_jsonl(selected_path)
    selected_rows[0]["source_scores_sha256"] = "wrong"
    selected_path.write_text(
        "".join(json.dumps(row) + "\n" for row in selected_rows),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="source_scores_sha256"):
        SFTDataset(
            str(tmp_path / "latents"),
            str(tmp_path / "embeds"),
            str(scores_csv),
            selected_samples_path=str(selected_path),
        )

    pair_rows = _read_jsonl(pairs_path)
    pair_rows[0]["winner_score"], pair_rows[0]["loser_score"] = (
        pair_rows[0]["loser_score"],
        pair_rows[0]["winner_score"],
    )
    pairs_path.write_text(
        "".join(json.dumps(row) + "\n" for row in pair_rows),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="does not match source score"):
        DPODataset(
            str(tmp_path / "latents"),
            str(tmp_path / "embeds"),
            str(scores_csv),
            preference_pairs_path=str(pairs_path),
        )


def test_materialized_sft_rejects_below_threshold_version_substitution(tmp_path: Path) -> None:
    scores_csv = _write_scores(
        tmp_path / "scores.csv",
        [
            {"id": "p1", "version": 0, "score": "0.1", "target_text": "Ёж"},
            {"id": "p1", "version": 1, "score": "0.9", "target_text": "Ёж"},
        ],
    )
    selected_path = tmp_path / "selected.jsonl"
    materialize_sft_samples(scores_csv, selected_path, threshold=0.3)
    selected_rows = _read_jsonl(selected_path)
    selected_rows[0]["version"] = 0
    selected_path.write_text(
        "".join(json.dumps(row) + "\n" for row in selected_rows),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="does not match source score"):
        SFTDataset(
            str(tmp_path / "latents"),
            str(tmp_path / "embeds"),
            str(scores_csv),
            selected_samples_path=str(selected_path),
        )


def test_materialized_dpo_rejects_swapped_winner_loser_versions(tmp_path: Path) -> None:
    scores_csv = _write_scores(
        tmp_path / "scores.csv",
        [
            {"id": "p1", "version": 0, "score": "0.1", "target_text": "Ёж"},
            {"id": "p1", "version": 1, "score": "0.9", "target_text": "Ёж"},
        ],
    )
    pairs_path = tmp_path / "pairs.jsonl"
    materialize_dpo_pairs(scores_csv, pairs_path, threshold=0.5, margin=0.1)
    pair_rows = _read_jsonl(pairs_path)
    pair_rows[0]["winner_version"], pair_rows[0]["loser_version"] = (
        pair_rows[0]["loser_version"],
        pair_rows[0]["winner_version"],
    )
    pairs_path.write_text(
        "".join(json.dumps(row) + "\n" for row in pair_rows),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="does not match source score"):
        DPODataset(
            str(tmp_path / "latents"),
            str(tmp_path / "embeds"),
            str(scores_csv),
            preference_pairs_path=str(pairs_path),
        )


@pytest.mark.parametrize(
    ("dataset_size", "message"),
    [(0, "selected dataset is empty"), (3, "drop_last=True would yield no batches")],
)
def test_drop_last_training_requires_at_least_one_full_batch(
    dataset_size: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        require_drop_last_batch(dataset_size, 4, stage="sft")


def test_masked_sft_dataset_fails_fast_for_missing_required_directories(tmp_path: Path) -> None:
    missing_root = tmp_path / "missing-root"
    with pytest.raises(FileNotFoundError, match="Missing latents dir"):
        MaskedSFTDataset(str(missing_root))

    latents_only = tmp_path / "latents-only"
    (latents_only / "latents").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="Missing text_embeds dir"):
        MaskedSFTDataset(str(latents_only))


def test_masked_sft_dataset_loads_matched_latents_masks_and_text_embeds(tmp_path: Path) -> None:
    data_dir = tmp_path / "masked"
    _write_masked_sample(
        data_dir,
        "sample-b",
        latent=torch.full((2, 3, 4), 2.0),
        mask_lat=torch.ones(3, 4),
        prompt_embeds=torch.arange(12, dtype=torch.float32).reshape(3, 4),
    )
    _write_masked_sample(
        data_dir,
        "sample-a",
        latent=torch.full((2, 3, 4), 1.0),
        mask_lat=torch.zeros(3, 4),
        prompt_embeds=torch.arange(8, dtype=torch.float32).reshape(2, 4),
    )

    dataset = MaskedSFTDataset(str(data_dir))

    assert len(dataset) == 2
    assert dataset.sample_ids == ["sample-a", "sample-b"]
    item = dataset[0]
    assert item["sample_id"] == "sample-a"
    assert item["latent"].shape == (2, 3, 4)
    assert item["mask_lat"].shape == (3, 4)
    assert item["prompt_embeds"].shape == (2, 4)
    assert torch.equal(item["latent"], torch.full((2, 3, 4), 1.0))


def test_masked_sft_collate_stacks_masks_latents_and_pads_text_embeddings() -> None:
    batch = [
        {
            "sample_id": "sample-a",
            "latent": torch.ones(2, 2, 2),
            "mask_lat": torch.ones(2, 2),
            "prompt_embeds": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        },
        {
            "sample_id": "sample-b",
            "latent": torch.full((2, 2, 2), 2.0),
            "mask_lat": torch.zeros(2, 2),
            "prompt_embeds": torch.tensor([[5.0, 6.0]]),
        },
    ]

    collated = masked_sft_collate_fn(batch)

    assert collated["sample_ids"] == ["sample-a", "sample-b"]
    assert collated["latent"].shape == (2, 2, 2, 2)
    assert collated["mask_lat"].shape == (2, 2, 2)
    assert collated["prompt_embeds"].shape == (2, 2, 2)
    assert torch.equal(collated["prompt_embeds"][1, 0], torch.tensor([5.0, 6.0]))
    assert torch.equal(collated["prompt_embeds"][1, 1], torch.zeros(2))


def test_resolution_bucket_sampler_uses_complete_shapes_csv_without_latent_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_dir = tmp_path / "masked"
    for sample_id, shape in {
        "sample-a": (2, 2),
        "sample-b": (2, 2),
        "sample-c": (3, 2),
        "sample-d": (3, 2),
    }.items():
        height, width = shape
        _write_masked_sample(
            data_dir,
            sample_id,
            latent=torch.zeros(1, height, width),
            mask_lat=torch.zeros(height, width),
            prompt_embeds=torch.ones(1, 2),
        )
    _write_shapes(
        data_dir / "shapes.csv",
        [
            {"id": "sample-a", "H": 2, "W": 2},
            {"id": "sample-b", "H": 2, "W": 2},
            {"id": "sample-c", "H": 3, "W": 2},
            {"id": "sample-d", "H": 3, "W": 2},
        ],
    )
    dataset = MaskedSFTDataset(str(data_dir))

    def fail_if_called(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("ResolutionBucketSampler should use complete shapes.csv")

    monkeypatch.setattr(torch, "load", fail_if_called)
    sampler_a = ResolutionBucketSampler(
        dataset,
        batch_size=2,
        shuffle=True,
        drop_last=False,
        seed=17,
    )
    sampler_b = ResolutionBucketSampler(
        dataset,
        batch_size=2,
        shuffle=True,
        drop_last=False,
        seed=17,
    )

    batches = list(sampler_a)
    assert batches == list(sampler_b)
    assert len(sampler_a) == 2
    batch_shapes = {
        dataset.shapes[dataset.sample_ids[index]] for batch in batches for index in batch
    }
    assert sorted(batch_shapes) == [
        (2, 2),
        (3, 2),
    ]
    for batch in batches:
        assert len({dataset.shapes[dataset.sample_ids[index]] for index in batch}) == 1


def test_materialized_sft_selections_match_sft_dataset_threshold_semantics(
    tmp_path: Path,
) -> None:
    latents_dir = tmp_path / "latents"
    text_embeds_dir = tmp_path / "text_embeds"
    scores_csv = _write_scores(
        tmp_path / "scores.csv",
        [
            {"id": "p2", "version": 2, "score": "0.90", "target_text": "Жук"},
            {"id": "p1", "version": 1, "score": "0.29", "target_text": "Ёж"},
            {"id": "p1", "version": 2, "score": "0.30", "target_text": "Ёж"},
            {"id": "p2", "version": 1, "score": "0.70", "target_text": "Жук"},
        ],
    )
    for prompt_id, versions in {"p1": [2], "p2": [1, 2]}.items():
        for version in versions:
            _write_latent(latents_dir, prompt_id, version, torch.zeros(1, 2, 2))
        _write_text_embed(text_embeds_dir, prompt_id, torch.ones(1, 2))
    dataset = SFTDataset(
        str(latents_dir),
        str(text_embeds_dir),
        str(scores_csv),
        score_threshold=0.3,
    )

    summary = materialize_sft_samples(
        scores_csv,
        tmp_path / "selected_samples.jsonl",
        threshold=0.3,
    )

    dataset_keys = sorted((sample["prompt_id"], sample["version"]) for sample in dataset.samples)
    materialized_keys = [
        (str(row["prompt_id"]), int(row["version"]))
        for row in _read_jsonl(tmp_path / "selected_samples.jsonl")
    ]
    assert summary["selected_count"] == len(dataset)
    assert materialized_keys == dataset_keys == [("p1", 2), ("p2", 1), ("p2", 2)]


def test_materialized_dpo_pairs_reject_equal_and_ambiguous_scores(tmp_path: Path) -> None:
    scores_csv = _write_scores(
        tmp_path / "scores.csv",
        [
            {"id": "equal", "version": 1, "score": "0.70", "target_text": "Ёж"},
            {"id": "equal", "version": 2, "score": "0.70", "target_text": "Ёж"},
            {"id": "ambiguous", "version": 1, "score": "0.50", "target_text": "Жук"},
            {"id": "ambiguous", "version": 2, "score": "0.55", "target_text": "Жук"},
            {"id": "selected", "version": 1, "score": "0.20", "target_text": "Цех"},
            {"id": "selected", "version": 2, "score": "0.90", "target_text": "Цех"},
        ],
    )

    summary = materialize_dpo_pairs(
        scores_csv,
        tmp_path / "preference_pairs.jsonl",
        threshold=0.5,
        margin=0.1,
        ambiguity_margin=0.0,
    )

    rows = _read_jsonl(tmp_path / "preference_pairs.jsonl")
    assert summary["pair_count"] == 1
    assert summary["filtering_stats"]["ambiguous_below_margin"] == 2
    assert [(row["prompt_id"], row["winner_version"], row["loser_version"]) for row in rows] == [
        ("selected", 2, 1)
    ]
    assert float(rows[0]["winner_score"]) > float(rows[0]["loser_score"])
    assert rows[0]["margin"] == 0.7
