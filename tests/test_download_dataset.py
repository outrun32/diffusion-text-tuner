"""Tests for pinned public prompt-dataset downloads."""

from __future__ import annotations

import json
import sys
from types import ModuleType, SimpleNamespace


def test_download_dataset_records_revision_and_hashes(monkeypatch, tmp_path):
    source = tmp_path / "source.parquet"
    source.write_bytes(b"parquet-fixture")
    output = tmp_path / "prompts.jsonl"
    manifest = tmp_path / "manifest.json"

    datasets_module = ModuleType("datasets")
    datasets_module.load_dataset = lambda *_args, **_kwargs: [
        {
            "id": "p1",
            "prompt": "Render ТЕСТ",
            "target_text": "ТЕСТ",
            "font": "sans",
            "color": "red",
            "effect": None,
            "text_size": "large",
            "char_coverage": '{"т": 2}',
        }
    ]

    class FakeApi:
        def dataset_info(self, *_args, **_kwargs):
            return SimpleNamespace(sha="resolved-commit")

    hub_module = ModuleType("huggingface_hub")
    hub_module.HfApi = FakeApi
    hub_module.hf_hub_download = lambda *_args, **_kwargs: str(source)

    monkeypatch.setitem(sys.modules, "datasets", datasets_module)
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub_module)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "download_dataset.py",
            "--output",
            str(output),
            "--manifest",
            str(manifest),
            "--revision",
            "requested-ref",
        ],
    )

    from scripts import download_dataset

    download_dataset.main()

    row = json.loads(output.read_text(encoding="utf-8"))
    provenance = json.loads(manifest.read_text(encoding="utf-8"))
    assert row["style"] == {"font": "sans", "color": "red", "size": "large"}
    assert provenance["requested_revision"] == "requested-ref"
    assert provenance["resolved_revision"] == "resolved-commit"
    assert provenance["row_count"] == 1
    assert len(provenance["source_sha256"]) == 64
    assert len(provenance["output_sha256"]) == 64
