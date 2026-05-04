from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

from src.data_quality.synthetic_quality import inspect_synthetic_dataset


def _write_masked_fixture(root: Path) -> tuple[Path, Path]:
    data_dir = root / "masked_sft"
    raw_dir = root / "raw"
    for directory in (
        data_dir / "raw_imgs",
        data_dir / "raw_masks",
        raw_dir / "meta",
    ):
        directory.mkdir(parents=True, exist_ok=True)

    rows = [
        {
            "id": "sample-1",
            "resolution": "10",
            "n_words": "1",
            "text": "Ёж",
            "caption": "Render Ёж",
        },
        {
            "id": "sample-2",
            "resolution": "10",
            "n_words": "1",
            "text": "Щит",
            "caption": "Render Щит",
        },
    ]
    with (data_dir / "index.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["id", "resolution", "n_words", "text", "caption"]
        )
        writer.writeheader()
        writer.writerows(rows)
    with (data_dir / "prompts.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps({"id": row["id"], "prompt": row["caption"]}, ensure_ascii=False) + "\n"
            )
    with (data_dir / "shapes.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "H", "W"])
        writer.writeheader()
        writer.writerows(
            [{"id": "sample-1", "H": "2", "W": "2"}, {"id": "sample-2", "H": "2", "W": "2"}]
        )

    _write_image_and_mask(data_dir, "sample-1", text_box=(2, 2, 7, 7), fill=240, background=20)
    _write_image_and_mask(data_dir, "sample-2", text_box=(1, 1, 8, 3), fill=120, background=100)
    (raw_dir / "meta" / "sample-1.json").write_text(
        json.dumps(
            {
                "id": "sample-1",
                "resolution": 10,
                "annotations": [{"text": "Ёж", "bbox": [2, 2, 5, 5], "font": "Serif"}],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (raw_dir / "meta" / "sample-2.json").write_text(
        json.dumps(
            {
                "id": "sample-2",
                "resolution": 10,
                "annotations": [{"text": "Щит", "bbox": [1, 1, 7, 2], "font": "Sans"}],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return data_dir, raw_dir


def _write_image_and_mask(
    data_dir: Path,
    sample_id: str,
    *,
    text_box: tuple[int, int, int, int],
    fill: int,
    background: int,
) -> None:
    image = Image.new("RGB", (10, 10), color=(background, background, background))
    mask = Image.new("L", (10, 10), color=0)
    image_draw = ImageDraw.Draw(image)
    mask_draw = ImageDraw.Draw(mask)
    image_draw.rectangle(text_box, fill=(fill, fill, fill))
    mask_draw.rectangle(text_box, fill=255)
    image.save(data_dir / "raw_imgs" / f"{sample_id}.png")
    mask.save(data_dir / "raw_masks" / f"{sample_id}.png")


def test_synthetic_quality_reports_metrics_and_cpu_safe_imports(tmp_path: Path) -> None:
    data_dir, raw_dir = _write_masked_fixture(tmp_path)

    report = inspect_synthetic_dataset(data_dir, raw_dir=raw_dir)

    assert report.ok
    assert report.schema_version == "synthetic-quality/v1"
    assert report.sample_count == 2
    assert report.missing_files == {}
    assert report.mask_area_fraction["count"] == 2
    assert report.mask_area_fraction["min"] > 0
    assert report.bbox_height_fraction["max"] == 0.5
    assert report.contrast["min"] == 20.0
    assert report.character_coverage["counts"]["ё"] == 1
    assert report.character_coverage["counts"]["щ"] == 1
    assert report.font_coverage == {"Sans": 1, "Serif": 1}
    assert report.resolution_distribution == {"10x10": 2}
    assert "paddleocr" not in sys.modules
    assert "diffusers" not in sys.modules
    assert "transformers" not in sys.modules


def test_synthetic_quality_reports_threshold_rejections_and_optional_ocr(tmp_path: Path) -> None:
    data_dir, raw_dir = _write_masked_fixture(tmp_path)
    ocr_path = tmp_path / "ocr.csv"
    with ocr_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "target_text", "ocr_text"])
        writer.writeheader()
        writer.writerows(
            [
                {"id": "sample-1", "target_text": "Ёж", "ocr_text": "Ёж"},
                {"id": "sample-2", "target_text": "Щит", "ocr_text": "Шит"},
            ]
        )

    report = inspect_synthetic_dataset(
        data_dir,
        raw_dir=raw_dir,
        ocr_results=ocr_path,
        thresholds={
            "min_mask_area_fraction": 0.2,
            "min_contrast": 30.0,
            "min_bbox_height_fraction": 0.3,
        },
    )

    assert not report.ok
    assert report.accepted_count == 1
    assert report.rejected_count == 1
    assert report.rejection_reasons == {
        "bbox_height_fraction_below_min": 1,
        "contrast_below_min": 1,
    }
    assert report.ocr_summary == {
        "count": 2,
        "exact_matches": 1,
        "exact_match_rate": 0.5,
        "mean_cer": 0.166667,
    }


def test_synthetic_inspection_cli_writes_report_manifest_and_contact_sheet(
    tmp_path: Path,
) -> None:
    from scripts.inspect_synthetic_dataset import main

    data_dir, raw_dir = _write_masked_fixture(tmp_path)
    report_path = tmp_path / "reports" / "synthetic-quality.json"
    manifest_path = tmp_path / "reports" / "synthetic-manifest.json"
    contact_sheet_path = tmp_path / "reports" / "contact-sheet.png"

    exit_code = main(
        [
            "--data-dir",
            str(data_dir),
            "--raw-dir",
            str(raw_dir),
            "--report",
            str(report_path),
            "--manifest",
            str(manifest_path),
            "--contact-sheet",
            str(contact_sheet_path),
            "--contact-sheet-samples",
            "2",
        ]
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert report["schema_version"] == "synthetic-quality/v1"
    assert report["sample_count"] == 2
    assert manifest["schema_version"] == "dataset-manifest/v1"
    assert manifest["dataset_kind"] == "synthetic"
    assert manifest["filtering_stats"]["accepted"] == 2
    assert contact_sheet_path.is_file()
    with Image.open(contact_sheet_path) as sheet:
        assert sheet.size[0] > 10
        assert sheet.size[1] > 10


def test_synthetic_inspection_cli_returns_nonzero_for_blocking_thresholds(
    tmp_path: Path,
) -> None:
    from scripts.inspect_synthetic_dataset import main

    data_dir, raw_dir = _write_masked_fixture(tmp_path)
    report_path = tmp_path / "quality.json"

    exit_code = main(
        [
            "--data-dir",
            str(data_dir),
            "--raw-dir",
            str(raw_dir),
            "--report",
            str(report_path),
            "--min-contrast",
            "300",
        ]
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert exit_code == 1
    assert report["accepted_count"] == 0
    assert report["rejection_reasons"] == {"contrast_below_min": 2}


def test_synthetic_quality_docs_cover_reports_manifests_and_artifact_safety() -> None:
    docs = Path("docs/synthetic_quality.md").read_text(encoding="utf-8")
    required = [
        "inspect_synthetic_dataset.py",
        "scripts/synth/build_dataset.py",
        "raw/imgs/{sid}.png",
        "masked_sft/index.csv",
        "mask area fraction",
        "bbox height fraction",
        "foreground/background contrast",
        "character coverage",
        "font coverage",
        "resolution distribution",
        "OCR verification is optional",
        "--ocr-results runs/synthetic-quality/ocr-results.csv",
        "--contact-sheet runs/synthetic-quality/contact-sheet.png",
        "dataset-manifest/v1",
        "Generated reports, manifests, contact sheets, images, masks, tensors, and private OCR outputs are runtime artifacts",
    ]
    missing = [item for item in required if item not in docs]
    assert not missing
