"""CPU-safe thesis output bundle generation from recorded evidence.

The functions in this module consume local run manifests, score reports,
diagnostic reports, and optional image paths that already exist on disk. They do
not load diffusion models, OCR engines, CUDA, tensors, checkpoints, or external
model weights. SVG plots are written with deterministic text output and contact
sheets use PIL only when explicitly requested by the bundle config.
"""

from __future__ import annotations

import csv
import html
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from src.runtime.manifests import ManifestError, RunManifest, load_run_manifest

SCHEMA_VERSION = "thesis-output-bundle/v1"
CONFIG_SCHEMA_VERSION = "thesis-output-config/v1"
MISSING_MARKERS = (None, "", "none", "null", "nan", "NaN")


class ThesisOutputError(ValueError):
    """Raised when thesis output evidence is malformed or not thesis-ready."""


def build_thesis_output_bundle(
    config: Mapping[str, Any] | str | Path,
    *,
    require_ready: bool = True,
) -> dict[str, Any]:
    """Build tables, SVG plots, and contact sheets from recorded evidence.

    Parameters
    ----------
    config:
        Either a mapping using ``thesis-output-config/v1`` fields or a JSON file
        path containing that mapping.
    require_ready:
        When true, raise :class:`ThesisOutputError` if required provenance is
        missing or malformed. When false, return the bundle with blocking
        readiness errors for inspection.
    """

    config_payload = _load_config(config)
    output_dir = Path(str(config_payload.get("output_dir") or "thesis_outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)
    blocking_errors: list[str] = []
    warnings: list[str] = []

    source_manifests = _load_source_manifests(
        config_payload.get("source_manifests", []), blocking_errors
    )
    score_reports = _load_evidence_reports(
        config_payload.get("score_reports", []), "score report", blocking_errors
    )
    diagnostic_reports = _load_evidence_reports(
        config_payload.get("diagnostic_reports", []),
        "diagnostic report",
        blocking_errors,
    )

    tables = _build_tables(
        config_payload.get("table_specs", []), output_dir, blocking_errors
    )
    svg_plots = _build_svg_plots(
        config_payload.get("svg_plot_specs", []), output_dir, blocking_errors
    )
    contact_sheets = _build_contact_sheets(
        config_payload.get("contact_sheet_specs", []),
        output_dir,
        blocking_errors,
        warnings,
    )

    ready = not blocking_errors
    bundle = _sort_mapping(
        {
            "schema_version": SCHEMA_VERSION,
            "config_schema_version": str(
                config_payload.get("schema_version") or CONFIG_SCHEMA_VERSION
            ),
            "source_manifests": source_manifests,
            "evidence": {
                "score_reports": score_reports,
                "diagnostic_reports": diagnostic_reports,
            },
            "tables": tables,
            "svg_plots": svg_plots,
            "contact_sheets": contact_sheets,
            "readiness": {
                "ready": ready,
                "blocking_errors": blocking_errors,
                "warnings": warnings,
            },
        }
    )
    if require_ready and not ready:
        raise ThesisOutputError(
            "thesis output bundle is not thesis-ready: " + "; ".join(blocking_errors)
        )
    return bundle


def format_thesis_output_markdown(bundle: Mapping[str, Any]) -> str:
    """Render a deterministic Markdown summary for a thesis output bundle."""

    lines = [
        "# Thesis output bundle",
        "",
        f"- schema_version: `{bundle.get('schema_version', '')}`",
        f"- ready: `{bundle.get('readiness', {}).get('ready', False)}`",
        "",
        "## Source manifests",
        "",
        "| Run ID | Stage | Git commit | Config snapshot | Manifest path |",
        "| --- | --- | --- | --- | --- |",
    ]
    for manifest in bundle.get("source_manifests", []):
        lines.append(
            "| {run_id} | {stage} | {commit} | `{config}` | `{path}` |".format(
                run_id=manifest.get("run_id", ""),
                stage=manifest.get("stage", ""),
                commit=manifest.get("git", {}).get("commit", ""),
                config=manifest.get("config_snapshot_path", ""),
                path=manifest.get("path", ""),
            )
        )
    lines.extend(
        [
            "",
            "## Evidence warnings",
            "",
            *_format_list_or_none(bundle.get("readiness", {}).get("warnings", []), "No warnings."),
            "",
            "## Blocking readiness errors",
            "",
            *_format_list_or_none(
                bundle.get("readiness", {}).get("blocking_errors", []), "No blocking errors."
            ),
            "",
            "## Score reports",
            "",
            "| Path | Schema | Records |",
            "| --- | --- | ---: |",
        ]
    )
    for report in bundle.get("evidence", {}).get("score_reports", []):
        lines.append(
            f"| `{report.get('path', '')}` | {report.get('schema_version', '')} | "
            f"{report.get('record_count', 0)} |"
        )
    lines.extend(
        [
            "",
            "## Diagnostic reports",
            "",
            "| Path | Schema | Records |",
            "| --- | --- | ---: |",
        ]
    )
    for report in bundle.get("evidence", {}).get("diagnostic_reports", []):
        lines.append(
            f"| `{report.get('path', '')}` | {report.get('schema_version', '')} | "
            f"{report.get('record_count', 0)} |"
        )
    lines.extend(_format_artifact_section("Tables", bundle.get("tables", [])))
    lines.extend(_format_artifact_section("SVG plots", bundle.get("svg_plots", [])))
    lines.extend(_format_artifact_section("Contact sheets", bundle.get("contact_sheets", [])))
    return "\n".join(lines) + "\n"


def write_thesis_output_bundle(bundle: Mapping[str, Any], path: str | Path) -> None:
    """Write deterministic thesis output bundle JSON."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(_sort_mapping(dict(bundle)), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def write_thesis_output_markdown(bundle: Mapping[str, Any], path: str | Path) -> None:
    """Write a deterministic Markdown thesis output summary."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(format_thesis_output_markdown(bundle), encoding="utf-8")


def _load_config(config: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(config, Mapping):
        payload = dict(config)
    else:
        config_path = Path(config)
        try:
            loaded = json.loads(config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ThesisOutputError(f"{config_path}: malformed config JSON") from exc
        except OSError as exc:
            raise ThesisOutputError(f"{config_path}: could not read config") from exc
        if not isinstance(loaded, dict):
            raise ThesisOutputError(f"{config_path}: config must be a JSON object")
        payload = loaded
    if not isinstance(payload.get("source_manifests", []), list):
        raise ThesisOutputError("source_manifests must be a list")
    for field in ("score_reports", "diagnostic_reports", "table_specs"):
        if not isinstance(payload.get(field, []), list):
            raise ThesisOutputError(f"{field} must be a list")
    for field in ("svg_plot_specs", "contact_sheet_specs"):
        if not isinstance(payload.get(field, []), list):
            raise ThesisOutputError(f"{field} must be a list")
    return payload


def _load_source_manifests(paths: Sequence[Any], errors: list[str]) -> list[dict[str, Any]]:
    manifests: list[dict[str, Any]] = []
    for raw_path in paths:
        manifest_path = Path(str(raw_path))
        if not manifest_path.is_file():
            errors.append(f"missing manifest: {manifest_path}")
            continue
        try:
            manifest = load_run_manifest(manifest_path)
        except ManifestError as exc:
            errors.append(f"malformed manifest {manifest_path}: {exc}")
            continue
        manifests.append(_manifest_summary(manifest))
    return manifests


def _manifest_summary(manifest: RunManifest) -> dict[str, Any]:
    return {
        "path": str(manifest.manifest_path),
        "run_id": manifest.run_id,
        "stage": manifest.stage,
        "git": dict(manifest.git),
        "config_snapshot_path": str(manifest.config_snapshot_path),
        "config_snapshot": dict(manifest.config_snapshot),
        "inputs": dict(manifest.inputs),
        "outputs": dict(manifest.outputs),
        "metrics": dict(manifest.metrics),
    }


def _load_evidence_reports(
    paths: Sequence[Any], report_kind: str, errors: list[str]
) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for raw_path in paths:
        report_path = Path(str(raw_path))
        if not report_path.is_file():
            errors.append(f"missing {report_kind}: {report_path}")
            continue
        try:
            payload, records = _read_records(report_path)
        except ThesisOutputError as exc:
            errors.append(str(exc))
            continue
        reports.append(
            {
                "path": str(report_path),
                "schema_version": str(payload.get("schema_version") or ""),
                "record_count": len(records),
                "keys": sorted(str(key) for key in payload.keys()),
            }
        )
    return reports


def _build_tables(
    specs: Sequence[Any], output_dir: Path, errors: list[str]
) -> list[dict[str, Any]]:
    tables: list[dict[str, Any]] = []
    for raw_spec in specs:
        spec = _mapping_spec(raw_spec, "table", errors)
        if spec is None:
            continue
        name = str(spec.get("name") or "table")
        source = Path(str(spec.get("source") or ""))
        columns = [str(column) for column in spec.get("columns", [])]
        output_csv = str(spec.get("output_csv") or f"tables/{name}.csv")
        if not source.is_file():
            errors.append(f"table {name}: missing source {source}")
            continue
        if not columns:
            errors.append(f"table {name}: columns must not be empty")
            continue
        try:
            _, records = _read_records(source)
        except ThesisOutputError as exc:
            errors.append(f"table {name}: {exc}")
            continue
        path = output_dir / output_csv
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            for record in records:
                writer.writerow({column: record.get(column, "") for column in columns})
        tables.append(
            {
                "name": name,
                "path": str(path),
                "relative_path": output_csv,
                "row_count": len(records),
                "columns": columns,
                "source_paths": [str(source)],
                "markdown_title": str(spec.get("markdown_title") or name),
            }
        )
    return tables


def _build_svg_plots(
    specs: Sequence[Any], output_dir: Path, errors: list[str]
) -> list[dict[str, Any]]:
    plots: list[dict[str, Any]] = []
    for raw_spec in specs:
        spec = _mapping_spec(raw_spec, "svg plot", errors)
        if spec is None:
            continue
        name = str(spec.get("name") or "plot")
        source = Path(str(spec.get("source") or ""))
        output_svg = str(spec.get("output_svg") or f"plots/{name}.svg")
        if not source.is_file():
            errors.append(f"svg plot {name}: missing source {source}")
            continue
        try:
            _, records = _read_records(source)
        except ThesisOutputError as exc:
            errors.append(f"svg plot {name}: {exc}")
            continue
        x_field = str(spec.get("x") or "sample_id")
        y_field = str(spec.get("y") or "product_score")
        path = output_dir / output_svg
        _write_svg_plot(
            records,
            path,
            x_field=x_field,
            y_field=y_field,
            title=str(spec.get("title") or name),
        )
        plots.append(
            {
                "name": name,
                "path": str(path),
                "relative_path": output_svg,
                "source_paths": [str(source)],
                "point_count": len(records),
                "x": x_field,
                "y": y_field,
            }
        )
    return plots


def _write_svg_plot(
    records: Sequence[Mapping[str, Any]],
    output_path: Path,
    *,
    x_field: str,
    y_field: str,
    title: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    width = 640
    height = 360
    margin_left = 52
    margin_bottom = 48
    plot_width = width - margin_left - 24
    plot_height = height - 72
    numeric_values = [_coerce_float(record.get(y_field)) for record in records]
    finite_values = [value for value in numeric_values if value is not None]
    y_min = min(finite_values) if finite_values else 0.0
    y_max = max(finite_values) if finite_values else 1.0
    if math.isclose(y_min, y_max):
        y_min = min(0.0, y_min)
        y_max = max(1.0, y_max)
    point_count = max(1, len(records) - 1)
    point_lines: list[str] = []
    for index, record in enumerate(records):
        value = _coerce_float(record.get(y_field))
        if value is None:
            continue
        x = margin_left + (index / point_count) * plot_width
        y = 32 + (1 - ((value - y_min) / (y_max - y_min))) * plot_height
        sample_id = html.escape(str(record.get(x_field) or record.get("sample_id") or index))
        point_lines.append(
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="5" fill="#2176AE" '
            f'data-sample-id="{sample_id}" />'
        )
        point_lines.append(
            f'<text x="{x:.2f}" y="{y - 8:.2f}" font-size="10" '
            f'text-anchor="middle">{sample_id}</text>'
        )
    escaped_title = html.escape(title)
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        'viewBox="0 0 640 360">',
        '<rect width="640" height="360" fill="white" />',
        f'<text x="320" y="22" text-anchor="middle" font-size="18">{escaped_title}</text>',
        f'<line x1="{margin_left}" y1="32" x2="{margin_left}" y2="{32 + plot_height}" '
        'stroke="black" />',
        f'<line x1="{margin_left}" y1="{32 + plot_height}" '
        f'x2="{margin_left + plot_width}" y2="{32 + plot_height}" stroke="black" />',
        f'<text x="18" y="180" transform="rotate(-90 18 180)" font-size="12">{html.escape(y_field)}</text>',
        f'<text x="320" y="346" text-anchor="middle" font-size="12">{html.escape(x_field)}</text>',
        *point_lines,
        "</svg>",
    ]
    output_path.write_text("\n".join(svg) + "\n", encoding="utf-8")


def _build_contact_sheets(
    specs: Sequence[Any], output_dir: Path, errors: list[str], warnings: list[str]
) -> list[dict[str, Any]]:
    sheets: list[dict[str, Any]] = []
    for raw_spec in specs:
        spec = _mapping_spec(raw_spec, "contact sheet", errors)
        if spec is None:
            continue
        name = str(spec.get("name") or "contact_sheet")
        output_image = str(spec.get("output_image") or f"contact_sheets/{name}.png")
        limit = max(0, int(spec.get("limit", 12)))
        raw_entries = spec.get("images", [])
        if not isinstance(raw_entries, list):
            errors.append(f"contact sheet {name}: images must be a list")
            continue
        entries: list[dict[str, str]] = []
        for index, raw_entry in enumerate(raw_entries[:limit]):
            if not isinstance(raw_entry, Mapping):
                warnings.append(f"contact sheet {name}: skipped non-object image entry {index}")
                continue
            image_path = Path(str(raw_entry.get("path") or ""))
            if not image_path.is_file():
                warnings.append(f"contact sheet {name}: missing image {image_path}")
            entries.append(
                {
                    "sample_id": str(raw_entry.get("sample_id") or index),
                    "path": str(image_path),
                    "caption": str(raw_entry.get("caption") or raw_entry.get("sample_id") or index),
                }
            )
        path = output_dir / output_image
        _write_contact_sheet(entries, path)
        sheets.append(
            {
                "name": name,
                "path": str(path),
                "relative_path": output_image,
                "entry_count": len(entries),
                "limit": limit,
                "entries": entries,
                "source_paths": [entry["path"] for entry in entries],
            }
        )
    return sheets


def _write_contact_sheet(entries: Sequence[Mapping[str, str]], output_path: Path) -> None:
    from PIL import Image, ImageDraw

    output_path.parent.mkdir(parents=True, exist_ok=True)
    thumb_width = 96
    thumb_height = 72
    caption_height = 24
    sheet_width = thumb_width * max(1, len(entries))
    sheet = Image.new("RGB", (sheet_width, thumb_height + caption_height), "white")
    draw = ImageDraw.Draw(sheet)
    if not entries:
        draw.text((4, 20), "empty", fill="black")
        sheet.save(output_path)
        return
    for index, entry in enumerate(entries):
        x = index * thumb_width
        try:
            with Image.open(entry["path"]) as source:
                image = source.convert("RGB")
                image.thumbnail((thumb_width, thumb_height))
                sheet.paste(image, (x, 0))
        except OSError:
            draw.rectangle((x, 0, x + thumb_width - 1, thumb_height - 1), outline="black")
            draw.text((x + 4, 20), "missing", fill="black")
        draw.text((x + 2, thumb_height + 2), entry["caption"][:18], fill="black")
    sheet.save(output_path)


def _read_records(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    suffix = path.suffix.lower()
    try:
        if suffix == ".csv":
            with path.open("r", encoding="utf-8", newline="") as handle:
                records = [dict(row) for row in csv.DictReader(handle)]
            return {"schema_version": "csv", "records": records}, records
        if suffix == ".jsonl":
            records = []
            for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ThesisOutputError(
                        f"{path}: line {line_number}: malformed JSON"
                    ) from exc
                if not isinstance(payload, dict):
                    raise ThesisOutputError(
                        f"{path}: line {line_number}: record must be a JSON object"
                    )
                records.append(payload)
            return {"schema_version": "jsonl", "records": records}, records
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ThesisOutputError(f"{path}: could not read evidence file") from exc
    except json.JSONDecodeError as exc:
        raise ThesisOutputError(f"{path}: malformed JSON") from exc
    if isinstance(payload, list):
        records = payload
        payload = {"schema_version": "json-list", "records": records}
    elif isinstance(payload, dict):
        raw_records = payload.get("records", [])
        if not isinstance(raw_records, list):
            raise ThesisOutputError(f"{path}: records must be a list")
        records = raw_records
    else:
        raise ThesisOutputError(f"{path}: evidence JSON must be an object or list")
    if not all(isinstance(record, dict) for record in records):
        raise ThesisOutputError(f"{path}: records must contain only objects")
    return dict(payload), list(records)


def _mapping_spec(
    raw_spec: Any, spec_kind: str, errors: list[str]
) -> dict[str, Any] | None:
    if not isinstance(raw_spec, Mapping):
        errors.append(f"{spec_kind} spec must be an object")
        return None
    return dict(raw_spec)


def _coerce_float(value: Any) -> float | None:
    if value in MISSING_MARKERS or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _format_list_or_none(values: Sequence[Any], empty_text: str) -> list[str]:
    if not values:
        return [f"- {empty_text}"]
    return [f"- {value}" for value in values]


def _format_artifact_section(title: str, artifacts: Sequence[Mapping[str, Any]]) -> list[str]:
    lines = [
        "",
        f"## {title}",
        "",
        "| Name | Destination | Source paths |",
        "| --- | --- | --- |",
    ]
    for artifact in artifacts:
        destination = artifact.get("relative_path") or artifact.get("path", "")
        source_paths = ", ".join(f"`{path}`" for path in artifact.get("source_paths", []))
        lines.append(
            f"| {artifact.get('name', '')} | `{destination}` | {source_paths} |"
        )
    return lines


def _sort_mapping(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _sort_mapping(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_sort_mapping(item) for item in value]
    return value
