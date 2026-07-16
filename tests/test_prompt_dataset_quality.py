from __future__ import annotations

import json
from pathlib import Path

from src.data_quality.prompt_validation import validate_prompt_dataset


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    return path


def _write_jsonl(path: Path, rows: list[dict[str, object] | str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            if isinstance(row, str):
                handle.write(row + "\n")
            else:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return path


def _prompt_record(
    record_id: str,
    target_text: str,
    *,
    content_type: str = "poster",
    style: dict[str, str] | None = None,
    lang: str = "ru",
) -> dict[str, object]:
    style_payload = style or {"font": "serif", "color": "white", "effect": "clean", "size": "large"}
    return {
        "id": record_id,
        "prompt": f"Render {target_text} on a poster",
        "target_text": target_text,
        "content_type": content_type,
        "style": style_payload,
        "lang": lang,
        "char_coverage": {ch.lower(): 1 for ch in target_text if ch.isalpha()},
    }


def test_prompt_quality_report_counts_valid_records_and_distributions(tmp_path: Path) -> None:
    prompts = _write_jsonl(
        tmp_path / "prompts.jsonl",
        [
            _prompt_record("p1", "Ёж Цех", content_type="poster"),
            _prompt_record(
                "p2",
                "Шрифт 42!",
                content_type="typography",
                style={"font": "gothic", "color": "red"},
            ),
            _prompt_record("p3", "Hello Ж readable", content_type="product", lang="en"),
        ],
    )

    report = validate_prompt_dataset(
        prompts,
        thresholds={
            "required_rare_characters": ["ё", "ж", "ц", "ш"],
            "min_rare_character_coverage": 1.0,
            "allowed_scripts": ["cyrillic", "latin", "digits", "punctuation"],
        },
    )

    assert report.ok
    assert report.schema_version == "prompt-quality/v1"
    assert report.valid_records == 3
    assert report.length_buckets == {"1-4": 0, "5-12": 2, "13-24": 1, "25+": 0}
    assert report.script_coverage["cyrillic"] == 3
    assert report.script_coverage["latin"] == 1
    assert report.script_coverage["digits"] == 1
    assert report.script_coverage["punctuation"] == 1
    assert report.rare_character_coverage["coverage_ratio"] == 1.0
    assert report.content_type_distribution == {"poster": 1, "product": 1, "typography": 1}
    assert report.style_distribution["font"] == {"gothic": 1, "serif": 2}
    assert len(report.metadata["source"]["sha256"]) == 64
    assert report.metadata["source"]["size_bytes"] == prompts.stat().st_size


def test_prompt_quality_report_aggregates_malformed_rows_missing_fields_and_duplicates(
    tmp_path: Path,
) -> None:
    prompts = _write_jsonl(
        tmp_path / "prompts.jsonl",
        [
            _prompt_record("p1", "Привет"),
            "{not-json",
            {"id": "p2", "prompt": "missing target"},
            _prompt_record("p3", "Привет"),
        ],
    )

    report = validate_prompt_dataset(prompts, thresholds={"max_duplicate_rate": 0.1})

    assert not report.ok
    assert report.total_lines == 4
    assert report.valid_records == 2
    assert any("line 2" in error and "malformed JSON" in error for error in report.errors)
    assert any(
        "line 3" in error and "missing required field: target_text" in error
        for error in report.errors
    )
    assert report.duplicate_rate == 0.5
    assert report.duplicate_examples == ["Привет"]
    assert any("duplicate_rate" in warning for warning in report.warnings)


def test_prompt_quality_report_warns_for_rare_character_gaps_and_distribution_drift(
    tmp_path: Path,
) -> None:
    prompts = _write_jsonl(
        tmp_path / "prompts.jsonl",
        [
            _prompt_record("p1", "Афиша", content_type="poster"),
            _prompt_record("p2", "Баннер", content_type="poster"),
            _prompt_record("p3", "Логотип", content_type="poster"),
        ],
    )

    report = validate_prompt_dataset(
        prompts,
        thresholds={
            "required_rare_characters": ["ё", "щ", "ъ"],
            "min_rare_character_coverage": 0.67,
            "expected_content_distribution": {"poster": [0.0, 0.5], "product": [0.2, 1.0]},
        },
    )

    assert report.ok
    assert report.rare_character_coverage["missing"] == ["ё", "щ", "ъ"]
    assert any("rare_character_coverage" in warning for warning in report.warnings)
    assert any("content_type poster" in warning for warning in report.warnings)
    assert any("content_type product" in warning for warning in report.warnings)


def test_prompt_quality_report_flags_cpu_safe_naturalness_and_script_heuristics(
    tmp_path: Path,
) -> None:
    prompts = _write_jsonl(
        tmp_path / "prompts.jsonl",
        [
            _prompt_record("p1", ""),
            _prompt_record("p2", "Ответь одной строкой: вывеска"),
            _prompt_record("p3", "слово слово слово слово слово"),
            _prompt_record("p4", '"Незакрытая цитата'),
            _prompt_record("p5", "Valid🙂"),
        ],
    )

    report = validate_prompt_dataset(prompts, thresholds={"allowed_scripts": ["cyrillic"]})

    assert not report.ok
    assert any("line 1" in error and "empty target_text" in error for error in report.errors)
    assert any("instruction-like" in warning for warning in report.warnings)
    assert any("repeated token" in warning for warning in report.warnings)
    assert any("unmatched quote" in warning for warning in report.warnings)
    assert any("illegal character" in error and "line 5" in error for error in report.errors)
    assert any("disallowed script latin" in error and "line 5" in error for error in report.errors)


def test_prompt_quality_accepts_currency_and_unicode_minus_as_text_symbols(tmp_path: Path) -> None:
    prompts = _write_jsonl(
        tmp_path / "prompts.jsonl",
        [_prompt_record("p1", "СКИДКА 500 ₽ − 10%")],
    )

    report = validate_prompt_dataset(prompts)

    assert report.ok
    assert report.script_coverage["digits"] == 1
    assert report.script_coverage["punctuation"] == 1


def test_prompt_quality_applies_entropy_and_stage_provenance_thresholds(tmp_path: Path) -> None:
    missing_stage = _prompt_record("p1", "АФИША")
    prompts = _write_jsonl(tmp_path / "prompts.jsonl", [missing_stage])

    report = validate_prompt_dataset(
        prompts,
        thresholds={
            "require_stage_provenance": True,
            "min_content_type_entropy": 0.5,
            "min_style_entropy": 0.5,
        },
    )

    assert not report.ok
    assert report.valid_records == 0
    assert any("missing required provenance field" in error for error in report.errors)
    assert report.metadata["distribution_entropy"] == {
        "content_type": 0.0,
        "style_signature": 0.0,
    }


def test_prompt_validation_cli_uses_config_thresholds_and_cli_overrides(tmp_path: Path) -> None:
    from scripts.validate_prompt_dataset import main

    prompts = _write_jsonl(tmp_path / "prompts.jsonl", [_prompt_record("p1", "ДЛИННАЯ СТРОКА")])
    config = _write_json(
        tmp_path / "config.json",
        {
            "seed": 11,
            "validation_thresholds": {
                "max_target_length": 3,
                "min_rare_char_coverage": 0.0,
            },
        },
    )
    configured_report = tmp_path / "configured.json"
    overridden_report = tmp_path / "overridden.json"

    assert (
        main(
            [
                "--input",
                str(prompts),
                "--config",
                str(config),
                "--report",
                str(configured_report),
                "--strict-warnings",
            ]
        )
        == 1
    )
    assert (
        main(
            [
                "--input",
                str(prompts),
                "--config",
                str(config),
                "--report",
                str(overridden_report),
                "--max-target-length",
                "64",
                "--strict-warnings",
            ]
        )
        == 0
    )
    configured = json.loads(configured_report.read_text(encoding="utf-8"))
    overridden = json.loads(overridden_report.read_text(encoding="utf-8"))
    assert configured["metadata"]["thresholds"]["min_rare_character_coverage"] == 0.0
    assert configured["metadata"]["thresholds"]["max_target_length"] == 3
    assert overridden["metadata"]["thresholds"]["max_target_length"] == 64


def test_dataset_manifest_records_deterministic_prompt_provenance(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from src.data_quality import manifests

    monkeypatch.setattr(manifests, "_utc_now", lambda: "2026-05-04T00:00:00Z")
    monkeypatch.setattr(
        manifests,
        "collect_git_state",
        lambda root=None: {"available": True, "commit": "abc1234", "dirty": False},
    )
    config = _write_json(
        tmp_path / "configs" / "prompt.json",
        {"schema_version": "prompt-generation/v1", "seed": 7, "model_id": "Qwen/demo"},
    )
    prompts = _write_jsonl(tmp_path / "data" / "prompts.jsonl", [_prompt_record("p1", "Ёж")])

    manifest = manifests.create_dataset_manifest(
        dataset_kind="prompt",
        dataset_paths=[prompts],
        config_path=config,
        seed_strategy={"prompt.seed": 7},
        source_paths=[prompts, config],
        filtering_stats={"accepted": 1, "rejected": 0},
        output_counts={"prompts": 1},
        root=tmp_path,
    )

    payload = manifest.to_json_payload()
    assert payload["schema_version"] == "dataset-manifest/v1"
    assert payload["dataset_kind"] == "prompt"
    assert payload["dataset_paths"] == [str(prompts)]
    assert payload["config"]["path"] == str(config)
    assert len(payload["config"]["sha256"]) == 64
    assert payload["seed_strategy"] == {"prompt.seed": 7}
    assert payload["git"] == {"available": True, "commit": "abc1234", "dirty": False}
    assert payload["models"] == {"model_id": "Qwen/demo", "model_revision": None}
    assert payload["source_hashes"][str(config)]["sha256"] == payload["config"]["sha256"]
    assert payload["filtering_stats"] == {"accepted": 1, "rejected": 0}
    assert payload["output_counts"] == {"prompts": 1}


def test_dataset_manifest_loading_rejects_malformed_and_missing_fields(tmp_path: Path) -> None:
    from src.data_quality.manifests import DatasetManifestError, load_dataset_manifest

    malformed = tmp_path / "manifest.json"
    malformed.write_text("{not-json", encoding="utf-8")
    missing = tmp_path / "missing-fields.json"
    missing.write_text(json.dumps({"schema_version": "dataset-manifest/v1"}), encoding="utf-8")

    try:
        load_dataset_manifest(malformed)
    except DatasetManifestError as exc:
        assert "malformed JSON" in str(exc)
    else:  # pragma: no cover - documents required exception path
        raise AssertionError("malformed manifest should be rejected")

    try:
        load_dataset_manifest(missing)
    except DatasetManifestError as exc:
        assert "dataset_kind" in str(exc)
    else:  # pragma: no cover - documents required exception path
        raise AssertionError("missing top-level fields should be rejected")


def test_hash_source_file_references_binary_generated_artifacts_by_default(tmp_path: Path) -> None:
    from src.data_quality.manifests import hash_source_file

    text_path = tmp_path / "prompts.csv"
    text_path.write_text("id,target_text\np1,Ёж\n", encoding="utf-8")
    tensor_path = tmp_path / "outputs" / "generated" / "latent.pt"
    tensor_path.parent.mkdir(parents=True)
    tensor_path.write_bytes(b"\x80\x04binary")

    text_hash = hash_source_file(text_path)
    tensor_reference = hash_source_file(tensor_path)
    tensor_forced = hash_source_file(tensor_path, safe_hash_inputs={tensor_path})

    assert text_hash["hashed"] is True
    assert len(text_hash["sha256"]) == 64
    assert tensor_reference == {
        "path": str(tensor_path),
        "hashed": False,
        "reason": "unsafe generated or binary artifact",
    }
    assert tensor_forced["hashed"] is True
    assert len(tensor_forced["sha256"]) == 64


def test_prompt_validation_cli_writes_report_and_manifest(tmp_path: Path) -> None:
    from scripts.validate_prompt_dataset import main

    prompts = _write_jsonl(tmp_path / "prompts.jsonl", [_prompt_record("p1", "Ёж")])
    config = _write_json(tmp_path / "config.json", {"seed": 11, "model_id": "Qwen/demo"})
    report_path = tmp_path / "reports" / "prompt-quality.json"
    manifest_path = tmp_path / "reports" / "dataset-manifest.json"

    exit_code = main(
        [
            "--input",
            str(prompts),
            "--report",
            str(report_path),
            "--manifest",
            str(manifest_path),
            "--config",
            str(config),
            "--min-rare-character-coverage",
            "0.0",
            "--required-rare-characters",
            "ё,ж",
        ]
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert report["schema_version"] == "prompt-quality/v1"
    assert report["valid_records"] == 1
    assert manifest["schema_version"] == "dataset-manifest/v1"
    assert manifest["dataset_kind"] == "prompt"
    assert manifest["output_counts"] == {"valid_prompts": 1}


def test_prompt_validation_cli_exit_codes_for_errors_and_strict_warnings(tmp_path: Path) -> None:
    from scripts.validate_prompt_dataset import main

    warning_only = _write_jsonl(tmp_path / "warning.jsonl", [_prompt_record("p1", "Афиша")])
    invalid = _write_jsonl(tmp_path / "invalid.jsonl", [{"id": "p1", "prompt": "missing target"}])

    warning_args = [
        "--input",
        str(warning_only),
        "--min-rare-character-coverage",
        "1.0",
        "--required-rare-characters",
        "ё",
    ]
    assert main(warning_args) == 0
    assert main([*warning_args, "--strict-warnings"]) == 1
    assert main(["--input", str(invalid)]) == 2


def test_prompt_validation_cli_prints_json_stdout_when_no_report_path(
    tmp_path: Path,
    capsys,
) -> None:
    from scripts.validate_prompt_dataset import main

    prompts = _write_jsonl(tmp_path / "prompts.jsonl", [_prompt_record("p1", "Ёж")])

    assert main(["--input", str(prompts)]) == 0
    stdout = capsys.readouterr().out
    payload = json.loads(stdout)
    assert payload["valid_records"] == 1


def test_dataset_quality_docs_cover_cli_and_artifact_safety() -> None:
    docs = Path("docs/dataset_quality.md").read_text(encoding="utf-8")
    required = [
        "validate_prompt_dataset.py",
        "--input data/prompts/simple.jsonl",
        "--manifest runs/prompt-quality/dataset-manifest.json",
        "simple prompt datasets",
        "full prompt datasets",
        "curriculum prompt datasets",
        "Generated reports and manifests are runtime artifacts",
    ]
    missing = [item for item in required if item not in docs]
    assert not missing
