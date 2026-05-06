from __future__ import annotations

import csv
import importlib
import inspect
import json
import sys
from pathlib import Path


def _write_raw_metadata_fixture(raw_dir: Path) -> None:
    (raw_dir / "meta").mkdir(parents=True, exist_ok=True)
    (raw_dir / "index.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"id": "sample-1"}, ensure_ascii=False),
                json.dumps({"id": "sample-empty"}, ensure_ascii=False),
                json.dumps({"id": "sample-2"}, ensure_ascii=False),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (raw_dir / "meta" / "sample-1.json").write_text(
        json.dumps(
            {
                "resolution": 768,
                "bg_path": "backgrounds/bg-1.png",
                "annotations": [
                    {"text": "Ёж", "bbox": [1, 2, 30, 20], "font": "Serif"}
                ],
                "label": "Render Ёж on a poster",
                "caption_anyword": "A poster with *Ёж*",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (raw_dir / "meta" / "sample-empty.json").write_text(
        json.dumps(
            {
                "resolution": 768,
                "bg_path": "backgrounds/bg-empty.png",
                "annotations": [],
                "label": "No visible text",
                "caption_anyword": "No visible text",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (raw_dir / "meta" / "sample-2.json").write_text(
        json.dumps(
            {
                "resolution": 1024,
                "bg_path": "backgrounds/bg-2.png",
                "annotations": [
                    {"text": "Привет мир", "bbox": [4, 5, 80, 24], "font": "Sans"},
                    {"text": "42", "bbox": [8, 40, 18, 12], "font": "Mono"},
                ],
                "label": "Render Привет мир and 42",
                "caption_anyword": "Street sign says *Привет мир* and *42*",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def _records() -> list[dict]:
    return [
        {
            "id": "sample-1",
            "resolution": 768,
            "bg": "backgrounds/bg-1.png",
            "annotations": [
                {"text": "Ёж", "bbox": [1, 2, 30, 20], "font": "Serif"}
            ],
            "caption_human": "Render Ёж on a poster",
            "caption_anyword": "A poster with *Ёж*",
        },
        {
            "id": "sample-2",
            "resolution": 1024,
            "bg": "backgrounds/bg-2.png",
            "annotations": [
                {"text": "Привет мир", "bbox": [4, 5, 80, 24], "font": "Sans"},
                {"text": "42", "bbox": [8, 40, 18, 12], "font": "Mono"},
            ],
            "caption_human": "Render Привет мир and 42",
            "caption_anyword": "Street sign says *Привет мир* and *42*",
        },
    ]


def test_collate_records_drops_annotationless_samples_and_preserves_fields(
    tmp_path: Path,
) -> None:
    from src.synthesis.dataset_builder import collate_records

    raw_dir = tmp_path / "raw"
    _write_raw_metadata_fixture(raw_dir)

    records = collate_records(raw_dir)

    assert records == _records()


def test_index_writers_preserve_masked_sft_and_anyword_schemas(tmp_path: Path) -> None:
    from src.synthesis.dataset_builder import (
        fan_out,
        write_anyword_json,
        write_masked_index,
    )

    out_masked = tmp_path / "masked_sft"
    out_anyword = tmp_path / "anyword_format"
    out_masked.mkdir()
    out_anyword.mkdir()
    raw_dir = tmp_path / "raw"
    for subdir in (raw_dir / "imgs", raw_dir / "masks"):
        subdir.mkdir(parents=True)
    for record in _records():
        (raw_dir / "imgs" / f"{record['id']}.png").write_text(
            f"image-{record['id']}", encoding="utf-8"
        )
        (raw_dir / "masks" / f"{record['id']}.png").write_text(
            f"mask-{record['id']}", encoding="utf-8"
        )

    fan_out(_records(), raw_dir, out_masked, out_anyword)
    prompts_path = write_masked_index(_records(), out_masked)
    write_anyword_json(_records(), out_anyword)

    with (out_masked / "index.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    prompt_rows = [
        json.loads(line)
        for line in prompts_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    anyword = json.loads((out_anyword / "data.json").read_text(encoding="utf-8"))

    assert (
        out_masked / "raw_imgs" / "sample-1.png"
    ).read_text(encoding="utf-8") == "image-sample-1"
    assert (
        out_masked / "raw_masks" / "sample-2.png"
    ).read_text(encoding="utf-8") == "mask-sample-2"
    assert (
        out_anyword / "imgs" / "sample-1.png"
    ).read_text(encoding="utf-8") == "image-sample-1"
    assert (
        out_anyword / "masks" / "sample-2.png"
    ).read_text(encoding="utf-8") == "mask-sample-2"
    assert prompts_path == out_masked / "prompts.jsonl"
    assert rows == [
        {
            "id": "sample-1",
            "resolution": "768",
            "n_words": "1",
            "text": "Ёж",
            "caption": "Render Ёж on a poster",
        },
        {
            "id": "sample-2",
            "resolution": "1024",
            "n_words": "3",
            "text": "Привет мир | 42",
            "caption": "Render Привет мир and 42",
        },
    ]
    assert prompt_rows == [
        {"id": "sample-1", "prompt": "Render Ёж on a poster"},
        {"id": "sample-2", "prompt": "Render Привет мир and 42"},
    ]
    assert anyword == {
        "data_root": str(out_anyword.resolve()),
        "data_list": [
            {
                "img_name": "imgs/sample-1.png",
                "width": 768,
                "height": 768,
                "caption": "A poster with *Ёж*",
                "wm_score": 0.0,
                "annotations": _records()[0]["annotations"],
            },
            {
                "img_name": "imgs/sample-2.png",
                "width": 1024,
                "height": 1024,
                "caption": "Street sign says *Привет мир* and *42*",
                "wm_score": 0.0,
                "annotations": _records()[1]["annotations"],
            },
        ],
    }


def test_synthesis_build_config_preserves_cli_defaults() -> None:
    from src.synthesis.dataset_builder import SynthesisBuildConfig

    config = SynthesisBuildConfig(num=25_000)

    assert config.num == 25_000
    assert config.workers == 8
    assert config.template == Path("scripts/synth/synthtiger_template.py")
    assert config.template_name == "CyrillicScene"
    assert config.config == Path("configs/synth/cyrillic.yaml")
    assert config.runner == Path("scripts/synth/run_synthtiger.py")
    assert config.raw_dir == Path("data/synth_cyrillic/raw")
    assert config.out_masked == Path("data/synth_cyrillic/masked_sft")
    assert config.out_anyword == Path("data/synth_cyrillic/anyword_format")
    assert config.seed == 0
    assert config.skip_render is False
    assert config.clean is False
    assert config.bake_latents is False
    assert config.encode_text is False
    assert config.model_id == "black-forest-labs/FLUX.2-klein-base-4B"
    assert config.device == "cuda"


def test_dataset_builder_import_does_not_load_heavy_synthesis_or_model_stacks() -> None:
    sys.modules.pop("src.synthesis.dataset_builder", None)
    before_modules = set(sys.modules)

    importlib.import_module("src.synthesis.dataset_builder")

    imported_modules = set(sys.modules) - before_modules
    forbidden_imports = {
        "diffusers",
        "numpy",
        "PIL",
        "PIL.Image",
        "torch",
        "src.training.flux2_utils",
        "src.training.losses",
    }
    assert forbidden_imports.isdisjoint(imported_modules)


def test_build_dataset_runs_phases_in_order_and_honors_gates(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from src.synthesis import dataset_builder as builder

    phase_calls: list[str] = []
    expected_records = _records()
    prompts_path = tmp_path / "masked" / "prompts.jsonl"

    def fake_render_phase(**kwargs) -> None:
        phase_calls.append(
            f"render:{kwargs['num']}:{kwargs['workers']}:{kwargs['template_name']}"
        )

    def fake_collate_records(raw_dir: Path) -> list[dict]:
        phase_calls.append(f"collate:{raw_dir.name}")
        return expected_records

    def fake_fan_out(
        records: list[dict], raw_dir: Path, out_masked: Path, out_anyword: Path
    ) -> None:
        assert records == expected_records
        phase_calls.append(
            f"fan_out:{raw_dir.name}:{out_masked.name}:{out_anyword.name}"
        )

    def fake_write_anyword_json(records: list[dict], out_anyword: Path) -> None:
        assert records == expected_records
        phase_calls.append(f"anyword:{out_anyword.name}")

    def fake_write_masked_index(records: list[dict], out_masked: Path) -> Path:
        assert records == expected_records
        phase_calls.append(f"masked:{out_masked.name}")
        return prompts_path

    def fake_bake_latents_phase(
        records: list[dict], out_masked: Path, model_id: str, device: str
    ) -> None:
        assert records == expected_records
        phase_calls.append(f"bake:{out_masked.name}:{model_id}:{device}")

    def fake_encode_text_phase(
        received_prompts_path: Path, out_masked: Path, model_id: str, device: str
    ) -> None:
        assert received_prompts_path == prompts_path
        phase_calls.append(f"encode:{out_masked.name}:{model_id}:{device}")

    monkeypatch.setattr(builder, "render_phase", fake_render_phase)
    monkeypatch.setattr(builder, "collate_records", fake_collate_records)
    monkeypatch.setattr(builder, "fan_out", fake_fan_out)
    monkeypatch.setattr(builder, "write_anyword_json", fake_write_anyword_json)
    monkeypatch.setattr(builder, "write_masked_index", fake_write_masked_index)
    monkeypatch.setattr(builder, "bake_latents_phase", fake_bake_latents_phase)
    monkeypatch.setattr(builder, "encode_text_phase", fake_encode_text_phase)

    exit_code = builder.build_dataset(
        builder.SynthesisBuildConfig(
            num=3,
            workers=2,
            template_name="CyrillicScene",
            raw_dir=tmp_path / "raw",
            out_masked=tmp_path / "masked",
            out_anyword=tmp_path / "anyword",
        )
    )

    assert exit_code == 0
    assert phase_calls == [
        "render:3:2:CyrillicScene",
        "collate:raw",
        "fan_out:raw:masked:anyword",
        "anyword:anyword",
        "masked:masked",
    ]

    phase_calls.clear()
    preserved = tmp_path / "reuse_raw" / "keep.txt"
    preserved.parent.mkdir()
    preserved.write_text("do-not-clean-when-skipping-render", encoding="utf-8")
    exit_code = builder.build_dataset(
        builder.SynthesisBuildConfig(
            num=3,
            skip_render=True,
            clean=True,
            bake_latents=True,
            encode_text=True,
            raw_dir=preserved.parent,
            out_masked=tmp_path / "masked-gpu",
            out_anyword=tmp_path / "anyword-gpu",
            model_id="test-model",
            device="cpu",
        )
    )

    assert exit_code == 0
    assert preserved.read_text(encoding="utf-8") == "do-not-clean-when-skipping-render"
    assert phase_calls == [
        "collate:reuse_raw",
        "fan_out:reuse_raw:masked-gpu:anyword-gpu",
        "anyword:anyword-gpu",
        "masked:masked-gpu",
        "bake:masked-gpu:test-model:cpu",
        "encode:masked-gpu:test-model:cpu",
    ]


def test_gpu_model_phases_are_exposed_but_lazy_loaded() -> None:
    from src.synthesis.dataset_builder import bake_latents_phase, encode_text_phase

    assert callable(bake_latents_phase)
    assert callable(encode_text_phase)


def test_cli_main_builds_config_and_delegates(monkeypatch, tmp_path: Path) -> None:
    from scripts.synth import build_dataset as cli
    from src.synthesis.dataset_builder import SynthesisBuildConfig

    captured: list[SynthesisBuildConfig] = []

    def fake_build_dataset(config: SynthesisBuildConfig) -> int:
        captured.append(config)
        return 7

    monkeypatch.setattr(cli, "build_dataset", fake_build_dataset)

    exit_code = cli.main(
        [
            "--num",
            "12",
            "--workers",
            "3",
            "--template",
            str(tmp_path / "template.py"),
            "--template-name",
            "CustomScene",
            "--config",
            str(tmp_path / "config.yaml"),
            "--runner",
            str(tmp_path / "runner.py"),
            "--raw-dir",
            str(tmp_path / "raw"),
            "--out-masked",
            str(tmp_path / "masked"),
            "--out-anyword",
            str(tmp_path / "anyword"),
            "--seed",
            "123",
            "--skip-render",
            "--clean",
            "--bake-latents",
            "--encode-text",
            "--model-id",
            "test-model",
            "--device",
            "cpu",
        ]
    )

    assert exit_code == 7
    assert captured == [
        SynthesisBuildConfig(
            num=12,
            workers=3,
            template=tmp_path / "template.py",
            template_name="CustomScene",
            config=tmp_path / "config.yaml",
            runner=tmp_path / "runner.py",
            raw_dir=tmp_path / "raw",
            out_masked=tmp_path / "masked",
            out_anyword=tmp_path / "anyword",
            seed=123,
            skip_render=True,
            clean=True,
            bake_latents=True,
            encode_text=True,
            model_id="test-model",
            device="cpu",
        )
    ]


def test_cli_preserves_defaults_and_compatibility_reexports(monkeypatch) -> None:
    from scripts.synth import build_dataset as cli
    from src.synthesis import dataset_builder as builder

    for name in [
        "SynthesisBuildConfig",
        "render_phase",
        "collate_records",
        "fan_out",
        "write_anyword_json",
        "write_masked_index",
        "bake_latents_phase",
        "encode_text_phase",
        "build_dataset",
    ]:
        assert getattr(cli, name) is getattr(builder, name)

    captured: list[builder.SynthesisBuildConfig] = []

    def fake_build_dataset(config: builder.SynthesisBuildConfig) -> int:
        captured.append(config)
        return 0

    monkeypatch.setattr(cli, "build_dataset", fake_build_dataset)

    assert cli.main(["--num", "25"]) == 0
    assert captured == [builder.SynthesisBuildConfig(num=25)]


def test_cli_wrapper_is_thin_and_keeps_heavy_imports_in_builder() -> None:
    from scripts.synth import build_dataset as cli

    source = inspect.getsource(cli)

    assert "from src.synthesis.dataset_builder import" in source
    assert "def render_phase" not in source
    assert "def collate_records" not in source
    assert "def bake_latents_phase" not in source
    assert "import torch" not in source
    assert "from PIL" not in source
    assert "from diffusers" not in source
    assert "src.training.flux2_utils" not in source
