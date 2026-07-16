"""Checks for the installed package and console command surface."""

from __future__ import annotations

import subprocess
import sys
from importlib import metadata


def test_distribution_exposes_supported_console_scripts():
    distribution = metadata.distribution("diffusion-text-tuner")
    scripts = {
        entry.name: entry.value
        for entry in distribution.entry_points
        if entry.group == "console_scripts"
    }

    assert scripts == {
        "dtt-evidence": "scripts.build_evidence_manifest:main",
        "dtt-final-benchmark": "src.evaluation.final_benchmark:main",
        "dtt-generate": "scripts.generate_images:main",
        "dtt-manifest": "scripts.run_manifest:main",
        "dtt-preflight": "scripts.preflight_runtime:main",
        "dtt-prompts": "src.prompt_pipeline.generate:main",
        "dtt-score": "scripts.score_images:main",
    }


def test_prompt_and_runtime_imports_do_not_require_torch() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import src.runtime; import src.prompt_pipeline.generate; "
                "assert 'torch' not in sys.modules"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
