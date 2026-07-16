"""Launch the historical ReFL CUDA profiler without import-time model loading."""

from __future__ import annotations

import runpy
from pathlib import Path


def main() -> int:
    profiler = (
        Path(__file__).resolve().parents[1] / "experiments" / "legacy" / "profile_refl_step.py"
    )
    runpy.run_path(str(profiler), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
