"""Import-safe checkpoint path and interval helper contracts."""

from __future__ import annotations

from pathlib import Path


def should_save_checkpoint(step: int, interval: int) -> bool:
    """Return whether a positive training step should save a checkpoint."""

    return interval > 0 and step > 0 and step % interval == 0


def checkpoint_dir(output_dir: str | Path, step: int) -> Path:
    """Return the standard checkpoint directory for ``step`` under ``output_dir``."""

    return Path(output_dir) / "checkpoints" / f"step_{step:06d}"
