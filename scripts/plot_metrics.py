"""Thin CLI wrapper for plotting training metrics from CSV logs."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.plotting.training_metrics import load_metrics, plot_training_metrics, smooth

__all__ = ["build_parser", "load_metrics", "main", "plot", "plot_training_metrics", "smooth"]


def plot(metrics_csv: str | Path, output_dir: str | Path | None = None) -> Path:
    """Compatibility alias for callers importing ``scripts.plot_metrics.plot``."""

    return plot_training_metrics(metrics_csv, output_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics_csv", help="Path to metrics.csv")
    parser.add_argument("--output-dir", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    plot_training_metrics(args.metrics_csv, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
