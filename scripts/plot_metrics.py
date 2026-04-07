"""Plot training metrics from CSV log."""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_metrics(csv_path: str) -> dict[str, list[float]]:
    data: dict[str, list[float]] = {
        "step": [], "loss": [], "reward": [], "grad_norm": [], "lr": [], "elapsed_s": [],
    }
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key in data:
                data[key].append(float(row[key]))
    return data


def smooth(values: list[float], window: int = 5) -> np.ndarray:
    if len(values) < window:
        return np.array(values)
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def plot(metrics_csv: str, output_dir: str | None = None):
    data = load_metrics(metrics_csv)
    if output_dir is None:
        output_dir = str(Path(metrics_csv).parent)

    steps = data["step"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("ReFL Training Metrics", fontsize=16, fontweight="bold")

    # 1. Reward curve
    ax = axes[0, 0]
    ax.plot(steps, data["reward"], alpha=0.3, color="tab:blue", label="raw")
    if len(steps) >= 5:
        s = smooth(data["reward"])
        ax.plot(steps[len(steps)-len(s):], s, color="tab:blue", linewidth=2, label=f"smooth(5)")
    ax.set_xlabel("Step")
    ax.set_ylabel("Reward (P(yes))")
    ax.set_title("VLM Reward")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Loss curve
    ax = axes[0, 1]
    ax.plot(steps, data["loss"], alpha=0.3, color="tab:red", label="raw")
    if len(steps) >= 5:
        s = smooth(data["loss"])
        ax.plot(steps[len(steps)-len(s):], s, color="tab:red", linewidth=2, label="smooth(5)")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Gradient norm
    ax = axes[1, 0]
    ax.plot(steps, data["grad_norm"], color="tab:green", alpha=0.5)
    if len(steps) >= 5:
        s = smooth(data["grad_norm"])
        ax.plot(steps[len(steps)-len(s):], s, color="tab:green", linewidth=2)
    ax.set_xlabel("Step")
    ax.set_ylabel("Gradient Norm")
    ax.set_title("Gradient Norm")
    ax.grid(True, alpha=0.3)

    # 4. Speed (steps/sec derived from elapsed)
    ax = axes[1, 1]
    if len(steps) > 1:
        speed = [s / max(e, 0.1) for s, e in zip(steps, data["elapsed_s"])]
        ax.plot(steps, speed, color="tab:purple", alpha=0.5)
        if len(speed) >= 5:
            s = smooth(speed)
            ax.plot(steps[len(steps)-len(s):], s, color="tab:purple", linewidth=2)
    ax.set_xlabel("Step")
    ax.set_ylabel("Steps/sec")
    ax.set_title("Training Speed")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = Path(output_dir) / "training_curves.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Charts saved to {out_path}")

    # Print summary stats
    rewards = data["reward"]
    print(f"\n--- Summary ---")
    print(f"Steps: {len(rewards)}")
    print(f"Reward: start={rewards[0]:.4f}, end={rewards[-1]:.4f}, "
          f"max={max(rewards):.4f}, mean={np.mean(rewards):.4f}")
    print(f"Grad norm: mean={np.mean(data['grad_norm']):.4f}, max={max(data['grad_norm']):.4f}")
    if len(rewards) >= 10:
        first_10 = np.mean(rewards[:10])
        last_10 = np.mean(rewards[-10:])
        print(f"Reward trend: first-10-avg={first_10:.4f}, last-10-avg={last_10:.4f}, "
              f"delta={last_10 - first_10:+.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics_csv", help="Path to metrics.csv")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    plot(args.metrics_csv, args.output_dir)
