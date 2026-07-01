"""
Generate thesis-ready training curve plots from TensorBoard event files.

Produces a 4-panel figure (reward_mean, entropy, loss_critic, episode_len_max)
with a light smoothed overlay.

Usage:
    python plot_training.py            # base training, 4x1 stacked
    python plot_training.py --2x2      # base training, 2x2 grid
    python plot_training.py --ft       # fine-tuning logs, 4x1
    python plot_training.py --ft --2x2 # fine-tuning logs, 2x2

Output: docs/training_curves[_ft][_2x2].pdf + .png
"""

import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

LOG_DIR_BASE = os.path.join(os.path.dirname(__file__), "..", "logs")
LOG_DIR_FT   = os.path.join(os.path.dirname(__file__), "..", "logs_ft")
OUT_DIR      = os.path.join(os.path.dirname(__file__), "..", "docs")

TAGS    = ["reward_mean", "entropy", "loss_critic", "episode_len_max"]
YLABELS = ["Mean reward", "Entropy", "Critic loss", "Max episode length"]
SMOOTH_WINDOW = 20

# Episode metrics are logged in env-steps (multiples of 2048).
# Loss/entropy metrics are logged per gradient update (sequential integers).
EPOCH_DIVISOR = {
    "reward_mean":     32_000,
    "episode_len_max": 32_000,
    "entropy":         16,
    "loss_critic":     16,
}
SKIP_INITIAL = 0


def load_scalars(log_dir: str, tag: str):
    ea = EventAccumulator(log_dir, size_guidance={"scalars": 0})
    ea.Reload()
    available = ea.Tags().get("scalars", [])
    if tag not in available:
        print(f"  [warn] tag '{tag}' not found. Available: {available}")
        return np.array([]), np.array([])

    events     = ea.Scalars(tag)
    wall_times = np.array([e.wall_time for e in events])
    steps      = np.array([e.step      for e in events])
    values     = np.array([e.value     for e in events])

    # Sort by wall_time so multiple sequential runs merge into a single
    # chronological series regardless of per-run step resets.
    order      = np.argsort(wall_times, kind="stable")
    steps      = steps[order]
    values     = values[order]

    # Detect run restarts (step decreases) and stitch into a continuous
    # step sequence so the x-axis remains monotonically increasing.
    stitched = steps.copy().astype(float)
    offset   = 0.0
    for i in range(1, len(steps)):
        if steps[i] <= steps[i - 1]:          # restart detected
            offset += steps[i - 1] + 1        # offset by end of previous run
        stitched[i] = steps[i] + offset

    return stitched, values


def smooth(values: np.ndarray, window: int) -> np.ndarray:
    if len(values) < window:
        return values
    return np.convolve(values, np.ones(window) / window, mode="valid")


def plot_panel(ax, tag, ylabel, log_dir, xlabel=False):
    steps, values = load_scalars(log_dir, tag)
    if len(steps) == 0:
        ax.set_ylabel(ylabel, fontsize=11)
        ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                ha="center", va="center", color="grey")
        return

    divisor     = EPOCH_DIVISOR.get(tag, 32_000)
    epochs      = steps / divisor

    skip        = max(0, min(SKIP_INITIAL, len(epochs) - SMOOTH_WINDOW - 1))
    epochs_plot = epochs[skip:]
    values_plot = values[skip:]

    smoothed_values = smooth(values_plot, SMOOTH_WINDOW)
    smoothed_epochs = epochs_plot[SMOOTH_WINDOW - 1:]

    ax.plot(epochs_plot, values_plot,
            color="#bfbfbf", linewidth=0.6, alpha=0.7, zorder=1)
    ax.plot(smoothed_epochs, smoothed_values,
            color="#1f4e79", linewidth=1.6, zorder=2)

    ax.set_ylabel(ylabel, fontsize=11)
    if xlabel:
        ax.set_xlabel("Fine-tuning epoch" if log_dir == LOG_DIR_FT else "Training epoch",
                      fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=9)


def build_4x1(log_dir):
    fig, axes = plt.subplots(
        4, 1,
        figsize=(8, 10),
        sharex=True,
        gridspec_kw={"hspace": 0.08},
    )
    xlabel_text = "Fine-tuning epoch" if log_dir == LOG_DIR_FT else "Training epoch"
    for ax, tag, ylabel in zip(axes, TAGS, YLABELS):
        plot_panel(ax, tag, ylabel, log_dir)

    axes[-1].set_xlabel(xlabel_text, fontsize=11)
    for ax in axes[:-1]:
        plt.setp(ax.get_xticklabels(), visible=False)

    return fig


def build_2x2(log_dir):
    fig, axes = plt.subplots(
        2, 2,
        figsize=(11, 7),
        gridspec_kw={"hspace": 0.35, "wspace": 0.18},
    )
    flat = axes.flatten()
    for ax, tag, ylabel in zip(flat, TAGS, YLABELS):
        plot_panel(ax, tag, ylabel, log_dir, xlabel=(ax in flat[2:]))

    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--2x2", dest="grid2x2", action="store_true",
                        help="Use 2×2 grid layout instead of 4×1 stacked")
    parser.add_argument("--ft", dest="ft", action="store_true",
                        help="Read from logs_ft (fine-tuning) instead of logs")
    args = parser.parse_args()

    log_dir = LOG_DIR_FT if args.ft else LOG_DIR_BASE
    os.makedirs(OUT_DIR, exist_ok=True)

    suffix = ("_ft" if args.ft else "") + ("_2x2" if args.grid2x2 else "")
    fig    = build_2x2(log_dir) if args.grid2x2 else build_4x1(log_dir)

    out_pdf = os.path.join(OUT_DIR, f"training_curves{suffix}.pdf")
    out_png = os.path.join(OUT_DIR, f"training_curves{suffix}.png")
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_pdf}")
    print(f"Saved: {out_png}")
    plt.close(fig)


if __name__ == "__main__":
    main()
