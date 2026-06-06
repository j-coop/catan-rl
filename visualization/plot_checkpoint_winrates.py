"""
Plot win-rate evolution over base-training checkpoints for each bot level.

Reads the JSON produced by marl/tests/evaluate_checkpoints.py.

Usage:
    python visualization/plot_checkpoint_winrates.py \
        --input results/checkpoint_winrates.json \
        [--2x1]   # side-by-side instead of stacked (unused, single panel only)

Output: docs/checkpoint_winrates.pdf + .png
"""

import argparse
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "docs")

LEVEL_STYLES = {
    "1": dict(color="#2196F3", label="Bot Level 1"),   # blue
    "2": dict(color="#FF9800", label="Bot Level 2"),   # orange
    "3": dict(color="#E53935", label="Bot Level 3"),   # red
}
SMOOTH_WINDOW = 5


def smooth(values, window):
    if len(values) < window:
        return np.array(values)
    return np.convolve(values, np.ones(window) / window, mode="valid")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="results/checkpoint_winrates.json",
                        help="JSON file produced by evaluate_checkpoints.py")
    parser.add_argument("--smooth", type=int, default=SMOOTH_WINDOW,
                        help="Smoothing window (set to 1 to disable)")
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    results = data["results"]
    checkpoints = np.array([r["checkpoint"] for r in results])
    levels = [str(l) for l in data["metadata"]["bot_levels"]]

    fig, ax = plt.subplots(figsize=(8, 4.5))

    for level in levels:
        raw = np.array([r["win_rates"][level] * 100 for r in results])
        style = LEVEL_STYLES.get(level, dict(color="grey", label=f"Bot Level {level}"))

        # raw line, faint
        ax.plot(checkpoints, raw,
                color=style["color"], linewidth=0.6, alpha=0.35, zorder=1)

        # smoothed line
        if args.smooth > 1:
            sv = smooth(raw, args.smooth)
            se = checkpoints[args.smooth - 1:]
            ax.plot(se, sv,
                    color=style["color"], linewidth=2.0,
                    label=style["label"], zorder=2)
        else:
            ax.plot(checkpoints, raw,
                    color=style["color"], linewidth=2.0,
                    label=style["label"], zorder=2)

    # 25% random-chance reference
    ax.axhline(25, color="#555555", linewidth=1.0, linestyle="--",
               label="Random baseline (25%)")

    ax.set_xlabel("Training epoch", fontsize=11)
    ax.set_ylabel("Win rate (%)", fontsize=11)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=9, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=9)

    os.makedirs(OUT_DIR, exist_ok=True)
    out_pdf = os.path.join(OUT_DIR, "checkpoint_winrates.pdf")
    out_png = os.path.join(OUT_DIR, "checkpoint_winrates.png")
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_pdf}")
    print(f"Saved: {out_png}")
    plt.close(fig)


if __name__ == "__main__":
    main()
