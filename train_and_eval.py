"""Train an agent and compute exact exploitability at every checkpoint, then plot.

Usage:
    python train_and_eval.py configs/mmd_goofspiel.yaml

Requires trainer.checkpoint_dir and trainer.checkpoint_every to be set in the
config so that checkpoints are available for the exploitability eval phase.
"""

from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from train import main as _train
from eval_checkpoints import main as _eval


def _plot(checkpoint_dir: str) -> None:
    data = np.load(os.path.join(checkpoint_dir, "exploitability.npz"))
    steps = data["step"]
    nashconv = data["nashconv"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    for ax, yscale in zip(axes, ("linear", "log")):
        ax.plot(steps, nashconv, linewidth=1.5)
        ax.set_xlabel("Training step")
        ax.set_ylabel("NashConv")
        ax.set_yscale(yscale)
        ax.set_title(f"NashConv ({yscale} scale)")
        ax.grid(True, alpha=0.3, which="both")

    fig.tight_layout()
    out = os.path.join(checkpoint_dir, "nashconv.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"plot saved → {out}")


def main(config_path: str) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    trainer_cfg = cfg.get("trainer", {})
    checkpoint_dir = trainer_cfg.get("checkpoint_dir")
    if not checkpoint_dir:
        raise ValueError("trainer.checkpoint_dir must be set in the config")
    if not trainer_cfg.get("checkpoint_every"):
        raise ValueError("trainer.checkpoint_every must be set in the config")

    # ── Phase 1: Train ────────────────────────────────────────────────────
    print("=" * 60)
    print("PHASE 1: Training")
    print("=" * 60 + "\n")
    _train(config_path)

    # ── Phase 2: Exploitability ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PHASE 2: Exploitability")
    print("=" * 60 + "\n")
    _eval(checkpoint_dir)

    # ── Phase 3: Plot ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PHASE 3: Plot")
    print("=" * 60 + "\n")
    _plot(checkpoint_dir)


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/mmd_goofspiel.yaml"
    main(config_path)
