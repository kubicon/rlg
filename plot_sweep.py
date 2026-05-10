"""
Load exploitability.npz from a sweep experiment directory and plot results.

Subdirectory names are parsed as key=value pairs (separated by underscores).
One figure is produced per unique combination of all keys except --curve-key.
Within each figure, one curve is drawn per value of --curve-key.

Usage:
    python plot_sweep.py public_experiments/magnet_strength --curve-key magnet_coef
    python plot_sweep.py public_experiments/magnet_strength --curve-key magnet_coef --out-dir plots/magnet/
    python plot_sweep.py public_experiments/magnet_strength --curve-key magnet_coef --metric nashconv
"""

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ── directory-name parsing ────────────────────────────────────────────────────

def _parse_name(name: str) -> dict[str, str]:
    """Parse 'magnet_coef=0.2_loss_type=rnad_env=leduc' into the corresponding dict.

    Keys may themselves contain underscores. Parts without '=' are accumulated
    as key prefixes until a part with '=' is found.
    """
    result = {}
    key_parts: list[str] = []
    for part in name.split("_"):
        if "=" in part:
            k, v = part.split("=", 1)
            key_parts.append(k)
            result["_".join(key_parts)] = v
            key_parts = []
        else:
            key_parts.append(part)
    return result


def _try_numeric(v: str):
    try:
        return float(v)
    except ValueError:
        return v


# ── plotting ──────────────────────────────────────────────────────────────────

def _figure_title(group_keys: dict[str, str]) -> str:
    return "  |  ".join(f"{k}={v}" for k, v in group_keys.items())


def _out_filename(group_keys: dict[str, str]) -> str:
    return "_".join(f"{k}={v}" for k, v in group_keys.items()) + ".png"


def plot_sweep(exp_dir: Path, curve_key: str, metric: str, out_dir: Path) -> None:
    subdirs = [d for d in sorted(exp_dir.iterdir()) if d.is_dir()]

    # collect all runs with parseable names and existing metric file
    runs: list[tuple[dict, np.ndarray, np.ndarray]] = []
    for d in subdirs:
        params = _parse_name(d.name)
        if curve_key not in params:
            continue
        npz = d / "exploitability.npz"
        if not npz.exists():
            print(f"  skipping {d.name}: no exploitability.npz")
            continue
        data = np.load(npz)
        if metric not in data:
            print(f"  skipping {d.name}: metric '{metric}' not in {list(data.keys())}")
            continue
        runs.append((params, data["step"], data[metric]))

    if not runs:
        print("No valid runs found.")
        return

    # group by all keys except curve_key, preserving insertion order
    groups: dict[tuple, list] = {}
    for params, steps, values in runs:
        group_key = tuple((k, v) for k, v in params.items() if k != curve_key)
        groups.setdefault(group_key, []).append((params[curve_key], steps, values))

    out_dir.mkdir(parents=True, exist_ok=True)

    for group_key, curves in groups.items():
        group_dict = dict(group_key)
        # sort curves by numeric value of curve_key where possible
        curves.sort(key=lambda x: _try_numeric(x[0]))

        fig, ax = plt.subplots(figsize=(7, 4))
        for label, steps, values in curves:
            ax.plot(steps, values, linewidth=1.5, label=f"{curve_key}={label}")

        ax.set_xlabel("Step")
        ax.set_ylabel(metric)
        ax.set_title(_figure_title(group_dict))
        ax.legend()
        fig.tight_layout()

        out_path = out_dir / _out_filename(group_dict)
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"  saved {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("exp_dir", type=Path, help="Sweep experiment directory")
    parser.add_argument(
        "--curve-key",
        required=True,
        help="Hyperparameter to vary across curves within each figure",
    )
    parser.add_argument(
        "--metric",
        default="nashconv",
        help="Metric to plot from exploitability.npz (default: nashconv)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for PNGs (default: plots/<exp_dir_name>/)",
    )
    args = parser.parse_args()

    out_dir = args.out_dir or Path("plots") / args.exp_dir.name
    print(f"Plotting {args.exp_dir} → {out_dir}/")
    plot_sweep(args.exp_dir, args.curve_key, args.metric, out_dir)


if __name__ == "__main__":
    main()
