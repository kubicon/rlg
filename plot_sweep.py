"""
Load exploitability.npz from a sweep experiment directory and plot results.

Subdirectory names are parsed as key=value pairs (separated by underscores).
One figure is produced per unique combination of all keys except --curve-key.
Within each figure, one curve is drawn per value of --curve-key.

Usage:
    python plot_sweep.py public_experiments/magnet_strength --curve-key magnet_coef
    python plot_sweep.py public_experiments/magnet_strength --curve-key magnet_coef --out-dir plots/magnet/
    python plot_sweep.py public_experiments/magnet_strength --curve-key magnet_coef --metric nashconv

Plots are saved into exp_dir by default (one PNG per group, containing both linear
and log y-scale side-by-side).
"""

import argparse
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


def _out_filename(group_keys: dict[str, str], yscale: str = "linear") -> str:
  base = "_".join(f"{k}={v}" for k, v in group_keys.items()) if group_keys else "all"
  suffix = "_log" if yscale == "log" else ""
  return base + suffix + ".png"


def _get_colors(n: int) -> list:
  """Return *n* perceptually distinct colors."""
  import colorsys
  if n <= 10:
    cmap = matplotlib.colormaps["tab10"]
    return [cmap(i) for i in range(n)]
  if n <= 20:
    cmap = matplotlib.colormaps["tab20"]
    return [cmap(i) for i in range(n)]
  # evenly spaced hues in HLS space for arbitrarily many curves
  return [colorsys.hls_to_rgb(i / n, 0.45, 0.85) for i in range(n)]


def _lighten_color(color, amount: float = 0.35):
  """Blend *color* toward white by *amount* (0 = unchanged, 1 = white)."""
  import matplotlib.colors as mc
  c = np.array(mc.to_rgb(color))
  return tuple(1.0 - amount * (1.0 - c))


def plot_sweep(exp_dir: Path, curve_keys: list[str], metric: str, out_dir: Path) -> None:
  subdirs = [d for d in sorted(exp_dir.iterdir()) if d.is_dir()]

  # collect all runs with parseable names and existing metric file
  runs: list[tuple[dict, np.ndarray, np.ndarray]] = []
  for d in subdirs:
    params = _parse_name(d.name)
    if not all(k in params for k in curve_keys):
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

  # when seed is a sweep param, aggregate over it instead of splitting curves
  has_seed = any("seed" in params for params, _, _ in runs)
  seed_is_curve = has_seed and curve_keys == ["seed"]
  exclude_from_group = set(curve_keys) | ({"seed"} if has_seed else set())

  def _curve_label(params: dict) -> str:
    if seed_is_curve:
      return "_"
    if len(curve_keys) == 1:
      return params[curve_keys[0]]
    return " | ".join(f"{k}={params[k]}" for k in curve_keys)

  def _curve_sort_key(label: str):
    # sort single-key labels numerically; multi-key labels lexicographically
    return _try_numeric(label)

  # group_key -> curve_label -> [(steps, values), ...]
  groups: dict[tuple, dict[str, list]] = {}
  for params, steps, values in runs:
    group_key = tuple((k, v) for k, v in params.items() if k not in exclude_from_group)
    cv = _curve_label(params)
    groups.setdefault(group_key, {}).setdefault(cv, []).append((steps, values))

  out_dir.mkdir(parents=True, exist_ok=True)

  for group_key, curve_map in groups.items():
    group_dict = dict(group_key)
    sorted_curves = sorted(curve_map.items(), key=lambda x: _curve_sort_key(x[0]))

    colors = _get_colors(len(sorted_curves))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for ax, yscale in zip(axes, ("linear", "log")):
      for (label, seed_runs), color in zip(sorted_curves, colors):
        all_values = np.stack([v for _, v in seed_runs])
        steps = seed_runs[0][0]
        if seed_is_curve:
          plot_label = None
        elif len(curve_keys) == 1:
          plot_label = f"{curve_keys[0]}={label}"
        else:
          plot_label = label  # already contains "k=v | k=v" pairs

        if has_seed and len(seed_runs) > 1:
          mean = np.mean(all_values, axis=0)
          sem = np.std(all_values, axis=0, ddof=1) / np.sqrt(len(seed_runs))
          ci = 1.96 * sem
          (line,) = ax.plot(steps, mean, linewidth=1.5, label=plot_label, color=color)
          ax.fill_between(steps, mean - ci, mean + ci, color=_lighten_color(line.get_color()))
        else:
          ax.plot(steps, all_values[0], linewidth=1.5, label=plot_label, color=color)

      ax.set_xlabel("Step")
      ax.set_ylabel(metric)
      ax.set_yscale(yscale)
      ax.set_title(f"{_figure_title(group_dict)}  ({yscale} scale)" if group_dict else f"({yscale} scale)")
      ax.grid(True, alpha=0.3, which="both")

    legend_below = False
    if not seed_is_curve:
      n_curves = len(sorted_curves)
      if n_curves <= 6:
        for ax in axes:
          ax.legend(fontsize="small")
      else:
        # shared legend below the figure to avoid covering the plots
        handles, labels = axes[0].get_legend_handles_labels()
        ncols = min(4, n_curves)
        n_rows = -(-n_curves // ncols)  # ceiling division
        fig.legend(
          handles, labels,
          loc="lower center",
          bbox_to_anchor=(0.5, 0),
          ncol=ncols,
          fontsize="small",
          framealpha=0.9,
        )
        legend_below = True

    fig.tight_layout()
    if legend_below:
      fig.subplots_adjust(bottom=0.06 + 0.06 * n_rows)
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
    nargs="+",
    help="Hyperparameter(s) to vary across curves within each figure. "
         "Multiple keys produce combined labels (e.g. --curve-key feature_dim activation norm).",
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
    help="Output directory for PNGs (default: exp_dir itself)",
  )
  args = parser.parse_args()

  out_dir = args.out_dir or args.exp_dir
  print(f"Plotting {args.exp_dir} → {out_dir}/")
  plot_sweep(args.exp_dir, args.curve_key, args.metric, out_dir)


if __name__ == "__main__":
  main()
