"""
Generate config files from a sweep spec, and optionally run them.

The spec is a normal YAML config with an optional top-level `sweep:` block:

    sweep:
      algorithm.lr: [0.0001, 0.001, 0.003]
      algorithm.batch_size: [32, 64]

    seed: 0
    env:
      name: leduc
    algorithm:
      lr: 0.0001
      ...

One output file is written per element of the Cartesian product of all sweep
lists.  With no `sweep:` block the input is copied as-is (one output file).

Usage:
    python make_configs.py <spec.yaml> [--out-dir <dir>]
    python make_configs.py <spec.yaml> --run-experiments [--runner train.py]
"""

import argparse
import copy
import itertools
import subprocess
import sys
from pathlib import Path

import yaml


# ── nested-dict helpers ───────────────────────────────────────────────────────


def _set_dotted(d: dict, dotted_key: str, value) -> None:
  """Set d[a][b][c] given dotted_key='a.b.c'."""
  keys = dotted_key.split(".")
  for k in keys[:-1]:
    d = d.setdefault(k, {})
  d[keys[-1]] = value


def _apply_overrides(base: dict, overrides: dict) -> dict:
  """Return a deep copy of base with each dotted key set to its value."""
  result = copy.deepcopy(base)
  for dotted_key, value in overrides.items():
    _set_dotted(result, dotted_key, value)
  return result


# ── file-name helpers ─────────────────────────────────────────────────────────


def _short_key(dotted_key: str) -> str:
  """Use the last segment of a dotted key for the filename."""
  return dotted_key.split(".")[-1]


def _value_label(value) -> str:
  if isinstance(value, dict):
    return value.get("name", value.get("type", str(value)))
  return str(value)


def _combo_suffix(sweep_keys: list[str], combo: tuple) -> str:
  parts = [f"{_short_key(k)}={_value_label(v)}" for k, v in zip(sweep_keys, combo)]
  return "_".join(parts)


# ── main ──────────────────────────────────────────────────────────────────────


def _config_out_path(config: dict, fallback_dir: Path, suffix: str) -> Path:
  if "trainer" in config and "checkpoint_dir" in config["trainer"]:
    return Path(config["trainer"]["checkpoint_dir"]) / "config.yaml"
  return fallback_dir / f"config_{suffix}.yaml"


def generate(spec_path: Path, fallback_dir: Path) -> list[Path]:
  with open(spec_path) as f:
    spec = yaml.safe_load(f)

  sweep_spec: dict = spec.pop("sweep", {})
  sweep_zip_spec: dict = spec.pop("sweep_zip", {})

  # Build paired zip combos (all zip keys advance together, not crossed).
  if sweep_zip_spec:
    zip_keys = list(sweep_zip_spec.keys())
    zip_rows = list(zip(*[sweep_zip_spec[k] for k in zip_keys]))
    zip_combos: list[dict] = [dict(zip(zip_keys, row)) for row in zip_rows]
    # Only scalar-valued zip keys appear in the suffix (skip lists/dicts).
    zip_label_keys = [k for k in zip_keys if not isinstance(sweep_zip_spec[k][0], (list, dict))]
  else:
    zip_combos = [{}]
    zip_label_keys = []

  if not sweep_spec and not sweep_zip_spec:
    out_path = _config_out_path(spec, fallback_dir, "")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
      yaml.dump(spec, f, default_flow_style=False, sort_keys=False)
    return [out_path]

  sweep_keys = list(sweep_spec.keys())
  sweep_values = [sweep_spec[k] for k in sweep_keys]

  written: list[Path] = []
  for zip_combo in zip_combos:
    for combo in (itertools.product(*sweep_values) if sweep_keys else [()]):
      overrides = {**zip_combo, **dict(zip(sweep_keys, combo))}
      config = _apply_overrides(spec, overrides)
      zip_parts = [f"{_short_key(k)}={_value_label(zip_combo[k])}" for k in zip_label_keys]
      # Full `algorithm:` dicts in sweep_zip are not scalar zip keys; still label by type.
      if "algorithm" in zip_combo and isinstance(zip_combo["algorithm"], dict):
        zip_parts.insert(0, f"type={zip_combo['algorithm']['type']}")
      cart_parts = [f"{_short_key(k)}={_value_label(v)}" for k, v in zip(sweep_keys, combo)]
      suffix = "_".join(zip_parts + cart_parts)
      if "trainer" in config and "checkpoint_dir" in config["trainer"]:
        config["trainer"]["checkpoint_dir"] += f"/{suffix}"
      out_path = _config_out_path(config, fallback_dir, suffix)
      out_path.parent.mkdir(parents=True, exist_ok=True)
      with open(out_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
      written.append(out_path)

  return written


def main() -> None:
  parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
  )
  parser.add_argument("spec", type=Path, help="Sweep spec YAML file")
  parser.add_argument(
    "--out-dir",
    type=Path,
    default=None,
    help="Output directory (default: same dir as spec)",
  )
  parser.add_argument(
    "--run-experiments",
    action="store_true",
    help="Run each generated config sequentially after writing all files",
  )
  parser.add_argument(
    "--runner",
    type=str,
    default="train_and_eval.py",
    help="Script to run each config with (default: train.py)",
  )
  args = parser.parse_args()

  fallback_dir = args.out_dir if args.out_dir is not None else args.spec.parent
  written = generate(args.spec, fallback_dir)
  print(f"Wrote {len(written)} config(s):")
  for p in written:
    print(f"  {p}")

  if args.run_experiments:
    print()
    for i, config_path in enumerate(written, 1):
      print(f"[{i}/{len(written)}] Running {config_path.name} ...")
      result = subprocess.run([sys.executable, args.runner, str(config_path)])
      if result.returncode != 0:
        print(f"  FAILED (exit code {result.returncode}), stopping.")
        sys.exit(result.returncode)


if __name__ == "__main__":
  main()
