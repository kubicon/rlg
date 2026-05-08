"""Standard Trainer implementation with stdout logging and file checkpointing."""

from __future__ import annotations
import os
import time
import math
import pickle
from typing import Any

import numpy as np
import jax
import jax.numpy as jnp

from .base import Logger, Trainer
from ..algorithms.base import Algorithm


class StdoutLogger(Logger):
  """Prints metrics to stdout, one line per step."""

  def __init__(self, keys: list[str] | None = None) -> None:
    self._keys = keys

  def write(self, metrics: dict[str, float], step: int) -> None:
    keys = self._keys or sorted(metrics)
    parts = "  ".join(f"{k}: {metrics[k]:.4f}" for k in keys if k in metrics)
    print(f"step {step:6d} | {parts}")

  def close(self) -> None:
    pass


def load_metrics(checkpoint_dir: str) -> dict[str, np.ndarray]:
  """Load the full per-step metrics array saved alongside checkpoints.

  Returns a dict mapping metric name → 1-D numpy array of length n_steps_logged.
  """
  path = os.path.join(checkpoint_dir, "metrics.npz")
  data = np.load(path)
  return dict(data)


class StandardTrainer(Trainer):
  """Thin outer loop: calls algorithm.step, logs metrics, saves checkpoints.

  Args:
    algorithm:         Any Algorithm instance.
    log_every:         Log metrics every this many steps.
    checkpoint_every:  Save a checkpoint every this many steps (None to disable).
                       Auto-rounded up to the nearest multiple of log_every.
    checkpoint_dir:    Directory for checkpoint and metrics files.
    logger:            Logger instance; falls back to StdoutLogger if None.
    save_metrics:      If True, accumulate raw per-step metric arrays and write
                       them to {checkpoint_dir}/metrics.npz after every log
                       chunk. Disabled by default because it adds disk I/O.
  """

  def __init__(
    self,
    algorithm: Algorithm,
    log_every: int = 1,
    checkpoint_every: int | None = None,
    checkpoint_dir: str | None = None,
    logger: Logger | None = None,
    save_metrics: bool = False,
  ) -> None:
    self.algorithm = algorithm
    self.log_every = log_every
    self.checkpoint_dir = checkpoint_dir
    self.logger = logger or StdoutLogger()
    self.save_metrics = save_metrics

    if checkpoint_every is not None and log_every > 1:
      rounded = math.ceil(checkpoint_every / log_every) * log_every
      if rounded != checkpoint_every:
        print(
          f"checkpoint_every rounded {checkpoint_every} → {rounded} "
          f"to be divisible by log_every={log_every}"
        )
      self.checkpoint_every = rounded
    else:
      self.checkpoint_every = checkpoint_every

  def train(self, state: Any, n_steps: int) -> Any:
    """Run n_steps training iterations.

    n_steps is rounded up to the nearest multiple of log_every so the scan
    chunk length is always consistent and no second JIT compilation is needed.

    Args:
      state:   AlgorithmState returned by algorithm.init(key).
      n_steps: Number of calls to algorithm.step.

    Returns:
      Final AlgorithmState after all steps.
    """
    chunk = self.log_every
    rounded = math.ceil(n_steps / chunk) * chunk
    if rounded != n_steps:
      print(
        f"n_steps rounded {n_steps} → {rounded} to be divisible by log_every={chunk}"
      )
    n_steps = rounded

    def scan_fn(carry, _):
      new_state, metrics = self.algorithm.step(carry)
      return new_state, metrics

    run_chunk = jax.jit(lambda s: jax.lax.scan(scan_fn, s, None, length=chunk))

    t0 = time.perf_counter()
    steps_done = 0
    accumulated: dict[str, list[np.ndarray]] = {}

    def _after_chunk(state: Any, chunk_metrics: dict) -> None:
      nonlocal steps_done
      steps_done += chunk
      step = int(state.step)
      elapsed = time.perf_counter() - t0

      # Scalar means for stdout logging.
      scalar_metrics = {k: float(jnp.mean(v)) for k, v in chunk_metrics.items()}
      scalar_metrics["steps_per_sec"] = steps_done / elapsed
      self.logger.write(scalar_metrics, step)

      # Accumulate raw arrays for metrics file.
      if self.save_metrics and self.checkpoint_dir:
        for k, v in chunk_metrics.items():
          arr = np.asarray(v)  # shape (chunk,) — one scalar per step
          accumulated.setdefault(k, []).append(arr)
        path = os.path.join(self.checkpoint_dir, "metrics.npz")
        arrays: dict[str, np.ndarray] = {
          k: np.concatenate(vs) for k, vs in accumulated.items()
        }
        np.savez(path, **arrays)  # type: ignore[arg-type]

      if self.checkpoint_every and step % self.checkpoint_every == 0:
        self._save(state)

    for _ in range(n_steps // chunk):
      state, chunk_metrics = run_chunk(state)
      _after_chunk(state, chunk_metrics)

    self.logger.close()
    return state

  def _save(self, state: Any) -> None:
    if self.checkpoint_dir is None:
      return
    os.makedirs(self.checkpoint_dir, exist_ok=True)
    step = int(state.step)
    path = os.path.join(self.checkpoint_dir, f"step_{step:08d}.pkl")
    with open(path, "wb") as f:
      pickle.dump(jax.device_get(state), f)
    print(f"checkpoint saved → {path}")
