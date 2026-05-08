"""Standard Trainer implementation with stdout logging and file checkpointing."""
from __future__ import annotations
import os
import time
import pickle
from typing import Any

import jax
import jax.numpy as jnp

from .base import Logger, Trainer
from ..algorithms.base import Algorithm


class StdoutLogger(Logger):
  """Prints metrics to stdout, one line per step."""

  def __init__(self, keys: list[str] | None = None) -> None:
    self._keys = keys  # if None, print all keys in sorted order

  def write(self, metrics: dict[str, float], step: int) -> None:
    keys = self._keys or sorted(metrics)
    parts = "  ".join(f"{k}: {metrics[k]:.4f}" for k in keys if k in metrics)
    print(f"step {step:6d} | {parts}")

  def close(self) -> None:
    pass


class StandardTrainer(Trainer):
  """Thin outer loop: calls algorithm.step, logs metrics, saves checkpoints.

  Args:
    algorithm:         Any Algorithm instance (PPO, DQN, SAC, …).
    log_every:         Log metrics every this many steps.
    checkpoint_every:  Save a checkpoint every this many steps (None to disable).
    checkpoint_dir:    Directory for checkpoint files.
    logger:            Logger instance; falls back to StdoutLogger if None.
  """

  def __init__(
    self,
    algorithm:        Algorithm,
    log_every:        int         = 1,
    checkpoint_every: int | None  = None,
    checkpoint_dir:   str | None  = None,
    logger:           Logger | None = None,
  ) -> None:
    self.algorithm        = algorithm
    self.log_every        = log_every
    self.checkpoint_every = checkpoint_every
    self.checkpoint_dir   = checkpoint_dir
    self.logger           = logger or StdoutLogger()

  def train(self, state: Any, n_steps: int) -> Any:
    """Run n_steps training iterations.

    Uses lax.scan to fuse log_every steps into a single XLA call.  After each
    fused chunk the Python closure _after_chunk handles logging and checkpointing
    (which cannot run inside XLA).

    Args:
      state:   AlgorithmState returned by algorithm.init(key).
      n_steps: Number of calls to algorithm.step.

    Returns:
      Final AlgorithmState after all steps.
    """
    def scan_fn(carry, _):
      new_state, metrics = self.algorithm.step(carry)
      return new_state, metrics

    chunk = self.log_every
    run_chunk = jax.jit(lambda s: jax.lax.scan(scan_fn, s, None, length=chunk))

    t0 = time.perf_counter()
    steps_done = 0

    def _after_chunk(state: Any, chunk_metrics: dict, n: int) -> None:
      nonlocal steps_done
      steps_done += n
      step    = int(state.step)
      elapsed = time.perf_counter() - t0
      scalar_metrics = {k: float(jnp.mean(v)) for k, v in chunk_metrics.items()}
      scalar_metrics['steps_per_sec'] = steps_done / elapsed
      self.logger.write(scalar_metrics, step)
      if self.checkpoint_every and step % self.checkpoint_every == 0:
        self._save(state)

    n_full, remainder = divmod(n_steps, chunk)
    for _ in range(n_full):
      state, chunk_metrics = run_chunk(state)
      _after_chunk(state, chunk_metrics, chunk)

    if remainder:
      run_tail = jax.jit(lambda s: jax.lax.scan(scan_fn, s, None, length=remainder))
      state, tail_metrics = run_tail(state)
      _after_chunk(state, tail_metrics, remainder)

    self.logger.close()
    return state

  def _save(self, state: Any) -> None:
    if self.checkpoint_dir is None:
      return
    os.makedirs(self.checkpoint_dir, exist_ok=True)
    step = int(state.step)
    path = os.path.join(self.checkpoint_dir, f"step_{step:08d}.pkl")
    # Convert JAX arrays to numpy before pickling for portability
    cpu_state = jax.tree.map(lambda x: x if not hasattr(x, 'devices') else jnp.asarray(x), state)
    with open(path, 'wb') as f:
      pickle.dump(cpu_state, f)
    print(f"checkpoint saved → {path}")
