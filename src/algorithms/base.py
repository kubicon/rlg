"""Layer 5 — Algorithm abstract base.

An Algorithm is a fully self-contained training unit. It owns the environment,
the network, the optimizer, the rollout/replay buffer, and every piece of logic
needed to go from the current state to an updated state in one call.

The Trainer's only job is to call algorithm.step in a loop and handle
side-effects (logging, checkpointing). It has no influence on how training
proceeds internally.
"""
from __future__ import annotations
import abc
from typing import Any

import jax

AlgorithmState = Any  # concrete algorithms define their own pytree


class Algorithm(abc.ABC):
  """Self-contained training unit: owns env, network, optimizer, and buffer."""

  @abc.abstractmethod
  def init(self, key: jax.Array) -> AlgorithmState:
    """Initialize all state: params, opt_state, env_state, rng, step counter."""

  @abc.abstractmethod
  def step(self, state: AlgorithmState) -> tuple[AlgorithmState, dict[str, jax.Array]]:
    """One complete training iteration.

    Internally handles: data collection from the env, advantage estimation,
    and all gradient updates (epochs × minibatches for on-policy).

    Returns updated AlgorithmState and a dict of scalar metrics.
    """
