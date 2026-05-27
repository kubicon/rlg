from __future__ import annotations

import jax
import jax.numpy as jnp
import flax.linen as nn

from .base import Head


class CategoricalHead(Head):
  """Outputs logits over n_actions for a categorical distribution."""

  n_actions: int

  @nn.compact
  def __call__(self, x: jax.Array) -> jax.Array:
    return nn.Dense(self.n_actions)(x)


class GaussianHead(Head):
  """Outputs (mean, log_std) for a Gaussian distribution."""

  action_dim: int
  log_std_min: float = -20.0
  log_std_max: float = 2.0

  @nn.compact
  def __call__(self, x: jax.Array) -> tuple[jax.Array, jax.Array]:
    mean = nn.Dense(self.action_dim)(x)
    log_std = jnp.clip(nn.Dense(self.action_dim)(x), self.log_std_min, self.log_std_max)
    return mean, log_std


class QHead(Head):
  """Outputs Q-values for each action: shape (..., n_actions)."""

  n_actions: int
  # Small init keeps early Q-values near zero so Retrace targets stay in
  # the reward range at the start of training (avoids bootstrap instability).
  output_scale: float = 0.01

  @nn.compact
  def __call__(self, x: jax.Array) -> jax.Array:
    return nn.Dense(
      self.n_actions,
      kernel_init=nn.initializers.variance_scaling(
        self.output_scale, "fan_in", "truncated_normal"
      ),
    )(x)


class DistributionalQHead(Head):
  """Outputs atom logits for distributional RL: shape (..., n_actions, n_atoms)."""

  n_actions: int
  n_atoms: int

  @nn.compact
  def __call__(self, x: jax.Array) -> jax.Array:
    logits = nn.Dense(self.n_actions * self.n_atoms)(x)
    return logits.reshape(logits.shape[:-1] + (self.n_actions, self.n_atoms))


class ValueHead(Head):
  """Outputs a scalar state value: shape (...)."""

  @nn.compact
  def __call__(self, x: jax.Array) -> jax.Array:
    return nn.Dense(1)(x).squeeze(-1)


class AdvantageHead(Head):
  """Outputs mean-centered advantages for dueling DQN: shape (..., n_actions)."""

  n_actions: int

  @nn.compact
  def __call__(self, x: jax.Array) -> jax.Array:
    a = nn.Dense(self.n_actions)(x)
    return a - a.mean(axis=-1, keepdims=True)
