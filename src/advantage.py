"""Advantage estimation functions.

All functions are pure: arrays in, arrays out. Safe to use inside jax.jit,
jax.vmap, or jax.lax.scan without modification.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def compose_affine_transforms(
  transform_i: tuple[jax.Array, jax.Array],
  transform_j: tuple[jax.Array, jax.Array],
  reverse: bool = False,
) -> tuple[jax.Array, jax.Array]:
  """Composes two affine transforms (bias, scale)."""
  if reverse:
    transform_i, transform_j = transform_j, transform_i
  bias_i, scale_i = transform_i
  bias_j, scale_j = transform_j
  return bias_j + scale_j * bias_i, scale_j * scale_i


def retrace(
  rewards: jax.Array,      # (T,)
  q_taken: jax.Array,      # (T,) — Q_target(s_t, a_t)
  v_target: jax.Array,     # (T,) — E_{a~π_target}[Q_target(s_t, a)]
  c: jax.Array,            # (T,) — λ·min(1, π(a_t)/μ(a_t)) trace coefficients
  discount: jax.Array,     # (T,) — γ*(1 - done_t)
  bootstrap_q: float | jax.Array = 0.0,
) -> jax.Array:            # (T,) — Q^ret(s_t, a_t) retrace targets
  """Retrace(λ) Q-value targets (Munos et al., 2016).
  Implemented as an associative backward scan (same structure as vtrace) so
  it is fully compatible with jax.jit and jax.vmap.
  """
  bootstrap_q = jnp.broadcast_to(bootstrap_q, jnp.shape(q_taken)[1:])

  # Shift quantities one step forward: index t holds the t+1 value.
  next_v = jnp.concatenate([v_target[1:], jnp.zeros_like(v_target[:1])])
  next_q = jnp.concatenate([q_taken[1:], jnp.expand_dims(bootstrap_q, 0)])
  next_c = jnp.concatenate([c[1:], jnp.zeros_like(c[:1])])

  bias = rewards + discount * (next_v - next_c * next_q)
  scale = discount * next_c

  rev = (bias[::-1], scale[::-1])
  rev_q_ret, _ = jax.lax.associative_scan(compose_affine_transforms, rev)
  return rev_q_ret[::-1]

def vtrace(
  reward: jax.Array,
  value: jax.Array,
  discount: jax.Array,
  importance_sampling: float | jax.Array = 1.0,
  bootstrap_value: float | jax.Array = 0.0,
  lambda_: float | jax.Array = 1.0,
  delta_clip: float | jax.Array = 1.0,
  trace_clip: float | jax.Array = 1.0,
) -> tuple[jax.Array, jax.Array]:
  clipped_delta_ratio = jnp.minimum(delta_clip, importance_sampling)
  clipped_trace_ratio = lambda_ * jnp.minimum(trace_clip, importance_sampling)

  bootstrap_value = jnp.broadcast_to(bootstrap_value, jnp.shape(value)[1:])
  next_value = jnp.concatenate([value[1:], jnp.expand_dims(bootstrap_value, 0)])

  raw_td_error = reward + discount * next_value - value
  weighted_td_error = clipped_delta_ratio * raw_td_error

  recursive_decay = discount * clipped_trace_ratio

  rev_transforms = weighted_td_error[::-1], recursive_decay[::-1]
  rev_transform = jax.lax.associative_scan(compose_affine_transforms, rev_transforms)
  rev_advantage, _ = rev_transform
  advantage = rev_advantage[::-1]

  target = value + advantage

  return target, advantage
