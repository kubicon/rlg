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
