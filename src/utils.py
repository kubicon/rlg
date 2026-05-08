import jax
from jax import numpy as jnp


def safe_log_softmax(logits: jax.Array, legal_actions: jax.Array) -> jax.Array:
  log_probs = jax.nn.log_softmax(logits, where=legal_actions)
  log_probs = jnp.where(legal_actions, log_probs, 0.0)
  return log_probs

def kl_divergence(log_probs1: jax.Array, log_probs2: jax.Array) -> jax.Array:
  kl_divergence = jnp.sum(jnp.exp(log_probs1) * (log_probs1 - log_probs2), axis=-1)
  return kl_divergence



def weighted_sum(values: jax.Array, reach_probs: jax.Array, axis: int = 0) -> jax.Array:
  return jnp.vecdot(values, reach_probs, axis=axis)

def weighted_mean(values: jax.Array, reach_probs: jax.Array, axis: int = 0) -> jax.Array:
  reach_norm = reach_probs.sum(axis)
  denominator = jnp.where(reach_norm == 0, 1, reach_norm)
  return weighted_sum(values, reach_probs, axis) / denominator