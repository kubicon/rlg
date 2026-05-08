import jax
import jax.numpy as jnp

def neurd_loss(
  logits: jax.Array,
  legal_actions: jax.Array,
  advantages: jax.Array,
  clip: float = 10000,
  threshold: float = 2.0,
) -> jax.Array:
  """
  Computes a Neural Replicator Dynamics (NeuRD) Loss.
  It is same as replicator dynamics, but it forces the difference between any 2 logits for the same input to be at most 2 * threshold.
  References:

  [1] https://arxiv.org/abs/1906.00190
  """ 

  logits_centered = logits * legal_actions

  advantages = jnp.clip(advantages, -clip, clip)

  logit_can_decrease = logits_centered > -threshold
  logit_can_increase = logits_centered < threshold
  positive_advantages = jnp.maximum(0.0, advantages)
  negative_advantages = jnp.minimum(0.0, advantages)

  advantage_weights = (
    logit_can_decrease * negative_advantages + logit_can_increase * positive_advantages
  )

  loss = logits_centered * jax.lax.stop_gradient(advantage_weights)
  loss = jnp.sum(loss * legal_actions, axis=-1)
  return loss