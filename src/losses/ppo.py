"""PPO (clipped surrogate) loss.

All functions are pure and operate on a single (timestep, player) sample.
ppo_loss expects scalar / (A,) shaped inputs and returns a scalar.
Callers vmap over T and P (and B) and reduce with a trajectory-aware
weighted mean that masks out padding steps after episode termination.

Dimension convention:
  T — timesteps in the episode        (vmap'd externally)
  P — number of players               (vmap'd externally)
  A — number of actions
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.lax as lax
from ..policy import policy_probs, policy_log_probs, policy_entropy_loss


def ppo_loss(
  values: jax.Array,  # ()      — current value estimate
  logits: jax.Array,  # (A,)    — current policy logits
  legal_actions: jax.Array,  # (A,)    — boolean action mask
  actions: jax.Array,  # ()  int — action taken during rollout
  sample_logits: jax.Array,  # (A,)    — old policy logits
  sample_values: jax.Array,  # ()      — old value estimate
  advantages: jax.Array,  # ()      — advantage estimate
  returns: jax.Array,  # ()      — value target
  clip_eps: float = 0.2,
  vf_coef: float = 0.5,
  ent_coef: float = 0.01,
  alpha: float = 1.0,
) -> tuple[jax.Array, dict]:
  """PPO loss for a single (timestep, player) sample → scalar."""
  strategy = policy_probs(logits, legal_actions, alpha)
  log_probs_all = policy_log_probs(logits, legal_actions, alpha)
  sample_log_probs_all = policy_log_probs(sample_logits, legal_actions, alpha)

  # Log prob of the action actually taken → scalar
  log_prob = log_probs_all[actions]
  sample_log_prob = sample_log_probs_all[actions]

  policy_loss = ppo_policy_loss(log_prob, sample_log_prob, advantages, clip_eps)
  value_loss = ppo_value_loss(values, sample_values, returns, clip_eps)
  entropy_loss = policy_entropy_loss(strategy, log_probs_all, alpha)

  total = policy_loss + vf_coef * value_loss + ent_coef * entropy_loss
  return total, {
    "policy_loss": policy_loss,
    "value_loss": value_loss,
    "entropy_loss": entropy_loss,
  }


def ppo_policy_loss(
  log_prob: jax.Array,  # () — log prob of chosen action, current policy
  sample_log_prob: jax.Array,  # () — log prob of chosen action, old policy
  advantage: jax.Array,  # ()
  clip_eps: float = 0.2,
) -> jax.Array:
  ratio = jnp.exp(log_prob - sample_log_prob)
  advantage = lax.stop_gradient(advantage)
  return -jnp.minimum(
    ratio * advantage,
    jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantage,
  )


def ppo_value_loss(
  values: jax.Array,  # () — current value estimate
  sample_values: jax.Array,  # () — old value estimate
  returns: jax.Array,  # () — value target
  clip_eps: float = 0.2,
) -> jax.Array:
  v_clipped = sample_values + jnp.clip(values - sample_values, -clip_eps, clip_eps)
  return 0.5 * jnp.maximum(
    (values - returns) ** 2,
    (v_clipped - returns) ** 2,
  )
