from ..losses.neurd import neurd_loss
from .ppo import ppo_value_loss, ppo_entropy_loss
import jax
import jax.numpy as jnp
from ..utils import softmax, safe_log_softmax


def rnad_regularization(
  log_probs: jax.Array, magnet_log_probs: jax.Array, magnet_coef: float
) -> jax.Array:
  return magnet_coef * (log_probs - magnet_log_probs)


def estimate_baseline_regrets(
  value: jax.Array,
  advanatge: jax.Array,
  strategy: jax.Array,
  sampling_strategy: jax.Array,
  action: jax.Array,
) -> jax.Array:
  # Taken from https://arxiv.org/abs/1809.03057
  # Baseline + (Utility - Baseline)/pi. Advantage is basically Utility-baseline.
  q_values = value + jax.nn.one_hot(
    action, value.shape[-1]
  ) * advanatge / jnp.take_along_axis(sampling_strategy, action[None], axis=-1)
  regrets = q_values - jnp.sum(q_values * strategy, axis=-1, keepdims=True)
  return regrets


def rnad_loss(
  values: jax.Array,
  logits: jax.Array,
  legal_actions: jax.Array,
  actions: jax.Array,
  sample_logits: jax.Array,
  sample_values: jax.Array,
  magnet_logits: jax.Array,
  advantages: jax.Array,
  returns: jax.Array,
  clip_eps: float,
  vf_coef: float,
  ent_coef: float,
  magnet_coef: float,
  neurd_clip: float,
  neurd_threshold: float,
) -> tuple[jax.Array, dict]:
  log_probs_all = safe_log_softmax(logits, legal_actions)
  strategy = softmax(logits, legal_actions)
  magnet_log_probs_all = safe_log_softmax(magnet_logits, legal_actions)
  sampling_strategy = softmax(sample_logits, legal_actions)

  regularized_value = values - rnad_regularization(
    log_probs_all, magnet_log_probs_all, magnet_coef
  )

  regrets = estimate_baseline_regrets(
    regularized_value, advantages, strategy, sampling_strategy, actions
  )

  policy_loss = -neurd_loss(logits, legal_actions, regrets, neurd_clip, neurd_threshold)

  value_loss = ppo_value_loss(values, sample_values, returns, clip_eps)
  entropy_loss = ppo_entropy_loss(log_probs_all, strategy)

  loss = policy_loss + vf_coef * value_loss + ent_coef * entropy_loss
  metrics = {
    "policy_loss": policy_loss,
    "value_loss": value_loss,
    "entropy_loss": entropy_loss,
  }
  return loss, metrics
