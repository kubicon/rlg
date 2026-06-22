from .ppo import ppo_policy_loss, ppo_value_loss
import jax
from ..policy import policy_probs, policy_log_probs, policy_entropy_loss, policy_kl


def mmd_loss(
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
  old_policy_coef: float,
  alpha: float = 1.0,
) -> tuple[jax.Array, dict]:
  strategy = policy_probs(logits, legal_actions, alpha)  # (A,)
  log_probs_all = policy_log_probs(logits, legal_actions, alpha)  # (A,)
  sample_strategy = policy_probs(sample_logits, legal_actions, alpha)  # (A,)
  sample_log_probs_all = policy_log_probs(sample_logits, legal_actions, alpha)  # (A,)
  magnet_strategy = policy_probs(magnet_logits, legal_actions, alpha)  # (A,)
  magnet_log_probs_all = policy_log_probs(magnet_logits, legal_actions, alpha)  # (A,)

  log_prob = log_probs_all[actions]
  sample_log_prob = sample_log_probs_all[actions]

  policy_loss = ppo_policy_loss(log_prob, sample_log_prob, advantages, clip_eps)
  value_loss = ppo_value_loss(values, sample_values, returns, clip_eps)
  magnet_loss = policy_kl(strategy, log_probs_all, magnet_strategy, magnet_log_probs_all, alpha)
  old_kl_loss = policy_kl(strategy, log_probs_all, sample_strategy, sample_log_probs_all, alpha)
  entropy_loss = policy_entropy_loss(strategy, log_probs_all, alpha)
  total = (
    policy_loss
    + vf_coef * value_loss
    + ent_coef * entropy_loss
    + magnet_coef * magnet_loss
    + old_policy_coef * old_kl_loss
  )
  return total, {
    "policy_loss": policy_loss,
    "value_loss": value_loss,
    "entropy_loss": entropy_loss,
    "magnet_loss": magnet_loss,
    "old_kl_loss": old_kl_loss,
  }
