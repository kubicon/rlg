from .ppo import ppo_policy_loss, ppo_value_loss, ppo_entropy_loss
import jax
import jax.numpy as jnp
from ..utils import safe_log_softmax, kl_divergence

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
  old_policy_coef: float
) -> jax.Array:
  strategy= jax.nn.softmax(logits, where=legal_actions)                   # (A,)
  log_probs_all= safe_log_softmax(logits, legal_actions)         # (A,)
  sample_log_probs_all = safe_log_softmax(sample_logits, legal_actions)  # (A,)
  magnet_log_probs_all = safe_log_softmax(magnet_logits, legal_actions)  # (A,)
  
  log_prob        = log_probs_all[actions]
  sample_log_prob = sample_log_probs_all[actions]

  policy_loss  = ppo_policy_loss(log_prob, sample_log_prob, advantages, clip_eps)
  value_loss   = ppo_value_loss(values, sample_values, returns, clip_eps)
  magnet_loss = kl_divergence(log_probs_all, magnet_log_probs_all)
  old_kl_loss = kl_divergence(log_probs_all, sample_log_probs_all)
  entropy_loss = ppo_entropy_loss(log_probs_all, strategy) 
  return policy_loss + vf_coef * value_loss + ent_coef * entropy_loss + magnet_loss * magnet_coef + old_kl_loss * old_policy_coef