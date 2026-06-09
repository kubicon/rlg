from .ppo import ppo_policy_loss, ppo_value_loss, ppo_entropy_loss
import jax
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
  local_ratio: jax.Array,
  clip_eps: float,
  vf_coef: float,
  ent_coef: float,
  magnet_coef: float,
  old_policy_coef: float,
) -> tuple[jax.Array, dict]:
  strategy = jax.nn.softmax(logits, where=legal_actions)  # (A,)
  log_probs_all = safe_log_softmax(logits, legal_actions)  # (A,)
  sample_log_probs_all = safe_log_softmax(sample_logits, legal_actions)  # (A,)
  magnet_log_probs_all = safe_log_softmax(magnet_logits, legal_actions)  # (A,)

  log_prob = log_probs_all[actions]
  sample_log_prob = sample_log_probs_all[actions]
  # magnet_log_prob = magnet_log_probs_all[actions]

  # advantages = advantages - magnet_coef *( log_prob - magnet_log_prob)

  # local_ratio = ∏_p π_old(a_p)/μ(a_p) is the JOINT off-policy correction for
  # the draw at this node (1.0 when on-policy). The transition depends on every
  # player's action, so the joint ratio is needed; the trust-region ratio inside
  # ppo_policy_loss stays π/π_old (the acting player's own score function), and
  # this factor multiplies the surrogate to keep the gradient unbiased — the
  # acting player's own ratio pairs with ∇log π, the opponents' ratios correct
  # the realised advantage for their exploratory draws.
  policy_loss = local_ratio * ppo_policy_loss(log_prob, sample_log_prob, advantages, clip_eps)
  value_loss = ppo_value_loss(values, sample_values, returns, clip_eps)
  magnet_loss = kl_divergence(log_probs_all, magnet_log_probs_all)
  old_kl_loss = kl_divergence(log_probs_all, sample_log_probs_all)
  entropy_loss = ppo_entropy_loss(log_probs_all, strategy)
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
