from .ppo import ppo_policy_loss, ppo_value_loss
from .neurd import neurd_loss
import jax
import jax.numpy as jnp
from ..policy import policy_probs, policy_log_probs, policy_entropy_loss, policy_kl
from ..algorithms.types import KLDirection, LossType


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
  # nan-safe baseline: with α-entmax, zero-mass actions may carry ±inf
  # regularised values; masking by strategy>0 avoids inf·0 = nan. Any residual
  # non-finite per-action regret is bounded by NeuRD's advantage clip downstream.
  baseline = jnp.sum(
    jnp.where(strategy > 0, q_values * strategy, 0.0), axis=-1, keepdims=True
  )
  regrets = q_values - baseline
  return regrets


def rnad_regularization(
  log_probs: jax.Array, magnet_log_probs: jax.Array, magnet_coef: float
) -> jax.Array:
  return magnet_coef * (log_probs - magnet_log_probs)


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
  old_policy_coef: float = 0.0,
  neurd_clip: float = 5.0,
  neurd_threshold: float = 2.0,
  alpha: float = 1.0,
  kl_direction: KLDirection = KLDirection.REVERSE,
  loss_type: LossType = LossType.MMD,
) -> tuple[jax.Array, dict]:
  strategy = policy_probs(logits, legal_actions, alpha)
  log_probs_all = policy_log_probs(logits, legal_actions, alpha)
  sample_strategy = policy_probs(sample_logits, legal_actions, alpha)
  sample_log_probs_all = policy_log_probs(sample_logits, legal_actions, alpha)
  magnet_strategy = policy_probs(magnet_logits, legal_actions, alpha)
  magnet_log_probs_all = policy_log_probs(magnet_logits, legal_actions, alpha)

  if kl_direction == KLDirection.FORWARD:
    _kl = lambda p, lp, q, lq: policy_kl(q, lq, p, lp, alpha)
  else:
    _kl = lambda p, lp, q, lq: policy_kl(p, lp, q, lq, alpha)

  if loss_type == LossType.RNAD:
    regrets = estimate_baseline_regrets(values + jnp.zeros_like(logits), advantages, strategy, sample_strategy, actions)
    policy_loss = -neurd_loss(logits, legal_actions, regrets, neurd_clip, neurd_threshold)
  else:
    log_prob = log_probs_all[actions]
    sample_log_prob = sample_log_probs_all[actions]
    policy_loss = ppo_policy_loss(log_prob, sample_log_prob, advantages, clip_eps)

  value_loss = ppo_value_loss(values, sample_values, returns, clip_eps)
  magnet_loss = _kl(strategy, log_probs_all, magnet_strategy, magnet_log_probs_all)
  old_kl_loss = _kl(strategy, log_probs_all, sample_strategy, sample_log_probs_all)
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
