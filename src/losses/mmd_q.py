"""MMD-Q loss: policy head + Q-head, no V-head.

The policy gradient uses an all-actions advantage built from the Polyak
target network's Q-vector (stop-gradiented), with the sampled action
replaced by its multi-step Retrace(λ) target:

    q_vec(a) = Q_target(s, a),   q_vec(a_taken) = q_target

The advantage is centered by the baseline E_π[q_vec] over the SAME vector,
so that Σ_a π(a)·A(s, a) = 0 — an unbiased, variance-reduced per-action
advantage. Using one consistent (target-net) Q-vector for both the
per-action values and the baseline is what keeps it centered; mixing the
live Q-head, a target-net baseline, and a multi-step target across actions
would leave a systematic offset that biases the update.

For NeuRD (loss_type=RNAD) the per-action regret is this same centered
Q-vector, with the magnet regularization applied as a separate KL penalty
term rather than being subtracted from the Q-values.

The live Q-head is supervised only via the Q-loss (live Q(s, a_taken) →
q_target) and never enters the advantage. No separate V-head or V-loss.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .ppo import ppo_policy_loss, ppo_value_loss
from .neurd import neurd_loss
from ..policy import policy_probs, policy_log_probs, policy_entropy_loss, policy_kl
from ..algorithms.types import KLDirection, LossType


def mmd_q_loss(
  q_values: jax.Array,        # (A,) — current Q(s, :)
  logits: jax.Array,          # (A,) — current policy logits
  legal_actions: jax.Array,   # (A,) — boolean action mask
  actions: jax.Array,         # ()   int — action taken during rollout
  sample_logits: jax.Array,   # (A,) — sampling policy logits (from rollout)
  sample_q_values: jax.Array, # (A,) — sampling Q(s,:) — for Q-loss clipping
  magnet_logits: jax.Array,   # (A,) — magnet (periodically-reset) policy
  q_target: jax.Array,        # ()   — Retrace(λ) target for Q(s, a_taken) (stop-grad)
  target_q_values: jax.Array, # (A,) — target-net Q(s,:) (stop-grad) — advantage
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
  """MMD-Q / RNaD-Q loss for a single (timestep, player) sample → scalar."""
  pi = policy_probs(logits, legal_actions, alpha)
  log_probs_all = policy_log_probs(logits, legal_actions, alpha)
  sample_pi = policy_probs(sample_logits, legal_actions, alpha)
  sample_log_probs_all = policy_log_probs(sample_logits, legal_actions, alpha)
  magnet_pi = policy_probs(magnet_logits, legal_actions, alpha)
  magnet_log_probs_all = policy_log_probs(magnet_logits, legal_actions, alpha)

  if kl_direction == KLDirection.FORWARD:
    _kl = lambda p, lp, q, lq: policy_kl(q, lq, p, lp, alpha)
  else:
    _kl = lambda p, lp, q, lq: policy_kl(p, lp, q, lq, alpha)

  # Q-vector: target-net for all actions, Retrace target for the sampled action.
  q_vec = target_q_values.at[actions].set(q_target)

  if loss_type == LossType.RNAD:
    # nan-safe centering: α-entmax may assign zero mass to legal actions,
    # where the regularised Q can be ±inf; mask by pi>0 to avoid inf·0=nan.
    baseline = jnp.sum(jnp.where(pi > 0, q_vec * pi, 0.0))
    regrets = q_vec - baseline
    policy_loss = -neurd_loss(logits, legal_actions, regrets, neurd_clip, neurd_threshold)
  else:
    log_prob = log_probs_all[actions]
    sample_log_prob = sample_log_probs_all[actions]
    v_baseline = jnp.dot(q_vec, pi)
    advantage = q_target - v_baseline
    policy_loss = ppo_policy_loss(log_prob, sample_log_prob, advantage, clip_eps)

  # ── Q-loss: live Q(s, a_taken) → Retrace target, PPO-clipped ─────────────
  q_taken = q_values[actions]
  sample_q_taken = sample_q_values[actions]
  q_loss = ppo_value_loss(q_taken, sample_q_taken, q_target, clip_eps)

  # ── Auxiliary terms ───────────────────────────────────────────────────────
  magnet_loss = _kl(pi, log_probs_all, magnet_pi, magnet_log_probs_all)
  old_kl_loss = _kl(pi, log_probs_all, sample_pi, sample_log_probs_all)
  entropy_loss = policy_entropy_loss(pi, log_probs_all, alpha)

  total = (
    policy_loss
    + vf_coef * q_loss
    + ent_coef * entropy_loss
    + magnet_coef * magnet_loss
    + old_policy_coef * old_kl_loss
  )
  return total, {
    "policy_loss": policy_loss,
    "q_loss": q_loss,
    "entropy_loss": entropy_loss,
    "magnet_loss": magnet_loss,
    "old_kl_loss": old_kl_loss,
  }
