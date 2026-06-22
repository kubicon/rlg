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

For NeuRD (rnad_q_loss) the per-action regret is this same centered
quantity with the magnet regularization subtracted per action.

The live Q-head is supervised only via the Q-loss (live Q(s, a_taken) →
q_target) and never enters the advantage. No separate V-head or V-loss.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .ppo import ppo_policy_loss, ppo_value_loss
from .neurd import neurd_loss
from .rnad import estimate_baseline_regrets, rnad_regularization
from ..policy import policy_probs, policy_log_probs, policy_entropy_loss, policy_kl


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
  old_policy_coef: float,
  alpha: float = 1.0,
) -> tuple[jax.Array, dict]:
  """MMD-Q loss for a single (timestep, player) sample → scalar."""
  pi = policy_probs(logits, legal_actions, alpha)            # (A,)
  log_probs_all = policy_log_probs(logits, legal_actions, alpha)    # (A,)
  sample_pi = policy_probs(sample_logits, legal_actions, alpha)
  sample_log_probs_all = policy_log_probs(sample_logits, legal_actions, alpha)
  magnet_pi = policy_probs(magnet_logits, legal_actions, alpha)
  magnet_log_probs_all = policy_log_probs(magnet_logits, legal_actions, alpha)

  # ── Policy gradient: all-actions mean-actor-critic ───────────────────────
  # NOTE: Tried to do the gradient update across all actions, it had much more variance and with smaller batches it was unstable.
  # With this approach it works and is comparable to V-value function.
  log_prob = log_probs_all[actions]
  sample_log_prob = sample_log_probs_all[actions]
  
  q_vec = target_q_values.at[actions].set(q_target)
  v_baseline = jnp.dot(q_vec, pi)
  advantage = q_target - v_baseline
  policy_loss = ppo_policy_loss(log_prob, sample_log_prob, advantage, clip_eps) 
  
  # ── Q-loss: live Q(s, a_taken) → Retrace target, PPO-clipped ─────────────
  q_taken = q_values[actions]
  sample_q_taken = sample_q_values[actions]
  q_loss = ppo_value_loss(q_taken, sample_q_taken, q_target, clip_eps)

  # ── Auxiliary terms (unchanged from MMD) ─────────────────────────────────
  magnet_loss = policy_kl(pi, log_probs_all, magnet_pi, magnet_log_probs_all, alpha)
  old_kl_loss = policy_kl(pi, log_probs_all, sample_pi, sample_log_probs_all, alpha)
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


def rnad_q_loss(
  q_values: jax.Array,        # (A,) — current Q(s, :)
  logits: jax.Array,          # (A,) — current policy logits
  legal_actions: jax.Array,   # (A,) — boolean action mask
  actions: jax.Array,         # ()   int — action taken during rollout
  sample_logits: jax.Array,   # (A,) — sampling policy logits (from rollout)
  sample_q_values: jax.Array, # (A,) — sampling Q(s,:) — for Q-loss clipping
  magnet_logits: jax.Array,   # (A,) — magnet (periodically-reset) policy
  q_target: jax.Array,        # ()   — Retrace(λ) target for Q(s, a_taken) (stop-grad)
  target_q_values: jax.Array, # (A,) — target-net Q(s,:) (stop-grad) — regrets
  clip_eps: float,
  vf_coef: float,
  ent_coef: float,
  magnet_coef: float,
  neurd_clip: float,
  neurd_threshold: float,
  alpha: float = 1.0,
) -> tuple[jax.Array, dict]:
  """RNaD-Q loss for a single (timestep, player) sample → scalar.

  Uses NeuRD for the policy gradient. Per-action regrets are built from the
  stop-grad target-net Q-vector (sampled action replaced by its Retrace
  target), with the magnet regularization subtracted per action:
      Q_reg(s, a) = q_vec(a) − magnet_coef · (log π(a) − log π_magnet(a))
      regret(s, a) = Q_reg(s, a) − E_π[Q_reg(s, ·)]
  so the baseline is centered over the same regularized vector. The live
  Q-head never enters the regret (it is trained only by the Q-loss).
  """
  log_probs_all = policy_log_probs(logits, legal_actions, alpha)    # (A,)
  strategy = policy_probs(logits, legal_actions, alpha)     # (A,)
  magnet_strategy = policy_probs(magnet_logits, legal_actions, alpha)
  magnet_log_probs_all = policy_log_probs(magnet_logits, legal_actions, alpha)

  # ── Per-action regularized Q-vector: target-net Q, sampled → Retrace ─────
  q_vec = target_q_values.at[actions].set(q_target)
  q_reg = q_vec - rnad_regularization(
    log_probs_all, magnet_log_probs_all, magnet_coef
  )

  # ── Regret = Q_reg(s,a) − E_π[Q_reg(s,·)] (centered over the same vector) ─
  # nan-safe baseline: α-entmax may zero a legal action, where the log-ratio
  # regularisation makes q_reg ±inf; masking by strategy>0 avoids inf·0 = nan.
  # Such per-action regrets are bounded by NeuRD's advantage clip downstream.
  baseline = jnp.sum(jnp.where(strategy > 0, q_reg * strategy, 0.0))
  regrets = q_reg - baseline

  # ── NeuRD policy loss ────────────────────────────────────────────────────
  policy_loss = -neurd_loss(logits, legal_actions, regrets, neurd_clip, neurd_threshold)

  # ── Q-loss: live Q(s, a_taken) → Retrace target, PPO-clipped ─────────────
  q_taken = q_values[actions]
  sample_q_taken = sample_q_values[actions]
  q_loss = ppo_value_loss(q_taken, sample_q_taken, q_target, clip_eps)

  # ── Entropy ───────────────────────────────────────────────────────────────
  entropy_loss = policy_entropy_loss(strategy, log_probs_all, alpha)

  # Magnet KL logged for monitoring; already embedded in regrets above.
  magnet_kl = policy_kl(strategy, log_probs_all, magnet_strategy, magnet_log_probs_all, alpha)

  total = policy_loss + vf_coef * q_loss + ent_coef * entropy_loss
  return total, {
    "policy_loss": policy_loss,
    "q_loss": q_loss,
    "entropy_loss": entropy_loss,
    "magnet_kl": magnet_kl,
  }
