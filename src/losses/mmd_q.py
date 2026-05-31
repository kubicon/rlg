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

from .ppo import ppo_policy_loss, ppo_value_loss, ppo_entropy_loss
from .neurd import neurd_loss
from .rnad import estimate_baseline_regrets, rnad_regularization
from ..utils import safe_log_softmax, kl_divergence


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
) -> tuple[jax.Array, dict]:
  """MMD-Q loss for a single (timestep, player) sample → scalar."""
  pi = jax.nn.softmax(logits, where=legal_actions)           # (A,)
  log_probs_all = safe_log_softmax(logits, legal_actions)    # (A,)
  sample_log_probs_all = safe_log_softmax(sample_logits, legal_actions)
  magnet_log_probs_all = safe_log_softmax(magnet_logits, legal_actions)

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
  magnet_loss = kl_divergence(log_probs_all, magnet_log_probs_all)
  old_kl_loss = kl_divergence(log_probs_all, sample_log_probs_all)
  entropy_loss = ppo_entropy_loss(log_probs_all, pi)

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
  log_probs_all = safe_log_softmax(logits, legal_actions)    # (A,)
  strategy = jax.nn.softmax(logits, where=legal_actions)     # (A,)
  magnet_log_probs_all = safe_log_softmax(magnet_logits, legal_actions)

  # ── Per-action regularized Q-vector: target-net Q, sampled → Retrace ─────
  q_vec = target_q_values.at[actions].set(q_target)
  q_reg = q_vec - rnad_regularization(
    log_probs_all, magnet_log_probs_all, magnet_coef
  )

  # ── Regret = Q_reg(s,a) − E_π[Q_reg(s,·)] (centered over the same vector) ─
  regrets = q_reg - jnp.dot(q_reg, strategy)

  # ── NeuRD policy loss ────────────────────────────────────────────────────
  policy_loss = -neurd_loss(logits, legal_actions, regrets, neurd_clip, neurd_threshold)

  # ── Q-loss: live Q(s, a_taken) → Retrace target, PPO-clipped ─────────────
  q_taken = q_values[actions]
  sample_q_taken = sample_q_values[actions]
  q_loss = ppo_value_loss(q_taken, sample_q_taken, q_target, clip_eps)

  # ── Entropy ───────────────────────────────────────────────────────────────
  entropy_loss = ppo_entropy_loss(log_probs_all, strategy)

  # Magnet KL logged for monitoring; already embedded in regrets above.
  magnet_kl = kl_divergence(log_probs_all, magnet_log_probs_all)

  total = policy_loss + vf_coef * q_loss + ent_coef * entropy_loss
  return total, {
    "policy_loss": policy_loss,
    "q_loss": q_loss,
    "entropy_loss": entropy_loss,
    "magnet_kl": magnet_kl,
  }
