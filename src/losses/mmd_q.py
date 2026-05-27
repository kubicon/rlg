"""MMD-Q loss: policy head + Q-head, no V-head.

The policy gradient never reads Q(s, ·) for non-sampled actions (those
entries receive no training signal — Retrace only supervises the played
action). Instead it uses two stop-gradiented quantities computed in the
algorithm before the epoch loop:

    q_target   — Retrace(λ) target for Q(s, a_taken)        (utility)
    v_baseline — E_{π_target}[Q_target(s, ·)] from the      (baseline)
                 Polyak target network

The advantage at the sampled action is A(s, a) = q_target − v_baseline.
For NeuRD (rnad_q_loss) the full per-action regret vector is reconstructed
from this single sampled-action advantage via the all-actions baseline
trick (Srinivasan et al. 2018, https://arxiv.org/abs/1809.03057), exactly
as the V-based RNaD loss does.

The Q-head is supervised only via the Q-loss (Q(s, a_taken) → q_target).
No separate V-head or V-loss exists.
"""

from __future__ import annotations

import jax

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
  q_target: jax.Array,        # ()   — Retrace(λ) target for Q(s, a_taken)
  v_baseline: jax.Array,      # ()   — E_{π_target}[Q_target(s,:)] (stop-grad)
  clip_eps: float,
  qf_coef: float,
  ent_coef: float,
  magnet_coef: float,
  old_policy_coef: float,
) -> tuple[jax.Array, dict]:
  """MMD-Q loss for a single (timestep, player) sample → scalar."""
  pi = jax.nn.softmax(logits, where=legal_actions)           # (A,)
  log_probs_all = safe_log_softmax(logits, legal_actions)    # (A,)
  sample_log_probs_all = safe_log_softmax(sample_logits, legal_actions)
  magnet_log_probs_all = safe_log_softmax(magnet_logits, legal_actions)

  # ── Policy gradient (PPO-clipped, IS ratio vs. sampling policy) ──────────
  # Advantage uses the Retrace target at the sampled action and a target-net
  # baseline — both stop-gradiented — so it never reads untrained Q-values.
  log_prob = log_probs_all[actions]
  sample_log_prob = sample_log_probs_all[actions]
  advantage = q_target - v_baseline
  policy_loss = ppo_policy_loss(log_prob, sample_log_prob, advantage, clip_eps)

  # ── Q-loss: Q(s, a_taken) → Retrace target, PPO-clipped ──────────────────
  q_taken = q_values[actions]
  sample_q_taken = sample_q_values[actions]
  q_loss = ppo_value_loss(q_taken, sample_q_taken, q_target, clip_eps)

  # ── Auxiliary terms (unchanged from MMD) ─────────────────────────────────
  magnet_loss = kl_divergence(log_probs_all, magnet_log_probs_all)
  old_kl_loss = kl_divergence(log_probs_all, sample_log_probs_all)
  entropy_loss = ppo_entropy_loss(log_probs_all, pi)

  total = (
    policy_loss
    + qf_coef * q_loss
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
  q_target: jax.Array,        # ()   — Retrace(λ) target for Q(s, a_taken)
  v_baseline: jax.Array,      # ()   — E_{π_target}[Q_target(s,:)] (stop-grad)
  clip_eps: float,
  qf_coef: float,
  ent_coef: float,
  magnet_coef: float,
  neurd_clip: float,
  neurd_threshold: float,
) -> tuple[jax.Array, dict]:
  """RNaD-Q loss for a single (timestep, player) sample → scalar.

  Uses NeuRD for the policy gradient. The magnet regularization is applied
  per-action to a regularized state baseline (as in the V-based RNaD loss):
      V_reg(s, a) = v_baseline(s) − magnet_coef · (log π(a) − log π_magnet(a))

  Per-action regrets are reconstructed from the single sampled-action
  advantage (Retrace target − baseline) via the all-actions baseline trick,
  so the regret vector never depends on untrained Q-values for non-sampled
  actions.
  """
  log_probs_all = safe_log_softmax(logits, legal_actions)    # (A,)
  strategy = jax.nn.softmax(logits, where=legal_actions)     # (A,)
  magnet_log_probs_all = safe_log_softmax(magnet_logits, legal_actions)
  sampling_strategy = jax.nn.softmax(sample_logits, where=legal_actions)

  # ── Regularised per-action baseline: subtract magnet KL per action ───────
  regularized_value = v_baseline - rnad_regularization(
    log_probs_all, magnet_log_probs_all, magnet_coef
  )

  # ── Sampled-action advantage: Retrace target − target-net baseline ───────
  advantage = q_target - v_baseline

  # ── Per-action regrets via the all-actions baseline trick ────────────────
  regrets = estimate_baseline_regrets(
    regularized_value, advantage, strategy, sampling_strategy, actions
  )

  # ── NeuRD policy loss ────────────────────────────────────────────────────
  policy_loss = -neurd_loss(logits, legal_actions, regrets, neurd_clip, neurd_threshold)

  # ── Q-loss: Q(s, a_taken) → Retrace target, PPO-clipped ──────────────────
  q_taken = q_values[actions]
  sample_q_taken = sample_q_values[actions]
  q_loss = ppo_value_loss(q_taken, sample_q_taken, q_target, clip_eps)

  # ── Entropy ───────────────────────────────────────────────────────────────
  entropy_loss = ppo_entropy_loss(log_probs_all, strategy)

  # Magnet KL logged for monitoring; already embedded in regrets above.
  magnet_kl = kl_divergence(log_probs_all, magnet_log_probs_all)

  total = policy_loss + qf_coef * q_loss + ent_coef * entropy_loss
  return total, {
    "policy_loss": policy_loss,
    "q_loss": q_loss,
    "entropy_loss": entropy_loss,
    "magnet_kl": magnet_kl,
  }
