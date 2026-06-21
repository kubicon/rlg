"""Gumbel-Q MMD loss: pathwise policy gradient via a Gumbel-Softmax sample.

Mirror of ``losses/mmd_q.py``'s ``mmd_q_loss`` that swaps ONLY the policy
term. Everything that gives MMD its convergence — the closed-form magnet KL,
the entropy bonus, and the PPO-clipped Q-loss — is reused unchanged.

Original (score-function) policy term in ``mmd_q_loss``:

    q_vec      = target_q_values.at[a].set(q_target)      # (A,) stop-grad
    advantage  = q_target - E_pi[q_vec]
    policy_loss = ppo_policy_loss(log pi(a), log mu(a), advantage, clip_eps)

Here we instead maximize the expected Q under a reparameterized policy sample:

    y           = gumbel_softmax(logits, legal, key, tau)  # differentiable in logits
    policy_loss = - <stop_grad(q_vec), y>

The gradient flows pathwise through ``y`` into ``logits`` rather than through
the log-prob ratio. No baseline is needed: subtracting a constant from
``q_vec`` is gradient-invariant since the entries of ``y`` sum to 1. There is
no importance ratio, so PPO clipping is dropped on the policy term — the
across-epoch trust region is the (unchanged) ``old_policy_coef`` KL.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.lax as lax

from .ppo import ppo_value_loss, ppo_entropy_loss
from ..utils import safe_log_softmax, kl_divergence
from ..gumbel import gumbel_softmax


def gumbel_q_loss(
  q_values: jax.Array,        # (A,) — current Q(s, :)
  logits: jax.Array,          # (A,) — current policy logits
  legal_actions: jax.Array,   # (A,) — boolean action mask
  actions: jax.Array,         # ()   int — action taken during rollout
  sample_logits: jax.Array,   # (A,) — sampling policy logits (from rollout)
  sample_q_values: jax.Array, # (A,) — sampling Q(s,:) — for Q-loss clipping
  magnet_logits: jax.Array,   # (A,) — magnet (periodically-reset) policy
  q_target: jax.Array,        # ()   — Retrace(λ) target for Q(s, a_taken) (stop-grad)
  target_q_values: jax.Array, # (A,) — target-net Q(s,:) (stop-grad) — advantage
  key: jax.Array,             # PRNG key for the Gumbel-Softmax sample
  clip_eps: float,
  vf_coef: float,
  ent_coef: float,
  magnet_coef: float,
  old_policy_coef: float,
  tau: float,                 # Gumbel-Softmax temperature
) -> tuple[jax.Array, dict]:
  """Gumbel-Q MMD loss for a single (timestep, player) sample → scalar."""
  pi = jax.nn.softmax(logits, where=legal_actions)           # (A,)
  log_probs_all = safe_log_softmax(logits, legal_actions)    # (A,)
  sample_log_probs_all = safe_log_softmax(sample_logits, legal_actions)
  magnet_log_probs_all = safe_log_softmax(magnet_logits, legal_actions)

  # ── Pathwise policy gradient through a Gumbel-Softmax sample ──────────────
  # Same target-net Q-vector as mmd_q_loss (sampled action → Retrace target),
  # fully stop-gradiented. Maximize <q_vec, y> ⇒ minimize -<q_vec, y>.
  q_vec = lax.stop_gradient(target_q_values.at[actions].set(q_target))  # (A,)
  y = gumbel_softmax(logits, legal_actions, key, tau)                   # (A,)
  policy_loss = -jnp.dot(q_vec, y)

  # ── Q-loss: live Q(s, a_taken) → Retrace target, PPO-clipped (reused) ────
  q_taken = q_values[actions]
  sample_q_taken = sample_q_values[actions]
  q_loss = ppo_value_loss(q_taken, sample_q_taken, q_target, clip_eps)

  # ── Auxiliary terms (identical to mmd_q_loss) ────────────────────────────
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
