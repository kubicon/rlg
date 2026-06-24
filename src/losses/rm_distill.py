"""Regret-matching *distillation* loss — a softmax-compatible RM training signal.

This is the complement to the regret-matching policy *parametrization* used by
``loss_type=rm`` (see ``policy.py``'s RM transform). Here the actor stays a
softmax / α-entmax policy and regret matching enters only as a **target**:

    π_RM(a) ∝ [π(a) + η·regret(a)]₊      (η = rm_step_size),

a single regret-matching⁺ step taken *from the current policy*, and the policy is
pulled toward it by the cross-entropy H(π_RM, π) = −Σ π_RM·log π. Because π_RM is a
stop-gradient target this has the same minimiser as the forward KL(π_RM ‖ π) but
avoids ``log π_RM`` on the (exact) zeros that RM produces.

The step is *incremental* on purpose. The absolute RM map [regret]₊/Σ[regret]₊ has
no fixed point at the equilibrium: there the regrets vanish, so the target degrades
to uniform and the cross-entropy keeps dragging the policy toward uniform (the
magnet cannot overcome an O(1) pull). The incremental target equals π when the
regrets are 0, so the distillation term vanishes at the fixed point and the magnet
holds the strategy — mirroring how mmd_loss's PPO advantage term goes to 0.

Why have both:
  * ``rm``         — the actor *is* the regret-matching map [z]₊/Σ[z]₊. Literal RM,
                     but the map is flat where z ≤ 0 (dead-regret gradients).
  * ``rm_distill`` — a softmax/entmax actor *driven* by an RM target. Less literally
                     RM, but full-support smooth gradients (no dead units).

As in ``mmd_loss``, last-iterate convergence comes from the **weight-space** KL
regularisation: the ``magnet_coef`` magnet term plus the ``old_policy_coef``
proximal term. The regrets are the *instantaneous* per-step regrets (no
cumulative-regret network). The signature matches ``mmd_loss`` so the trainer
reuses the same vmap axes.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.lax as lax

from .ppo import ppo_value_loss
from .rnad import estimate_baseline_regrets
from ..policy import policy_probs, policy_log_probs, policy_entropy_loss, policy_kl


def rm_distill_loss(
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
  rm_step_size: float = 1.0,
  alpha: float = 1.0,
) -> tuple[jax.Array, dict]:
  strategy = policy_probs(logits, legal_actions, alpha)  # (A,)
  log_probs_all = policy_log_probs(logits, legal_actions, alpha)  # (A,)
  sample_strategy = policy_probs(sample_logits, legal_actions, alpha)  # (A,)
  sample_log_probs_all = policy_log_probs(sample_logits, legal_actions, alpha)  # (A,)
  magnet_strategy = policy_probs(magnet_logits, legal_actions, alpha)  # (A,)
  magnet_log_probs_all = policy_log_probs(magnet_logits, legal_actions, alpha)  # (A,)

  # Per-action regrets via the baseline + (U−baseline)/π control variate. The
  # whole regret vector — and hence the RM target — is a fixed target: stop the
  # gradient so it does not back-propagate into the policy being trained.
  value_vec = jnp.broadcast_to(values, legal_actions.shape)  # (A,) constant baseline
  regrets = estimate_baseline_regrets(
    value_vec, advantages, lax.stop_gradient(strategy), sample_strategy, actions
  )
  regrets = lax.stop_gradient(regrets)

  # *Incremental* RM⁺ target: a single regret-matching step taken FROM the current
  # policy, π_RM ∝ [π + η·regret]₊ (η = rm_step_size), rather than the absolute RM
  # map [regret]₊/Σ[regret]₊. This is essential for convergence: with the absolute
  # map, at (or near) the equilibrium all regrets → 0, so the target collapses to
  # *uniform* and the cross-entropy term keeps applying an O(1) pull toward uniform
  # that the magnet cannot overcome — the policy never settles on a non-uniform
  # equilibrium. The incremental target is the *identity at the fixed point*: when
  # regret = 0 it returns π, so H(π_RM, π) and its gradient (π − sg π) vanish and
  # the magnet/proximal terms hold the strategy, exactly as the PPO advantage term
  # does in mmd_loss. Positive regrets nudge mass toward those actions; η scales
  # the step. Falls back to uniform-over-legal only in the degenerate case where
  # the whole shifted vector is ≤ 0.
  shifted = lax.stop_gradient(strategy) + rm_step_size * regrets
  pos = jnp.where(legal_actions, jnp.maximum(shifted, 0.0), 0.0)
  z = pos.sum()
  n_legal = jnp.maximum(legal_actions.sum(), 1.0)
  uniform = legal_actions / n_legal
  rm_target = jnp.where(z > 0.0, pos / jnp.where(z > 0.0, z, 1.0), uniform)

  # Cross-entropy H(π_RM, π) = −Σ π_RM log π — the only term that drives the
  # policy toward the regret-matching target. Restricted to π's support: for
  # softmax every legal action has π > 0 so this is the full CE; for sparse
  # α-entmax π can be 0 where π_RM > 0 (log π = −∞), so those terms are dropped
  # (their entmax gradient is 0 anyway) and the magnet/entropy terms carry the
  # out-of-support pressure instead.
  ce_mask = (rm_target > 0.0) & (strategy > 0.0)
  policy_loss = -jnp.sum(jnp.where(ce_mask, rm_target * log_probs_all, 0.0))

  # Weight-space KL regularization (the last-iterate mechanism, as in MMD).
  magnet_loss = policy_kl(strategy, log_probs_all, magnet_strategy, magnet_log_probs_all, alpha)
  old_kl_loss = policy_kl(strategy, log_probs_all, sample_strategy, sample_log_probs_all, alpha)

  value_loss = ppo_value_loss(values, sample_values, returns, clip_eps)
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
