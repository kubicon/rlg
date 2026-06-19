"""Regret-Matching R-NaD loss.

A regret-matching (RM / NeuRD) variant of :func:`rnad_loss`. The network output
is reinterpreted as per-action CUMULATIVE REGRETS R; the strategy is the RM
projection ``relu(R) / Σ relu(R)`` (uniform over legal actions when every regret
is ≤ 0); and the policy update is a squared-error NeuRD surrogate whose gradient
w.r.t. R equals minus the instantaneous regret -- so a (momentum-free) SGD step
literally accumulates regret in the weights.

v-trace (advantages/returns), the magnet/value-space regularization, and the
baseline-corrected regret estimator are shared with the softmax ``rnad`` path
(:mod:`src.losses.rnad`); only the regret→strategy map, the exploration floor,
and the policy backward differ. Kept in a separate module so the original
softmax implementation is untouched.

All functions operate on a single (timestep, player) sample: scalars and
``(A,)``-shaped arrays. Callers vmap over B, T, P.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .ppo import ppo_value_loss
from .rnad import estimate_baseline_regrets, rnad_regularization


# RM produces exact zeros for dominated actions; clip before taking logs so the
# magnet/entropy terms stay finite (softmax never hits zero, hence safe there).
_LOG_FLOOR = 1e-9


def rm_policy(
  logits: jax.Array, legal_actions: jax.Array, eps: float = 1e-12
) -> jax.Array:
  """Regret matching: ``s_a = relu(R_a)·mask / Σ_b relu(R_b)``.

  ``logits`` carries the per-action cumulative regret R. Falls back to uniform
  over legal actions when every regret is ≤ 0. Trailing axis is actions.
  """
  regrets = jnp.where(legal_actions > 0, logits, 0.0)
  pos = jax.nn.relu(regrets) * legal_actions
  denom = pos.sum(-1, keepdims=True)
  n_legal = jnp.clip(legal_actions.sum(-1, keepdims=True), eps, None)
  uniform = legal_actions / n_legal
  return jnp.where(denom > 0, pos / jnp.clip(denom, eps, None), uniform)


def rm_behavior(
  logits: jax.Array,
  legal_actions: jax.Array,
  epsilon: float,
  eps: float = 1e-12,
) -> jax.Array:
  """ε-on-policy sampling distribution ``μ = (1-ε)·s + ε·uniform``.

  The floor keeps ``μ(a) > 0`` on every legal action so the ``1/μ(a*)``
  importance correction stays bounded -- RM sends dominated actions to exactly
  0, which would make a pure on-policy ``1/μ`` blow up. ``epsilon`` tunes how
  much exploration is mixed in (``epsilon=0`` → pure RM on-policy sampling).
  """
  s = rm_policy(logits, legal_actions, eps)
  n_legal = jnp.clip(legal_actions.sum(-1, keepdims=True), eps, None)
  uniform = legal_actions / n_legal
  return (1.0 - epsilon) * s + epsilon * uniform


def _masked_log(p: jax.Array, legal_actions: jax.Array) -> jax.Array:
  """log(p) clipped away from -inf, zeroed on illegal actions."""
  return jnp.where(legal_actions > 0, jnp.log(jnp.clip(p, _LOG_FLOOR, 1.0)), 0.0)


def rm_entropy_loss(strategy: jax.Array, legal_actions: jax.Array) -> jax.Array:
  """Negative entropy of the RM strategy (scalar). Minimising maximises entropy."""
  log_s = _masked_log(strategy, legal_actions)
  entropy = -(strategy * log_s).sum(-1)
  return -entropy


def rm_neurd_surrogate(
  logits: jax.Array,
  legal_actions: jax.Array,
  regrets: jax.Array,
  clip: float,
) -> jax.Array:
  """Squared-error NeuRD surrogate (Hennes et al. 2020, eq. form from the RM kit).

      L = ½ Σ_a ( R_a − stopgrad(R_a + ρ_a) )²   ⇒   dL/dR_a = −ρ_a

  so an SGD step does ``R += lr · ρ`` == accumulate the (clipped) instantaneous
  regret ρ in the weights. The squared form keeps the loss magnitude ~ regret,
  which is gentler for adaptive optimisers than the bare linear NeuRD term.
  """
  rho = jnp.clip(regrets, -clip, clip) * legal_actions
  target = jax.lax.stop_gradient(logits + rho)
  se = (logits - target) ** 2 * legal_actions
  return 0.5 * se.sum(-1)


def rm_rnad_loss(
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
  neurd_clip: float,
  epsilon: float,
) -> tuple[jax.Array, dict]:
  """RM analogue of :func:`rnad_loss` for a single (timestep, player) sample."""
  strategy = rm_policy(logits, legal_actions)
  magnet_strategy = rm_policy(magnet_logits, legal_actions)
  # Behaviour policy that generated the action: the SAME ε-floor used at rollout,
  # recomputed from the stored sampling regrets, so 1/μ(a*) matches what was
  # actually sampled.
  sampling_strategy = rm_behavior(sample_logits, legal_actions, epsilon)

  log_s = _masked_log(strategy, legal_actions)
  log_m = _masked_log(magnet_strategy, legal_actions)

  # Value-space (R-NaD) magnet: subtract the per-action KL-to-magnet cost from
  # the per-action baseline before forming regrets.
  regularized_value = values - rnad_regularization(log_s, log_m, magnet_coef)

  regrets = estimate_baseline_regrets(
    regularized_value, advantages, strategy, sampling_strategy, actions
  )

  policy_loss = rm_neurd_surrogate(logits, legal_actions, regrets, neurd_clip)
  value_loss = ppo_value_loss(values, sample_values, returns, clip_eps)
  entropy_loss = rm_entropy_loss(strategy, legal_actions)

  loss = policy_loss + vf_coef * value_loss + ent_coef * entropy_loss
  metrics = {
    "policy_loss": policy_loss,
    "value_loss": value_loss,
    "entropy_loss": entropy_loss,
  }
  return loss, metrics
