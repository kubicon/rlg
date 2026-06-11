"""Natural Policy Gradient (NPG) loss.

Implements the policy-space update of alternating regularized natural policy
gradient from Kalogiannis & Farina (NeurIPS 2025), here in the *simultaneous*
(non-alternating) setting. The natural gradient never materialises a Fisher
matrix: for a softmax policy the Fisher-preconditioned step is exactly a
mirror-descent / multiplicative-weights update in policy space, so we compute
that closed-form target and fit the network to it by distillation.

Update target (per infoset s), with step size η and regularization temperature τ:

    log π̄(·|s) = (1 − ητ)·log π_t(·|s) + ητ·log π_ref(·|s) + η·Q(s,·) − logZ

  • π_t      — policy at the start of the step (the on-policy rollout policy).
               The (1 − ητ) self-power is the proximal/KL-to-self term that the
               Fisher metric induces; it is what makes this "natural".
  • π_ref    — moving reference policy (the magnet). This is the entropic
               bidilated regularizer: with π_ref uniform it reduces to plain
               entropy regularization (Thm 3.3); with a moving reference it is
               the empirical KL-to-moving-reference variant of the paper.
  • Q(s,·)   — all-action action-value vector from the Polyak target network,
               with the sampled action overwritten by its Retrace(λ) target
               (all stop-gradiented), exactly as in losses/mmd_q.py.

The policy loss is the projection the paper writes, π_{t+1} ≈ argmin_π KL(π ‖ π̄),
i.e. KL(π_θ ‖ stop_grad(π̄)) — the same KL direction the magnet term uses
elsewhere in this codebase. Entropy / proximal regularization is already baked
into π̄, so there is no separate entropy or old-policy KL term.

The live Q-head is supervised only by the Q-loss (PPO-clipped regression of
Q(s, a_taken) to its Retrace target) and never feeds gradients into the policy.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.lax as lax

from .ppo import ppo_value_loss, ppo_entropy_loss
from ..utils import safe_log_softmax, kl_divergence


def npg_loss(
  q_values: jax.Array,        # (A,) — current Q(s, :)
  logits: jax.Array,          # (A,) — current policy logits
  legal_actions: jax.Array,   # (A,) — boolean action mask
  actions: jax.Array,         # ()   int — action taken during rollout
  sample_logits: jax.Array,   # (A,) — rollout (== step-start) policy logits  → π_t
  sample_q_values: jax.Array, # (A,) — rollout Q(s,:) — for Q-loss clipping
  magnet_logits: jax.Array,   # (A,) — reference (magnet) policy logits        → π_ref
  q_target: jax.Array,        # ()   — Retrace(λ) target for Q(s, a_taken) (stop-grad)
  target_q_values: jax.Array, # (A,) — target-net Q(s,:) (stop-grad) — exponent
  clip_eps: float,
  vf_coef: float,
  eta: float,
  tau: float,
) -> tuple[jax.Array, dict]:
  """NPG loss for a single (timestep, player) sample → scalar."""
  log_pi = safe_log_softmax(logits, legal_actions)            # current network (moves over epochs)
  log_pi_t = safe_log_softmax(sample_logits, legal_actions)   # π_t   (step-start, on-policy)
  log_ref = safe_log_softmax(magnet_logits, legal_actions)    # π_ref (magnet)

  # All-action Q-vector: target-net Q with sampled action → its Retrace target.
  q_vec = target_q_values.at[actions].set(q_target)

  # Closed-form natural-gradient / MWU target in policy space. safe_log_softmax
  # re-normalises (the −logZ) over legal actions; illegal entries never leak
  # because they are masked out of the normalisation and zeroed afterwards.
  raw_target = (1.0 - eta * tau) * log_pi_t + eta * tau * log_ref + eta * q_vec
  log_target = lax.stop_gradient(safe_log_softmax(raw_target, legal_actions))

  # Distillation onto the target: π_{t+1} ≈ argmin_π KL(π ‖ π̄).
  policy_loss = kl_divergence(log_pi, log_target)

  # Q-loss: live Q(s, a_taken) → Retrace target, PPO-clipped.
  q_taken = q_values[actions]
  sample_q_taken = sample_q_values[actions]
  q_loss = ppo_value_loss(q_taken, sample_q_taken, q_target, clip_eps)

  total = policy_loss + vf_coef * q_loss

  pi = jax.nn.softmax(logits, where=legal_actions)
  return total, {
    "policy_loss": policy_loss,
    "q_loss": q_loss,
    "target_kl": policy_loss,                       # KL(π_θ ‖ π̄), distillation gap
    "step_kl": kl_divergence(log_pi, log_pi_t),     # KL(π_θ ‖ π_t), how far we moved
    "ref_kl": kl_divergence(log_pi, log_ref),       # KL(π_θ ‖ π_ref), regularizer pull
    "entropy": -ppo_entropy_loss(log_pi, pi),       # policy entropy (monitoring)
  }
