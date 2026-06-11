"""NPG — Natural Policy Gradient (simultaneous, regularized).

Implements the natural-policy-gradient method of Kalogiannis & Farina
(NeurIPS 2025) in the *simultaneous* setting (both players updated every step;
no alternation). Reuses QMMD's Q-value infrastructure unchanged:

  - twin_head agent (CategoricalHead logits + QHead), no V-head;
  - Retrace(λ) Q-targets (``_compute_q_targets``);
  - Polyak target network (``target_params``) for stable Q-values;
  - moving reference policy (``magnet_params``) = the bidilated regularizer.

The only thing that differs from QMMD is the policy update: instead of a
PPO-clipped surrogate, the policy is fit to the closed-form natural-gradient /
mirror-descent target (see losses/npg.py):

    log π̄(·|s) = (1 − ητ)·log π_t(·|s) + ητ·log π_ref(·|s) + η·Q(s,·) − logZ
    loss       = KL(π_θ ‖ stop_grad(π̄))

Value-space (dilated) regularization
-------------------------------------
The paper regularizes in *value* space: the regularized value (Sec. 2.2)
sums τ·ψ(π(·|h)) over every history h in the subtree, so the cost charged at
infoset s accounts for the regularization of all *subsequent* infosets, not
just s itself. With ``regularize_value=True`` (default) we honour this by
computing the Retrace bootstrap from the *soft* value

    V_τ(s) = E_μ[Q(s,·)] − τ·KL(μ(·|s) ‖ π_ref(·|s)),

so the Retrace return — and therefore the Q-head trained on it — carries the
downstream regularization, and the all-action Q-vector fed into the exponent is
the genuine soft Q_τ. The ``log π_ref`` / ``(1 − ητ)`` terms in the policy
update cover only node s, so there is no double counting. With
``regularize_value=False`` the bootstrap is the raw value and the regularizer
acts only per-step (the myopic policy-space variant).

No ε-truncation, no alternation, no parameter schedules — by design.

Requires a Q-head agent (same as QMMD). mmd.py and mmd_q.py are untouched.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.lax as lax
import optax

from .base import TrainingState
from .episode import collect_episodes
from .mmd_q import QMMD
from .types import LossType, MagnetUpdateType
from ..advantage import retrace
from ..agents.base import Agent
from ..envs.base import Env
from ..losses.npg import npg_loss
from ..utils import safe_log_softmax


class NPG(QMMD):
  """Regularized natural policy gradient with simultaneous updates.

  Args:
      env, agent, n_epochs, batch_size, lr,
      clip_eps, gamma, gae_lambda:  as in QMMD (clip_eps clips the Q-loss only).
      vf_coef:               Weight of the Q-value (Retrace regression) loss.
      eta:                   Policy-space NPG step size η.
      tau:                   Regularization temperature τ. The policy self-decays
                             as π_t^(1−ητ) and is pulled toward the reference as
                             π_ref^(ητ); 0 ≤ ητ ≤ 1 keeps the power non-negative.
      target_update_rate:    Polyak τ for the Q target network.
      magnet_interval:       Steps between hard resets of the reference policy
                             (magnet_update_type=periodic).
      magnet_update_rate:    Polyak τ for the reference policy
                             (magnet_update_type=incremental).
      magnet_update_type:    "periodic" or "incremental" reference-policy update.
      regularize_value:      If True (default), bootstrap Retrace from the soft
                             value so Q_τ carries downstream regularization
                             (the paper's dilated/value-space regularizer). If
                             False, use the raw value (myopic per-step variant).
      optimizer, grad_clip:  as in QMMD.
  """

  def __init__(
    self,
    env: Env,
    agent: Agent,
    n_epochs: int = 4,
    batch_size: int = 4,
    lr: float = 3e-4,
    clip_eps: float = 0.2,
    vf_coef: float = 0.5,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    eta: float = 0.5,
    tau: float = 0.1,
    target_update_rate: float = 0.001,
    magnet_interval: int = 2000,
    magnet_update_rate: float = 0.001,
    magnet_update_type: MagnetUpdateType = MagnetUpdateType.PERIODIC,
    regularize_value: bool = True,
    optimizer: optax.GradientTransformation | None = None,
    grad_clip: float | None = None,
  ) -> None:
    super().__init__(
      env,
      agent,
      n_epochs=n_epochs,
      batch_size=batch_size,
      lr=lr,
      clip_eps=clip_eps,
      vf_coef=vf_coef,
      gamma=gamma,
      gae_lambda=gae_lambda,
      target_update_rate=target_update_rate,
      magnet_interval=magnet_interval,
      magnet_update_rate=magnet_update_rate,
      magnet_update_type=magnet_update_type,
      loss_type=LossType.MMD,   # unused: NPG.step dispatches npg_loss directly
      optimizer=optimizer,
      grad_clip=grad_clip,
      alternating=False,        # NPG here is simultaneous by design
      schedules=None,           # no schedules by design
    )
    self.eta = eta
    self.tau = tau
    self.regularize_value = regularize_value

  # init() is inherited from QMMD: extras = {target_params, magnet_params}.

  # ── Soft Q-target computation (value-space regularization) ─────────────────

  def _compute_soft_q_targets(
    self,
    rewards: jax.Array,          # (B, T, P)
    target_q_values: jax.Array,  # (B, T, P, A) — Polyak target-net Q
    sample_logits: jax.Array,    # (B, T, P, A) — rollout (on-policy) policy μ
    magnet_logits: jax.Array,    # (B, T, P, A) — reference policy π_ref
    actions: jax.Array,          # (B, T, P)
    legal_actions: jax.Array,    # (B, T, P, A)
    dones: jax.Array,            # (B, T)
    tau: float,
  ) -> jax.Array:                # (B, T, P) — soft Q targets
    """Retrace(λ) Q-targets bootstrapped from the *soft* value.

    Identical to QMMD._compute_q_targets except the bootstrap value subtracts
    the per-node regularization, V_τ(s) = E_μ[Q(s,·)] − τ·KL(μ(·|s) ‖ π_ref(·|s)).
    Because Retrace recursively bootstraps V_τ, this propagates the τ·KL cost of
    every downstream infoset into the return — the dilated/value-space
    regularizer (paper Sec. 2.2). The node's *own* KL is not included here; it is
    handled by the policy update's log π_ref / (1 − ητ) terms.
    """
    mu = jax.nn.softmax(sample_logits, where=legal_actions)        # (B,T,P,A)
    log_mu = safe_log_softmax(sample_logits, legal_actions)        # (B,T,P,A)
    log_ref = safe_log_softmax(magnet_logits, legal_actions)       # (B,T,P,A)
    kl = jnp.sum(mu * (log_mu - log_ref), axis=-1)                 # (B,T,P) — KL(μ‖ref)

    # Soft bootstrap value: V_τ(s) = E_μ[Q_target(s,·)] − τ·KL(μ(·|s)‖ref).
    v_target = (mu * target_q_values).sum(-1) - tau * kl           # (B,T,P)

    q_taken = jnp.take_along_axis(
      target_q_values, actions[..., None], axis=-1
    ).squeeze(-1)                                                  # (B,T,P)

    discount = (1.0 - dones) * self.gamma                          # (B,T)

    _retrace = lambda r, q, v, d: retrace(r, q, v, d, lambda_=self.gae_lambda)
    _retrace_P = jax.vmap(_retrace, in_axes=(1, 1, 1, None), out_axes=1)
    _retrace_BP = jax.vmap(_retrace_P, in_axes=(0, 0, 0, 0), out_axes=0)
    return _retrace_BP(rewards, q_taken, v_target, discount)       # (B,T,P)

  # ── Public step ───────────────────────────────────────────────────────────

  def step(self, state: TrainingState) -> tuple[TrainingState, dict[str, jax.Array]]:
    rng, collect_key = jax.random.split(state.rng)
    _, _, _, episodes = collect_episodes(
      self.env, self.agent, state.params, collect_key, self.batch_size
    )

    target_out = self._eval_params(state.extras["target_params"], episodes)

    # Reference (magnet) logits — the bidilated regularizer.
    magnet_out = self._eval_params(state.extras["magnet_params"], episodes)
    magnet_logits = lax.stop_gradient(magnet_out.logits)  # (B,T,P,A)

    # ── Retrace Q-targets ──────────────────────────────────────────────────
    # With regularize_value, the bootstrap is the *soft* value, so the Retrace
    # return (and the Q-head it trains) carries the regularization of every
    # downstream infoset — the paper's dilated/value-space regularizer.
    if self.regularize_value:
      q_targets = lax.stop_gradient(
        self._compute_soft_q_targets(
          episodes.rewards,
          target_out.q_values,           # (B,T,P,A) — target-net Q (stability)
          episodes.agent_output.logits,  # (B,T,P,A) — rollout (on-policy) μ
          magnet_logits,                 # (B,T,P,A) — π_ref
          episodes.actions,              # (B,T,P)
          episodes.legal_actions,        # (B,T,P,A)
          episodes.dones,                # (B,T)
          self.tau,
        )
      )  # (B,T,P)
    else:
      q_targets = lax.stop_gradient(
        self._compute_q_targets(
          episodes.rewards,
          target_out.q_values,
          episodes.agent_output.logits,
          episodes.actions,
          episodes.legal_actions,
          episodes.dones,
        )
      )  # (B,T,P)

    # All-action target-net Q-vector for the NPG exponent (stop-grad). Once the
    # Q-head is trained on the (soft) targets above, this is the soft Q_τ(s,·).
    target_q_values = lax.stop_gradient(target_out.q_values)  # (B,T,P,A)

    params, opt_state = state.params, state.opt_state
    valid = self._valid_mask(episodes.dones)

    def epoch_fn(carry, _):
      params, opt_state = carry

      def total_loss(params):
        agent_out = self._eval_params(params, episodes)

        # 9 arrays + 4 scalars
        _axes = (0, 0, 0, 0, 0, 0, 0, 0, 0, None, None, None, None)
        loss_P = jax.vmap(npg_loss, in_axes=_axes)
        loss_TP = jax.vmap(loss_P, in_axes=_axes)
        loss_BTP = jax.vmap(loss_TP, in_axes=_axes)
        losses, metrics = loss_BTP(
          agent_out.q_values,                 # (B,T,P,A)
          agent_out.logits,                   # (B,T,P,A)
          episodes.legal_actions,             # (B,T,P,A)
          episodes.actions,                   # (B,T,P)
          episodes.agent_output.logits,       # (B,T,P,A) — π_t (sampling)
          episodes.agent_output.q_values,     # (B,T,P,A) — sampling Q
          magnet_logits,                      # (B,T,P,A) — π_ref
          q_targets,                          # (B,T,P)
          target_q_values,                    # (B,T,P,A)
          self.clip_eps,
          self.vf_coef,
          self.eta,
          self.tau,
        )
        wmean = lambda x: self._wmean(x, valid)
        return wmean(losses), jax.tree.map(wmean, metrics)  # type: ignore

      (_, metrics), grads = jax.value_and_grad(total_loss, has_aux=True)(params)
      updates, new_opt_state = self.optimizer.update(grads, opt_state, params)
      return (optax.apply_updates(params, updates), new_opt_state), metrics

    (params, opt_state), epoch_metrics = jax.lax.scan(
      epoch_fn, (params, opt_state), None, length=self.n_epochs
    )

    # Polyak update: target ← τ·params + (1−τ)·target
    target_params = optax.incremental_update(
      params, state.extras["target_params"], self.target_update_rate
    )
    if self.magnet_update_type == MagnetUpdateType.INCREMENTAL:
      magnet_params = optax.incremental_update(
        params, state.extras["magnet_params"], self.magnet_update_rate
      )
    else:
      magnet_params = optax.periodic_update(
        params, state.extras["magnet_params"], state.step, self.magnet_interval
      )

    return TrainingState(
      params=params,
      opt_state=opt_state,
      env_state=state.env_state,
      agent_state=state.agent_state,
      rng=rng,
      step=state.step + 1,
      extras={"target_params": target_params, "magnet_params": magnet_params},
    ), jax.tree.map(jnp.mean, epoch_metrics)
