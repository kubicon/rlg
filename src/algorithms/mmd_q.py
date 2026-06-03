"""QMMD — MMD variant with an explicit Q-value function and no V-head.

Architecture differences from MMD:
  - Network is a twin_head: CategoricalHead (logits) + QHead (Q-values).
  - No V-head. State value is derived on-the-fly as V_π(s) = E_π[Q(s,:)].

Q-target computation:
  - Uses Retrace(λ) (Munos et al., 2016) rather than vtrace.
  - Retrace produces proper multi-step Q-targets satisfying the Q-Bellman
    equation rather than conflating V and Q targets.
  - The rollout data is on-policy, so the bootstrap and trace use the rollout
    policy μ over the Polyak target-net Q-values:
    V(s) = E_{a~μ}[Q_target(s,:)]. With π = μ the Retrace IS factor
    min(1, π/μ) is identically 1, so the trace coefficient reduces to the
    constant gae_lambda (i.e. Expected SARSA(λ) / Q-boosting). The target
    network is used only for value stability (its Q-values), never its policy.

Policy gradient:
  - Advantage at the sampled action A(s,a) = q_target − V(s), where
    q_target is the Retrace(λ) target and V(s) = E_{a~μ}[Q_target] is the
    rollout-policy baseline over target-net Q-values. Both are stop-gradiented,
    so the policy update never reads untrained Q-values for non-sampled actions
    and the Q-head only sees gradients via the Q-loss.
  - For NeuRD (loss_type=rnad) the full per-action regret vector is
    reconstructed from this single sampled-action advantage via the
    all-actions baseline trick (Srinivasan et al. 2018).
  - IS correction against the rollout policy uses the standard PPO clip.

Everything else (Polyak target, magnet reset, KL terms, entropy,
multi-epoch scan, alternating training) is inherited from PPOBase / MMD.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.lax as lax
import optax
from .base import TrainingState
from .episode import collect_episodes
from .ppo import PPOBase
from .types import LossType, MagnetUpdateType
from ..agents.base import Agent
from ..envs.base import Env
from ..advantage import retrace
from ..losses.mmd_q import mmd_q_loss, rnad_q_loss


class QMMD(PPOBase):
  """MMD with a Q-value function (no V-head) trained via Retrace(λ).

  Args:
      env, agent, n_epochs, batch_size, lr,
      clip_eps, ent_coef, gamma, gae_lambda,
      delta_clip, trace_clip:   inherited from PPOBase (delta_clip/trace_clip
                                 unused — kept for API compatibility).
      vf_coef:                  Weight of the Q-value loss.
      magnet_coef:              Weight of KL(current ‖ magnet) term.
      old_policy_coef:          Weight of KL(current ‖ old policy) term.
      target_update_rate:       Polyak τ for target_params update.
      magnet_interval:          Steps between hard resets (used when magnet_update_type=periodic).
      magnet_update_rate:       Polyak τ for magnet update (used when magnet_update_type=incremental).
      magnet_update_type:       "periodic" (hard reset every k steps) or "incremental" (Polyak).
      gae_lambda:               Trace decay λ for Retrace (1.0 = full traces).
      loss_type:                "mmd" (PPO policy gradient) or "rnad" (NeuRD).
      neurd_clip:               NeuRD logit clip (only used when loss_type=rnad).
      neurd_threshold:          NeuRD logit threshold (only used when loss_type=rnad).
      alternating:              If True, update one player per step.
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
    ent_coef: float = 0.01,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    delta_clip: float = 1.0,
    trace_clip: float = 1.0,
    magnet_coef: float = 0.15,
    old_policy_coef: float = 0.05,
    target_update_rate: float = 0.001,
    magnet_interval: int = 2000,
    magnet_update_rate: float = 0.001,
    magnet_update_type: MagnetUpdateType = MagnetUpdateType.PERIODIC,
    loss_type: LossType = LossType.MMD,
    neurd_clip: float = 5.0,
    neurd_threshold: float = 2.0,
    optimizer: optax.GradientTransformation | None = None,
    grad_clip: float | None = None,
    alternating: bool = False,
  ) -> None:
    super().__init__(
      env,
      agent,
      n_epochs,
      batch_size,
      lr,
      clip_eps,
      vf_coef=0.0,   # base V-head disabled; self.vf_coef below is for Q-loss
      ent_coef=ent_coef,
      gamma=gamma,
      gae_lambda=gae_lambda,
      delta_clip=delta_clip,
      trace_clip=trace_clip,
      optimizer=optimizer,
      grad_clip=grad_clip,
    )
    self.vf_coef = vf_coef
    self.loss_type = loss_type
    self.neurd_clip = neurd_clip
    self.neurd_threshold = neurd_threshold
    self.magnet_coef = magnet_coef
    self.old_policy_coef = old_policy_coef
    self.target_update_rate = target_update_rate
    self.magnet_interval = magnet_interval
    self.magnet_update_rate = magnet_update_rate
    self.magnet_update_type = magnet_update_type
    self.alternating = alternating

  # ── Init ──────────────────────────────────────────────────────────────────

  def init(self, key: jax.Array) -> TrainingState:
    key, params, opt_state, env_state, agent_state = self._init_common(key)
    return TrainingState(
      params=params,
      opt_state=opt_state,
      env_state=env_state,
      agent_state=agent_state,
      rng=key,
      step=jnp.zeros((), jnp.int32),
      extras={"target_params": params, "magnet_params": params},
    )

  # ── Q-target computation ──────────────────────────────────────────────────

  def _compute_q_targets(
    self,
    rewards: jax.Array,          # (B, T, P)
    target_q_values: jax.Array,  # (B, T, P, A) — Polyak target-net Q (stability)
    sample_logits: jax.Array,    # (B, T, P, A) — rollout (on-policy) policy
    actions: jax.Array,          # (B, T, P)
    legal_actions: jax.Array,    # (B, T, P, A)
    dones: jax.Array,            # (B, T)
  ) -> jax.Array:                # (B, T, P) — Q targets
    # On-policy: bootstrap and trace use the rollout policy μ over the target
    # net's Q-values. Since π = μ the Retrace IS factor min(1, π/μ) is exactly
    # 1, so the trace coefficient is the constant gae_lambda (Expected
    # SARSA(λ) / Q-boosting).
    mu = jax.nn.softmax(sample_logits, where=legal_actions)           # (B,T,P,A)

    # V(s) = E_{a~μ}[Q_target(s, a)]
    v_target = (mu * target_q_values).sum(-1)                         # (B,T,P)

    # Q_target for the observed action
    q_taken = jnp.take_along_axis(
      target_q_values, actions[..., None], axis=-1
    ).squeeze(-1)                                                      # (B,T,P)

    discount = (1.0 - dones) * self.gamma                             # (B,T)

    # vmap retrace over P (axis 1 in T×P arrays) then over B (axis 0)
    _retrace = lambda r, q, v, d: retrace(r, q, v, d, lambda_=self.gae_lambda)
    _retrace_P = jax.vmap(_retrace, in_axes=(1, 1, 1, None), out_axes=1)
    _retrace_BP = jax.vmap(_retrace_P, in_axes=(0, 0, 0, 0), out_axes=0)
    return _retrace_BP(rewards, q_taken, v_target, discount)                           # (B,T,P)

  # ── Public step ───────────────────────────────────────────────────────────

  def step(self, state: TrainingState) -> tuple[TrainingState, dict[str, jax.Array]]:
    rng, collect_key = jax.random.split(state.rng)
    _, _, _, episodes = collect_episodes(
      self.env, self.agent, state.params, collect_key, self.batch_size
    )

    # ── Retrace targets: target-net Q-values, on-policy rollout policy ──────
    target_out = self._eval_params(state.extras["target_params"], episodes)
    q_targets = lax.stop_gradient(
      self._compute_q_targets(
        episodes.rewards,
        target_out.q_values,           # (B,T,P,A) — target-net Q (stability)
        episodes.agent_output.logits,  # (B,T,P,A) — rollout (on-policy) π
        episodes.actions,              # (B,T,P)
        episodes.legal_actions,        # (B,T,P,A)
        episodes.dones,                # (B,T)
      )
    )  # (B,T,P)

    # ── Target-net Q-vector for the advantage/regrets ──────────────────────
    # The loss builds the per-action advantage from this (stop-grad) vector,
    # overriding the sampled action with its Retrace target and centering by
    # E_π[·] over the same vector — see losses/mmd_q.py.
    target_q_values = lax.stop_gradient(target_out.q_values)  # (B,T,P,A)

    # ── Magnet logits ──────────────────────────────────────────────────────
    magnet_out = self._eval_params(state.extras["magnet_params"], episodes)
    magnet_logits = lax.stop_gradient(magnet_out.logits)  # (B,T,P,A)

    params, opt_state = state.params, state.opt_state
    valid = self._valid_mask(episodes.dones)

    if self.alternating:
      n_players = self.env.num_players
      active = state.step % n_players
      player_mask = jnp.zeros(n_players).at[active].set(float(n_players))

    def epoch_fn(carry, _):
      params, opt_state = carry

      def total_loss(params):
        agent_out = self._eval_params(params, episodes)

        if self.loss_type == LossType.MMD:
          # 9 arrays + 5 scalars
          _axes = (0, 0, 0, 0, 0, 0, 0, 0, 0, None, None, None, None, None)
          loss_P = jax.vmap(mmd_q_loss, in_axes=_axes)
          loss_TP = jax.vmap(loss_P, in_axes=_axes)
          loss_BTP = jax.vmap(loss_TP, in_axes=_axes)
          losses, metrics = loss_BTP(
            agent_out.q_values,                 # (B,T,P,A)
            agent_out.logits,                   # (B,T,P,A)
            episodes.legal_actions,             # (B,T,P,A)
            episodes.actions,                   # (B,T,P)
            episodes.agent_output.logits,       # (B,T,P,A) — sampling π
            episodes.agent_output.q_values,     # (B,T,P,A) — sampling Q
            magnet_logits,                      # (B,T,P,A)
            q_targets,                          # (B,T,P)
            target_q_values,                    # (B,T,P,A)
            self.clip_eps,
            self.vf_coef,
            self.ent_coef,
            self.magnet_coef,
            self.old_policy_coef,
          )
        elif self.loss_type == LossType.RNAD:
          # 9 arrays + 6 scalars
          _axes = (0, 0, 0, 0, 0, 0, 0, 0, 0, None, None, None, None, None, None)
          loss_P = jax.vmap(rnad_q_loss, in_axes=_axes)
          loss_TP = jax.vmap(loss_P, in_axes=_axes)
          loss_BTP = jax.vmap(loss_TP, in_axes=_axes)
          losses, metrics = loss_BTP(
            agent_out.q_values,                 # (B,T,P,A)
            agent_out.logits,                   # (B,T,P,A)
            episodes.legal_actions,             # (B,T,P,A)
            episodes.actions,                   # (B,T,P)
            episodes.agent_output.logits,       # (B,T,P,A) — sampling π
            episodes.agent_output.q_values,     # (B,T,P,A) — sampling Q
            magnet_logits,                      # (B,T,P,A)
            q_targets,                          # (B,T,P)
            target_q_values,                    # (B,T,P,A)
            self.clip_eps,
            self.vf_coef,
            self.ent_coef,
            self.magnet_coef,
            self.neurd_clip,
            self.neurd_threshold,
          )

        if self.alternating:
          wmean = lambda x: self._wmean(x * player_mask, valid)
        else:
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
