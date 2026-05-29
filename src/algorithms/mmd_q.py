"""QMMD — MMD variant with an explicit Q-value function and no V-head.

Architecture differences from MMD:
  - Network is a twin_head: CategoricalHead (logits) + QHead (Q-values).
  - No V-head. State value is derived on-the-fly as V_π(s) = E_π[Q(s,:)].

Q-target computation:
  - Uses Retrace(λ) (Munos et al., 2016) rather than vtrace.
  - Retrace produces proper multi-step Q-targets with safe IS correction,
    satisfying the Q-Bellman equation rather than conflating V and Q targets.
  - Bootstrapping uses V_target(s) = E_{π_target}[Q_target(s,:)] from the
    Polyak-averaged target network.

Policy gradient:
  - Advantage at the sampled action A(s,a) = q_target − V_target(s), where
    q_target is the Retrace(λ) target and V_target(s) = E_{π_target}[Q_target]
    is a target-network baseline. Both are stop-gradiented, so the policy
    update never reads untrained Q-values for non-sampled actions and the
    Q-head only sees gradients via the Q-loss.
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
from enum import StrEnum

from .base import TrainingState
from .episode import collect_episodes
from .ppo import PPOBase
from ..agents.base import Agent
from ..envs.base import Env
from ..advantage import retrace, qboost
from ..losses.mmd_q import mmd_q_loss, rnad_q_loss


class LossType(StrEnum):
  MMD = "mmd"
  RNAD = "rnad"


class ValueType(StrEnum):
  RETRACE = "retrace"
  QBOOST = "qboost"


class QMMD(PPOBase):
  """MMD with a Q-value function (no V-head) trained via Retrace(λ).

  Args:
      env, agent, n_epochs, batch_size, lr,
      clip_eps, ent_coef, gamma, gae_lambda,
      delta_clip, trace_clip:   inherited from PPOBase (gae_lambda/delta_clip
                                 unused — kept for API compatibility).
      qf_coef:                  Weight of the Q-value loss.
      magnet_coef:              Weight of KL(current ‖ magnet) term.
      old_policy_coef:          Weight of KL(current ‖ old policy) term.
      target_update_rate:       Polyak τ for target_params update.
      magnet_interval:          Steps between hard resets of magnet_params.
      retrace_lambda:           Trace decay λ for Retrace (1.0 = full traces).
      value_type:               "retrace" (Retrace(λ)) or "qboost" (Q-boost Expected SARSA(λ), default).
      qboost_lambda:            Constant trace λ for Q-boost (ignored when value_type=retrace).
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
    qf_coef: float = 0.5,
    ent_coef: float = 0.01,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    delta_clip: float = 1.0,
    trace_clip: float = 1.0,
    magnet_coef: float = 0.15,
    old_policy_coef: float = 0.05,
    target_update_rate: float = 0.001,
    magnet_interval: int = 2000,
    retrace_lambda: float = 1.0,
    value_type: ValueType = ValueType.QBOOST,
    qboost_lambda: float = 0.95,
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
      vf_coef=0.0,   # no V-head; kept in base for API compatibility
      ent_coef=ent_coef,
      gamma=gamma,
      gae_lambda=gae_lambda,
      delta_clip=delta_clip,
      trace_clip=trace_clip,
      optimizer=optimizer,
      grad_clip=grad_clip,
    )
    self.qf_coef = qf_coef
    self.loss_type = loss_type
    self.neurd_clip = neurd_clip
    self.neurd_threshold = neurd_threshold
    self.magnet_coef = magnet_coef
    self.old_policy_coef = old_policy_coef
    self.target_update_rate = target_update_rate
    self.magnet_interval = magnet_interval
    self.retrace_lambda = retrace_lambda
    self.value_type = value_type
    self.qboost_lambda = qboost_lambda
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
    target_q_values: jax.Array,  # (B, T, P, A)
    target_logits: jax.Array,    # (B, T, P, A)
    sample_logits: jax.Array,    # (B, T, P, A) — behavior policy at rollout
    actions: jax.Array,          # (B, T, P)
    legal_actions: jax.Array,    # (B, T, P, A)
    dones: jax.Array,            # (B, T)
  ) -> jax.Array:                # (B, T, P) — Q targets
    pi_target = jax.nn.softmax(target_logits, where=legal_actions)   # (B,T,P,A)
    mu = jax.nn.softmax(sample_logits, where=legal_actions)           # (B,T,P,A)

    # V_target(s) = E_{a~π_target}[Q_target(s, a)]
    v_target = (pi_target * target_q_values).sum(-1)                  # (B,T,P)

    # Q_target for the observed action
    q_taken = jnp.take_along_axis(
      target_q_values, actions[..., None], axis=-1
    ).squeeze(-1)                                                      # (B,T,P)

    discount = (1.0 - dones) * self.gamma                             # (B,T)

    if self.value_type == ValueType.RETRACE:
      # IS trace coefficient: λ · min(1, π(a)/μ(a))
      pi_a = jnp.take_along_axis(
        pi_target, actions[..., None], axis=-1
      ).squeeze(-1)                                                    # (B,T,P)
      mu_a = jnp.take_along_axis(
        mu, actions[..., None], axis=-1
      ).squeeze(-1)                                                    # (B,T,P)
      c = self.retrace_lambda * jnp.minimum(1.0, pi_a / (mu_a + 1e-8)) # (B,T,P)

      # vmap retrace over P (axis 1 in T×P arrays) then over B (axis 0)
      _retrace_P = jax.vmap(retrace, in_axes=(1, 1, 1, 1, None), out_axes=1)
      _retrace_BP = jax.vmap(_retrace_P, in_axes=(0, 0, 0, 0, 0), out_axes=0)
      return _retrace_BP(rewards, q_taken, v_target, c, discount)     # (B,T,P)
    else:
      # vmap qboost over P (axis 1 in T×P arrays) then over B (axis 0)
      _qboost_P = jax.vmap(qboost, in_axes=(1, 1, 1, None, None), out_axes=1)
      _qboost_BP = jax.vmap(_qboost_P, in_axes=(0, 0, 0, None, 0), out_axes=0)
      return _qboost_BP(rewards, q_taken, v_target, self.qboost_lambda, discount)  # (B,T,P)

  # ── Public step ───────────────────────────────────────────────────────────

  def step(self, state: TrainingState) -> tuple[TrainingState, dict[str, jax.Array]]:
    rng, collect_key = jax.random.split(state.rng)
    _, _, _, episodes = collect_episodes(
      self.env, self.agent, state.params, collect_key, self.batch_size
    )

    # ── Retrace targets from the Polyak target network ─────────────────────
    target_out = self._eval_params(state.extras["target_params"], episodes)
    q_targets = lax.stop_gradient(
      self._compute_q_targets(
        episodes.rewards,
        target_out.q_values,        # (B,T,P,A)
        target_out.logits,          # (B,T,P,A)
        episodes.agent_output.logits,  # (B,T,P,A) — behavior π at rollout
        episodes.actions,           # (B,T,P)
        episodes.legal_actions,     # (B,T,P,A)
        episodes.dones,             # (B,T)
      )
    )  # (B,T,P)

    # ── Target-net state baseline: V_target(s) = E_{π_target}[Q_target(s,:)] ─
    # Used as the policy-gradient baseline so the advantage never reads
    # untrained Q-values for non-sampled actions (stable, stop-gradiented).
    target_pi = jax.nn.softmax(target_out.logits, where=episodes.legal_actions)
    v_baseline = lax.stop_gradient(
      (target_pi * target_out.q_values).sum(-1)
    )  # (B,T,P)

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
            v_baseline,                         # (B,T,P)
            self.clip_eps,
            self.qf_coef,
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
            v_baseline,                         # (B,T,P)
            self.clip_eps,
            self.qf_coef,
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
