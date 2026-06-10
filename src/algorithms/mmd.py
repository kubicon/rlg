"""MMD algorithm — PPO variant with two auxiliary parameter sets.

Two extra parameter sets are maintained alongside the main params:

  target_params  — Polyak-averaged copy, updated every step:
                       target ← τ · params + (1 − τ) · target
                   Used for stable GAE bootstrapping and value-loss clipping,
                   exactly as in PPO.

  magnet_params  — Reference policy for the KL magnet loss term. Update rule
                   is controlled by magnet_update_type:
                     periodic:    magnet ← params  if step % magnet_interval == 0
                                           magnet  otherwise
                     incremental: magnet ← τ·params + (1−τ)·magnet  every step

Both are stored in TrainingState.extras:
    {'target_params': ..., 'magnet_params': ...}
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
import jax.lax as lax
import optax

from ..losses.rnad import rnad_loss

from .base import TrainingState
from .episode import collect_episodes
from .ppo import PPOBase
from ..agents.base import Agent
from ..envs.base import Env
from ..losses.mmd import mmd_loss
from .types import LossType, MagnetUpdateType, MMD_SCHEDULABLE


class MMD(PPOBase):
  """PPO with a Polyak target network and a periodically-reset magnet policy.

  Args:
      env, agent, rollout_len, n_epochs, batch_size, lr,
      clip_eps, vf_coef, ent_coef, gamma, gae_lambda,
      delta_clip, trace_clip:  inherited from PPOBase.
      magnet_coef:             Weight of KL(current ‖ magnet) term.
      old_policy_coef:         Weight of KL(current ‖ old policy) term.
      target_update_rate:      Polyak τ for target_params update.
      magnet_interval:         Steps between hard resets (used when magnet_update_type=periodic).
      magnet_update_rate:      Polyak τ for magnet update (used when magnet_update_type=incremental).
      magnet_update_type:      "periodic" (hard reset every k steps) or "incremental" (Polyak).
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
    neurd_clip: float = 5.0,
    neurd_threshold: float = 2.0,
    loss_type: LossType = LossType.MMD,
    optimizer: optax.GradientTransformation | None = None,
    grad_clip: float | None = None,
    alternating: bool = False,
    schedules: dict[str, Callable[[int], float]] | None = None,
  ) -> None:
    super().__init__(
      env,
      agent,
      n_epochs,
      batch_size,
      lr,
      clip_eps,
      vf_coef,
      ent_coef,
      gamma,
      gae_lambda,
      delta_clip,
      trace_clip,
      optimizer,
      grad_clip,
    )
    self.magnet_coef = magnet_coef
    self.old_policy_coef = old_policy_coef
    self.target_update_rate = target_update_rate
    self.magnet_interval = magnet_interval
    self.magnet_update_rate = magnet_update_rate
    self.magnet_update_type = magnet_update_type
    self.loss_type = loss_type
    self.neurd_clip = neurd_clip
    self.neurd_threshold = neurd_threshold
    self.alternating = alternating
    schedules = schedules or {}
    unknown = set(schedules) - MMD_SCHEDULABLE
    if unknown:
      raise ValueError(f"Unknown schedule keys: {unknown}. Valid keys: {MMD_SCHEDULABLE}")
    self.schedules = schedules

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

  # ── Public step ───────────────────────────────────────────────────────────

  def step(self, state: TrainingState) -> tuple[TrainingState, dict[str, jax.Array]]:
    def _get(name: str, default):
      s = self.schedules.get(name)
      return s(state.step) if s is not None else default

    clip_eps        = _get("clip_eps",        self.clip_eps)
    vf_coef         = _get("vf_coef",         self.vf_coef)
    ent_coef        = _get("ent_coef",         self.ent_coef)
    magnet_coef     = _get("magnet_coef",     self.magnet_coef)
    old_policy_coef = _get("old_policy_coef", self.old_policy_coef)
    target_update_rate = _get("target_update_rate", self.target_update_rate)
    magnet_update_rate = _get("magnet_update_rate", self.magnet_update_rate)
    neurd_clip      = _get("neurd_clip",      self.neurd_clip)
    neurd_threshold = _get("neurd_threshold", self.neurd_threshold)

    rng, collect_key = jax.random.split(state.rng)
    _, _, _, episodes = collect_episodes(
      self.env, self.agent, state.params, collect_key, self.batch_size
    )

    # Both auxiliary networks are fixed across all epochs — precompute once.
    target_out = self._eval_params(state.extras["target_params"], episodes)
    target_values = lax.stop_gradient(target_out.value)  # (B, T, P)

    magnet_out = self._eval_params(state.extras["magnet_params"], episodes)
    magnet_logits = lax.stop_gradient(magnet_out.logits)  # (B, T, P, A)

    # Use target values for GAE bootstrapping (stable value estimates).
    advantages, targets = self._compute_advantages(
      episodes.rewards, target_values, episodes.dones
    )

    params, opt_state = state.params, state.opt_state
    valid = self._valid_mask(episodes.dones)

    if self.alternating:
      n_players = self.env.num_players
      active = state.step % n_players
      # Scale by n_players so the loss magnitude is unchanged relative to the
      # joint case (where all players' losses are summed).
      player_mask = jnp.zeros(n_players).at[active].set(float(n_players))

    def epoch_fn(carry, _):
      params, opt_state = carry

      def total_loss(params):
        agent_out = self._eval_params(params, episodes)

        if self.loss_type == LossType.MMD:
          _axes = (0, 0, 0, 0, 0, 0, 0, 0, 0, None, None, None, None, None)
          loss_P = jax.vmap(mmd_loss, in_axes=_axes)
          loss_TP = jax.vmap(loss_P, in_axes=_axes)
          loss_BTP = jax.vmap(loss_TP, in_axes=_axes)
          losses, metrics = loss_BTP(
            agent_out.value,
            agent_out.logits,
            episodes.legal_actions,
            episodes.actions,
            episodes.agent_output.logits,  # trajectory sampling policy
            episodes.agent_output.value,
            magnet_logits,
            advantages,
            targets,
            clip_eps,
            vf_coef,
            ent_coef,
            magnet_coef,
            old_policy_coef,
          )
        elif self.loss_type == LossType.RNAD:
          _axes = (0, 0, 0, 0, 0, 0, 0, 0, 0, None, None, None, None, None, None)
          loss_P = jax.vmap(rnad_loss, in_axes=_axes)
          loss_TP = jax.vmap(loss_P, in_axes=_axes)
          loss_BTP = jax.vmap(loss_TP, in_axes=_axes)
          losses, metrics = loss_BTP(
            agent_out.value,
            agent_out.logits,
            episodes.legal_actions,
            episodes.actions,
            episodes.agent_output.logits,  # trajectory sampling policy
            episodes.agent_output.value,
            magnet_logits,
            advantages,
            targets,
            clip_eps,
            vf_coef,
            ent_coef,
            magnet_coef,
            neurd_clip,
            neurd_threshold,
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
      params, state.extras["target_params"], target_update_rate
    )
    if self.magnet_update_type == MagnetUpdateType.INCREMENTAL:
      magnet_params = optax.incremental_update(
        params, state.extras["magnet_params"], magnet_update_rate
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
