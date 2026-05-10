"""Best-response training: one player trains against a fixed opponent policy.

BRAlgorithm is a PPO variant where:
  - opp_params are frozen throughout training (the fixed opponent)
  - br_player is the only player whose policy is updated
  - The loss is treated as a 1-player problem using only br_player's rewards

This is structurally identical to PPO (with a Polyak target network) except
that episode collection uses the mixed-policy collector and the advantage /
loss computation is sliced to br_player only.
"""

from __future__ import annotations
from typing import Any

import jax
import jax.numpy as jnp
import jax.lax as lax
import optax

from .base import TrainingState
from .episode import collect_episodes_br
from .ppo import PPOBase
from ..advantage import vtrace
from ..agents.base import Agent
from ..envs.base import Env
from ..losses.ppo import ppo_loss
from ..utils import weighted_mean


class BRAlgorithm(PPOBase):
  """PPO best-response: trains br_player against a fixed opp_params opponent.

  Args:
      env, agent, n_epochs, batch_size, lr, clip_eps, vf_coef, ent_coef,
      gamma, gae_lambda, delta_clip, trace_clip:  inherited from PPOBase.
      opp_params:          Frozen parameters for all players except br_player.
      br_player:           Index of the player being trained.
      target_update_rate:  Polyak τ for the target network update.
  """

  def __init__(
    self,
    env: Env,
    agent: Agent,
    opp_params: Any,
    br_player: int,
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
    target_update_rate: float = 0.005,
  ) -> None:
    super().__init__(
      env, agent, n_epochs, batch_size, lr, clip_eps,
      vf_coef, ent_coef, gamma, gae_lambda, delta_clip, trace_clip,
    )
    self.opp_params = opp_params
    self.br_player = br_player
    self.target_update_rate = target_update_rate

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
      extras={"target_params": params},
    )

  # ── Public step ───────────────────────────────────────────────────────────

  def step(self, state: TrainingState) -> tuple[TrainingState, dict[str, jax.Array]]:
    rng, collect_key = jax.random.split(state.rng)
    _, _, _, episodes = collect_episodes_br(
      self.env, self.agent,
      state.params, self.opp_params, self.br_player,
      collect_key, self.batch_size,
    )

    # Target values for br_player only — (B, T)
    target_out = self._eval_params(state.extras["target_params"], episodes)
    target_values = lax.stop_gradient(target_out.value[:, :, self.br_player])

    advantages, targets = self._compute_advantages_1p(
      episodes.rewards[:, :, self.br_player], target_values, episodes.dones
    )

    params, opt_state = state.params, state.opt_state
    valid = self._valid_mask(episodes.dones)  # (B, T)

    def epoch_fn(carry, _):
      params, opt_state = carry

      def total_loss(params):
        agent_out = self._eval_params(params, episodes)
        p = self.br_player

        losses, metrics = jax.vmap(jax.vmap(ppo_loss, in_axes=(0,)*8 + (None,)*3), in_axes=(0,)*8 + (None,)*3)(
          agent_out.value[:, :, p],
          agent_out.logits[:, :, p, :],
          episodes.legal_actions[:, :, p, :],
          episodes.actions[:, :, p],
          episodes.agent_output.logits[:, :, p, :],
          target_values,  # used for PPO value clipping, same as PPO._step
          advantages,
          targets,
          self.clip_eps,
          self.vf_coef,
          self.ent_coef,
        )
        # weighted mean over T then mean over B — no P axis
        loss = weighted_mean(losses, valid, axis=1).mean()
        metrics = jax.tree.map(lambda x: weighted_mean(x, valid, axis=1).mean(), metrics)
        return loss, metrics

      (_, metrics), grads = jax.value_and_grad(total_loss, has_aux=True)(params)
      updates, new_opt_state = self.optimizer.update(grads, opt_state, params)
      return (optax.apply_updates(params, updates), new_opt_state), metrics

    (params, opt_state), epoch_metrics = jax.lax.scan(
      epoch_fn, (params, opt_state), None, length=self.n_epochs
    )

    target_params = optax.incremental_update(
      params, state.extras["target_params"], self.target_update_rate
    )

    return TrainingState(
      params=params,
      opt_state=opt_state,
      env_state=state.env_state,
      agent_state=state.agent_state,
      rng=rng,
      step=state.step + 1,
      extras={"target_params": target_params},
    ), jax.tree.map(jnp.mean, epoch_metrics)

  # ── Helpers ───────────────────────────────────────────────────────────────

  def _compute_advantages_1p(
    self,
    rewards: jax.Array,  # (B, T)
    values: jax.Array,   # (B, T)
    dones: jax.Array,    # (B, T)
  ) -> tuple[jax.Array, jax.Array]:
    """vtrace advantage estimation for a single player, returning (B, T) arrays."""
    discount = (1.0 - dones) * self.gamma
    targets, advantages = jax.vmap(
      vtrace,
      in_axes=(0, 0, 0, None, None, None, None, None),
    )(rewards, values, discount, 1.0, 0.0, self.gae_lambda, self.delta_clip, self.trace_clip)
    return advantages, targets
