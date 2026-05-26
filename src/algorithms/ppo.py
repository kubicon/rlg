"""PPO algorithm — thin orchestrator over modular pure functions.

Wires together:
  agents/actor_critic.py   — ActorCriticAgent (player_evaluate)
  algorithms/episode.py    — collect_episodes
  advantage.py             — vtrace
  losses/ppo.py            — ppo_loss

Dimension convention (episode fields after collect_episodes):
  B — batch_size          (independent episodes collected in parallel)
  T — rollout length      (env.max_length for finite games like Goofspiel)
  P — num_players
  A — num_actions

step collects B episodes, computes (B, T, P) advantages via vtrace, then
runs n_epochs gradient updates using jax.lax.scan for full XLA fusion.
ppo_loss handles a single (timestep, player) sample → scalar, so the outer
loss vmaps over B, T, P and reduces with a weighted mean that masks padding
steps after episode termination (using the dones flag).
"""

from __future__ import annotations
from typing import Any

import jax
import jax.numpy as jnp
import jax.lax as lax
import optax

from .base import Algorithm, TrainingState
from .episode import collect_episodes
from ..advantage import vtrace
from ..agents.base import Agent
from ..envs.base import Env
from ..losses.ppo import ppo_loss
from ..utils import weighted_mean


class PPOBase(Algorithm):
  """Shared infrastructure for PPO-family algorithms."""

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
    optimizer: optax.GradientTransformation | None = None,
    grad_clip: float | None = None,
  ) -> None:
    self.env = env
    self.agent = agent
    self.n_epochs = n_epochs
    self.batch_size = batch_size
    self.clip_eps = clip_eps
    self.vf_coef = vf_coef
    self.ent_coef = ent_coef
    self.gamma = gamma
    self.gae_lambda = gae_lambda
    self.delta_clip = delta_clip
    self.trace_clip = trace_clip
    base_optimizer = optimizer if optimizer is not None else optax.adam(lr)
    self.optimizer = (
      optax.chain(optax.clip_by_global_norm(grad_clip), base_optimizer)
      if grad_clip is not None
      else base_optimizer
    )

  def _init_common(self, key: jax.Array):
    key, env_key, net_key = jax.random.split(key, 3)
    env_state = self.env.init_state(env_key)
    dummy_obs = self.env.information_set(env_state, 0, key)
    params = self.agent.init_params(net_key, dummy_obs)
    opt_state = self.optimizer.init(params)
    agent_state = self.agent.init_state(params)
    return key, params, opt_state, env_state, agent_state

  def _eval_params(self, params: Any, episodes) -> Any:
    """Re-evaluate params over all (B, T) steps → agent_out shaped (B, T, P, ...)."""
    eval_T = jax.vmap(self.agent.player_evaluate, in_axes=(None, 0, 0))
    eval_BT = jax.vmap(eval_T, in_axes=(None, 0, 0))
    agent_out, _ = eval_BT(params, episodes.agent_states, episodes.infosets)
    return agent_out

  def _compute_advantages(
    self,
    rewards: jax.Array,  # (B, T, P)
    values: jax.Array,  # (B, T, P)
    dones: jax.Array,  # (B, T)
  ) -> tuple[jax.Array, jax.Array]:
    """Returns (advantages, targets) both shaped (B, T, P)."""
    discount = (1.0 - dones) * self.gamma  # (B, T)

    vtrace_P = jax.vmap(
      vtrace,
      in_axes=(-1, -1, None, None, None, None, None, None),
      out_axes=(-1, -1),
    )
    vtrace_BP = jax.vmap(
      vtrace_P,
      in_axes=(0, 0, 0, None, None, None, None, None),
    )
    targets, advantages = vtrace_BP(
      rewards,
      values,
      discount,
      1.0,
      0.0,
      self.gae_lambda,
      self.delta_clip,
      self.trace_clip,
    )
    return advantages, targets

  @staticmethod
  def _valid_mask(dones: jax.Array) -> jax.Array:
    """Boolean mask (B, T) that is False for padding steps after episode end."""
    valid = jnp.cumulative_prod(1 - dones, axis=1, include_initial=True)
    return valid[..., :-1]

  @staticmethod
  def _wmean(x: jax.Array, valid: jax.Array) -> jax.Array:
    """Weighted mean over (B, T, P) losses using valid mask."""
    x = x.sum(-1)  # sum over P
    x = weighted_mean(x, valid, 1)  # mean over T
    return x.mean(-1)  # mean over B


class PPO(PPOBase):
  """Proximal Policy Optimisation with vtrace advantage estimation.

  A Polyak-averaged copy of the parameters (`target_params`) is maintained
  in `state.extras` and used as the reference policy/value in the losses,
  providing more stable training targets than the per-rollout old policy.

  Args:
      env:        Env instance (simultaneous or single-player).
      agent:      Agent whose player_evaluate drives rollout and re-evaluation.
      n_epochs:   Gradient passes over each rollout.
      batch_size:  Independent episodes per iteration (B).
      lr:          Learning rate used when `optimizer` is None (builds Adam).
      optimizer:   Any optax optimizer (e.g. ``optax.adamw(lr)``,
                   ``optax.sgd(lr)``). Overrides `lr` when provided.
      clip_eps:    PPO clip ratio ε; also used for value-function clipping.
      vf_coef:     Value-function loss coefficient.
      ent_coef:    Entropy bonus coefficient.
      gamma:       Discount factor.
      gae_lambda:  vtrace λ parameter.
      delta_clip:  vtrace δ-clipping ratio (1.0 = standard GAE).
      trace_clip:  vtrace ρ-clipping ratio (1.0 = standard GAE).
      polyak_tau:  Polyak averaging rate for target update (closer to 1
                   = target tracks params faster).
  """

  def __init__(self, *args, target_update_rate: float = 0.005, **kwargs) -> None:
    super().__init__(*args, **kwargs)
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
    _, _, _, episodes = collect_episodes(
      self.env, self.agent, state.params, collect_key, self.batch_size
    )

    # Target values are fixed across all epochs — precompute once.
    target_out = self._eval_params(state.extras["target_params"], episodes)
    target_values = lax.stop_gradient(target_out.value)  # (B, T, P)

    # Use target values for GAE bootstrapping (stable value estimates).
    advantages, targets = self._compute_advantages(
      episodes.rewards, target_values, episodes.dones
    )

    params, opt_state = state.params, state.opt_state
    valid = self._valid_mask(episodes.dones)

    def epoch_fn(carry, _):
      params, opt_state = carry

      def total_loss(params):
        agent_out = self._eval_params(params, episodes)

        _axes = (0, 0, 0, 0, 0, 0, 0, 0, None, None, None)
        loss_P = jax.vmap(ppo_loss, in_axes=_axes)
        loss_TP = jax.vmap(loss_P, in_axes=_axes)
        loss_BTP = jax.vmap(loss_TP, in_axes=_axes)
        losses, metrics = loss_BTP(
          agent_out.value,
          agent_out.logits,
          episodes.legal_actions,
          episodes.actions,
          episodes.agent_output.logits,
          target_values,
          advantages,
          targets,
          self.clip_eps,
          self.vf_coef,
          self.ent_coef,
        )
        wmean = lambda x: self._wmean(x, valid)
        return wmean(losses), jax.tree.map(wmean, metrics)

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

    return TrainingState(
      params=params,
      opt_state=opt_state,
      env_state=state.env_state,
      agent_state=state.agent_state,
      rng=rng,
      step=state.step + 1,
      extras={"target_params": target_params},
    ), jax.tree.map(jnp.mean, epoch_metrics)
