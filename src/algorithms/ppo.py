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
from typing import NamedTuple, Any

import jax
import jax.numpy as jnp
import optax

from .base import Algorithm
from .episode import collect_episodes
from ..advantage import vtrace
from ..agents.base import Agent
from ..envs.base import Env
from ..losses.ppo import ppo_loss
from ..utils import weighted_mean

class PPOState(NamedTuple):
    params:      Any        # agent / network parameters pytree
    opt_state:   Any        # optax optimizer state
    env_state:   Any        # current environment state (carried but not used by collect)
    agent_state: Any        # recurrent carry (None for stateless agents)
    rng:         jax.Array  # PRNG key, threaded through every step
    step:        jax.Array  # int32 scalar, counts training iterations


class PPO(Algorithm):
    """Proximal Policy Optimisation with vtrace advantage estimation.

    Args:
        env:         Env instance (simultaneous or single-player).
        agent:       Agent whose player_evaluate drives rollout and re-evaluation.
        rollout_len: Steps per episode (defaults to env.max_length).
        n_epochs:    Gradient passes over each rollout.
        batch_size:  Independent episodes per iteration (B).
        lr:          Adam learning rate.
        clip_eps:    PPO clip ratio ε; also used for value-function clipping.
        vf_coef:     Value-function loss coefficient.
        ent_coef:    Entropy bonus coefficient.
        gamma:       Discount factor.
        gae_lambda:  vtrace λ parameter.
        delta_clip:  vtrace δ-clipping ratio (1.0 = standard GAE).
        trace_clip:  vtrace ρ-clipping ratio (1.0 = standard GAE).
    """

    def __init__(
        self,
        env:         Env,
        agent:       Agent,
        rollout_len: int   = 128,
        n_epochs:    int   = 4,
        batch_size:  int   = 4,
        lr:          float = 3e-4,
        clip_eps:    float = 0.2,
        vf_coef:     float = 0.5,
        ent_coef:    float = 0.01,
        gamma:       float = 0.99,
        gae_lambda:  float = 0.95,
        delta_clip:  float = 1.0,
        trace_clip:  float = 1.0,
    ) -> None:
        self.env         = env
        self.agent       = agent
        self.rollout_len = rollout_len
        self.n_epochs    = n_epochs
        self.batch_size  = batch_size
        self.clip_eps    = clip_eps
        self.vf_coef     = vf_coef
        self.ent_coef    = ent_coef
        self.gamma       = gamma
        self.gae_lambda  = gae_lambda
        self.delta_clip  = delta_clip
        self.trace_clip  = trace_clip
        self.optimizer   = optax.adam(lr)

    # ── Init ──────────────────────────────────────────────────────────────────

    def init(self, key: jax.Array) -> PPOState:
        key, env_key, net_key = jax.random.split(key, 3)
        env_state   = self.env.init_state(env_key)
        dummy_obs   = self.env.information_set(env_state, 0)
        params      = self.agent.init_params(net_key, dummy_obs)
        opt_state   = self.optimizer.init(params)
        agent_state = self.agent.init_state(params)
        return PPOState(
            params      = params,
            opt_state   = opt_state,
            env_state   = env_state,
            agent_state = agent_state,
            rng         = key,
            step        = jnp.zeros((), jnp.int32),
        )

    # ── Public step ───────────────────────────────────────────────────────────

    def step(self, state: PPOState) -> tuple[PPOState, dict[str, jax.Array]]:
        # 1. Collect B full episodes.
        #    Each episode: T steps, P players → fields shaped (B, T, P, ...).
        rng, collect_key = jax.random.split(state.rng)
        _, _, _, episodes = collect_episodes(
            self.env, self.agent, state.params, collect_key, self.batch_size)

        # 2. Advantage estimates: (B, T, P) each.
        advantages, targets = self._compute_advantages(
            episodes.rewards, episodes.agent_output.value, episodes.dones)

        params, opt_state = state.params, state.opt_state

        # 3. n_epochs gradient steps on the frozen rollout.
        def epoch_fn(carry, _):
            params, opt_state = carry

            def total_loss(params):
                # Re-evaluate with current params on all (B, T) step/player combos.
                # player_evaluate(params, state, obs (P, D)) → output (P, ...).
                # Two vmaps: outer over B, inner over T.
                eval_T  = jax.vmap(self.agent.player_evaluate, in_axes=(None, 0, 0))
                eval_BT = jax.vmap(eval_T,                     in_axes=(None, 0, 0))
                agent_out, _ = eval_BT(params, episodes.agent_states, episodes.infosets)
                # agent_out.logits: (B, T, P, A)   agent_out.value: (B, T, P)

                # ppo_loss handles a single (timestep, player) sample → scalar.
                # Three nested vmaps produce (B, T, P) per-sample losses.
                _axes = (0, 0, 0, 0, 0, 0, 0, 0, None, None, None)
                loss_P   = jax.vmap(ppo_loss,  in_axes=_axes)
                loss_TP  = jax.vmap(loss_P,    in_axes=_axes)
                loss_BTP = jax.vmap(loss_TP,   in_axes=_axes)
                losses, metrics = loss_BTP(
                    agent_out.value,              # (B, T, P)
                    agent_out.logits,             # (B, T, P, A)
                    episodes.legal_actions,       # (B, T, P, A)
                    episodes.actions,             # (B, T, P)
                    episodes.agent_output.logits, # (B, T, P, A) — frozen old policy
                    episodes.agent_output.value,  # (B, T, P)    — frozen old values
                    advantages,                   # (B, T, P)
                    targets,                      # (B, T, P)
                    self.clip_eps, self.vf_coef, self.ent_coef,
                )
                # losses / metrics: (B, T, P)

                # Weighted mean: mask out padding steps after episode termination.
                # dones[b, t] = True at the terminal step; steps t+1… are padding.
                # valid[b, t] = True while no done has occurred strictly before t. 
                
                valid = jnp.cumulative_prod(1 - episodes.dones, axis=1, include_initial=True) 
                valid = valid[..., :-1] # The last element of trajectory is not used, because we included the initial (which is always 1) in cummlative product.

                def wmean(x):
                  x = x.sum(-1) # Sum over players
                  x = weighted_mean(x, valid, 1) # Mean over trajectory
                  x = x.mean(-1) # Mean over batch
                  return x

                return wmean(losses), jax.tree.map(wmean, metrics)

            (_, metrics), grads = jax.value_and_grad(total_loss, has_aux=True)(params)
            updates, new_opt_state = self.optimizer.update(grads, opt_state, params)
            return (optax.apply_updates(params, updates), new_opt_state), metrics

        (params, opt_state), epoch_metrics = jax.lax.scan(
            epoch_fn, (params, opt_state), None, length=self.n_epochs)

        return PPOState(
            params      = params,
            opt_state   = opt_state,
            env_state   = state.env_state,
            agent_state = state.agent_state,
            rng         = rng,
            step        = state.step + 1,
        ), jax.tree.map(jnp.mean, epoch_metrics)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _compute_advantages(
        self,
        rewards: jax.Array,  # (B, T, P)
        values:  jax.Array,  # (B, T, P)
        dones:   jax.Array,  # (B, T)
    ) -> tuple[jax.Array, jax.Array]:
        """Returns (advantages, targets) both shaped (B, T, P)."""
        discount = (1.0 - dones) * self.gamma  # (B, T)

        # vtrace operates on (T,) sequences per player per episode.
        # Inner vmap: over P (last axis of rewards/values; discount is shared).
        # Outer vmap: over B (axis 0 of all).
        vtrace_P  = jax.vmap(
            vtrace,
            in_axes=(-1, -1, None, None, None, None, None, None),
            out_axes=(-1, -1),
        )
        vtrace_BP = jax.vmap(
            vtrace_P,
            in_axes=(0, 0, 0, None, None, None, None, None),
        )
        targets, advantages = vtrace_BP(
            rewards, values, discount,
            1.0, 0.0, self.gae_lambda, self.delta_clip, self.trace_clip,
        )
        return advantages, targets
