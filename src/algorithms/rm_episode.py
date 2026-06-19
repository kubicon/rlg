"""Regret-matching episode rollout.

A copy of :mod:`src.algorithms.episode` whose only difference is action
selection: the network output is read as cumulative regrets, projected to a
strategy by regret matching, mixed with a uniform exploration floor
(``μ = (1-ε)·s + ε·uniform``), and sampled. ``epsilon`` tunes exploration
(``epsilon=0`` → pure RM on-policy).

The returned :class:`~src.algorithms.episode.Episode` is identical in shape to
the softmax path -- ``agent_output.logits`` simply holds regrets now -- so the
RM loss can recover the same behaviour distribution from the stored regrets.

Kept separate from the softmax rollout so the original is untouched.
"""

from __future__ import annotations
from typing import Any

import jax
import jax.numpy as jnp

from ..agents.base import Agent
from ..envs.base import Env, EnvState
from ..losses.rm_rnad import rm_behavior
from .episode import Episode


def collect_episodes_rm(
  env: Env,
  agent: Agent,
  params: Any,
  rng: Any,
  batch_size: int,
  epsilon: float,
  normalize_rewards: bool = True,
  opp_params: Any = None,
  br_player: int | None = None,
) -> tuple[EnvState, Any, Any, Episode]:
  """Batched RM rollout. See :func:`collect_episodes` for the softmax version."""
  rng = jax.random.split(rng, batch_size)
  return jax.vmap(
    collect_episode_rm,
    in_axes=(None, None, None, 0, None, None, None, None, None, None, None),
  )(
    env,
    agent,
    params,
    rng,
    epsilon,
    None,
    None,
    None,
    normalize_rewards,
    opp_params,
    br_player,
  )


def _sample_logits(logits, legal_actions, epsilon):
  """RM behaviour as categorical logits: log μ, with illegal actions at -inf."""
  mu = rm_behavior(logits, legal_actions, epsilon)
  log_mu = jnp.log(jnp.clip(mu, 1e-12, 1.0))
  return jnp.where(legal_actions, log_mu, -jnp.inf)


def collect_episode_rm(
  env: Env,
  agent: Agent,
  params: Any,
  rng: Any,
  epsilon: float,
  agent_state: Any = None,
  env_state: EnvState = None,
  rollout_len: None | int = None,
  normalize_rewards: bool = True,
  opp_params: Any = None,
  br_player: int | None = None,
) -> tuple[EnvState, Any, Any, Episode]:
  """RM analogue of :func:`collect_episode`. Identical except action sampling."""
  P = env.num_players
  if rollout_len is None:
    rollout_len = env.max_length

  if agent_state is None:
    agent_state = agent.init_state(params)
  if env_state is None:
    rng, env_key = jax.random.split(rng)
    env_state = env.init_state(env_key)

  def scan_step(carry, _):
    env_state, agent_state, rng = carry
    rng, act_key, env_key, obs_key, state_key = jax.random.split(rng, 5)
    obs_keys = jax.random.split(obs_key, P)

    infosets = jax.vmap(env.information_set, in_axes=(None, 0, 0))(
      env_state, jnp.arange(P), obs_keys
    )
    state_rep = env.state_representation(env_state, state_key)
    legal_actions = jax.vmap(env.legal_actions, in_axes=(None, 0))(
      env_state, jnp.arange(P)
    )

    obs = agent.make_obs_step(infosets, state_rep)
    agent_out, new_agent_state = agent.player_evaluate(params, agent_state, obs)

    if opp_params is None:
      sample_logits = _sample_logits(agent_out.logits, legal_actions, epsilon)
    else:
      opp_obs = agent.make_obs_step(infosets, state_rep)
      opp_out, _ = agent.player_evaluate(opp_params, agent_state, opp_obs)
      br_logits = _sample_logits(agent_out.logits, legal_actions, epsilon)
      opp_logits = _sample_logits(opp_out.logits, legal_actions, epsilon)
      is_br = jnp.arange(P) == br_player
      sample_logits = jnp.where(is_br[:, None], br_logits, opp_logits)

    player_keys = jax.random.split(act_key, P)
    actions = jax.vmap(jax.random.categorical)(player_keys, sample_logits)

    new_env_state, rewards, done, _ = env.apply_action(env_state, actions, env_key)

    if normalize_rewards:
      rewards = rewards / env.max_reward

    step = Episode(
      env_state,
      agent_state,
      legal_actions,
      infosets,
      state_rep,
      done,
      rewards,
      actions,
      agent_out,
    )
    return (new_env_state, new_agent_state, rng), step

  (new_env_state, new_agent_state, new_rng), episode = jax.lax.scan(
    scan_step, (env_state, agent_state, rng), None, length=rollout_len
  )
  return new_env_state, new_agent_state, new_rng, episode
