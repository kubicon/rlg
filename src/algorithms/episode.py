"""Episode rollout collection.

collect_rollout is a pure function: no Python-side mutation, no callbacks.
Auto-resets the environment on episode end so a single call can span multiple
episodes. The lax.scan carry holds (env_state, agent_state, rng) so both
recurrent and stateless agents work without branching.

The returned Rollout stores the full agent_output pytree alongside transitions,
so the loss function can access whatever fields the agent produces (logits,
values, embeddings, …) without knowing the network architecture.
"""
from __future__ import annotations
from typing import NamedTuple, Any

import jax
import jax.numpy as jnp

from ..agents.base import Agent
from ..envs.base import Env, EnvState


class Episode(NamedTuple):
  """Trajectory collected from the environment.

  All arrays have a leading time axis T.
  Multi-player fields have an additional player axis P.
  agent_output is an agent-specific pytree (e.g. ActorCriticOutput) stacked
  over T and P — the algorithm extracts whatever fields it needs from it.
  """
  env_states:    Any      # (T,)             — environment states
  agent_states:  Any      # (T,)             — agent states
  legal_actions: Any      # (T, P)             — legal actions # Can be extracted from env_states
  infosets:      Any      # (T, P)             — infosets # Can be extracted from env_state
  dones:         jax.Array  # (T,)  bool         — True when episode ended
  rewards:       jax.Array  # (T, P)             — per-player rewards
  actions:       jax.Array  # (T, P)             — 0-indexed sampled actions
  agent_output:  Any        # (T, P, ...)        — full agent evaluate output



def collect_episodes(
  env:         Env,
  agent:       Agent,
  params:      Any,
  rng:         Any,
  batch_size: int,
) -> tuple[EnvState, Any, Any, list[Episode]]:
  """Collect episodes from the environment."""
  rng = jax.random.split(rng, batch_size)
  return jax.vmap(collect_episode, in_axes=(None, None, None, 0))(env, agent, params, rng)

def collect_episode(
  env:         Env,
  agent:       Agent,
  params:      Any,
  rng:         Any,
  agent_state: Any = None,
  env_state:   EnvState = None,
  rollout_len: None |  int = None,
) -> tuple[EnvState, Any, Any, Episode]:
  """Collect rollout_len steps from env under the current policy.

  Each step:
    1. Gets player observations and legal-action masks for all players.
    2. Calls agent.player_evaluate (batched over P players) to get logits,
       values, or any other output the agent produces.
    3. Applies legal-action masking, samples actions, records log-probs.
    4. Steps the env; auto-resets on episode end.

  For stateless agents pass agent_state=None; it is carried through unchanged.
  For recurrent agents the updated state is returned and threaded into the
  next step — each step processes all P players in one batched forward pass,
  so the state is shared across players (suitable for self-play with a single
  shared recurrent state).

  Args:
    env:         Environment (defines num_players, num_actions, etc.).
    agent:       Agent whose player_evaluate drives action selection.
    params:      Current agent parameters.
    agent_state: Current recurrent carry (None for stateless agents).
    env_state:   Starting environment state.
    rng:         PRNG key (threaded through the scan).
    rollout_len: Number of steps T to collect.

  Returns:
    new_env_state:   Environment state after the last step (possibly reset).
    new_agent_state: Updated recurrent carry (None for stateless agents).
    new_rng:         Updated PRNG key.
    rollout:         Rollout with all fields shaped (T, P, ...).
  """
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
    rng, act_key, env_key = jax.random.split(rng, 3)

    infosets = jax.vmap(env.information_set, in_axes=(None, 0))(env_state, jnp.arange(P))
    legal_actions = jax.vmap(env.legal_actions, in_axes=(None, 0))(env_state, jnp.arange(P))

    # Batched forward pass: obs (P, obs_dim) → agent_output with (P, ...) fields
    agent_out, new_agent_state = agent.player_evaluate(params, agent_state, infosets)

    logits_masked = jnp.where(legal_actions, agent_out.logits, -jnp.inf)
    player_keys   = jax.random.split(act_key, P)
    actions       = jax.vmap(jax.random.categorical)(player_keys, logits_masked) 

    new_env_state, rewards, done, _ = env.apply_action(env_state, actions, env_key)

    step = Episode(env_state, agent_state, legal_actions, infosets, done, rewards, actions, agent_out)
    return (new_env_state, new_agent_state, rng), step

  (new_env_state, new_agent_state, new_rng), episode = jax.lax.scan(
    scan_step, (env_state, agent_state, rng), None, length=rollout_len)
  return new_env_state, new_agent_state, new_rng, episode
