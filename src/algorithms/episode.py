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

  env_states: Any  # (T,)             — environment states
  agent_states: Any  # (T,)             — agent states
  legal_actions: (
    Any  # (T, P)             — legal actions # Can be extracted from env_states
  )
  infosets: Any  # (T, P)             — infosets # Can be extracted from env_state
  dones: jax.Array  # (T,)  bool         — True when episode ended
  rewards: jax.Array  # (T, P)             — per-player rewards
  actions: jax.Array  # (T, P)             — 0-indexed sampled actions
  agent_output: Any  # (T, P, ...)        — full agent evaluate output


def collect_episodes(
  env: Env,
  agent: Agent,
  params: Any,
  rng: Any,
  batch_size: int,
  normalize_rewards: bool = True,
  opp_params: Any = None,
  br_player: int | None = None,
) -> tuple[EnvState, Any, Any, Episode]:
  """Collect a batch of episodes from the environment.

  When opp_params and br_player are given, params is used only for br_player
  and opp_params drives all other players (best-response mode).
  """
  rng = jax.random.split(rng, batch_size)
  return jax.vmap(
    collect_episode,
    in_axes=(None, None, None, 0, None, None, None, None, None, None),
  )(env, agent, params, rng, None, None, None, normalize_rewards, opp_params, br_player)


def collect_episodes_br(
  env: Env,
  agent: Agent,
  br_params: Any,
  opp_params: Any,
  br_player: int,
  rng: Any,
  batch_size: int,
  normalize_rewards: bool = True,
) -> tuple[EnvState, Any, Any, Episode]:
  """Convenience wrapper: collect episodes with a fixed opponent policy."""
  return collect_episodes(
    env,
    agent,
    br_params,
    rng,
    batch_size,
    normalize_rewards,
    opp_params=opp_params,
    br_player=br_player,
  )


def collect_episode(
  env: Env,
  agent: Agent,
  params: Any,
  rng: Any,
  agent_state: Any = None,
  env_state: EnvState = None,
  rollout_len: None | int = None,
  normalize_rewards: bool = True,
  opp_params: Any = None,
  br_player: int | None = None,
) -> tuple[EnvState, Any, Any, Episode]:
  """Collect rollout_len steps from env under the current policy.

  Each step:
    1. Gets player observations and legal-action masks for all players.
    2. Calls agent.player_evaluate (batched over P players) to get logits,
       values, or any other output the agent produces.
    3. Applies legal-action masking, samples actions, records log-probs.
    4. Steps the env; auto-resets on episode end.

  When opp_params and br_player are provided the episode is collected in
  best-response mode: params drives br_player, opp_params drives all others.
  agent_output in the returned Episode always reflects params (br_player's
  network), so the BR algorithm can read its slice without special-casing.

  For stateless agents pass agent_state=None; it is carried through unchanged.
  For recurrent agents the updated state is returned and threaded into the
  next step — each step processes all P players in one batched forward pass,
  so the state is shared across players (suitable for self-play with a single
  shared recurrent state).

  Args:
    env:               Environment (defines num_players, num_actions, etc.).
    agent:             Agent whose player_evaluate drives action selection.
    params:            Current agent parameters (or br_params in BR mode).
    rng:               PRNG key (threaded through the scan).
    agent_state:       Current recurrent carry (None for stateless agents).
    env_state:         Starting environment state.
    rollout_len:       Number of steps T to collect.
    normalize_rewards: If True, divide rewards by ``env.max_reward`` so they
                       lie in [-1, 1].
    opp_params:        Fixed opponent parameters. When set, br_player must
                       also be provided.
    br_player:         Player index trained by params; all others use opp_params.

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
    rng, act_key, env_key, obs_key = jax.random.split(rng, 4)
    obs_keys = jax.random.split(obs_key, P)

    infosets = jax.vmap(env.information_set, in_axes=(None, 0, 0))(
      env_state, jnp.arange(P), obs_keys
    )
    legal_actions = jax.vmap(env.legal_actions, in_axes=(None, 0))(
      env_state, jnp.arange(P)
    )

    # Batched forward pass: obs (P, obs_dim) → agent_output with (P, ...) fields
    agent_out, new_agent_state = agent.player_evaluate(params, agent_state, infosets)

    if opp_params is None:
      logits_masked = jnp.where(legal_actions, agent_out.logits, -jnp.inf)
    else:
      opp_out, _ = agent.player_evaluate(opp_params, agent_state, infosets)
      is_br = jnp.arange(P) == br_player
      mixed_logits = jnp.where(is_br[:, None], agent_out.logits, opp_out.logits)
      logits_masked = jnp.where(legal_actions, mixed_logits, -jnp.inf)

    player_keys = jax.random.split(act_key, P)
    actions = jax.vmap(jax.random.categorical)(player_keys, logits_masked)

    new_env_state, rewards, done, _ = env.apply_action(env_state, actions, env_key)

    if normalize_rewards:
      rewards = rewards / env.max_reward

    step = Episode(
      env_state, agent_state, legal_actions, infosets, done, rewards, actions, agent_out
    )
    return (new_env_state, new_agent_state, rng), step

  (new_env_state, new_agent_state, new_rng), episode = jax.lax.scan(
    scan_step, (env_state, agent_state, rng), None, length=rollout_len
  )
  return new_env_state, new_agent_state, new_rng, episode
