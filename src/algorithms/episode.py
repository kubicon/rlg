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
from ..utils import safe_log_softmax


class Episode(NamedTuple):
  """Trajectory collected from the environment.

  All arrays have a leading time axis T.
  Multi-player fields have an additional player axis P.
  agent_output is an agent-specific pytree (e.g. ActorCriticOutput) stacked
  over T and P — the algorithm extracts whatever fields it needs from it.
  """

  env_states: Any        # (T,)       — environment states
  agent_states: Any      # (T,)       — agent states
  legal_actions: Any     # (T, P)     — legal actions
  infosets: Any          # (T, P)     — per-player information sets
  state_reps: Any        # (T, ...)   — ground-truth state representations
  dones: jax.Array       # (T,)  bool — True when episode ended
  rewards: jax.Array     # (T, P)     — per-player rewards
  actions: jax.Array     # (T, P)     — 0-indexed sampled actions
  agent_output: Any      # (T, P, ...) — full agent evaluate output
  behavior_log_probs: jax.Array  # (T, P, A) — log μ(·), behavior (sampling)
                                 # policy NORMALISED log-probs over all actions
                                 # (logsumexp over legal = 0; −inf on illegal).
                                 # Not raw logits: gather directly for log μ(a).


def importance_weights(
  behavior_log_probs: jax.Array,  # (T, P, A) — log μ(·), behavior policy
  logits: jax.Array,              # (T, P, A) — raw π_old logits
  legal_actions: jax.Array,       # (T, P, A)
  actions: jax.Array,             # (T, P)
) -> tuple[jax.Array, jax.Array]:
  """Off-policy importance weights for a single trajectory.

  Relates the trajectory-sampling behavior policy μ to the on-policy π_old that
  produced the logits. Computed in log-space for numerical stability; every
  quantity is 1 when on-policy (μ = π_old, e.g. epsilon=0). vmap over the batch
  axis to apply to a batch of trajectories.

  In a simultaneous-move game the transition at step t (reward and next state)
  is determined by the JOINT action of all players, so its off-policy correction
  is the product over players of π_old/μ — not the per-player ratio. The acting
  player's own factor pairs with its score function ∇log π in the policy loss;
  the remaining (opponent) factors correct the realised return for the opponents'
  exploratory draws. Non-acting players in sequential games have a single forced
  legal action, so their per-player ratio is exactly 1 and the joint product is
  correct for sequential games too — no acting-player mask required.

  Returns:
    joint_ratio:  (T,)   — per-step JOINT action ratio ∏_p π_old(a_p)/μ(a_p).
                  Serves as the forward vtrace correction (transition) and, with
                  the score function supplying its own factor, the local
                  policy-gradient factor.
    reach_weight: (T,)   — unclipped joint (all-players) exclusive prefix product
                  of π_old/μ: the probability of reaching each timestep under
                  π_old relative to μ (the backward / reach correction).
  """
  behavior_logp = jnp.take_along_axis(
    behavior_log_probs, actions[..., None], axis=-1
  )[..., 0]  # log μ(a) of the taken action (T, P)
  pi_old_logp = jnp.take_along_axis(
    safe_log_softmax(logits, legal_actions), actions[..., None], axis=-1
  )[..., 0]  # log π_old(a) (T, P)

  log_ratio = pi_old_logp - behavior_logp  # (T, P)

  joint_log_ratio = log_ratio.sum(-1)  # (T,) — sum over players (joint action)
  joint_ratio = jnp.exp(joint_log_ratio)  # (T,) — ∏_p π_old/μ
  reach_log = jnp.cumulative_sum(
    joint_log_ratio, axis=0, include_initial=True
  )[:-1]  # exclusive prefix
  reach_weight = jnp.exp(reach_log)  # (T,)
  return joint_ratio, reach_weight


def collect_episodes(
  env: Env,
  agent: Agent,
  params: Any,
  rng: Any,
  batch_size: int,
  normalize_rewards: bool = True,
  opp_params: Any = None,
  br_player: int | None = None,
  epsilon: float | jax.Array = 0.0,
) -> tuple[EnvState, Any, Any, Episode]:
  """Collect a batch of episodes from the environment.

  When opp_params and br_player are given, params is used only for br_player
  and opp_params drives all other players (best-response mode).

  epsilon controls off-policy exploration: actions are sampled from the mixture
  behavior policy μ = (1 - epsilon)·π + epsilon·Uniform(legal). epsilon=0 is
  on-policy (μ = π); epsilon=1 samples uniformly over legal actions.
  """
  rng = jax.random.split(rng, batch_size)
  return jax.vmap(
    collect_episode,
    in_axes=(None, None, None, 0, None, None, None, None, None, None, None),
  )(env, agent, params, rng, None, None, None, normalize_rewards, opp_params, br_player, epsilon)


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
  epsilon: float | jax.Array = 0.0,
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
    rng, act_key, env_key, obs_key, state_key = jax.random.split(rng, 5)
    obs_keys = jax.random.split(obs_key, P)

    infosets = jax.vmap(env.information_set, in_axes=(None, 0, 0))(
      env_state, jnp.arange(P), obs_keys
    )
    state_rep = env.state_representation(env_state, state_key)
    legal_actions = jax.vmap(env.legal_actions, in_axes=(None, 0))(
      env_state, jnp.arange(P)
    )

    # Construct obs for this agent (infosets only, or richer tuple for privileged agents)
    obs = agent.make_obs_step(infosets, state_rep)

    # Batched forward pass: obs (P, obs_dim) → agent_output with (P, ...) fields
    agent_out, new_agent_state = agent.player_evaluate(params, agent_state, obs)

    if opp_params is None:
      logits_masked = jnp.where(legal_actions, agent_out.logits, -jnp.inf)
    else:
      opp_obs = agent.make_obs_step(infosets, state_rep)
      opp_out, _ = agent.player_evaluate(opp_params, agent_state, opp_obs)
      is_br = jnp.arange(P) == br_player
      mixed_logits = jnp.where(is_br[:, None], agent_out.logits, opp_out.logits)
      logits_masked = jnp.where(legal_actions, mixed_logits, -jnp.inf)

    # Behavior policy μ = (1 - epsilon)·π + epsilon·Uniform(legal). Sample from μ
    # and store its full NORMALISED log-probs (log μ over all actions) so the
    # algorithm can apply importance-sampling corrections — the per-action
    # log-prob is recovered by gathering at the taken action when needed. At
    # epsilon=0, μ = π and the log-probs equal log_softmax(logits_masked) (up to
    # a constant shift, to which categorical sampling is invariant), reproducing
    # on-policy sampling exactly.
    pi = jax.nn.softmax(logits_masked, axis=-1)  # (P, A), 0 on illegal
    n_legal = legal_actions.sum(-1, keepdims=True)
    uniform = legal_actions / n_legal  # (P, A), uniform over legal actions
    mu = (1.0 - epsilon) * pi + epsilon * uniform  # (P, A)
    behavior_log_probs = jnp.where(
      legal_actions, jnp.log(jnp.where(legal_actions, mu, 1.0)), -jnp.inf
    )  # (P, A) — log μ(·), normalised (logsumexp over legal = 0)

    player_keys = jax.random.split(act_key, P)
    actions = jax.vmap(jax.random.categorical)(player_keys, behavior_log_probs)

    new_env_state, rewards, done, _ = env.apply_action(env_state, actions, env_key)

    if normalize_rewards:
      rewards = rewards / env.max_reward

    step = Episode(
      env_state, agent_state, legal_actions, infosets, state_rep, done, rewards,
      actions, agent_out, behavior_log_probs,
    )
    return (new_env_state, new_agent_state, rng), step

  (new_env_state, new_agent_state, new_rng), episode = jax.lax.scan(
    scan_step, (env_state, agent_state, rng), None, length=rollout_len
  )
  return new_env_state, new_agent_state, new_rng, episode
