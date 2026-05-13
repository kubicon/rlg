"""Gaussian observation-noise wrapper for any Env.

Wraps an existing environment and adds zero-mean Gaussian noise (with a
configurable variance) to every observation produced by the six observation
methods.  All other behaviour (transitions, rewards, legal actions, turn
order) is forwarded unchanged.

Usage::

    env = NoisyEnv(Goofspiel(n_cards=5), variance=0.1)
    state = env.init_state(key)
    obs = env.information_set(state, player_id=0, key=obs_key)

Each call to an observation method produces independent noise drawn from
N(0, variance) as long as a distinct key is passed.  Passing the same key
twice for the same state and method will return identical results.
"""

from __future__ import annotations
from typing import NamedTuple, Any

import jax
import jax.numpy as jnp

from .base import Env, PRNGKey, Info, Action, Observation


class NoisyEnvState(NamedTuple):
  inner: Any


class NoisyEnv(Env):
  """Wraps any Env and adds N(0, variance) noise to all observation methods.

  Args:
    env:      The underlying environment to wrap.
    variance: Variance of the additive Gaussian noise (std = sqrt(variance)).
  """

  def __init__(self, env: Env, variance: float) -> None:
    self._env = env
    self._std = jnp.sqrt(jnp.asarray(variance, dtype=jnp.float32))

  def _noisy(self, obs: jax.Array, key: PRNGKey) -> jax.Array:
    return obs + self._std * jax.random.normal(key, obs.shape, dtype=obs.dtype)

  # ── Static properties ────────────────────────────────────────────────────

  @property
  def num_players(self) -> int:
    return self._env.num_players

  @property
  def max_length(self) -> int:
    return self._env.max_length

  @property
  def num_actions(self) -> int:
    return self._env.num_actions

  @property
  def max_reward(self) -> float:
    return self._env.max_reward

  # ── State lifecycle ──────────────────────────────────────────────────────

  def init_state(self, key: PRNGKey) -> NoisyEnvState:
    return NoisyEnvState(inner=self._env.init_state(key))

  def apply_action(
    self,
    state: NoisyEnvState,
    actions: Action,
    key: PRNGKey,
  ) -> tuple[NoisyEnvState, jax.Array, jax.Array, Info]:
    new_inner, rewards, done, info = self._env.apply_action(state.inner, actions, key)
    return NoisyEnvState(inner=new_inner), rewards, done, info

  # ── Observations — each call produces independent noise via key ──────────

  def player_observation(
    self, state: NoisyEnvState, player_id: jax.Array, key: PRNGKey
  ) -> Observation:
    obs = self._env.player_observation(state.inner, player_id, key)
    return self._noisy(obs, jax.random.fold_in(key, 0))

  def public_observation(self, state: NoisyEnvState, key: PRNGKey) -> Observation:
    obs = self._env.public_observation(state.inner, key)
    return self._noisy(obs, jax.random.fold_in(key, 1))

  def state_observation(self, state: NoisyEnvState, key: PRNGKey) -> Observation:
    obs = self._env.state_observation(state.inner, key)
    return self._noisy(obs, jax.random.fold_in(key, 2))

  def information_set(
    self, state: NoisyEnvState, player_id: jax.Array | int, key: PRNGKey
  ) -> Observation:
    obs = self._env.information_set(state.inner, player_id, key)
    noise_key = jax.random.fold_in(jax.random.fold_in(key, 3), player_id)
    return self._noisy(obs, noise_key)

  def public_state(self, state: NoisyEnvState, key: PRNGKey) -> Observation:
    obs = self._env.public_state(state.inner, key)
    return self._noisy(obs, jax.random.fold_in(key, 4))

  def state_representation(self, state: NoisyEnvState, key: PRNGKey) -> Observation:
    obs = self._env.state_representation(state.inner, key)
    return self._noisy(obs, jax.random.fold_in(key, 5))

  # ── Action legality & turn order ─────────────────────────────────────────

  def legal_actions(
    self, state: NoisyEnvState, player_id: jax.Array | int
  ) -> jax.Array:
    return self._env.legal_actions(state.inner, player_id)

  def current_player(self, state: NoisyEnvState) -> jax.Array:
    return self._env.current_player(state.inner)
