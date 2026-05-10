"""Normal-form (matrix) games.

Single-shot 2-player simultaneous-move games.

Observation sizes (float32):
  player_observation : 2   player_id one-hot (allows asymmetric strategies)
  public_observation : 1   constant zero
  state_observation  : 1   constant zero

There is no history or private signal, so the player-id one-hot is the only
information needed for an agent to learn the correct mixed strategy per role.
"""

from __future__ import annotations
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .base import Env, Info, PRNGKey


class NormalFormState(NamedTuple):
  done: jax.Array  # () bool


class NormalFormGame(Env):
  """Single-shot 2-player simultaneous-move game defined by a reward matrix.

  Args:
      matrix: array-like of shape (n_actions, n_actions, 2).
              matrix[a0, a1, p] is player p's reward when p0 plays a0
              and p1 plays a1.
  """

  def __init__(self, matrix) -> None:
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim != 3 or matrix.shape[2] != 2:
      raise ValueError("matrix must have shape (n_actions, n_actions, 2)")
    if matrix.shape[0] != matrix.shape[1]:
      raise ValueError("matrix must be square (same action count for both players)")
    self._matrix = jnp.array(matrix)
    self._n = int(matrix.shape[0])
    self._max_reward = float(jnp.max(matrix))

  # ── Static properties ────────────────────────────────────────────────────

  @property
  def num_players(self) -> int:
    return 2

  @property
  def max_length(self) -> int:
    return 1

  @property
  def num_actions(self) -> int:
    return self._n

  @property
  def max_reward(self) -> float:
    return self._max_reward

  # ── State lifecycle ──────────────────────────────────────────────────────

  def init_state(self, key: PRNGKey) -> NormalFormState:
    return NormalFormState(done=jnp.bool_(False))

  def apply_action(
    self,
    state: NormalFormState,
    actions: jax.Array,  # (2,) int32
    key: PRNGKey,
  ) -> tuple[NormalFormState, jax.Array, jax.Array, Info]:
    rewards = self._matrix[actions[0], actions[1]]  # (2,)
    done = jnp.bool_(True)
    return NormalFormState(done=done), rewards, done, {}

  # ── Observations ────────────────────────────────────────────────────────

  def player_observation(
    self, state: NormalFormState, player_id: jax.Array
  ) -> jax.Array:
    return jax.nn.one_hot(player_id, self.num_players, dtype=jnp.float32)

  def public_observation(self, state: NormalFormState) -> jax.Array:
    return jnp.zeros(1, dtype=jnp.float32)

  def state_observation(self, state: NormalFormState) -> jax.Array:
    return jnp.zeros(1, dtype=jnp.float32)

  # ── Action legality & turn order ────────────────────────────────────────

  def legal_actions(
    self, state: NormalFormState, player_id: jax.Array | int
  ) -> jax.Array:
    return jnp.ones(self._n, dtype=jnp.bool_)

  # ── Perfect-recall representations ──────────────────────────────────────

  def information_set(
    self, state: NormalFormState, player_id: jax.Array | int
  ) -> jax.Array:
    return self.player_observation(state, jnp.int32(player_id))

  def public_state(self, state: NormalFormState) -> jax.Array:
    return self.public_observation(state)

  def state_representation(self, state: NormalFormState) -> jax.Array:
    return self.state_observation(state)


# ── Concrete games ────────────────────────────────────────────────────────────


def _zero_sum(p0: np.ndarray) -> np.ndarray:
  """Build a (n, n, 2) matrix from player 0's (n, n) payoff table."""
  p0 = np.asarray(p0, dtype=np.float32)
  return np.stack([p0, -p0], axis=-1)


_RPS_P0 = np.array(
  [
    # Rock  Paper  Scissors
    [0.0, -1.0, 1.0],  # Rock
    [1.0, 0.0, -1.0],  # Paper
    [-1.0, 1.0, 0.0],  # Scissors
  ],
  dtype=np.float32,
)

_MP_P0 = np.array(
  [
    # Heads  Tails
    [1.0, -1.0],  # Heads
    [-1.0, 1.0],  # Tails
  ],
  dtype=np.float32,
)


class RockPaperScissors(NormalFormGame):
  """Rock-Paper-Scissors. Actions: 0=Rock, 1=Paper, 2=Scissors."""

  def __init__(self) -> None:
    super().__init__(_zero_sum(_RPS_P0))


class MatchingPennies(NormalFormGame):
  """Matching Pennies. Actions: 0=Heads, 1=Tails.

  Player 0 wins (gets +1) on a match; player 1 wins on a mismatch.
  """

  def __init__(self) -> None:
    super().__init__(_zero_sum(_MP_P0))


class BiasedMatchingPennies(NormalFormGame):
  """Matching Pennies with an inflated reward for the Tails-Tails outcome.

  Args:
      bias: amount added to player 0's Tails-Tails payoff (and subtracted
            from player 1's), shifting the Nash equilibrium away from 50/50.
  """

  def __init__(self, bias: float = 1.0) -> None:
    p0 = _MP_P0.copy()
    p0[1, 1] += bias
    super().__init__(_zero_sum(p0))
