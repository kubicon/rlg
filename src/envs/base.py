"""Layer 3 — Environments.

All methods are pure functions: EnvState flows in, new EnvState flows out.
No Python-side mutation, no callbacks. Concrete state types must be registered
as JAX pytrees so they pass through jit/vmap/scan unchanged.

Simultaneous games:   all players act every step.
Sequential games:     current_player(state) identifies who acts. Inactive
                      players receive a legal_actions mask with exactly one
                      True entry (a dummy no-op), which forces a deterministic
                      policy distribution and contributes zero policy gradient.
"""
from __future__ import annotations
import abc
from typing import Any

import jax
import jax.numpy as jnp

PRNGKey    = jax.Array
Action     = Any          # shape (num_players, *action_shape)
Observation = Any
Info       = dict[str, jax.Array]
EnvState   = Any          # concrete envs define their own pytree state


class Env(abc.ABC):
  """Abstract JAX environment for single- or multi-player games."""

  # ── Static properties (known at construction, used for network sizing) ──

  @property
  @abc.abstractmethod
  def num_players(self) -> int:
    """Number of players in the game."""

  @property
  @abc.abstractmethod
  def max_length(self) -> int:
    """Maximum number of steps per episode."""

  @property
  @abc.abstractmethod
  def num_actions(self) -> int:
    """Size of the action space. Static — used to size policy network outputs."""

  # ── State lifecycle ─────────────────────────────────────────────────────

  @abc.abstractmethod
  def init_state(self, key: PRNGKey) -> EnvState:
    """Sample a fresh initial state."""

  @abc.abstractmethod
  def apply_action(
    self,
    state:   EnvState,
    actions: Action,
    key:     PRNGKey,
  ) -> tuple[EnvState, jax.Array, jax.Array, Info]:
    """Advance the environment by one step.

    actions:  shape (num_players, *action_shape). In sequential games,
              actions from players other than current_player are ignored.
    Returns:
      next_state: EnvState
      rewards:    jax.Array of shape (num_players,)
      done:       jax.Array scalar bool
      info:       dict of auxiliary arrays
    """

  # ── Observations ────────────────────────────────────────────────────────

  @abc.abstractmethod
  def player_observation(self, state: EnvState, player_id: jax.Array) -> Observation:
    """Private observation for player_id (what only that player can see).

    player_id is a JAX scalar so this can be vmapped across all players:
      obs = jax.vmap(lambda p: env.player_observation(state, p))(
                jnp.arange(env.num_players))
    """

  @abc.abstractmethod
  def public_observation(self, state: EnvState) -> Observation:
    """Observation available to every player (common knowledge)."""

  @abc.abstractmethod
  def state_observation(self, state: EnvState) -> Observation:
    """Full ground-truth state — uniquely identifies the underlying game state.

    Not available to agents during play. Intended for value functions with
    privileged information (e.g. opponent hand in card games) or debugging.
    """

  # ── Action legality & turn order ────────────────────────────────────────

  @abc.abstractmethod
  def legal_actions(self, state: EnvState, player_id: jax.Array) -> jax.Array:
    """Boolean mask of shape (num_actions,) indicating which actions are legal.

    Sequential games must return jnp.zeros(num_actions, bool).at[0].set(True)
    for players who are not current_player — a single legal action forces a
    deterministic distribution and zeroes the policy gradient for that step.
    """

  # ── Perfect-recall representations ──────────────────────────────────────

  @abc.abstractmethod
  def information_set(self, state: EnvState, player_id: jax.Array) -> Observation:
    """Ordered history of everything player_id has observed and done.

    Unlike player_observation (a snapshot), this encodes the full sequence of
    past actions and observations — the information set in the game-theoretic
    sense. Two histories that lead to the same information set are strategically
    identical for that player.
    """

  @abc.abstractmethod
  def public_state(self, state: EnvState) -> Observation:
    """Ordered history of all public information so far.

    Unlike public_observation (a snapshot), this preserves the sequence of
    public events — prize cards, results, and any information revealed by the
    resolution of each turn (e.g. cards played in a draw).
    """

  @abc.abstractmethod
  def state_representation(self, state: EnvState) -> Observation:
    """Ordered history of the full ground-truth game trajectory.

    Uniquely identifies the entire history, including all hidden actions.
    Intended for value functions, debugging, and counterfactual reasoning.
    """

  def current_player(self, state: EnvState) -> jax.Array:
    """Index of the player to move this step.

    Returns -1 for simultaneous-move games (default).
    Override for sequential-move games; the training loop uses this to build
    the correct legal_actions masks without any Python branching.
    """
    return jnp.full((), -1, dtype=jnp.int32)
