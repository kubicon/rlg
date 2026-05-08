"""Layer — Agent abstract base.

An Agent is the bridge between an Algorithm and its network(s). It owns the
network construction and exposes three evaluate methods, one per observation
type defined by the environment:

  player_evaluate   — takes infoset or player_observation (what one player sees)
  public_evaluate   — takes public_state or public_observation (common knowledge)
  state_evaluate    — takes state_representation or state_observation (privileged)

Each method returns (output_pytree, new_agent_state). The output_pytree is
agent-specific (e.g. ActorCriticOutput) and carries everything the Algorithm
or loss function needs — logits, values, embeddings, etc.

An Agent may contain multiple networks (separate actor/critic, privileged critic
alongside a player network, etc.). The Algorithm never imports a concrete network
class; it only speaks the Agent interface.

Stateless agents return None from init_state and accept None in evaluate calls.
Recurrent agents carry an array-valued state (LSTM cell/hidden, etc.) through
each evaluate call. For multi-player self-play the caller is responsible for
managing per-player state if required.
"""

from __future__ import annotations
import abc
from typing import Any

import jax


class Agent(abc.ABC):
  """Bridge between an Algorithm and one or more networks."""

  @abc.abstractmethod
  def init_params(self, key: jax.Array, dummy_obs: Any) -> Any:
    """Initialise and return all network parameters."""

  @abc.abstractmethod
  def init_state(self, params: Any) -> Any:
    """Return the initial recurrent carry (None for stateless agents)."""

  @abc.abstractmethod
  def player_evaluate(self, params: Any, state: Any, obs: Any) -> tuple[Any, Any]:
    """Forward pass on player-private information.

    obs is player_observation or information_set — what exactly one player
    can see. obs may be batched (e.g. shape (P, obs_dim) for P players).

    Returns (output_pytree, new_state).
    """

  @abc.abstractmethod
  def public_evaluate(self, params: Any, state: Any, obs: Any) -> tuple[Any, Any]:
    """Forward pass on public information.

    obs is public_observation or public_state — common knowledge shared by
    all players. Useful for a centralised critic or a world model.

    Returns (output_pytree, new_state).
    """

  @abc.abstractmethod
  def state_evaluate(self, params: Any, state: Any, obs: Any) -> tuple[Any, Any]:
    """Forward pass on privileged ground-truth information.

    obs is state_observation or state_representation — the full hidden state,
    not available to agents during play. Intended for oracle critics, debugging,
    or counterfactual reasoning.

    Returns (output_pytree, new_state).
    """
