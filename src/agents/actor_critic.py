"""Actor-critic agent: one shared (or separate) network for actor and critic.

ActorCriticOutput is the pytree returned by all three evaluate methods.
For this agent the same network is used regardless of observation type —
the caller decides which observation to pass in.

A more specialised agent (e.g. one with a privileged critic that receives
state_observation) would override state_evaluate to use a different network.
"""

from __future__ import annotations
from typing import NamedTuple, Any

import jax
import flax.linen as nn

from .base import Agent


class ActorCriticOutput(NamedTuple):
  """Output pytree from a single actor-critic forward pass.

  logits: raw (unmasked) action logits — shape (..., n_actions).
  value:  critic estimate V(s)         — shape (...,).
  """

  logits: jax.Array
  value: jax.Array


class QActorCriticOutput(NamedTuple):
  """Output pytree from an actor-critic forward pass with an explicit Q-head.

  logits:   raw action logits                   — shape (..., n_actions).
  value:    V(s) critic estimate                 — shape (...,).
  q_values: Q(s, :) values for all actions      — shape (..., n_actions).
  """

  logits: jax.Array
  value: jax.Array
  q_values: jax.Array


class ActorCriticAgent(Agent):
  """Actor-critic agent backed by a single TwinHead-compatible network.

  All three evaluate methods run the same forward pass; the caller is
  responsible for passing the appropriate observation type.

  Args:
    network: TwinHead-compatible module: (obs, state) -> ((logits, value), state).
             Expects CategoricalHead (logits) and ValueHead (value) as the two heads.
  """

  def __init__(self, network: nn.Module) -> None:
    self.network = network

  # ── Lifecycle ─────────────────────────────────────────────────────────────

  def init_params(self, key: jax.Array, dummy_obs: Any) -> Any:
    return self.network.init_params(key, dummy_obs)

  def init_state(self, params: Any) -> Any:
    """Returns the learned initial recurrent state, or None for stateless nets."""
    return self.network.init_state(params)

  # ── Evaluate methods ──────────────────────────────────────────────────────

  def _forward(
    self, params: Any, state: Any, obs: Any
  ) -> tuple[ActorCriticOutput, Any]:
    (logits, value), new_state = self.network.apply({"params": params}, obs, state)
    return ActorCriticOutput(logits=logits, value=value), new_state

  def player_evaluate(
    self, params: Any, state: Any, obs: Any
  ) -> tuple[ActorCriticOutput, Any]:
    """Evaluate on player_observation or information_set."""
    return self._forward(params, state, obs)

  def public_evaluate(
    self, params: Any, state: Any, obs: Any
  ) -> tuple[ActorCriticOutput, Any]:
    """Evaluate on public_observation or public_state."""
    return self._forward(params, state, obs)

  def state_evaluate(
    self, params: Any, state: Any, obs: Any
  ) -> tuple[ActorCriticOutput, Any]:
    """Evaluate on state_observation or state_representation."""
    return self._forward(params, state, obs)


class PolicyQOutput(NamedTuple):
  """Output pytree from a policy + Q-value forward pass (no V-head).

  logits:   raw action logits             — shape (..., n_actions).
  q_values: Q(s, :) for all actions       — shape (..., n_actions).
  """

  logits: jax.Array
  q_values: jax.Array


class PolicyQAgent(Agent):
  """Agent backed by a TwinHead network with a policy head and a Q-head.

  head1 must be CategoricalHead (logits); head2 must be QHead (Q-values).
  There is no V-head — the state value is derived on-the-fly as E_π[Q(s,:)].

  Args:
    network: TwinHead-compatible module: (obs, state) -> ((logits, q_values), state).
  """

  def __init__(self, network: nn.Module) -> None:
    self.network = network

  def init_params(self, key: jax.Array, dummy_obs: Any) -> Any:
    return self.network.init_params(key, dummy_obs)

  def init_state(self, params: Any) -> Any:
    return self.network.init_state(params)

  def _forward(
    self, params: Any, state: Any, obs: Any
  ) -> tuple[PolicyQOutput, Any]:
    (logits, q_values), new_state = self.network.apply({"params": params}, obs, state)
    return PolicyQOutput(logits=logits, q_values=q_values), new_state

  def player_evaluate(
    self, params: Any, state: Any, obs: Any
  ) -> tuple[PolicyQOutput, Any]:
    return self._forward(params, state, obs)

  def public_evaluate(
    self, params: Any, state: Any, obs: Any
  ) -> tuple[PolicyQOutput, Any]:
    return self._forward(params, state, obs)

  def state_evaluate(
    self, params: Any, state: Any, obs: Any
  ) -> tuple[PolicyQOutput, Any]:
    return self._forward(params, state, obs)


class QActorCriticAgent(Agent):
  """Actor-critic agent backed by a TripleHead network (logits, V, Q).

  The network must return a 3-tuple (logits, value, q_values) from its forward
  pass. Use with the ``triple_head`` network config.

  Args:
    network: TripleHead-compatible module: (obs, state) -> ((logits, value, q_values), state).
  """

  def __init__(self, network: nn.Module) -> None:
    self.network = network

  def init_params(self, key: jax.Array, dummy_obs: Any) -> Any:
    return self.network.init_params(key, dummy_obs)

  def init_state(self, params: Any) -> Any:
    return self.network.init_state(params)

  def _forward(
    self, params: Any, state: Any, obs: Any
  ) -> tuple[QActorCriticOutput, Any]:
    (logits, value, q_values), new_state = self.network.apply(
      {"params": params}, obs, state
    )
    return QActorCriticOutput(logits=logits, value=value, q_values=q_values), new_state

  def player_evaluate(
    self, params: Any, state: Any, obs: Any
  ) -> tuple[QActorCriticOutput, Any]:
    return self._forward(params, state, obs)

  def public_evaluate(
    self, params: Any, state: Any, obs: Any
  ) -> tuple[QActorCriticOutput, Any]:
    return self._forward(params, state, obs)

  def state_evaluate(
    self, params: Any, state: Any, obs: Any
  ) -> tuple[QActorCriticOutput, Any]:
    return self._forward(params, state, obs)
