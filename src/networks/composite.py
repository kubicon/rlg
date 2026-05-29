from __future__ import annotations
from typing import Any

import jax
import flax.linen as nn

from .base import Head, Torso


class TwinHead(nn.Module):
  """Shared torso whose features feed two independent heads.

  Returns ((out1, out2), new_state). Use for any task that produces two
  outputs from a common representation — actor-critic, value/advantage
  decomposition, multi-task, etc.
  """

  torso: Torso
  head1: Head
  head2: Head

  def __call__(self, x: Any, state: Any) -> tuple[tuple[Any, Any], Any]:
    features, new_state = self.torso(x, state)
    return (self.head1(features), self.head2(features)), new_state

  def _zero_state(self) -> Any:
    return self.torso._zero_state()

  def init_state(self, params: Any) -> Any:
    return self.torso.init_state(params["torso"])

  def init_params(self, key: jax.Array, x: Any) -> Any:
    return self.init(key, x, self._zero_state())["params"]


class TripleHead(nn.Module):
  """Shared torso whose features feed three independent heads.

  Returns ((out1, out2, out3), new_state). Designed for actor-critic with an
  explicit Q-head: head1=CategoricalHead (logits), head2=ValueHead (V),
  head3=QHead (Q-values per action).
  """

  torso: Torso
  head1: Head
  head2: Head
  head3: Head

  def __call__(self, x: Any, state: Any) -> tuple[tuple[Any, Any, Any], Any]:
    features, new_state = self.torso(x, state)
    return (self.head1(features), self.head2(features), self.head3(features)), new_state

  def _zero_state(self) -> Any:
    return self.torso._zero_state()

  def init_state(self, params: Any) -> Any:
    return self.torso.init_state(params["torso"])

  def init_params(self, key: jax.Array, x: Any) -> Any:
    return self.init(key, x, self._zero_state())["params"]


class SplitTwinHead(nn.Module):
  """Two independent (torso, head) pathways fed from different observations.

  actor_torso + actor_head process actor_obs (e.g. information set).
  critic_torso + critic_head process critic_obs (e.g. ground-truth state).

  Input x must be a tuple (actor_obs, critic_obs).
  Returns ((actor_out, critic_out), (new_actor_state, new_critic_state)).

  Use with PrivilegedPolicyQAgent to train the Q-critic on privileged state
  information while the actor still sees only the player's information set.
  """

  actor_torso: Torso
  actor_head: Head
  critic_torso: Torso
  critic_head: Head

  def __call__(
    self, x: tuple[Any, Any], state: tuple[Any, Any]
  ) -> tuple[tuple[Any, Any], tuple[Any, Any]]:
    actor_obs, critic_obs = x
    state_actor, state_critic = state
    actor_features, new_state_actor = self.actor_torso(actor_obs, state_actor)
    critic_features, new_state_critic = self.critic_torso(critic_obs, state_critic)
    return (
      self.actor_head(actor_features),
      self.critic_head(critic_features),
    ), (new_state_actor, new_state_critic)

  def _zero_state(self) -> tuple[Any, Any]:
    return self.actor_torso._zero_state(), self.critic_torso._zero_state()

  def init_state(self, params: Any) -> tuple[Any, Any]:
    return (
      self.actor_torso.init_state(params["actor_torso"]),
      self.critic_torso.init_state(params["critic_torso"]),
    )

  def init_params(self, key: jax.Array, x: Any) -> Any:
    return self.init(key, x, self._zero_state())["params"]


class SeparateTwinHead(nn.Module):
  """Two fully independent (torso, head) pathways, each with its own parameters.

  Returns ((out1, out2), (new_state1, new_state2)). Use when the two outputs
  should not share any representation — e.g. twin critics in SAC/TD3, or
  actor and critic with entirely different architectures.
  """

  torso1: Torso
  head1: Head
  torso2: Torso
  head2: Head

  def __call__(
    self, x: Any, state: tuple[Any, Any]
  ) -> tuple[tuple[Any, Any], tuple[Any, Any]]:
    state1, state2 = state
    features1, new_state1 = self.torso1(x, state1)
    features2, new_state2 = self.torso2(x, state2)
    return (self.head1(features1), self.head2(features2)), (new_state1, new_state2)

  def _zero_state(self) -> tuple[Any, Any]:
    return self.torso1._zero_state(), self.torso2._zero_state()

  def init_state(self, params: Any) -> tuple[Any, Any]:
    return (
      self.torso1.init_state(params["torso1"]),
      self.torso2.init_state(params["torso2"]),
    )

  def init_params(self, key: jax.Array, x: Any) -> Any:
    return self.init(key, x, self._zero_state())["params"]
