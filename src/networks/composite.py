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
