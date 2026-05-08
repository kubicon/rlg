from __future__ import annotations
from typing import Any

import jax
import flax.linen as nn


class Torso(nn.Module):
  """Feature extractor: (x, state) -> (features, new_state).

  Stateless torsos ignore state and return it unchanged.
  Recurrent torsos carry meaningful state (e.g. LSTM (c, h)) between steps.
  The initial state is a learnable parameter: use init_state(params) to obtain it.
  """

  feature_dim: int

  def __call__(self, x: Any, state: Any) -> tuple[jax.Array, Any]:
    raise NotImplementedError

  def _zero_state(self) -> Any:
    """Dummy zero state used only for tracing inside init_params."""
    return None

  def init_state(self, _params: Any) -> Any:
    """Return the (learnable) initial state extracted from params."""
    return None

  def init_params(self, key: jax.Array, x: Any) -> Any:
    """Initialise all parameters, including any learnable initial state."""
    return self.init(key, x, self._zero_state())["params"]


class Head(nn.Module):
  """Maps features to task-specific output. Subclasses must implement __call__."""

  def __call__(self, x: jax.Array) -> Any:
    raise NotImplementedError

  def init_params(self, key: jax.Array, x: jax.Array) -> Any:
    return self.init(key, x)["params"]


class Network(nn.Module):
  """Combines a Torso and a Head: (x, state) -> (output, new_state)."""

  torso: Torso
  head: Head

  def __call__(self, x: Any, state: Any) -> tuple[Any, Any]:
    features, new_state = self.torso(x, state)
    return self.head(features), new_state

  def init_state(self, params: Any) -> Any:
    """Return the (learnable) initial state, delegated to the torso."""
    return self.torso.init_state(params["torso"])

  def init_params(self, key: jax.Array, x: Any) -> Any:
    return self.init(key, x, self.torso._zero_state())["params"]
