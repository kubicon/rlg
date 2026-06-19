from __future__ import annotations

import jax.nn as jnn
import jax.numpy as jnp
import jax
import flax.linen as nn

from .rm_simplex import rm_simplex


_ACTIVATIONS = {
  "none": lambda x: x,
  "relu": jnn.relu,
  "gelu": jnn.gelu,
  "tanh": jnp.tanh,
  "sigmoid": jnn.sigmoid,
  "silu": jnn.silu,
  "elu": jnn.elu,
  "leaky_relu": jnn.leaky_relu,
  "softplus": jnn.softplus,
  # Optional (D): hard RM-simplex hidden activation with a custom backward.
  "rm_simplex": rm_simplex,
}


class Activation(nn.Module):
  """Stateless activation selected by kind. No learnable parameters.

  Field is named 'kind' (not 'name') to avoid shadowing flax.linen.Module.name,
  which Flax uses internally as the module's scope identifier.
  """

  kind: str = "relu"

  def __call__(self, x: jax.Array) -> jax.Array:
    if self.kind not in _ACTIVATIONS:
      raise ValueError(
        f"Unknown activation '{self.kind}'. Options: {list(_ACTIVATIONS)}"
      )
    return _ACTIVATIONS[self.kind](x)


class Normalization(nn.Module):
  """Normalization layer selected by kind.

  Supported: 'layer', 'rms', 'group', 'none'.
  num_groups is only used when kind='group'.

  Field is named 'kind' (not 'name') to avoid shadowing flax.linen.Module.name.
  """

  kind: str = "none"
  num_groups: int = 32

  @nn.compact
  def __call__(self, x: jax.Array) -> jax.Array:
    if self.kind == "none":
      return x
    if self.kind == "layer":
      return nn.LayerNorm()(x)
    if self.kind == "rms":
      return nn.RMSNorm()(x)
    if self.kind == "group":
      return nn.GroupNorm(self.num_groups)(x)
    raise ValueError(
      f"Unknown normalization '{self.kind}'. Options: layer, rms, group, none"
    )
