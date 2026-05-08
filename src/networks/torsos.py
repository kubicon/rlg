from __future__ import annotations
from typing import Any

import jax
import jax.numpy as jnp
import flax.linen as nn

from .base import Torso
from .modules import Activation, Normalization


class ConvResBlock(nn.Module):
  """Pre-activation convolutional residual block (IMPALA-style)."""
  channels: int
  activation: Activation

  @nn.compact
  def __call__(self, x: jax.Array) -> jax.Array:
    residual = x
    x = self.activation(x)
    x = nn.Conv(self.channels, (3, 3), padding='SAME')(x)
    x = self.activation(x)
    x = nn.Conv(self.channels, (3, 3), padding='SAME')(x)
    return x + residual


class MLPTorso(Torso):
  """MLP feature extractor. Applies hidden layers then projects to feature_dim."""
  hidden: tuple[int, ...]
  activation: Activation = Activation(kind='relu')

  @nn.compact
  def __call__(self, x: Any, state: Any) -> tuple[jax.Array, Any]:
    for h in self.hidden:
      x = self.activation(nn.Dense(h)(x))
    return nn.Dense(self.feature_dim)(x), state


class ConvTorso(Torso):
  """Convolutional feature extractor for image inputs (H, W, C)."""
  channels: tuple[int, ...]
  kernel_sizes: tuple[int, ...]
  strides: tuple[int, ...]
  activation: Activation = Activation(kind='relu')

  @nn.compact
  def __call__(self, x: Any, state: Any) -> tuple[jax.Array, Any]:
    for c, k, s in zip(self.channels, self.kernel_sizes, self.strides):
      x = self.activation(nn.Conv(c, (k, k), strides=(s, s))(x))
    x = x.reshape((*x.shape[:-3], -1))
    return nn.Dense(self.feature_dim)(x), state


class ResidualTorso(Torso):
  """MLP with residual blocks. Projects input to hidden_dim, then applies n_blocks."""
  hidden_dim: int
  n_blocks: int
  activation: Activation = Activation(kind='relu')

  @nn.compact
  def __call__(self, x: Any, state: Any) -> tuple[jax.Array, Any]:
    x = nn.Dense(self.hidden_dim)(x)
    for _ in range(self.n_blocks):
      residual = x
      x = nn.Dense(self.hidden_dim)(self.activation(x))
      x = nn.Dense(self.hidden_dim)(self.activation(x))
      x = x + residual
    return nn.Dense(self.feature_dim)(self.activation(x)), state


class ResNetTorso(Torso):
  """IMPALA-style convolutional residual torso for image inputs (H, W, C).

  Each stage: Conv -> MaxPool -> n residual blocks.
  len(channels) must equal len(blocks_per_stage).
  """
  channels: tuple[int, ...]
  blocks_per_stage: tuple[int, ...]
  activation: Activation = Activation(kind='relu')

  @nn.compact
  def __call__(self, x: Any, state: Any) -> tuple[jax.Array, Any]:
    for c, n in zip(self.channels, self.blocks_per_stage):
      x = nn.Conv(c, (3, 3), padding='SAME')(x)
      x = nn.max_pool(x, (3, 3), strides=(2, 2), padding='SAME')
      for _ in range(n):
        x = ConvResBlock(c, self.activation)(x)
    x = self.activation(x)
    x = x.reshape((*x.shape[:-3], -1))
    return nn.Dense(self.feature_dim)(x), state


class TransformerTorso(Torso):
  """Pre-norm Transformer torso for sequential/relational inputs (..., seq_len, d_model).

  Applies n_layers of self-attention + FFN, mean-pools the sequence,
  then projects to feature_dim.
  """
  n_heads: int
  n_layers: int
  mlp_dim: int
  activation: Activation = Activation(kind='gelu')
  norm: Normalization = Normalization(kind='layer')

  @nn.compact
  def __call__(self, x: Any, state: Any) -> tuple[jax.Array, Any]:
    d_model = x.shape[-1]
    for _ in range(self.n_layers):
      residual = x
      # Fresh Normalization instances per layer so each has its own parameters.
      x = Normalization(kind=self.norm.kind, num_groups=self.norm.num_groups)(x)
      x = nn.MultiHeadDotProductAttention(self.n_heads)(x, x)
      x = x + residual
      residual = x
      x = Normalization(kind=self.norm.kind, num_groups=self.norm.num_groups)(x)
      x = self.activation(nn.Dense(self.mlp_dim)(x))
      x = nn.Dense(d_model)(x)
      x = x + residual
    x = Normalization(kind=self.norm.kind, num_groups=self.norm.num_groups)(x)
    x = x.mean(axis=-2)
    return nn.Dense(self.feature_dim)(x), state


class LSTMTorso(Torso):
  """LSTM recurrent torso. State is a (c, h) pair; both are learnable initial params."""
  hidden_dim: int

  def _zero_state(self) -> tuple[jax.Array, jax.Array]:
    zeros = jnp.zeros((self.hidden_dim,))
    return zeros, zeros

  @nn.compact
  def __call__(
    self, x: Any, state: tuple[jax.Array, jax.Array]
  ) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
    # Register initial state as learnable parameters.
    self.param('init_c', nn.initializers.zeros, (self.hidden_dim,))
    self.param('init_h', nn.initializers.zeros, (self.hidden_dim,))
    new_state, h = nn.LSTMCell(self.hidden_dim)(state, x)
    return nn.Dense(self.feature_dim)(h), new_state

  def init_state(self, params: Any) -> tuple[jax.Array, jax.Array]:
    return params['init_c'], params['init_h']


class GRUTorso(Torso):
  """GRU recurrent torso. State is a single h array; learnable as an initial param."""
  hidden_dim: int

  def _zero_state(self) -> jax.Array:
    return jnp.zeros((self.hidden_dim,))

  @nn.compact
  def __call__(
    self, x: Any, state: jax.Array
  ) -> tuple[jax.Array, jax.Array]:
    # Register initial state as a learnable parameter.
    self.param('init_h', nn.initializers.zeros, (self.hidden_dim,))
    new_state, h = nn.GRUCell(self.hidden_dim)(state, x)
    return nn.Dense(self.feature_dim)(h), new_state

  def init_state(self, params: Any) -> jax.Array:
    return params['init_h']
