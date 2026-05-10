"""Network config registry.

Each module type registers a builder function under a string key.
build_torso / build_head / build_network dispatch through these registries
and recursively build any nested sub-components from their dicts.

Adding a new module requires only decorating one function — no new class,
no union type to update:

    from rlg.networks.configs import register_torso

    @register_torso('my_torso')
    def _(feature_dim: int = 256, my_param: int = 42):
        from mymodule import MyTorso
        return MyTorso(feature_dim=feature_dim, my_param=my_param)

YAML usage (unchanged from before):

    import yaml
    from rlg.networks.configs import build_network

    with open('configs/example.yaml') as f:
        raw = yaml.safe_load(f)

    network = build_network(raw['network'])
    params  = network.init_params(key, x)
    state   = network.init_state(params)
"""

from __future__ import annotations
from collections.abc import Callable
from typing import Any

import flax.linen as nn


# ── Registries ─────────────────────────────────────────────────────────────

_torso_registry: dict[str, Callable] = {}
_head_registry: dict[str, Callable] = {}
_network_registry: dict[str, Callable] = {}


def register_torso(name: str) -> Callable:
  """Decorator: register a torso builder under `name`."""

  def decorator(fn: Callable) -> Callable:
    _torso_registry[name] = fn
    return fn

  return decorator


def register_head(name: str) -> Callable:
  """Decorator: register a head builder under `name`."""

  def decorator(fn: Callable) -> Callable:
    _head_registry[name] = fn
    return fn

  return decorator


def register_network(name: str) -> Callable:
  """Decorator: register a network builder under `name`."""

  def decorator(fn: Callable) -> Callable:
    _network_registry[name] = fn
    return fn

  return decorator


# ── Torso builders ─────────────────────────────────────────────────────────


@register_torso("mlp")
def _(
  feature_dim: int = 256,
  hidden=(256, 256),
  activation: str = "relu",
  norm: str = "none",
):
  from .modules import Activation, Normalization
  from .torsos import MLPTorso

  return MLPTorso(
    feature_dim=feature_dim,
    hidden=tuple(hidden),
    activation=Activation(kind=activation),
    norm=Normalization(kind=norm),
  )


@register_torso("conv")
def _(
  feature_dim: int = 256,
  channels=(32, 64, 64),
  kernel_sizes=(8, 4, 3),
  strides=(4, 2, 1),
  activation: str = "relu",
):
  from .modules import Activation
  from .torsos import ConvTorso

  return ConvTorso(
    feature_dim=feature_dim,
    channels=tuple(channels),
    kernel_sizes=tuple(kernel_sizes),
    strides=tuple(strides),
    activation=Activation(kind=activation),
  )


@register_torso("residual")
def _(
  feature_dim: int = 256,
  hidden_dim: int = 256,
  n_blocks: int = 4,
  activation: str = "relu",
):
  from .modules import Activation
  from .torsos import ResidualTorso

  return ResidualTorso(
    feature_dim=feature_dim,
    hidden_dim=hidden_dim,
    n_blocks=n_blocks,
    activation=Activation(kind=activation),
  )


@register_torso("resnet")
def _(
  feature_dim: int = 256,
  channels=(16, 32, 32),
  blocks_per_stage=(2, 2, 2),
  activation: str = "relu",
):
  from .modules import Activation
  from .torsos import ResNetTorso

  return ResNetTorso(
    feature_dim=feature_dim,
    channels=tuple(channels),
    blocks_per_stage=tuple(blocks_per_stage),
    activation=Activation(kind=activation),
  )


@register_torso("transformer")
def _(
  feature_dim: int = 256,
  n_heads: int = 4,
  n_layers: int = 3,
  mlp_dim: int = 512,
  activation: str = "gelu",
  norm: str = "layer",
  norm_num_groups: int = 32,
):
  from .modules import Activation, Normalization
  from .torsos import TransformerTorso

  return TransformerTorso(
    feature_dim=feature_dim,
    n_heads=n_heads,
    n_layers=n_layers,
    mlp_dim=mlp_dim,
    activation=Activation(kind=activation),
    norm=Normalization(kind=norm, num_groups=norm_num_groups),
  )


@register_torso("lstm")
def _(feature_dim: int = 256, hidden_dim: int = 256):
  from .torsos import LSTMTorso

  return LSTMTorso(feature_dim=feature_dim, hidden_dim=hidden_dim)


@register_torso("gru")
def _(feature_dim: int = 256, hidden_dim: int = 256):
  from .torsos import GRUTorso

  return GRUTorso(feature_dim=feature_dim, hidden_dim=hidden_dim)


# ── Head builders ──────────────────────────────────────────────────────────


@register_head("categorical")
def _(n_actions: int):
  from .heads import CategoricalHead

  return CategoricalHead(n_actions=n_actions)


@register_head("gaussian")
def _(action_dim: int, log_std_min: float = -20.0, log_std_max: float = 2.0):
  from .heads import GaussianHead

  return GaussianHead(
    action_dim=action_dim, log_std_min=log_std_min, log_std_max=log_std_max
  )


@register_head("q")
def _(n_actions: int):
  from .heads import QHead

  return QHead(n_actions=n_actions)


@register_head("distributional_q")
def _(n_actions: int, n_atoms: int = 51):
  from .heads import DistributionalQHead

  return DistributionalQHead(n_actions=n_actions, n_atoms=n_atoms)


@register_head("value")
def _():
  from .heads import ValueHead

  return ValueHead()


@register_head("advantage")
def _(n_actions: int):
  from .heads import AdvantageHead

  return AdvantageHead(n_actions=n_actions)


# ── Network builders ───────────────────────────────────────────────────────


@register_network("network")
def _(torso: dict, head: dict):
  from .base import Network

  return Network(torso=build_torso(torso), head=build_head(head))


@register_network("twin_head")
def _(torso: dict, head1: dict, head2: dict):
  from .composite import TwinHead

  return TwinHead(
    torso=build_torso(torso), head1=build_head(head1), head2=build_head(head2)
  )


@register_network("separate_twin_head")
def _(torso1: dict, head1: dict, torso2: dict, head2: dict):
  from .composite import SeparateTwinHead

  return SeparateTwinHead(
    torso1=build_torso(torso1),
    head1=build_head(head1),
    torso2=build_torso(torso2),
    head2=build_head(head2),
  )


# ── Public API ─────────────────────────────────────────────────────────────


def build_torso(data: dict[str, Any]):
  """Build a Torso from a config dict. 'type' selects the registered builder."""
  data = dict(data)
  name = data.pop("type")
  if name not in _torso_registry:
    raise ValueError(f"Unknown torso '{name}'. Registered: {sorted(_torso_registry)}")
  return _torso_registry[name](**data)


def build_head(data: dict[str, Any]):
  """Build a Head from a config dict. 'type' selects the registered builder."""
  data = dict(data)
  name = data.pop("type")
  if name not in _head_registry:
    raise ValueError(f"Unknown head '{name}'. Registered: {sorted(_head_registry)}")
  return _head_registry[name](**data)


def build_network(data: dict[str, Any]) -> nn.Module:
  """Build any network from a config dict (from yaml.safe_load / json.load)."""
  data = dict(data)
  name = data.pop("type")
  if name not in _network_registry:
    raise ValueError(
      f"Unknown network '{name}'. Registered: {sorted(_network_registry)}"
    )
  return _network_registry[name](**data)
