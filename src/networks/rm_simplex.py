"""Hard regret-matching simplex activation with a softmax-free custom backward.

Optional (D) component from the RM transfer kit: a hidden activation that maps
each layer's pre-activations to a hard RM simplex (whole layer; group_size=1,
scale=1) while routing a well-conditioned gradient back.

    forward : a  = relu(z) / Σ relu(z)             (uniform if all z ≤ 0)
    backward: s  = relu(z) / Σ relu(z)             (the same hard simplex)
              adv = grad_out − ⟨s, grad_out⟩        (centre = local advantage)
              dz  = adv · (z > 0)                    (KEEP the hard relu mask)

Why a custom backward: plain autograd through relu-normalise has dead gradients
(units relu'd to 0 get exactly 0 gradient and never recover). The baseline
subtraction turns the incoming value into a local advantage/regret and keeping
the hard mask is what makes multi-state learning work; dropping either fails.

NOTE (from the kit's measurements): this helped a multi-state capacity probe but
did NOT help a real extensive-form game (Leduc) and tightens the stability
basin. It is an opt-in toggle (``activation: rm_simplex``); start without it.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def _rm_strategy(z: jax.Array) -> jax.Array:
  pos = jax.nn.relu(z)
  ss = pos.sum(-1, keepdims=True)
  return jnp.where(
    ss > 0, pos / jnp.clip(ss, 1e-12, None), jnp.full_like(z, 1.0 / z.shape[-1])
  )


@jax.custom_vjp
def rm_simplex(z: jax.Array) -> jax.Array:
  return _rm_strategy(z)


def _rm_simplex_fwd(z: jax.Array):
  return _rm_strategy(z), z


def _rm_simplex_bwd(z: jax.Array, grad_out: jax.Array):
  s = _rm_strategy(z)
  adv = grad_out - (s * grad_out).sum(-1, keepdims=True)  # baseline subtraction
  dz = adv * (z > 0)  # hard relu cutoff
  return (dz,)


rm_simplex.defvjp(_rm_simplex_fwd, _rm_simplex_bwd)
