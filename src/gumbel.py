"""Gumbel-Softmax (Concrete) relaxation with legal-action masking.

Experimental helper used by the Gumbel-Q MMD variant
(``algorithms/gumbel_q.py`` + ``losses/gumbel_q.py``). Kept in its own module
so it does not touch the existing softmax-based usage anywhere else.

The relaxation produces a differentiable, soft one-hot vector

    y_i = softmax_i((logit_i + g_i) / tau),   g_i ~ Gumbel(0, 1)

which reparameterizes a categorical sample: the gradient flows pathwise
through ``y`` into ``logits``. As ``tau -> 0`` the sample approaches a hard
one-hot (high-variance gradient); larger ``tau`` is smoother but more biased.

Illegal actions never receive Gumbel noise and are forced to probability 0,
so the returned vector is always supported on the legal set.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.lax as lax


def gumbel_softmax(
  logits: jax.Array,         # (A,) — policy logits
  legal_actions: jax.Array,  # (A,) — boolean action mask
  key: jax.Array,            # PRNG key for the Gumbel draw
  tau: float = 1.0,          # temperature (> 0)
  hard: bool = False,        # if True, straight-through one-hot
) -> jax.Array:              # (A,) — soft (or ST-hard) sample on the simplex
  """Masked Gumbel-Softmax sample of a single categorical distribution.

  With ``hard=False`` returns the soft relaxation (a convex combination of the
  legal actions). With ``hard=True`` returns a one-hot vector in the forward
  pass while keeping the soft gradient (straight-through estimator).
  """
  neg_inf = jnp.full_like(logits, -jnp.inf)
  g = jax.random.gumbel(key, logits.shape, dtype=logits.dtype)
  # Gumbel noise only on legal actions; illegal stay at -inf -> prob 0.
  noisy = jnp.where(legal_actions, logits + g, neg_inf)
  y_soft = jax.nn.softmax(noisy / tau, axis=-1)

  if not hard:
    return y_soft

  # Straight-through: hard one-hot forward, soft gradient backward. argmax over
  # ``noisy`` (illegal = -inf) can never select an illegal action.
  idx = jnp.argmax(noisy, axis=-1)
  y_hard = jax.nn.one_hot(idx, logits.shape[-1], dtype=y_soft.dtype)
  return y_hard + (y_soft - lax.stop_gradient(y_soft))
