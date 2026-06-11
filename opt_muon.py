#!/usr/bin/env python
"""Optimistic Muon: Muon orthogonalization with optimistic-gradient extrapolation.

This composes two existing optax transforms:

  * ``optax.contrib.muon`` -- momentum + Newton-Schulz orthogonalization of the
    update (steepest descent under the spectral / Schatten-p norm).
  * ``optax.scale_by_optimistic_gradient`` -- the generalized optimistic /
    negative-momentum extrapolation ``u_t = (alpha + beta) g_t - beta g_{t-1}``
    that gives last-iterate convergence in min-max games (Mokhtari et al., 2019).

Why combine them
----------------
This library trains agents in self-play on two-player zero-sum games
(Leduc, Goofspiel, Battleship). Such problems are min-max / saddle-point
problems, where plain gradient descent can oscillate or diverge and where
optimistic gradient methods have strong last-iterate convergence guarantees.
Muon, meanwhile, gives very well-conditioned (orthogonalized) update directions.
Combining the two asks: take Muon's well-conditioned direction, then extrapolate
it optimistically to damp the oscillation typical of saddle-point dynamics.

Where the optimism goes (and the one conceptual caveat)
-------------------------------------------------------
Newton-Schulz orthogonalization is a *nonlinear* map that normalizes the update's
singular values to ~1, discarding magnitude information. The optimistic
extrapolation, by contrast, is a *linear* operation whose convergence theory is
derived for the raw gradient field. There are two ways to merge them:

  (A) optimism BEFORE orthogonalization -- feed ``(alpha+beta) g - beta g_prev``
      into Newton-Schulz. The orthogonalization then throws away the carefully
      constructed magnitude, so the OGD guarantee does not survive; the optimism
      degenerates into a mere direction perturbation. Available via
      ``position="before"`` for experimentation, but not the default.

  (B) optimism AFTER orthogonalization -- treat Muon's orthogonalized update as
      the "direction" and extrapolate *that*. This mirrors exactly how optax
      builds ``optimistic_adam_v2`` (``chain(scale_by_adam,
      scale_by_optimistic_gradient, scale_by_learning_rate)``): precondition
      first to get a normalized direction, then apply optimism. This is the
      principled placement and the default (``position="after"``).

Both placements (and applying optimism at both stages, ``position="both"``) are
selectable through the ``position`` argument of :func:`optimistic_muon`.

Caveat: even with placement (B), the formal last-iterate guarantee of OGD is not
preserved verbatim, because Muon's preconditioner is nonlinear (just as the
guarantee is not preserved for optimistic-Adam, whose Adam preconditioner is
also nonlinear). This is a well-motivated heuristic, exactly analogous to
optimistic-Adam, not a theorem. Empirically it is the right thing to try for
saddle-point training.

Implementation
--------------
Rather than re-implement Muon's ``combine.partition`` routing (which sends 2D
matrices through orthogonalization and everything else through AdamW, using
optax-private masking internals), we reuse ``optax.contrib.muon`` unchanged with
``learning_rate=1.0`` so that its output is the (negated) preconditioned
direction ``-D``. We then apply the optimistic extrapolation and finally the
real learning rate as a positive ``optax.scale(lr)`` (Muon already carries the
descent sign). The resulting update is

    u_t = -lr * [ (alpha + beta) D_t - beta D_{t-1} ],

which is the orthogonalized analog of ``optimistic_adam_v2``. As a bonus the
non-2D parameters (routed through AdamW inside Muon) become optimistic too,
consistently with the matrix parameters.
"""

from __future__ import annotations

from typing import Literal

import jax.typing
import optax


OptimismPosition = Literal["after", "before", "both", "none"]


def optimistic_muon(
    learning_rate: optax.ScalarOrSchedule,
    *,
    alpha: jax.typing.ArrayLike = 1.0,
    beta: jax.typing.ArrayLike = 1.0,
    position: OptimismPosition = "after",
    # ---- Muon hyper-parameters (forwarded to optax.contrib.muon) -------------
    momentum: jax.typing.ArrayLike = 0.95,
    nesterov: bool = True,
    ns_steps: jax.typing.ArrayLike = 5,
    weight_decay: jax.typing.ArrayLike = 0.0,
    adam_weight_decay: jax.typing.ArrayLike = 0.0,
    adam_learning_rate: optax.ScalarOrSchedule | None = None,
    **muon_kwargs,
) -> optax.GradientTransformation:
  r"""Muon with optimistic-gradient extrapolation around the orthogonalization.

  The optimistic extrapolation ``(alpha + beta) v_t - beta v_{t-1}`` can be
  applied to the *raw gradient* fed into Muon (``position="before"``), to Muon's
  *orthogonalized update direction* (``position="after"``, the principled
  default), to both stages independently (``position="both"``), or disabled
  (``position="none"``, plain Muon). See the module docstring for the trade-offs:
  "before" runs the extrapolation through the nonlinear Newton-Schulz step (which
  discards magnitude), whereas "after" mirrors ``optax.optimistic_adam_v2`` by
  extrapolating the already-preconditioned direction.

  Args:
    learning_rate: Global step size, fixed or a schedule.
    alpha: Optimistic-gradient coefficient (generalized OGD). ``alpha=1, beta=1``
      reproduces classic optimism ``2 v_t - v_{t-1}``.
    beta: Negative-momentum / optimism coefficient.
    position: Where to apply the optimism: ``"after"`` (default), ``"before"``,
      ``"both"``, or ``"none"``. When ``"both"``, the pre- and post-Muon stages
      keep separate extrapolation states but share ``alpha``/``beta``.
    momentum: Muon's gradient-momentum decay (``beta`` in ``optax.contrib.muon``;
      renamed here to avoid clashing with the optimism ``beta``).
    nesterov: Use Nesterov momentum inside Muon.
    ns_steps: Number of Newton-Schulz iterations.
    weight_decay: Decoupled weight decay applied to the matrix (Muon) params.
    adam_weight_decay: Weight decay for the AdamW branch (non-matrix params).
    adam_learning_rate: Auxiliary LR for the AdamW branch (non-matrix params),
      expressed *relative to* ``learning_rate``. Because the final scaling by
      ``learning_rate`` is applied to the whole update tree (after Muon's own
      internal LR of 1.0), the effective Adam learning rate is
      ``adam_learning_rate * learning_rate``. ``None`` (default) uses ``1.0``,
      i.e. the same effective LR as the Muon branch. NOTE: this differs from
      ``optax.contrib.muon``, where ``adam_learning_rate`` is absolute; here it
      is a multiplier, so e.g. with ``learning_rate=0.1`` you need
      ``adam_learning_rate=3e-3`` to get an effective Adam LR of ``3e-4``.
    **muon_kwargs: Any further keyword args forwarded to ``optax.contrib.muon``
      (e.g. ``preconditioning``, ``adaptive``, ``consistent_rms``).

  Returns:
    An ``optax.GradientTransformation`` implementing optimistic Muon.
  """
  if position not in ("after", "before", "both", "none"):
    raise ValueError(
        f"position must be one of 'after', 'before', 'both', 'none'; got "
        f"{position!r}")

  # Build Muon with unit learning rate so its output is the (sign-correct)
  # preconditioned direction; the real learning rate is applied after optimism.
  # The Adam branch likewise runs at a *relative* LR here: whatever we pass is
  # multiplied by the final scale(learning_rate), so the effective Adam LR is
  # `adam_learning_rate * learning_rate`. None -> 1.0 (same effective LR as
  # Muon). We must pass the value through explicitly: passing None to optax
  # would make it default Adam's LR to Muon's (1.0), discarding the caller's
  # choice.
  inner_muon = optax.contrib.muon(
      learning_rate=1.0,
      beta=momentum,
      nesterov=nesterov,
      ns_steps=ns_steps,
      weight_decay=weight_decay,
      adam_weight_decay=adam_weight_decay,
      adam_learning_rate=1.0 if adam_learning_rate is None else adam_learning_rate,
      **muon_kwargs,
  )

  chain = []
  # "before": extrapolate the raw gradient prior to orthogonalization.
  if position in ("before", "both"):
    chain.append(optax.scale_by_optimistic_gradient(alpha=alpha, beta=beta))
  chain.append(inner_muon)
  # "after": extrapolate Muon's orthogonalized direction (principled default).
  # Uses its own extrapolation state, independent of any "before" stage.
  if position in ("after", "both"):
    chain.append(optax.scale_by_optimistic_gradient(alpha=alpha, beta=beta))

  # Muon already carries the descent sign (its internal scale_by_learning_rate
  # negates), so apply the magnitude with a positive scale. A schedule is
  # supported via scale_by_schedule.
  if callable(learning_rate):
    chain.append(optax.scale_by_schedule(learning_rate))
  else:
    chain.append(optax.scale(learning_rate))

  return optax.chain(*chain)


if __name__ == "__main__":
  # Minimal sanity check: a bilinear saddle x*y, where vanilla GD spirals out
  # but optimistic methods converge. We just confirm the optimizer runs and
  # produces finite updates on a tiny matrix parameter (so the Muon branch,
  # i.e. orthogonalization, is actually exercised).
  import jax
  import jax.numpy as jnp

  def loss(p):
    # 2x2 matrix param; a simple smooth objective with a unique minimum at 0.
    return 0.5 * jnp.sum(p["w"] ** 2)

  init = jnp.array([[1.0, 2.0], [3.0, 4.0]])
  init_loss = 0.5 * float(jnp.sum(init ** 2))

  for position in ("after", "before", "both", "none"):
    params = {"w": init}
    opt = optimistic_muon(learning_rate=0.1, alpha=1.0, beta=1.0,
                          position=position)
    state = opt.init(params)

    @jax.jit
    def step(params, state):
      grads = jax.grad(loss)(params)
      updates, state = opt.update(grads, state, params)
      params = optax.apply_updates(params, updates)
      return params, state

    for _ in range(50):
      params, state = step(params, state)
    final = float(loss(params))
    print(f"position={position:7s} final loss = {final:.6e}")
    assert jnp.isfinite(final), f"non-finite loss for position={position}"
    assert final < init_loss, f"no progress for position={position}"
  print("optimistic_muon: OK")
