"""Policy transforms: softmax and α-entmax over (masked) logits.

The whole library produces *raw logits* from its networks and converts them to
a probability distribution at the point of use. Historically that conversion
was always ``jax.nn.softmax`` / ``safe_log_softmax``. This module generalises
it to the **α-entmax** family (Peters et al., 2019), which interpolates between
softmax (α = 1) and sparsemax (α = 2) and assigns *exact zero* probability to
low-scoring actions for α > 1.

Design:
  * ``alpha`` is a *static* Python float (a config hyperparameter, never traced).
    ``alpha == 1.0`` dispatches to the original softmax ops, so existing runs are
    bit-for-bit unchanged and pay no extra cost.
  * For α > 1 the entmax projection is computed by bisection on the threshold τ
    with a hand-written VJP (the closed-form entmax Jacobian) — far cheaper and
    more stable than differentiating through the bisection loop.

Sparsity (α > 1) means some *legal* actions get probability exactly 0, so
``log π(a)`` is ``-inf`` there. The honest log-probs are still useful for
importance ratios (the ratio collapses to 0, with a finite gradient), but the
entropy and KL *regularisers* must NOT go through logs — they use the matching
**Tsallis** entropy and the Bregman (Tsallis) divergence, which stay finite on
sparse supports. See ``policy_entropy_loss`` and ``policy_kl``.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import jax.lax as lax

from .utils import safe_log_softmax, kl_divergence

# Bisection iterations for the entmax threshold. ~30 gives float32-tight roots.
_ENTMAX_ITERS = 30


# ── α-entmax projection ──────────────────────────────────────────────────────


def _entmax_threshold_probs(z: jax.Array, alpha: float, n_iter: int) -> jax.Array:
  """Bisection for p(τ) = [ (α−1)·z − τ ]_+^{1/(α−1)} normalised to sum 1.

  ``z`` is expected to be masked already (illegal actions set to ``-inf``);
  those map to probability 0 through the clamp.
  """
  am1 = alpha - 1.0
  z = z * am1
  d = z.shape[-1]
  max_val = jnp.max(z, axis=-1, keepdims=True)

  def p_of(tau):
    return jnp.maximum(z - tau, 0.0) ** (1.0 / am1)

  # f(τ) = Σ p(τ) − 1 is monotonically decreasing; bracket the root in [lo, hi].
  tau_lo = max_val - 1.0                       # f(tau_lo) >= 0
  tau_hi = max_val - (1.0 / d) ** am1          # f(tau_hi) <= 0
  dm = tau_hi - tau_lo

  def body(carry, _):
    tau_lo, dm = carry
    dm = dm * 0.5
    tau_m = tau_lo + dm
    f_m = p_of(tau_m).sum(axis=-1, keepdims=True) - 1.0
    tau_lo = jnp.where(f_m >= 0.0, tau_m, tau_lo)
    return (tau_lo, dm), None

  (tau_lo, _), _ = lax.scan(body, (tau_lo, dm), None, length=n_iter)
  p = p_of(tau_lo)
  return p / p.sum(axis=-1, keepdims=True)


@functools.partial(jax.custom_vjp, nondiff_argnums=(1, 2))
def _entmax(z: jax.Array, alpha: float, n_iter: int) -> jax.Array:
  return _entmax_threshold_probs(z, alpha, n_iter)


def _entmax_fwd(z, alpha, n_iter):
  p = _entmax_threshold_probs(z, alpha, n_iter)
  return p, p


def _entmax_bwd(alpha, n_iter, p, g):
  # Closed-form entmax Jacobian-vector product (Peters et al., 2019, eq. 14):
  #   s_i = p_i^{2−α} on the support (0 elsewhere); dz = g·s − (⟨g, s⟩/⟨1, s⟩)·s
  s = jnp.where(p > 0, p ** (2.0 - alpha), 0.0)
  gs = g * s
  shift = gs.sum(axis=-1, keepdims=True) / s.sum(axis=-1, keepdims=True)
  return (gs - shift * s,)


_entmax.defvjp(_entmax_fwd, _entmax_bwd)


# ── Public transforms ────────────────────────────────────────────────────────


def policy_probs(
  logits: jax.Array, legal_actions: jax.Array, alpha: float = 1.0
) -> jax.Array:
  """π = softmax(logits) for α = 1, else α-entmax. Illegal actions get 0."""
  if alpha == 1.0:
    return jax.nn.softmax(logits, where=legal_actions)
  z = jnp.where(legal_actions, logits, -jnp.inf)
  return _entmax(z, float(alpha), _ENTMAX_ITERS)


def policy_log_probs(
  logits: jax.Array, legal_actions: jax.Array, alpha: float = 1.0
) -> jax.Array:
  """log π, masking illegal actions to 0.0 (the library's convention).

  For α > 1, legal actions with zero mass yield ``-inf`` (honest). The
  double-``where`` keeps the gradient finite (0) at those points, so importance
  ratios that read a single action's log-prob stay safe.
  """
  if alpha == 1.0:
    return safe_log_softmax(logits, legal_actions)
  p = policy_probs(logits, legal_actions, alpha)
  safe_p = jnp.where(p > 0, p, 1.0)
  log_p = jnp.where(p > 0, jnp.log(safe_p), -jnp.inf)
  return jnp.where(legal_actions, log_p, 0.0)


def policy_entropy_loss(
  probs: jax.Array, log_probs: jax.Array, alpha: float = 1.0
) -> jax.Array:
  """Negative (Tsallis) entropy for a single sample. Minimising it maximises H.

  α = 1 → Shannon entropy −Σ p log p (uses ``log_probs`` for stability).
  α > 1 → Tsallis entropy  (1 − Σ pⱼ^α) / (α(α−1)), finite on sparse supports.
  """
  if alpha == 1.0:
    entropy = -(probs * log_probs).sum(-1)
  else:
    entropy = (1.0 - (probs**alpha).sum(-1)) / (alpha * (alpha - 1.0))
  return -entropy


def tsallis_kl(p: jax.Array, q: jax.Array, alpha: float) -> jax.Array:
  """Bregman divergence generated by the negative Tsallis entropy (α > 1).

  D(p‖q) = [ (1/α)Σpⱼ^α + (1−1/α)Σqⱼ^α − Σ qⱼ^{α−1} pⱼ ] / (α−1).

  It is the divergence whose argmin under a linear term *is* α-entmax, and it
  is finite even where q puts zero mass (qⱼ^{α−1}→0 for α > 1) — the property
  that makes a sparse reference policy usable as a magnet. As α→1 it recovers
  KL(p‖q).
  """
  term_p = (p**alpha).sum(-1) / alpha
  term_q = (1.0 - 1.0 / alpha) * (q**alpha).sum(-1)
  cross = (q ** (alpha - 1.0) * p).sum(-1)
  return (term_p + term_q - cross) / (alpha - 1.0)


def policy_kl(
  p: jax.Array,
  log_p: jax.Array,
  q: jax.Array,
  log_q: jax.Array,
  alpha: float = 1.0,
) -> jax.Array:
  """KL(p‖q) for α = 1, else the Tsallis/Bregman divergence (probs only).

  Callers pass both probabilities and log-probabilities so the α = 1 path is
  bit-identical to the previous ``kl_divergence(log_p, log_q)``.
  """
  if alpha == 1.0:
    return kl_divergence(log_p, log_q)
  return tsallis_kl(p, q, alpha)
