"""Magnetic Mirror Descent for a *hybrid* discrete–continuous poker policy.

This is the continuous-action counterpart of ``mmd.py``. The policy is a
**two-level (hierarchical) distribution**:

  * a categorical "atom" head over ``A`` actions, whose **last ``K`` atoms are
    bet atoms** (the leading ``A − K`` are point actions such as fold / call /
    all-in), and
  * a continuous head emitting, for each of the ``K`` bet atoms, the mean and
    log-std ``(μ_k, log σ_k)`` of a **raw Gaussian** over the bet size.

The density on the mixed (counting + Lebesgue) measure is

    π(atom a)    = p_a                                   (a a point atom)
    π(bet_k = x) = p_{bet_k} · N(x ; μ_k, σ_k)           (k-th bet atom)

so it is a Gaussian *mixture* whose mixture weights live in the categorical
head — this is the design that keeps every MMD term in closed form.

Conventions (softmax only, α = 1):
  * The chosen action is a pair ``(actions, bet_actions)``: ``actions`` is the
    atom index; ``bet_actions`` is the **raw, unclipped** sampled bet size and
    is only meaningful when ``actions`` indexes a bet atom. The importance
    ratio is taken over this raw ``x`` — clipping to the legal range and
    rounding to chips happen *inside the environment* and never enter the
    log-density (see the discussion in the project notes). The trajectory
    buffer must therefore store the raw ``x``, not what the env received.
  * The continuous head emits **log σ** (not σ) for positivity; the algorithm
    is free to clamp it to a sane range before calling this loss.

Closed-form decomposition (chain rule, α = 1):
    KL(π‖ρ) = KL_cat(p‖r) + Σ_k p_{bet_k} · KL_gauss(N_k^π ‖ N_k^ρ)
    H(π)    = H_cat(p)     + Σ_k p_{bet_k} · H_gauss(σ_k)
where the gate weight ``p_{bet_k}`` is the *current* policy's mass on bet atom
``k``. The categorical halves reuse ``policy.py`` verbatim.

All functions are pure and operate on a single (timestep, player) sample;
callers vmap over T, P (and B) exactly as in ``mmd.py``.

Dimension convention:
  A — number of categorical atoms (point atoms + bet atoms)
  K — number of bet atoms (the *last* K of the A atoms); static
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp

from .ppo import ppo_policy_loss, ppo_value_loss
from ..policy import policy_probs, policy_log_probs, policy_entropy_loss, policy_kl

_LOG_2PI = math.log(2.0 * math.pi)
_HALF_LOG_2PI_E = 0.5 * (math.log(2.0 * math.pi) + 1.0)


# ── Gaussian helpers (per bet atom) ──────────────────────────────────────────


def gaussian_log_prob(x: jax.Array, mu: jax.Array, log_sigma: jax.Array) -> jax.Array:
  """log N(x ; μ, σ) with σ = exp(log_sigma). Elementwise over its inputs."""
  inv_sigma = jnp.exp(-log_sigma)
  return -0.5 * ((x - mu) * inv_sigma) ** 2 - log_sigma - 0.5 * _LOG_2PI


def gaussian_kl(
  mu_p: jax.Array,
  log_sigma_p: jax.Array,
  mu_q: jax.Array,
  log_sigma_q: jax.Array,
) -> jax.Array:
  """KL(N(μ_p,σ_p) ‖ N(μ_q,σ_q)), elementwise (one value per bet atom)."""
  var_ratio = jnp.exp(2.0 * (log_sigma_p - log_sigma_q))
  sq_diff = ((mu_p - mu_q) * jnp.exp(-log_sigma_q)) ** 2
  return (log_sigma_q - log_sigma_p) + 0.5 * (var_ratio + sq_diff - 1.0)


def gaussian_entropy(log_sigma: jax.Array) -> jax.Array:
  """Differential entropy ½·log(2πe σ²) = log σ + ½·log(2πe). Elementwise."""
  return log_sigma + _HALF_LOG_2PI_E


# ── Loss ─────────────────────────────────────────────────────────────────────


def mmd_cont_loss(
  values: jax.Array,            # ()    — current infoset value
  logits: jax.Array,            # (A,)  — current categorical logits
  bet_mu: jax.Array,            # (K,)  — current bet means
  bet_log_sigma: jax.Array,     # (K,)  — current bet log-stds
  legal_actions: jax.Array,     # (A,)  — boolean atom mask
  actions: jax.Array,           # ()int — atom index played
  bet_actions: jax.Array,       # ()    — raw (unclipped) bet size played
  sample_logits: jax.Array,     # (A,)  — trajectory (old) categorical logits
  sample_bet_mu: jax.Array,     # (K,)
  sample_bet_log_sigma: jax.Array,  # (K,)
  sample_values: jax.Array,     # ()
  magnet_logits: jax.Array,     # (A,)  — magnet categorical logits
  magnet_bet_mu: jax.Array,     # (K,)
  magnet_bet_log_sigma: jax.Array,  # (K,)
  advantages: jax.Array,        # ()
  returns: jax.Array,           # ()
  clip_eps: float,
  vf_coef: float,
  ent_coef: float,
  magnet_coef: float,
  old_policy_coef: float,
  num_bet_atoms: int,           # K (static) — bet atoms are the last K of A
) -> tuple[jax.Array, dict]:
  # ── Categorical head (reused verbatim from policy.py, softmax/α=1) ──
  strategy = policy_probs(logits, legal_actions)               # (A,)
  log_probs_all = policy_log_probs(logits, legal_actions)      # (A,)
  sample_strategy = policy_probs(sample_logits, legal_actions)
  sample_log_probs_all = policy_log_probs(sample_logits, legal_actions)
  magnet_strategy = policy_probs(magnet_logits, legal_actions)
  magnet_log_probs_all = policy_log_probs(magnet_logits, legal_actions)

  # Bet atoms are the last K atoms; map the played atom to a bet-component index.
  bet_offset = logits.shape[-1] - num_bet_atoms
  is_bet = actions >= bet_offset
  k = jnp.clip(actions - bet_offset, 0, num_bet_atoms - 1)  # safe even for point atoms
  gate = strategy[bet_offset:]  # (K,) current-policy mass on each bet atom

  # ── PPO importance ratio: log π(a) = log p_atom + [bet] · log N(x; μ_k, σ_k) ──
  log_prob = log_probs_all[actions] + jnp.where(
    is_bet, gaussian_log_prob(bet_actions, bet_mu[k], bet_log_sigma[k]), 0.0
  )
  sample_log_prob = sample_log_probs_all[actions] + jnp.where(
    is_bet,
    gaussian_log_prob(bet_actions, sample_bet_mu[k], sample_bet_log_sigma[k]),
    0.0,
  )

  policy_loss = ppo_policy_loss(log_prob, sample_log_prob, advantages, clip_eps)
  value_loss = ppo_value_loss(values, sample_values, returns, clip_eps)

  # ── Magnet KL: categorical part + gate-weighted Gaussian part ──
  magnet_cat_kl = policy_kl(
    strategy, log_probs_all, magnet_strategy, magnet_log_probs_all
  )
  magnet_gauss_kl = (
    gate * gaussian_kl(bet_mu, bet_log_sigma, magnet_bet_mu, magnet_bet_log_sigma)
  ).sum()
  magnet_loss = magnet_cat_kl + magnet_gauss_kl

  # ── Old-policy KL (current ‖ trajectory): same decomposition ──
  old_cat_kl = policy_kl(
    strategy, log_probs_all, sample_strategy, sample_log_probs_all
  )
  old_gauss_kl = (
    gate * gaussian_kl(bet_mu, bet_log_sigma, sample_bet_mu, sample_bet_log_sigma)
  ).sum()
  old_kl_loss = old_cat_kl + old_gauss_kl

  # ── Entropy loss = −H(π) = −H_cat − Σ_k p_{bet_k} H_gauss(σ_k) ──
  cat_entropy_loss = policy_entropy_loss(strategy, log_probs_all)  # = −H_cat
  gauss_entropy = (gate * gaussian_entropy(bet_log_sigma)).sum()
  entropy_loss = cat_entropy_loss - gauss_entropy

  total = (
    policy_loss
    + vf_coef * value_loss
    + ent_coef * entropy_loss
    + magnet_coef * magnet_loss
    + old_policy_coef * old_kl_loss
  )
  return total, {
    "policy_loss": policy_loss,
    "value_loss": value_loss,
    "entropy_loss": entropy_loss,
    "magnet_loss": magnet_loss,
    "magnet_gauss_kl": magnet_gauss_kl,
    "old_kl_loss": old_kl_loss,
    "old_gauss_kl": old_gauss_kl,
  }
