#!/usr/bin/env python
"""Train the hybrid discrete–continuous MMD policy on Heads-Up No-Limit Hold'em.

This is the continuous-action counterpart of ``train.py``. The generic
Trainer / Episode / MMD stack assumes a purely categorical action sampled by
``collect_episodes``; the continuous bet-sizing head needs the rollout to also
sample a real-valued bet and store the *raw* value for the importance ratio.
Rather than retrofit those shared types, this script keeps the whole continuous
pipeline self-contained:

  * ``ContActorCritic`` — a Flax MLP with four heads: categorical ``logits`` over
    ``A = 3 + K`` atoms (fold / call / all-in / K bet atoms), a state ``value``,
    and per-bet-atom Gaussian ``bet_mu`` / ``bet_log_sigma``.
  * ``rollout`` — self-play scan over the env; at each step both players are
    evaluated, the active player samples ``(atom, bet_x)`` (``bet_x`` from the
    chosen atom's Gaussian), and the **raw** ``bet_x`` is stored.
  * the MMD update reuses ``losses.mmd_cont.mmd_cont_loss`` (softmax / α=1), with
    a Polyak ``target`` network for value bootstrapping and a periodically reset
    ``magnet`` network for the KL term — exactly as ``algorithms/mmd.MMD`` does.

Usage:
    python train_mmd_cont.py                              # built-in defaults
    python train_mmd_cont.py configs/mmd_cont_hunl.yaml   # from a config
"""

from __future__ import annotations

import math
import os
import pickle
import sys
import time

import jax
import jax.numpy as jnp
import jax.lax as lax
import flax.linen as nn
import optax
import yaml

from src.envs import build_env
from src.advantage import vtrace
from src.losses.mmd_cont import mmd_cont_loss
from train import build_optimizer  # shared optimizer registry (adam/adamw/muon/…)


# ── Network ──────────────────────────────────────────────────────────────────


class ContActorCritic(nn.Module):
  """Shared MLP torso with categorical, value, and Gaussian bet-sizing heads.

  Returns ``(logits[A], value[], bet_mu[K], bet_log_sigma[K])``. Bet sizes are
  **normalized**: ``bet_mu`` and the sampled bet live in fraction-of-stack space
  ``[0, 1]`` (the mean is a sigmoid, initialising near half the stack), and the
  rollout converts to chips only at the env boundary. The Gaussian — and its
  log-prob in ``mmd_cont_loss`` — is therefore unitless and stack-independent,
  so the log-std clamp below is a fixed fraction band rather than chip-scaled.
  """

  n_actions: int
  num_bet_atoms: int
  hidden: tuple[int, ...] = (256, 256)
  log_sigma_min: float = -4.6   # σ ≈ 0.01  (~1% of stack) — floors how peaked the bet Gaussian gets
  log_sigma_max: float = -1.4   # σ ≈ 0.25  (quarter-stack std) — caps over-diffuse betting
  log_sigma_init: float = -2.3  # σ ≈ 0.10  (~10% of stack) — moderately exploratory start

  @nn.compact
  def __call__(self, x):
    h = x
    for w in self.hidden:
      h = nn.gelu(nn.Dense(w)(h))
    logits = nn.Dense(self.n_actions)(h)
    value = nn.Dense(1)(h)[..., 0]
    bet_mu = nn.sigmoid(nn.Dense(self.num_bet_atoms)(h))  # fraction of stack (0, 1)
    raw_ls = nn.Dense(
      self.num_bet_atoms, bias_init=nn.initializers.constant(self.log_sigma_init)
    )(h)
    bet_log_sigma = jnp.clip(raw_ls, self.log_sigma_min, self.log_sigma_max)
    return logits, value, bet_mu, bet_log_sigma


def make_forward(net: ContActorCritic):
  def forward(params, obs):
    return net.apply({"params": params}, obs)
  return forward


# ── Rollout (self-play) ──────────────────────────────────────────────────────


def rollout(env, forward, params, key):
  """Collect one self-play episode. All fields are stacked over time T.

  Per-player fields carry a trailing player axis P. Both players are evaluated
  every step (so the critic learns at every node); only the active player's
  sampled action drives the env, and inactive players have a single legal action
  (FOLD) so their policy is degenerate and contributes no policy gradient.
  """
  P = env.num_players
  A = env.num_actions
  K = env.num_bet_atoms
  bet_offset = A - K

  def step(carry, _):
    env_state, rng = carry
    rng, obs_key, act_key, bet_key, env_key = jax.random.split(rng, 5)
    pids = jnp.arange(P)
    infosets = jax.vmap(env.information_set, in_axes=(None, 0, 0))(
      env_state, pids, jax.random.split(obs_key, P)
    )  # (P, d)
    legal = jax.vmap(env.legal_actions, in_axes=(None, 0))(env_state, pids)  # (P, A)

    logits, value, bet_mu, bet_log_sigma = forward(params, infosets)
    masked = jnp.where(legal, logits, -jnp.inf)
    atoms = jax.vmap(jax.random.categorical)(jax.random.split(act_key, P), masked)  # (P,)

    # Sample the bet size from the chosen atom's Gaussian (only used if a bet).
    k = jnp.clip(atoms - bet_offset, 0, K - 1)
    mu_sel = jnp.take_along_axis(bet_mu, k[:, None], axis=1)[:, 0]
    ls_sel = jnp.take_along_axis(bet_log_sigma, k[:, None], axis=1)[:, 0]
    noise = jax.random.normal(bet_key, (P,))
    bet_x = mu_sel + jnp.exp(ls_sel) * noise  # raw, unclipped FRACTION — stored as-is
    # Convert fraction → chips only at the env boundary, scaling by each player's
    # *remaining* stack so bet_x ∈ [0, 1] maps onto exactly [0, all-in] at every
    # state (constant meaning regardless of stack depth). remaining_stack is a
    # function of the env state, not the policy, so the loss's importance ratio
    # stays a clean comparison in fraction space.
    remaining = env.starting_stack - env_state.committed  # (P,)
    bet_chips = bet_x * remaining
    actions = jnp.stack([atoms.astype(jnp.float32), bet_chips], axis=-1)  # (P, 2)

    next_state, rewards, done, _ = env.apply_action(env_state, actions, env_key)
    out = dict(
      infosets=infosets, legal=legal, atoms=atoms, bet_x=bet_x,
      logits=logits, value=value, bet_mu=bet_mu, bet_log_sigma=bet_log_sigma,
      rewards=rewards, done=done,
    )
    return (next_state, rng), out

  init = env.init_state(key)
  (_, _), traj = lax.scan(step, (init, key), None, length=env.max_length)
  return traj


# ── MMD-cont training step ───────────────────────────────────────────────────


def _valid_mask(dones):  # (B, T) -> (B, T), False after episode end
  valid = jnp.cumulative_prod(1 - dones, axis=1, include_initial=True)
  return valid[..., :-1]


def make_train_step(env, forward, optimizer, cfg):
  K = env.num_bet_atoms
  gamma, lam = cfg["gamma"], cfg["gae_lambda"]
  dclip, tclip = cfg["delta_clip"], cfg["trace_clip"]
  clip_eps, vf_coef, ent_coef = cfg["clip_eps"], cfg["vf_coef"], cfg["ent_coef"]
  magnet_coef, old_coef = cfg["magnet_coef"], cfg["old_policy_coef"]
  tau, magnet_interval = cfg["target_update_rate"], cfg["magnet_interval"]
  n_epochs, batch_size = cfg["n_epochs"], cfg["batch_size"]

  def eval_params(params, infosets):
    return forward(params, infosets)  # broadcasts over leading (B, T, P) axes

  def advantages_targets(rewards, values, dones):
    discount = (1.0 - dones) * gamma  # (B, T)
    vt_P = jax.vmap(vtrace, in_axes=(-1, -1, None, None, None, None, None, None),
                    out_axes=(-1, -1))
    vt_BP = jax.vmap(vt_P, in_axes=(0, 0, 0, None, None, None, None, None))
    targets, advs = vt_BP(rewards, values, discount, 1.0, 0.0, lam, dclip, tclip)
    return advs, targets

  _AX = (0,) * 16 + (None,) * 6  # 16 batched array args, 6 static

  def train_step(carry, _):
    params, opt_state, target_params, magnet_params, step, rng = carry
    rng, key = jax.random.split(rng)
    traj = jax.vmap(lambda k: rollout(env, forward, params, k))(
      jax.random.split(key, batch_size)
    )  # each field (B, T, P, ...) or (B, T)

    infosets = traj["infosets"]
    dones = traj["done"]  # (B, T)
    valid = _valid_mask(dones)

    _, t_value, _, _ = eval_params(target_params, infosets)
    target_values = lax.stop_gradient(t_value)  # (B, T, P)
    advs, targets = advantages_targets(traj["rewards"], target_values, dones)

    m_logits, _, m_mu, m_ls = eval_params(magnet_params, infosets)
    m_logits, m_mu, m_ls = map(lax.stop_gradient, (m_logits, m_mu, m_ls))

    def epoch(carry, _):
      params, opt_state = carry

      def loss_fn(params):
        c_logits, c_value, c_mu, c_ls = eval_params(params, infosets)
        loss_P = jax.vmap(mmd_cont_loss, in_axes=_AX)
        loss_TP = jax.vmap(loss_P, in_axes=_AX)
        loss_BTP = jax.vmap(loss_TP, in_axes=_AX)
        losses, metrics = loss_BTP(
          c_value, c_logits, c_mu, c_ls,
          traj["legal"], traj["atoms"], traj["bet_x"],
          traj["logits"], traj["bet_mu"], traj["bet_log_sigma"], traj["value"],
          m_logits, m_mu, m_ls,
          advs, targets,
          clip_eps, vf_coef, ent_coef, magnet_coef, old_coef, K,
        )
        # sum over players, valid-masked mean over time, mean over batch
        def wmean(x):
          x = x.sum(-1)
          x = (x * valid).sum(1) / jnp.maximum(valid.sum(1), 1.0)
          return x.mean()
        return wmean(losses), jax.tree.map(wmean, metrics)

      (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
      updates, opt_state = optimizer.update(grads, opt_state, params)
      params = optax.apply_updates(params, updates)
      metrics["total_loss"] = loss
      return (params, opt_state), metrics

    (params, opt_state), epoch_metrics = lax.scan(
      epoch, (params, opt_state), None, length=n_epochs
    )
    metrics = jax.tree.map(lambda x: x[-1], epoch_metrics)

    # Polyak target update every step; periodic hard reset of the magnet.
    target_params = jax.tree.map(lambda t, p: tau * p + (1.0 - tau) * t,
                                 target_params, params)
    reset = (step + 1) % magnet_interval == 0
    magnet_params = jax.tree.map(lambda m, p: jnp.where(reset, p, m),
                                 magnet_params, params)

    # Diagnostics: mean terminal return for player 0 (≈0 at equilibrium). The
    # env is not reset after a hand ends, so post-terminal steps re-fire rewards;
    # valid·done isolates the single real terminal step of each episode.
    terminal = valid * dones  # (B, T) — 1 only at the hand's terminal step
    p0_return = (traj["rewards"][..., 0] * terminal).sum(1).mean()
    metrics["p0_return"] = p0_return
    return (params, opt_state, target_params, magnet_params, step + 1, rng), metrics

  return train_step


# ── Config & main ────────────────────────────────────────────────────────────

_DEFAULTS = dict(
  n_epochs=2, batch_size=64, lr=3e-4,
  clip_eps=0.2, vf_coef=0.5, ent_coef=0.005,
  gamma=1.0, gae_lambda=0.95, delta_clip=1.0, trace_clip=1.0,
  magnet_coef=0.2, old_policy_coef=0.05,
  target_update_rate=0.001, magnet_interval=2000,
  hidden=(256, 256), grad_clip=1.0,
  # Bet-size Gaussian std clamp, in fraction-of-stack space (see ContActorCritic).
  log_sigma_min=-4.6, log_sigma_max=-1.4, log_sigma_init=-2.3,
)


def main(config_path: str | None = None) -> None:
  cfg = dict(seed=0, env=dict(name="hunl"), algorithm={}, trainer={})
  if config_path:
    with open(config_path) as f:
      loaded = yaml.safe_load(f)
    cfg.update(loaded)

  env = build_env(cfg["env"])
  alg = {**_DEFAULTS, **cfg.get("algorithm", {})}
  trainer = cfg.get("trainer", {})
  n_steps = int(trainer.get("n_steps", 20000))
  log_every = int(trainer.get("log_every", 200))
  ckpt_dir = trainer.get("checkpoint_dir")
  ckpt_every = int(trainer.get("checkpoint_every", 0))

  print(f"env: {env.__class__.__name__}  players={env.num_players}  "
        f"actions={env.num_actions}  bet_atoms={env.num_bet_atoms}  "
        f"max_len={env.max_length}")

  net = ContActorCritic(
    n_actions=env.num_actions, num_bet_atoms=env.num_bet_atoms,
    hidden=tuple(alg["hidden"]),
    log_sigma_min=alg["log_sigma_min"], log_sigma_max=alg["log_sigma_max"],
    log_sigma_init=alg["log_sigma_init"],
  )
  forward = make_forward(net)

  # Either an explicit `optimizer:` block (type + args, à la train.py) or the
  # flat `lr` (+ optional `grad_clip`) default.
  opt_cfg = alg.get("optimizer")
  if opt_cfg is not None:
    optimizer = build_optimizer(opt_cfg)
    opt_name = opt_cfg.get("type", "adam")
  else:
    optimizer = optax.adam(alg["lr"])
    if alg.get("grad_clip"):
      optimizer = optax.chain(optax.clip_by_global_norm(float(alg["grad_clip"])), optimizer)
    opt_name = "adam"
  print(f"optimizer: {opt_name}")

  key = jax.random.PRNGKey(cfg.get("seed", 0))
  key, init_key = jax.random.split(key)
  dummy = env.information_set(env.init_state(init_key), 0, init_key)
  params = net.init(init_key, dummy)["params"]
  opt_state = optimizer.init(params)
  n_params = sum(x.size for x in jax.tree.leaves(params))
  print(f"params: {n_params:,}  | training for {n_steps} steps\n")

  train_step = make_train_step(env, forward, optimizer, alg)
  carry = (params, opt_state, params, params, jnp.int32(0), key)

  # Run log_every steps per jitted scan, then surface metrics.
  @jax.jit
  def run_chunk(carry):
    return lax.scan(train_step, carry, None, length=log_every)

  start = time.time()
  done_steps = 0
  while done_steps < n_steps:
    carry, metrics = run_chunk(carry)
    m = jax.tree.map(lambda x: float(x[-1]), metrics)
    done_steps += log_every
    eps = done_steps * alg["batch_size"]
    print(f"step {done_steps:>7d} | {eps/ (time.time()-start):8.0f} eps/s | "
          f"loss {m['total_loss']:+.4f} | policy {m['policy_loss']:+.4f} | "
          f"value {m['value_loss']:.4f} | ent {m['entropy_loss']:+.4f} | "
          f"magnet {m['magnet_loss']:.4f} | p0_ret {m['p0_return']:+.4f}")
    if ckpt_dir and ckpt_every and done_steps % ckpt_every == 0:
      os.makedirs(ckpt_dir, exist_ok=True)
      with open(os.path.join(ckpt_dir, f"step_{done_steps}.pkl"), "wb") as f:
        pickle.dump(jax.tree.map(lambda x: x, carry[0]), f)

  print(f"\ndone. {done_steps} steps in {time.time()-start:.1f}s")


if __name__ == "__main__":
  main(sys.argv[1] if len(sys.argv) > 1 else None)
