#!/usr/bin/env python
"""Compute exploitability for every checkpoint in a directory.

Usage:
    python eval_checkpoints.py <checkpoint_dir>
    python eval_checkpoints.py data/mmd_goofspiel
    python eval_checkpoints.py data/mmd_goofspiel --threshold nucleus --epsilon 0.05
"""

from __future__ import annotations

import argparse
import glob
import os
import pickle
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import yaml

from src.envs import build_env, NoisyEnv
from src.envs.leduc_holdem import LeducHoldem
from src.networks.configs import build_network
from src.agents.actor_critic import ActorCriticAgent
from src.tree import (
  NK_PLAYER,
  NK_TERMINAL,
  LayeredGameTree,
  Strategy,
  extract_game_tree,
)
from src.tree.best_response import best_response_values
from train import _fill_env_dims, _AGENT_CLASSES, build_schedules


# ── Tree preprocessing (done once) ───────────────────────────────────────────


def _collect_player_infosets(
  tree: LayeredGameTree,
  env,
  player: int,
) -> tuple[list[bytes], np.ndarray, np.ndarray]:
  """Collect all unique info sets for one player.

  Returns:
      ids:        list of info_set_id bytes, length N
      obs_batch:  float32 array (N, obs_dim) — network inputs
      mask_batch: bool array (N, num_actions)
  """
  seen: dict[bytes, None] = {}
  ids: list[bytes] = []
  obs_list: list[np.ndarray] = []
  mask_list: list[np.ndarray] = []

  for lay in tree.layers:
    if lay.state_batch is None:
      continue

    iset_ids = lay.info_set_id_p0 if player == 0 else lay.info_set_id_p1
    masks = lay.legal_mask_p0 if player == 0 else lay.legal_mask_p1

    # Batch observation extraction for the whole layer at once.
    obs_all = np.asarray(
      jax.vmap(lambda s: env.information_set(s, jnp.int32(player), jax.random.key(0)))(lay.state_batch)
    )

    for i in range(lay.n_nodes):
      nk = int(lay.node_kind[i])
      if nk == NK_TERMINAL:
        continue
      if nk == NK_PLAYER and int(lay.acting_player[i]) != player:
        continue
      iset_id = iset_ids[i]
      if iset_id in seen:
        continue
      seen[iset_id] = None
      ids.append(iset_id)
      obs_list.append(obs_all[i])
      mask_list.append(masks[i])

  if not ids:
    return [], np.zeros((0, 1), dtype=np.float32), np.zeros((0, 1), dtype=bool)

  return ids, np.stack(obs_list).astype(np.float32), np.stack(mask_list).astype(bool)


def _build_apply_fn(agent, env):
  """Return a JIT-compiled (params, infoset_batch) -> logits_batch function."""
  return agent.build_eval_fn(env)


def _apply_threshold(probs: np.ndarray, mode: str, epsilon: float) -> np.ndarray:
  """Filter a probability vector and renormalize.

  Modes:
      "epsilon": zero out any action played with probability < epsilon.
      "nucleus": top-p filtering — keep the smallest set of highest-probability
                 actions whose cumulative mass reaches (1 - epsilon), drop the
                 rest (the tail of total mass epsilon).

  With epsilon == 0.0 both modes are the identity (no filtering).
  """
  if epsilon <= 0.0:
    return probs

  if mode == "epsilon":
    filtered = np.where(probs < epsilon, 0.0, probs)
  elif mode == "nucleus":
    order = np.argsort(probs)[::-1]  # descending
    cumulative = np.cumsum(probs[order])
    # Keep every action up to and including the one that first reaches the
    # (1 - epsilon) cumulative-mass cutoff.
    keep_sorted = cumulative < (1.0 - epsilon)
    if keep_sorted.size > 0:
      keep_sorted[np.argmax(~keep_sorted)] = True  # include the crossing action
    keep = np.zeros_like(probs, dtype=bool)
    keep[order] = keep_sorted
    filtered = np.where(keep, probs, 0.0)
  else:
    raise ValueError(f"unknown threshold mode {mode!r}")

  total = filtered.sum()
  if total <= 0.0:
    # Degenerate: everything got filtered out — fall back to the original.
    return probs
  return filtered / total


def _entmax_probs(logits: np.ndarray, mask: np.ndarray, alpha: float) -> np.ndarray:
  """α-entmax over legal actions (numpy), matching src/policy.py at eval time."""
  am1 = alpha - 1.0
  z = np.where(mask, logits, -np.inf) * am1
  d = int(mask.sum())
  max_val = z.max()
  tau_lo = max_val - 1.0
  tau_hi = max_val - (1.0 / d) ** am1
  p_of = lambda tau: np.maximum(z - tau, 0.0) ** (1.0 / am1)
  dm = tau_hi - tau_lo
  for _ in range(50):
    dm *= 0.5
    tau_m = tau_lo + dm
    if p_of(tau_m).sum() - 1.0 >= 0.0:
      tau_lo = tau_m
  p = p_of(tau_lo)
  return p / p.sum()


def _params_to_strategy(
  params: Any,
  ids: list[bytes],
  obs_batch: np.ndarray,
  mask_batch: np.ndarray,
  apply_fn,
  threshold_mode: str = "epsilon",
  epsilon: float = 0.0,
  alpha: float = 1.0,
) -> Strategy:
  """Single batched forward pass → Strategy dict."""
  if not ids:
    return {}

  logits_batch = np.asarray(apply_fn(params, obs_batch))  # (N, A)
  strategy: Strategy = {}
  for idx, iset_id in enumerate(ids):
    logits = logits_batch[idx]
    mask = mask_batch[idx]
    if alpha == 1.0:
      logits = np.where(mask, logits, -1e9)
      logits -= logits.max()
      probs = np.where(mask, np.exp(logits), 0.0)
      probs /= probs.sum()
    else:
      probs = _entmax_probs(logits, mask, alpha)
    probs = _apply_threshold(probs, threshold_mode, epsilon)
    strategy[iset_id] = probs

  return strategy


# ── Main ─────────────────────────────────────────────────────────────────────


def main(
  checkpoint_dir: str,
  threshold_mode: str = "epsilon",
  epsilon: float = 0.0,
) -> None:
  config_path = os.path.join(checkpoint_dir, "config.yaml")
  if not os.path.exists(config_path):
    raise FileNotFoundError(f"No config.yaml found in {checkpoint_dir!r}")

  with open(config_path) as f:
    cfg = yaml.safe_load(f)

  if epsilon > 0.0:
    print(f"thresholding: mode={threshold_mode}  epsilon={epsilon}")
  else:
    print("thresholding: disabled (epsilon=0.0)")

  # ── Build env and agent ───────────────────────────────────────────────
  env = build_env(cfg["env"])
  if isinstance(env, NoisyEnv):
    env = env._env
    print(f"env: NoisyEnv unwrapped → {env.__class__.__name__}  actions={env.num_actions}")
  else:
    print(f"env: {env.__class__.__name__}  actions={env.num_actions}")

  alg_cfg = dict(cfg["algorithm"])
  agent_cfg = alg_cfg.pop("agent")
  alg_type = alg_cfg.get("type", "mmd").lower()
  # Policy-transform exponent. If α was annealed during training, score each
  # checkpoint with α(step) — the value the agent was actually playing at that
  # step — rather than the converged value. Otherwise α is constant.
  alpha = float(alg_cfg.get("alpha", 1.0))
  alpha_schedule = None
  sched_cfg = alg_cfg.get("schedules") or {}
  if "alpha" in sched_cfg:
    alpha_schedule = build_schedules({"alpha": sched_cfg["alpha"]})["alpha"]
    print("policy transform: α-entmax (annealed; α reconstructed per checkpoint)")
  elif alpha != 1.0:
    print(f"policy transform: α-entmax (alpha={alpha})")
  net_cfg = _fill_env_dims(agent_cfg["network"], env)
  network = build_network(net_cfg)
  explicit_agent_type = agent_cfg.get("type")
  agent_class = _AGENT_CLASSES.get(explicit_agent_type or alg_type, ActorCriticAgent)
  agent = agent_class(network)

  # ── Extract game tree (once) ──────────────────────────────────────────
  print("extracting game tree …", flush=True)
  tree = extract_game_tree(env)
  print(f"  {tree.n_nodes} nodes, {tree.n_layers} layers")

  # ── Pre-collect info sets for both players (once) ─────────────────────
  print("collecting info sets …", flush=True)
  ids0, obs0, mask0 = _collect_player_infosets(tree, env, 0)
  ids1, obs1, mask1 = _collect_player_infosets(tree, env, 1)
  print(f"  player 0: {len(ids0)} unique info sets")
  print(f"  player 1: {len(ids1)} unique info sets")

  apply_fn = _build_apply_fn(agent, env)
  # Warm up JIT with dummy params from a throw-away init.
  dummy_key = jax.random.key(0)
  dummy_obs = obs0[:1] if len(obs0) > 0 else obs1[:1]
  init_env_state = env.init_state(dummy_key)
  dummy_obs_for_init = agent.dummy_obs(env, init_env_state, dummy_key)
  dummy_params = agent.init_params(dummy_key, dummy_obs_for_init)
  _ = apply_fn(dummy_params, dummy_obs)
  print("JIT compiled.\n")

  # ── Load and sort checkpoints ─────────────────────────────────────────
  paths = sorted(
    glob.glob(os.path.join(checkpoint_dir, "*.pkl")),
    key=lambda p: int(os.path.splitext(os.path.basename(p))[0]),
  )
  if not paths:
    raise FileNotFoundError(f"No .pkl checkpoints found in {checkpoint_dir!r}")

  print(f"{'ckpt':>6}  {'step':>8}  {'br_p0':>9}  {'br_p1':>9}  {'nashconv':>10}")
  print("-" * 52)
  init_state = env.init_state(jax.random.key(0))
  init_iset0 = np.asarray(env.information_set(init_state, jnp.int32(0), jax.random.key(0))).tobytes()
  init_iset1 = np.asarray(env.information_set(init_state, jnp.int32(1), jax.random.key(0))).tobytes()
  results: list[dict] = []
  for path in paths:
    ckpt_id = int(os.path.splitext(os.path.basename(path))[0])
    with open(path, "rb") as f:
      state = pickle.load(f)

    extras = state.extras if hasattr(state, "extras") else None
    params = (extras or {}).get("target_params", state.params)
    step = int(state.step)
    ckpt_alpha = float(alpha_schedule(step)) if alpha_schedule is not None else alpha

    strat0 = _params_to_strategy(params, ids0, obs0, mask0, apply_fn, threshold_mode, epsilon, ckpt_alpha)
    strat1 = _params_to_strategy(params, ids1, obs1, mask1, apply_fn, threshold_mode, epsilon, ckpt_alpha)

    if isinstance(env, LeducHoldem):
      _RANK_NAMES = ["J", "Q", "K"]
      dummy_state = env.init_state(jax.random.key(0))
      key0 = jax.random.key(0)
      for rank in range(3):
        s = dummy_state._replace(private_cards=jnp.array([rank, 0], dtype=jnp.int32))
        iset0 = np.asarray(env.information_set(s, jnp.int32(0), key0)).tobytes()
        iset1 = np.asarray(env.information_set(s._replace(private_cards=jnp.array([0, rank], dtype=jnp.int32)), jnp.int32(1), key0)).tobytes()
        p0 = strat0.get(iset0)
        p1 = strat1.get(iset1)
        p0_str = " ".join(f"{p:.3f}" for p in p0) if p0 is not None else "not found"
        p1_str = " ".join(f"{p:.3f}" for p in p1) if p1 is not None else "not found"
        print(f"  [ckpt {ckpt_id}] p0 card={_RANK_NAMES[rank]}: [{p0_str}]")
        print(f"  [ckpt {ckpt_id}] p1 card={_RANK_NAMES[rank]}: [{p1_str}]")
    else:
      p0_init = strat0.get(init_iset0)
      p1_init = strat1.get(init_iset1)
      p0_str = (
        " ".join(f"{p:.3f}" for p in p0_init) if p0_init is not None else "not found"
      )
      p1_str = (
        " ".join(f"{p:.3f}" for p in p1_init) if p1_init is not None else "not found"
      )
      print(f"  [ckpt {ckpt_id}] p0 init strategy: [{p0_str}]")
      print(f"  [ckpt {ckpt_id}] p1 init strategy: [{p1_str}]")

    bv = best_response_values(tree, strat0, strat1)
    # In a zero-sum game: NashConv = gain from p0 deviating + gain from p1 deviating.
    # v0_p0_br: best p0 can do against sigma_1 (upper bound for p0)
    # v1_p1_br: best p1 can do against sigma_0 (upper bound for p1)
    # At Nash both equal the game value; NashConv >= 0 always.
    nashconv = bv.v0_p0_br + bv.v1_p1_br

    print(
      f"{ckpt_id:>6}  {step:>8}  {bv.v0_p0_br:>9.4f}  {bv.v1_p1_br:>9.4f}  {nashconv:>10.4f}"
    )
    results.append(
      {
        "ckpt_id": ckpt_id,
        "step": step,
        "v0_p0_br": float(bv.v0_p0_br),
        "v1_p1_br": float(bv.v1_p1_br),
        "nashconv": float(nashconv),
      }
    )

  out_path = os.path.join(checkpoint_dir, "exploitability.npz")
  np.savez(
    out_path,
    ckpt_id=np.array([r["ckpt_id"] for r in results]),
    step=np.array([r["step"] for r in results]),
    v0_p0_br=np.array([r["v0_p0_br"] for r in results]),
    v1_p1_br=np.array([r["v1_p1_br"] for r in results]),
    nashconv=np.array([r["nashconv"] for r in results]),
  )
  print(f"\nresults saved → {out_path}")


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("checkpoint_dir", help="directory containing config.yaml and *.pkl checkpoints")
  parser.add_argument(
    "--threshold",
    dest="threshold_mode",
    choices=["epsilon", "nucleus"],
    default="epsilon",
    help="thresholding mode: 'epsilon' zeros out actions with prob < epsilon; "
    "'nucleus' applies top-p filtering keeping cumulative mass (1 - epsilon). "
    "(default: epsilon)",
  )
  parser.add_argument(
    "--epsilon",
    type=float,
    default=0.0,
    help="thresholding epsilon (default: 0.0, i.e. no filtering)",
  )
  args = parser.parse_args()
  main(args.checkpoint_dir, args.threshold_mode, args.epsilon)
