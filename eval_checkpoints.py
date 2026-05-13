#!/usr/bin/env python
"""Compute exploitability for every checkpoint in a directory.

Usage:
    python eval_checkpoints.py <checkpoint_dir>
    python eval_checkpoints.py data/mmd_goofspiel
"""

from __future__ import annotations

import glob
import os
import pickle
import sys
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import yaml

from src.envs import build_env
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
from train import _fill_env_dims


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


def _build_apply_fn(network):
  """Return a JIT-compiled (params, obs_batch) -> logits_batch function."""

  def _apply(params, obs_batch):
    def single(obs):
      (logits, _), _ = network.apply({"params": params}, obs, None)
      return logits

    return jax.vmap(single)(obs_batch)

  return jax.jit(_apply)


def _params_to_strategy(
  params: Any,
  ids: list[bytes],
  obs_batch: np.ndarray,
  mask_batch: np.ndarray,
  apply_fn,
) -> Strategy:
  """Single batched forward pass → Strategy dict."""
  if not ids:
    return {}

  logits_batch = np.asarray(apply_fn(params, obs_batch))  # (N, A)
  strategy: Strategy = {}
  for idx, iset_id in enumerate(ids):
    logits = logits_batch[idx]
    mask = mask_batch[idx]
    logits = np.where(mask, logits, -1e9)
    logits -= logits.max()
    probs = np.where(mask, np.exp(logits), 0.0)
    probs /= probs.sum()
    strategy[iset_id] = probs

  return strategy


# ── Main ─────────────────────────────────────────────────────────────────────


def main(checkpoint_dir: str) -> None:
  config_path = os.path.join(checkpoint_dir, "config.yaml")
  if not os.path.exists(config_path):
    raise FileNotFoundError(f"No config.yaml found in {checkpoint_dir!r}")

  with open(config_path) as f:
    cfg = yaml.safe_load(f)

  # ── Build env and agent ───────────────────────────────────────────────
  env = build_env(cfg["env"])
  print(f"env: {env.__class__.__name__}  actions={env.num_actions}")

  alg_cfg = dict(cfg["algorithm"])
  agent_cfg = alg_cfg.pop("agent")
  net_cfg = _fill_env_dims(agent_cfg["network"], env)
  network = build_network(net_cfg)
  agent = ActorCriticAgent(network)  # noqa: F841 (kept for symmetry with train.py)

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

  apply_fn = _build_apply_fn(network)
  # Warm up JIT with dummy params from a throw-away init.
  dummy_key = jax.random.key(0)
  dummy_obs = obs0[:1] if len(obs0) > 0 else obs1[:1]
  dummy_params = agent.init_params(dummy_key, dummy_obs[0])
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

    strat0 = _params_to_strategy(params, ids0, obs0, mask0, apply_fn)
    strat1 = _params_to_strategy(params, ids1, obs1, mask1, apply_fn)

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
  if len(sys.argv) != 2:
    print(__doc__)
    sys.exit(1)
  main(sys.argv[1])
