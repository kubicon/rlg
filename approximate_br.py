#!/usr/bin/env python
"""Train an approximate best response against a fixed opponent checkpoint.

The fixed opponent's player index is given by --player (or br.player in the
config). A fresh network is trained as the other player using PPO.

Usage:
    python approximate_br.py public_experiments/mmd_goofspiel/config.yaml \\
        --network public_experiments/mmd_goofspiel/50.pkl \\
        --player 0

Config extension (all existing fields unchanged, plus an optional br section):
    br:
      network_path: public_experiments/mmd_goofspiel/50.pkl
      player: 0
"""

from __future__ import annotations
import pickle
import sys
from typing import Any

import jax
import yaml

from src.envs import build_env
from src.networks.configs import build_network
from src.agents.actor_critic import ActorCriticAgent
from src.algorithms.br import BRAlgorithm
from src.trainers.trainer import StandardTrainer, StdoutLogger
from train import _fill_env_dims, _save_config

# Keys present in MMD config that BRAlgorithm (PPO-based) does not accept.
_MMD_ONLY_KEYS = frozenset({
  "magnet_coef", "old_policy_coef", "magnet_interval",
  "loss_type", "neurd_clip", "neurd_threshold",
})


def _load_opp_params(path: str) -> Any:
  with open(path, "rb") as f:
    state = pickle.load(f)
  extras = getattr(state, "extras", None) or {}
  if isinstance(extras, dict) and "target_params" in extras:
    return extras["target_params"]
  return state.params


def main(config_path: str, network_path: str | None, opp_player: int | None) -> None:
  with open(config_path) as f:
    cfg = yaml.safe_load(f)

  br_cfg = cfg.get("br", {})
  network_path = network_path or br_cfg.get("network_path")
  opp_player = opp_player if opp_player is not None else br_cfg.get("player", 0)

  if not network_path:
    raise ValueError("Provide --network or set br.network_path in the config")

  env = build_env(cfg["env"])
  print(
    f"env: {env.__class__.__name__}"
    f"  players={env.num_players}  actions={env.num_actions}  max_len={env.max_length}"
  )

  alg_cfg = dict(cfg["algorithm"])
  agent_cfg = alg_cfg.pop("agent")
  alg_cfg.pop("type")
  for k in _MMD_ONLY_KEYS:
    alg_cfg.pop(k, None)

  net_cfg = _fill_env_dims(agent_cfg["network"], env)
  network = build_network(net_cfg)
  agent = ActorCriticAgent(network)

  opp_params = _load_opp_params(network_path)
  br_player = (opp_player + 1) % env.num_players
  print(f"opponent loaded from {network_path}  (playing as player {opp_player})")
  print(f"training best response as player {br_player}")

  algorithm = BRAlgorithm(
    env=env,
    agent=agent,
    opp_params=opp_params,
    br_player=br_player,
    **alg_cfg,
  )
  print(
    f"algorithm: BRAlgorithm (PPO)"
    f" | rollout={env.max_length}"
    f" | {algorithm.n_epochs} epochs"
    f" | batch={algorithm.batch_size}"
  )

  trainer_cfg = dict(cfg["trainer"])
  n_steps = trainer_cfg.pop("n_steps")
  checkpoint_dir = trainer_cfg.get("checkpoint_dir")
  trainer = StandardTrainer(algorithm, logger=StdoutLogger(), **trainer_cfg)

  if checkpoint_dir:
    _save_config(cfg, checkpoint_dir)

  key = jax.random.PRNGKey(cfg.get("seed", 0))
  state = algorithm.init(key)
  print(f"params: {sum(x.size for x in jax.tree.leaves(state.params)):,}")
  print(f"training for {n_steps} iterations …\n")
  state = trainer.train(state, n_steps)
  print(f"\ndone. final step = {int(state.step)}")


if __name__ == "__main__":
  import argparse

  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("config", nargs="?", default="configs/mmd_goofspiel.yaml", help="Path to config YAML")
  p.add_argument("--network", default=None, metavar="PATH", help="Path to opponent checkpoint (.pkl)")
  p.add_argument("--player", type=int, default=None, metavar="N", help="Player index the opponent plays as (default: 0)")
  args = p.parse_args()
  main(args.config, args.network, args.player)
