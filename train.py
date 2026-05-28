#!/usr/bin/env python
"""Train an RL agent on any registered environment.

Usage:
    python train.py                                        # default PPO config
    python train.py configs/mmd_goofspiel.yaml             # MMD on Goofspiel
    python train.py configs/ppo_leduc.yaml                 # PPO on Leduc Hold'em
    python train.py configs/ppo_battleship.yaml            # PPO on Battleship
    python train.py configs/ppo_goofspiel.yaml --resume    # resume from latest checkpoint
"""

from __future__ import annotations
import os
import glob
import pickle
import sys
import yaml
import jax
import optax

from src.envs import build_env
from src.networks.configs import build_network
from src.agents.actor_critic import ActorCriticAgent, QActorCriticAgent, PolicyQAgent
from src.algorithms.ppo import PPO
from src.algorithms.mmd import MMD
from src.algorithms.mmd_q import QMMD
from src.trainers.trainer import StandardTrainer, StdoutLogger


_ALGORITHMS = {
  "ppo": PPO,
  "mmd": MMD,
  "qmmd": QMMD,
}

_AGENT_CLASSES = {
  "ppo": ActorCriticAgent,
  "mmd": ActorCriticAgent,
  "qmmd": PolicyQAgent,
}

_OPTIMIZERS = {
  "adam": lambda cfg: optax.adam(cfg["lr"], b1=cfg.get("b1", 0.9), b2=cfg.get("b2", 0.999)),
  "adamw": lambda cfg: optax.adamw(cfg["lr"], b1=cfg.get("b1", 0.9), b2=cfg.get("b2", 0.999), weight_decay=cfg.get("weight_decay", 1e-4)),
  "sgd": lambda cfg: optax.sgd(cfg["lr"], momentum=cfg.get("momentum", 0.0)),
  "muon": lambda cfg: optax.contrib.muon(cfg["lr"], beta=cfg.get("momentum", 0.95), nesterov=cfg.get("nesterov", True), ns_steps=cfg.get("ns_steps", 5), weight_decay=cfg.get("weight_decay", 0.0), adam_weight_decay=cfg.get("adam_weight_decay", 0.0), adam_learning_rate=cfg.get("adam_learning_rate", 3e-4)),
}


def build_optimizer(opt_cfg: dict) -> optax.GradientTransformation:
  opt_type = opt_cfg.get("type", "adam").lower()
  if opt_type not in _OPTIMIZERS:
    raise ValueError(f"Unknown optimizer '{opt_type}'. Choose from: {list(_OPTIMIZERS)}")
  return _OPTIMIZERS[opt_type](opt_cfg)


def _fill_env_dims(cfg: dict, env) -> dict:
  """Recursively replace null n_actions with env.num_actions."""
  out = {}
  for k, v in cfg.items():
    if isinstance(v, dict):
      out[k] = _fill_env_dims(v, env)
    elif k == "n_actions" and v is None:
      out[k] = env.num_actions
    else:
      out[k] = v
  return out


def _save_config(cfg: dict, checkpoint_dir: str) -> None:
  os.makedirs(checkpoint_dir, exist_ok=True)
  path = os.path.join(checkpoint_dir, "config.yaml")
  with open(path, "w") as f:
    yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
  print(f"config saved → {path}")


def load_config(checkpoint_dir: str) -> dict:
  """Load the config saved alongside a checkpoint directory."""
  path = os.path.join(checkpoint_dir, "config.yaml")
  with open(path) as f:
    return yaml.safe_load(f)


def load_latest_checkpoint(checkpoint_dir: str):
  """Return the TrainingState from the highest-numbered checkpoint in checkpoint_dir."""
  pattern = os.path.join(checkpoint_dir, "step_*.pkl")
  paths = sorted(glob.glob(pattern))
  if not paths:
    raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir!r}")
  latest = paths[-1]
  with open(latest, "rb") as f:
    state = pickle.load(f)
  print(f"resumed from checkpoint → {latest}  (step {int(state.step)})")
  return state


def main(config_path: str = "configs/ppo_goofspiel.yaml", resume: bool = False) -> None:
  with open(config_path) as f:
    cfg = yaml.safe_load(f)

  # ── Environment ────────────────────────────────────────────────────────
  env = build_env(cfg["env"])
  print(
    f"env: {env.__class__.__name__}  players={env.num_players}  actions={env.num_actions}  max_len={env.max_length}"
  )

  # ── Agent ──────────────────────────────────────────────────────────────
  alg_cfg = dict(cfg["algorithm"])
  agent_cfg = alg_cfg.pop("agent")
  alg_type = alg_cfg.pop("type").lower()

  net_cfg = _fill_env_dims(agent_cfg["network"], env)
  network = build_network(net_cfg)
  agent_class = _AGENT_CLASSES.get(alg_type, ActorCriticAgent)
  agent = agent_class(network)

  # ── Algorithm ──────────────────────────────────────────────────────────
  if alg_type not in _ALGORITHMS:
    raise ValueError(
      f"Unknown algorithm '{alg_type}'. Choose from: {list(_ALGORITHMS)}"
    )

  opt_cfg = alg_cfg.pop("optimizer", None)
  optimizer = build_optimizer(opt_cfg) if opt_cfg is not None else None
  algorithm = _ALGORITHMS[alg_type](env=env, agent=agent, optimizer=optimizer, **alg_cfg)
  print(
    f"algorithm: {alg_type.upper()}"
    f" | rollout={env.max_length}×{env.num_players} players"
    f" → {env.max_length * env.num_players} samples"
    f" | {algorithm.n_epochs} epochs"
    f" | batch={algorithm.batch_size}"
  )

  # ── Trainer ────────────────────────────────────────────────────────────
  trainer_cfg = dict(cfg["trainer"])
  n_steps = trainer_cfg.pop("n_steps")
  checkpoint_dir = trainer_cfg.get("checkpoint_dir")
  trainer = StandardTrainer(algorithm, logger=StdoutLogger(), **trainer_cfg)

  # ── Save config (once, before training starts) ─────────────────────────
  if checkpoint_dir:
    _save_config(cfg, checkpoint_dir)

  # ── Init or resume ─────────────────────────────────────────────────────
  if resume:
    if not checkpoint_dir:
      raise ValueError("Cannot resume: no checkpoint_dir set in trainer config")
    state = load_latest_checkpoint(checkpoint_dir)
    n_steps = max(0, n_steps - int(state.step))
    print(f"resuming — {n_steps} iterations remaining\n")
  else:
    key = jax.random.PRNGKey(cfg.get("seed", 0))
    state = algorithm.init(key)
    print(f"params: {sum(x.size for x in jax.tree.leaves(state.params)):,}")
    print(f"training for {n_steps} iterations …\n")
  # import chex
  # with chex.fake_jit():
  state = trainer.train(state, n_steps)
  print(f"\ndone. final step = {int(state.step)}")


if __name__ == "__main__":
  args = sys.argv[1:]
  resume = "--resume" in args
  args = [a for a in args if a != "--resume"]
  config_path = args[0] if args else "configs/qmmd.yaml"
  main(config_path, resume=resume)
