#!/usr/bin/env python
"""Train a PPO agent on Goofspiel.

Usage:
    python train_ppo.py                          # default config
    python train_ppo.py configs/my_config.yaml   # custom config
"""
from __future__ import annotations
import sys
import yaml
import jax
import chex

from src.envs.goofspiel import Goofspiel
from src.networks.configs import build_network
from src.agents.actor_critic import ActorCriticAgent
from src.algorithms.ppo import PPO
from src.trainers.trainer import StandardTrainer, StdoutLogger


def _fill_env_dims(cfg: dict, env) -> dict:
  """Recursively replace null n_actions with env.num_actions."""
  out = {}
  for k, v in cfg.items():
    if isinstance(v, dict):
      out[k] = _fill_env_dims(v, env)
    elif k == 'n_actions' and v is None:
      out[k] = env.num_actions
    else:
      out[k] = v
  return out


def main(config_path: str = 'configs/ppo_goofspiel.yaml') -> None:
  with open(config_path) as f:
    cfg = yaml.safe_load(f)

  # ── Environment ────────────────────────────────────────────────────────
  env = Goofspiel(**cfg['env'])
  print(f'env: Goofspiel(n_cards={env.n_cards}, prize_order={env.prize_order},'
        f' reward_type={env.reward_type})')

  # ── Agent (network lives inside agent config) ──────────────────────────
  alg_cfg   = dict(cfg['algorithm'])
  agent_cfg = alg_cfg.pop('agent')
  alg_type  = alg_cfg.pop('type')

  net_cfg = _fill_env_dims(agent_cfg['network'], env)
  network = build_network(net_cfg)
  agent   = ActorCriticAgent(network)

  # ── Algorithm ──────────────────────────────────────────────────────────
  assert alg_type == 'ppo', f"Unknown algorithm type '{alg_type}'"
  ppo = PPO(env=env, agent=agent, **alg_cfg)

  batch_size = ppo.batch_size
  print(f'PPO: rollout={ppo.rollout_len}×{env.num_players} players'
        f' → {ppo.rollout_len * env.num_players} samples'
        f' | {ppo.n_epochs} epochs '
        f' (batch={batch_size})')

  # ── Trainer ────────────────────────────────────────────────────────────
  trainer_cfg = dict(cfg['trainer'])
  n_steps     = trainer_cfg.pop('n_steps')
  trainer     = StandardTrainer(ppo, logger=StdoutLogger(), **trainer_cfg)

  # ── Init + train ───────────────────────────────────────────────────────
  key   = jax.random.PRNGKey(cfg.get('seed', 0))
  state = ppo.init(key)
  print(f'params: {sum(x.size for x in jax.tree.leaves(state.params)):,}')
  print(f'training for {n_steps} iterations …\n')

  # with chex.fake_jit():
  state = trainer.train(state, n_steps)
  print(f'\ndone. final step = {int(state.step)}')


if __name__ == '__main__':
  config_path = sys.argv[1] if len(sys.argv) > 1 else 'configs/ppo_goofspiel.yaml'
  main(config_path)
