from .base import Env
from .noisy import NoisyEnv, NoisyEnvState
from .goofspiel import Goofspiel
from .leduc_holdem import LeducHoldem
from .battleship import Battleship
from .liars_dice import LiarsDice
from .hex import Hex
from .mnk import MNK, TicTacToe
from .normal_form import (
  NormalFormGame,
  RockPaperScissors,
  BiasedRockPaperScissors,
  MatchingPennies,
  BiasedMatchingPennies,
)

_REGISTRY: dict[str, type] = {
  "goofspiel": Goofspiel,
  "leduc": LeducHoldem,
  "battleship": Battleship,
  "liars_dice": LiarsDice,
  "hex": Hex,
  "mnk": MNK,
  "tictactoe": TicTacToe,
  "normal_form": NormalFormGame,
  "rps": RockPaperScissors,
  "biased_rps": BiasedRockPaperScissors,
  "matching_pennies": MatchingPennies,
  "biased_matching_pennies": BiasedMatchingPennies,
}


def build_env(cfg: dict) -> Env:
  """Instantiate an environment from a config dict.

  The dict must contain a ``name`` key matching a registered environment.
  All remaining keys are passed as keyword arguments to the constructor.
  If ``obs_noise`` is present, the environment is wrapped in a NoisyEnv
  with that value as the noise variance.

  Example config::

      name: goofspiel
      n_cards: 5
      prize_order: random
      reward_type: binary
      obs_noise: 0.1   # optional — wraps in NoisyEnv(variance=0.1)
  """
  cfg = dict(cfg)
  obs_noise = cfg.pop("obs_noise", None)
  name = cfg.pop("name")
  if name not in _REGISTRY:
    raise ValueError(f"Unknown environment '{name}'. Available: {list(_REGISTRY)}")
  env = _REGISTRY[name](**cfg)
  if obs_noise is not None:
    env = NoisyEnv(env, variance=float(obs_noise))
  return env
