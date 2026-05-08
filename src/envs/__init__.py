from .base import Env
from .goofspiel import Goofspiel
from .leduc_holdem import LeducHoldem
from .battleship import Battleship

_REGISTRY: dict[str, type] = {
    "goofspiel": Goofspiel,
    "leduc": LeducHoldem,
    "battleship": Battleship,
}


def build_env(cfg: dict) -> Env:
    """Instantiate an environment from a config dict.

    The dict must contain a ``name`` key matching a registered environment.
    All remaining keys are passed as keyword arguments to the constructor.

    Example config::

        name: goofspiel
        n_cards: 5
        prize_order: random
        reward_type: binary
    """
    cfg = dict(cfg)
    name = cfg.pop("name")
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown environment '{name}'. "
            f"Available: {list(_REGISTRY)}"
        )
    return _REGISTRY[name](**cfg)
