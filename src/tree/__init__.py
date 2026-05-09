"""Layered game-tree extraction and algorithms.

``extract_game_tree`` expands the full game tree in BFS order.  Each BFS wave
becomes one `GameTreeLayer` (parallel rows per distinct history at that depth).
Edges between consecutive layers are stored sparsely in `LayerTransition`.

Histories are never merged: each root-to-leaf path has its own row indices.

Per-frontier work uses two fused ``jax.jit(jax.vmap(...))`` kernels — one for
``current_player`` / ``legal_actions`` / ``information_set``, one for
``apply_action`` — limiting XLA compiles per batch shape.  Next-frontier states
are gathered with one batched index into ``next_state`` (no Python per-row
``tree.map`` slices).

``best_response_values`` runs two passes (player 0 BR vs ``strategy_p1``, then
player 1 BR vs ``strategy_p0``), each with counterfactual reach propagation and
imperfect-information ``argmax`` per information set.

``counterfactual_regret_minimization`` implements tabular CFR on the same tree:
forward counterfactual reaches, backward expected values under the current
strategy, regret accumulation per information set, and average-strategy
tracking (two-player zero-sum).

Supported environments (full enumeration — small games only):
  - ``Goofspiel(...)`` — ``prize_order=='random'`` enumerates all prize permutations;
    ``'ascending'`` / ``'descending'`` use a single fixed deck.
  - ``Battleship(board_size=2, ship_lengths=[2])``
  - ``LeducHoldem()``
"""

from __future__ import annotations

from .best_response import BestResponseValues, best_response_values
from .cfr import CFRResult, InfosetKey, counterfactual_regret_minimization
from .extract import extract_game_tree
from .types import (
    EK_CHANCE,
    EK_PLAYER0,
    EK_PLAYER1,
    EK_SIMULTANEOUS,
    GameTreeLayer,
    LayerTransition,
    LayeredGameTree,
    NK_CHANCE,
    NK_PLAYER,
    NK_SIMULTANEOUS,
    NK_TERMINAL,
    Strategy,
    policy_prob,
    tree_num_actions,
)

__all__ = [
    "BestResponseValues",
    "CFRResult",
    "EK_CHANCE",
    "EK_PLAYER0",
    "EK_PLAYER1",
    "EK_SIMULTANEOUS",
    "GameTreeLayer",
    "InfosetKey",
    "LayerTransition",
    "LayeredGameTree",
    "NK_CHANCE",
    "NK_PLAYER",
    "NK_SIMULTANEOUS",
    "NK_TERMINAL",
    "Strategy",
    "best_response_values",
    "counterfactual_regret_minimization",
    "extract_game_tree",
    "policy_prob",
    "tree_num_actions",
]
