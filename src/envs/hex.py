"""Hex (and Dark Hex) for 2 sequential-move players.

Standard Hex rules:
  - Played on an n×n rhombus board.
  - Player 0 (Blue) must connect the top row to the bottom row.
  - Player 1 (Red) must connect the left column to the right column.
  - Players alternate placing one stone per turn on any empty cell.
  - First player to form a connected path between their two edges wins.
  - Draws are impossible.

Dark Hex variant (dark=True):
  - Each player only observes the cells they themselves have placed a stone on,
    plus cells where their placement attempt was blocked (i.e. the opponent's
    stone is revealed at that position).
  - A player may never attempt the same cell twice, regardless of outcome.
  - Maximum game length rises to 2·n² (both players may attempt every cell).
  - abrupt=True  (default): a blocked attempt loses the player's turn.
  - abrupt=False           : a blocked attempt does NOT switch the turn; the
                             player acts again until a stone is successfully placed.

Hex neighbours of (r, c): (r-1,c), (r-1,c+1), (r,c-1), (r,c+1), (r+1,c-1), (r+1,c).

Action encoding (num_actions = n²):
  action a: row = a // n, col = a % n

Win detection: iterative hex-grid flood-fill via jax.lax.scan for n² steps.

Observation sizes (float32); A = n²:

  Standard Hex:
    player_observation : 2·A + 1   own_stones(A) | opp_stones(A) | cur_player(1)
    public_observation : 2·A + 1   p0_stones(A) | p1_stones(A) | cur_player(1)
    state_observation  : 2·A + 1   (same — Hex is perfect-information)
    information_set    : 2·A + 1   (same as player_observation)
    public_state       : 2·A + 1   (same as public_observation)

  Dark Hex:
    player_observation : 2·A + 1   own_placed(A) | blocked(A) | cur_player(1)
    public_observation : 2·A + 1   zeros(2·A) | cur_player(1)  [board is hidden]
    state_observation  : 4·A + 1   p0_stones(A) | p1_stones(A)
                                   | attempted_p0(A) | attempted_p1(A) | cur_player(1)
    information_set    : 2·A + 1   ordered_attempts(2·A) | cur_player(1)
                                   per slot: (cell/A, was_blocked) — -1 for unused slots
    public_state       : 2·A+2        one_hot_step(2·A+1) | cur_player(1)  [abrupt]
                                     A+2                one_hot_placed(A+1) | cur_player(1)  [non-abrupt]
"""

from __future__ import annotations
from typing import NamedTuple

import jax
import jax.numpy as jnp

from .base import Env, Info, PRNGKey


class HexState(NamedTuple):
    board: jax.Array         # (n, n) int32  — 0=empty, 1=p0 stone, 2=p1 stone
    attempted: jax.Array     # (2, n, n) bool — cells each player has attempted
    cur_player: jax.Array    # ()      int32
    step: jax.Array          # ()      int32
    done: jax.Array          # ()      bool
    attempt_order: jax.Array # (2, n²) int32 — chronological attempt sequence per player; -1=unused


class Hex(Env):
    """Hex (and Dark Hex) for 2 sequential-move players.

    Args:
        board_size: side length n of the n×n board (default 11).
        dark:       if True, play the Dark Hex variant. Default False.
        abrupt:     only relevant when dark=True. If True (default), a blocked
                    attempt loses the player's turn. If False, the player acts
                    again until a stone is successfully placed.
    """

    def __init__(self, board_size: int = 11, dark: bool = False, abrupt: bool = True) -> None:
        self.board_size = board_size
        self.dark = dark
        self.abrupt = abrupt

    # ── Static properties ─────────────────────────────────────────────────────

    @property
    def num_players(self) -> int:
        return 2

    @property
    def max_length(self) -> int:
        n = self.board_size
        # Dark Hex: each player may attempt every cell once → 2·n² total steps.
        return 2 * n * n if self.dark else n * n

    @property
    def num_actions(self) -> int:
        return self.board_size ** 2

    @property
    def max_reward(self) -> float:
        return 1.0

    # ── State lifecycle ───────────────────────────────────────────────────────

    def init_state(self, key: PRNGKey) -> HexState:
        n = self.board_size
        return HexState(
            board=jnp.zeros((n, n), dtype=jnp.int32),
            attempted=jnp.zeros((2, n, n), dtype=jnp.bool_),
            cur_player=jnp.int32(0),
            step=jnp.int32(0),
            done=jnp.bool_(False),
            attempt_order=jnp.full((2, n * n), -1, dtype=jnp.int32),
        )

    def apply_action(
        self,
        state: HexState,
        actions: jax.Array,
        key: PRNGKey,
    ) -> tuple[HexState, jax.Array, jax.Array, Info]:
        cp = state.cur_player
        action = actions[cp]

        r = action // self.board_size
        c = action % self.board_size

        if self.dark:
            cell_empty = state.board[r, c] == 0
            # Place stone only when the cell is unoccupied; record attempt either way.
            new_board = jnp.where(cell_empty, state.board.at[r, c].set(cp + 1), state.board)
            new_attempted = state.attempted.at[cp, r, c].set(True)
            # Win is only possible when a stone was actually placed.
            won = cell_empty & self._check_win(new_board, cp)
            # keep_turn: stay with same player when won OR (non-abrupt + blocked).
            keep_turn = won | (~cell_empty if not self.abrupt else jnp.bool_(False))
            # Record ordered attempt history for perfect-recall information_set.
            p_attempt = state.attempted[cp].sum().astype(jnp.int32)
            new_attempt_order = state.attempt_order.at[cp, p_attempt].set(action)
        else:
            new_board = state.board.at[r, c].set(cp + 1)
            new_attempted = state.attempted
            won = self._check_win(new_board, cp)
            keep_turn = won
            new_attempt_order = state.attempt_order

        p0_reward = jnp.where(cp == 0, jnp.float32(1.0), jnp.float32(-1.0))
        terminal_rewards = jnp.stack([p0_reward, -p0_reward])
        rewards = jnp.where(won, terminal_rewards, jnp.zeros(2, dtype=jnp.float32))

        new_state = HexState(
            board=new_board,
            attempted=new_attempted,
            cur_player=jnp.where(keep_turn, state.cur_player, jnp.int32(1 - cp)),
            step=state.step + 1,
            done=won,
            attempt_order=new_attempt_order,
        )
        return new_state, rewards, won, {}

    # ── Win detection ─────────────────────────────────────────────────────────

    def _hex_expand(self, mask: jax.Array) -> jax.Array:
        """Expand a (n, n) bool reachability mask by one hex step (all 6 directions)."""
        n = self.board_size
        p = jnp.pad(mask.astype(jnp.int32), 1)
        # For cell (r, c), its 6 hex neighbours in the padded array at offset (r+1, c+1):
        return (
            p[0:n,   1:n+1]    # (r-1, c)
            | p[0:n,   2:n+2]  # (r-1, c+1)
            | p[1:n+1, 0:n]    # (r,   c-1)
            | p[1:n+1, 2:n+2]  # (r,   c+1)
            | p[2:n+2, 0:n]    # (r+1, c-1)
            | p[2:n+2, 1:n+1]  # (r+1, c)
        ).astype(jnp.bool_)

    def _check_win(self, board: jax.Array, player: jax.Array) -> jax.Array:
        """Return True if player (JAX int 0 or 1) has formed a winning connection."""
        n = self.board_size
        owns = board == (player + 1)

        reachable_p0 = jnp.zeros((n, n), dtype=jnp.bool_).at[0, :].set(owns[0, :])
        reachable_p1 = jnp.zeros((n, n), dtype=jnp.bool_).at[:, 0].set(owns[:, 0])
        reachable = jnp.where(player == 0, reachable_p0, reachable_p1)

        def step(r, _):
            return owns & (r | self._hex_expand(r)), None

        # n² steps guarantees convergence even for paths winding through all cells.
        final, _ = jax.lax.scan(step, reachable, None, length=n * n)

        win_p0 = jnp.any(final[n - 1, :])
        win_p1 = jnp.any(final[:, n - 1])
        return jnp.where(player == 0, win_p0, win_p1)

    # ── Observations ──────────────────────────────────────────────────────────

    def player_observation(
        self, state: HexState, player_id: jax.Array, key: PRNGKey
    ) -> jax.Array:
        if self.dark:
            # own_placed(A): cells where player successfully placed a stone.
            own_p0 = (state.board == 1).astype(jnp.float32).flatten()
            own_p1 = (state.board == 2).astype(jnp.float32).flatten()
            own = jnp.where(player_id == 0, own_p0, own_p1)
            # blocked(A): cells player attempted but found occupied by the opponent.
            blocked_p0 = (state.attempted[0] & (state.board != 1)).astype(jnp.float32).flatten()
            blocked_p1 = (state.attempted[1] & (state.board != 2)).astype(jnp.float32).flatten()
            blocked = jnp.where(player_id == 0, blocked_p0, blocked_p1)
            return jnp.concatenate([own, blocked, jnp.array([state.cur_player], dtype=jnp.float32)])
        else:
            own = (state.board == (player_id + 1)).astype(jnp.float32).flatten()
            opp = (state.board == (2 - player_id)).astype(jnp.float32).flatten()
            return jnp.concatenate([own, opp, jnp.array([state.cur_player], dtype=jnp.float32)])

    def public_observation(self, state: HexState, key: PRNGKey) -> jax.Array:
        if self.dark:
            # Board position is not common knowledge in Dark Hex.
            A = self.board_size ** 2
            return jnp.concatenate([
                jnp.zeros(2 * A, dtype=jnp.float32),
                jnp.array([state.cur_player], dtype=jnp.float32),
            ])
        else:
            p0 = (state.board == 1).astype(jnp.float32).flatten()
            p1 = (state.board == 2).astype(jnp.float32).flatten()
            return jnp.concatenate([p0, p1, jnp.array([state.cur_player], dtype=jnp.float32)])

    def state_observation(self, state: HexState, key: PRNGKey) -> jax.Array:
        """Full ground-truth state. Includes attempted masks for Dark Hex."""
        p0 = (state.board == 1).astype(jnp.float32).flatten()
        p1 = (state.board == 2).astype(jnp.float32).flatten()
        base = jnp.concatenate([p0, p1, jnp.array([state.cur_player], dtype=jnp.float32)])
        if self.dark:
            att_p0 = state.attempted[0].astype(jnp.float32).flatten()
            att_p1 = state.attempted[1].astype(jnp.float32).flatten()
            return jnp.concatenate([p0, p1, att_p0, att_p1,
                                    jnp.array([state.cur_player], dtype=jnp.float32)])
        return base

    # ── Action legality & turn order ──────────────────────────────────────────

    def legal_actions(self, state: HexState, player_id: jax.Array | int) -> jax.Array:
        """Active player: unattempted cells (dark) or empty cells (standard).
        Inactive player: only action 0 (dummy)."""
        is_active = jnp.int32(player_id) == state.cur_player

        if self.dark:
            # Any cell not yet attempted by this player is legal.
            not_tried_p0 = (~state.attempted[0]).flatten()
            not_tried_p1 = (~state.attempted[1]).flatten()
            active_mask = jnp.where(player_id == 0, not_tried_p0, not_tried_p1)
        else:
            active_mask = (state.board.flatten() == 0)

        inactive_mask = jnp.zeros(self.num_actions, dtype=jnp.bool_).at[0].set(True)
        return jnp.where(is_active, active_mask, inactive_mask)

    def current_player(self, state: HexState) -> jax.Array:
        return state.cur_player

    # ── Perfect-recall representations ────────────────────────────────────────

    def information_set(
        self, state: HexState, player_id: jax.Array | int, key: PRNGKey
    ) -> jax.Array:
        # Standard Hex: board is full history (perfect information).
        if not self.dark:
            return self.player_observation(state, jnp.int32(player_id), key)
        # Dark Hex: encode the ordered attempt sequence for perfect recall.
        # Each slot i holds (cell/n², was_blocked): -1.0 padding for unused slots.
        pid = jnp.int32(player_id)
        n2 = self.num_actions
        attempt_seq = jnp.where(pid == 0, state.attempt_order[0], state.attempt_order[1])

        def encode_one(cell_idx: jax.Array) -> jax.Array:
            valid = cell_idx >= 0
            safe = jnp.where(valid, cell_idx, jnp.int32(0))
            r = safe // self.board_size
            c = safe % self.board_size
            placed = state.board[r, c] == (pid + 1)
            norm = jnp.where(valid, safe.astype(jnp.float32) / n2, jnp.float32(-1.0))
            blocked = jnp.where(valid, (~placed).astype(jnp.float32), jnp.float32(-1.0))
            return jnp.stack([norm, blocked])

        seq = jax.vmap(encode_one)(attempt_seq).flatten()  # (2·n²,)
        return jnp.concatenate([seq, jnp.array([state.cur_player], dtype=jnp.float32)])

    def public_state(self, state: HexState, key: PRNGKey) -> jax.Array:
        # Standard Hex: same as public_observation.
        if not self.dark:
            return self.public_observation(state, key)
        # Abrupt: every step is a turn switch, so the step index is public.
        # Non-abrupt: step count leaks how many blocks occurred (private), so
        # only the number of successfully placed stones is public.
        if self.abrupt:
            indicator = jax.nn.one_hot(state.step, self.max_length + 1, dtype=jnp.float32)
        else:
            num_placed = (state.board > 0).sum()
            indicator = jax.nn.one_hot(num_placed, self.num_actions + 1, dtype=jnp.float32)
        return jnp.concatenate([indicator, jnp.array([state.cur_player], dtype=jnp.float32)])

    def state_representation(self, state: HexState, key: PRNGKey) -> jax.Array:
        return self.state_observation(state, key)
