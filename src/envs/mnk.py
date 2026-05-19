"""M,N,K game (generalized Tic-Tac-Toe) for 2 sequential-move players.

Standard rules:
  - Played on an m×n board.
  - Players alternate placing one stone per turn on any empty cell.
  - First player to place k stones in a row — horizontally, vertically, or
    diagonally — wins.
  - If the board fills with no winner, the game is a draw (reward 0).

Dark variant (dark=True):
  - Each player only observes the cells where they have successfully placed a
    stone, plus cells where their placement attempt was blocked (opponent's
    stone is revealed at that position).
  - A player may never attempt the same cell twice, regardless of outcome.
  - Maximum game length rises to 2·m·n (both players may attempt every cell).
  - abrupt=True  (default): a blocked attempt loses the player's turn.
  - abrupt=False           : a blocked attempt does NOT switch the turn; the
                             player acts again until a stone is successfully placed.

Action encoding (num_actions = m·n):
  action a: row = a // n, col = a % n

Win detection: all horizontal, vertical, and diagonal windows of length k are
checked by exhaustive static enumeration (unrolled at compile time).

Observation sizes (float32); A = m·n:

  Standard:
    player_observation : 2·A + 1   own_stones(A) | opp_stones(A) | cur_player(1)
    public_observation : 2·A + 1   p0_stones(A) | p1_stones(A) | cur_player(1)
    state_observation  : 2·A + 1   (same — standard MNK is perfect-information)

  Dark:
    player_observation : 2·A + 1   own_placed(A) | blocked(A) | cur_player(1)
    public_observation : 2·A + 1   zeros(2·A) | cur_player(1)  [board is hidden]
    state_observation  : 4·A + 1   p0_stones(A) | p1_stones(A)
                                   | attempted_p0(A) | attempted_p1(A) | cur_player(1)
    information_set    : 2·A + 1   ordered_attempts(2·A) | cur_player(1)
                                   per slot: (cell/A, was_blocked) — -1 for unused slots
    public_state       : 2            step/max_length(1) | cur_player(1)  [abrupt]
                                     num_placed/A(1)    | cur_player(1)  [non-abrupt]
"""

from __future__ import annotations
from typing import NamedTuple

import jax
import jax.numpy as jnp

from .base import Env, Info, PRNGKey


class MNKState(NamedTuple):
    board: jax.Array         # (m, n) int32  — 0=empty, 1=p0 stone, 2=p1 stone
    attempted: jax.Array     # (2, m, n) bool — cells each player has attempted
    cur_player: jax.Array    # ()      int32
    step: jax.Array          # ()      int32
    done: jax.Array          # ()      bool
    attempt_order: jax.Array # (2, m·n) int32 — chronological attempt sequence per player; -1=unused


class MNK(Env):
    """M,N,K game for 2 sequential-move players.

    Args:
        m:      number of rows.
        n:      number of columns.
        k:      length of the winning run.
        dark:   if True, play the dark variant. Default False.
        abrupt: only relevant when dark=True. If True (default), a blocked
                attempt loses the player's turn. If False, the player acts
                again until a stone is successfully placed.
    """

    def __init__(
        self,
        m: int,
        n: int,
        k: int,
        dark: bool = False,
        abrupt: bool = True,
    ) -> None:
        if k > m and k > n:
            raise ValueError(
                f"k={k} exceeds both m={m} and n={n}: no winning run is possible"
            )
        self.m = m
        self.n = n
        self.k = k
        self.dark = dark
        self.abrupt = abrupt

    # ── Static properties ─────────────────────────────────────────────────────

    @property
    def num_players(self) -> int:
        return 2

    @property
    def max_length(self) -> int:
        return 2 * self.m * self.n if self.dark else self.m * self.n

    @property
    def num_actions(self) -> int:
        return self.m * self.n

    @property
    def max_reward(self) -> float:
        return 1.0

    # ── State lifecycle ───────────────────────────────────────────────────────

    def init_state(self, key: PRNGKey) -> MNKState:
        a = self.m * self.n
        return MNKState(
            board=jnp.zeros((self.m, self.n), dtype=jnp.int32),
            attempted=jnp.zeros((2, self.m, self.n), dtype=jnp.bool_),
            cur_player=jnp.int32(0),
            step=jnp.int32(0),
            done=jnp.bool_(False),
            attempt_order=jnp.full((2, a), -1, dtype=jnp.int32),
        )

    def apply_action(
        self,
        state: MNKState,
        actions: jax.Array,
        key: PRNGKey,
    ) -> tuple[MNKState, jax.Array, jax.Array, Info]:
        cp = state.cur_player
        action = actions[cp]

        r = action // self.n
        c = action % self.n

        if self.dark:
            cell_empty = state.board[r, c] == 0
            new_board = jnp.where(cell_empty, state.board.at[r, c].set(cp + 1), state.board)
            new_attempted = state.attempted.at[cp, r, c].set(True)
            won = cell_empty & self._check_win(new_board, cp)
            # keep_turn: stay with the same player when the game ended (won) OR,
            # in non-abrupt mode, when the attempt was blocked (cell not empty).
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

        board_full = jnp.all(new_board != 0)
        done = won | board_full

        # Draw (board_full & ~won) gives zero reward; only wins give ±1.
        p0_reward = jnp.where(cp == 0, jnp.float32(1.0), jnp.float32(-1.0))
        win_rewards = jnp.stack([p0_reward, -p0_reward])
        rewards = jnp.where(won, win_rewards, jnp.zeros(2, dtype=jnp.float32))

        new_state = MNKState(
            board=new_board,
            attempted=new_attempted,
            cur_player=jnp.where(keep_turn, state.cur_player, jnp.int32(1 - cp)),
            step=state.step + 1,
            done=done,
            attempt_order=new_attempt_order,
        )
        return new_state, rewards, done, {}

    # ── Win detection ─────────────────────────────────────────────────────────

    def _check_win(self, board: jax.Array, player: jax.Array) -> jax.Array:
        """Return True if player has k in a row in any direction."""
        m, n, k = self.m, self.n, self.k
        owns = (board == (player + 1)).astype(jnp.int32)

        checks: list[jax.Array] = []

        # Horizontal: each column window of width k across all rows
        for c in range(n - k + 1):
            checks.append(jnp.any(owns[:, c:c + k].sum(axis=1) == k))

        # Vertical: each row window of height k across all columns
        for r in range(m - k + 1):
            checks.append(jnp.any(owns[r:r + k, :].sum(axis=0) == k))

        # Diagonal ↘
        for r in range(m - k + 1):
            for c in range(n - k + 1):
                checks.append(
                    jnp.stack([owns[r + i, c + i] for i in range(k)]).sum() == k
                )

        # Anti-diagonal ↙
        for r in range(m - k + 1):
            for c in range(n - k + 1):
                checks.append(
                    jnp.stack([owns[r + i, c + k - 1 - i] for i in range(k)]).sum() == k
                )

        if not checks:
            return jnp.bool_(False)
        return jnp.any(jnp.stack(checks))

    # ── Observations ──────────────────────────────────────────────────────────

    def player_observation(
        self, state: MNKState, player_id: jax.Array, key: PRNGKey
    ) -> jax.Array:
        if self.dark:
            own_p0 = (state.board == 1).astype(jnp.float32).flatten()
            own_p1 = (state.board == 2).astype(jnp.float32).flatten()
            own = jnp.where(player_id == 0, own_p0, own_p1)
            blocked_p0 = (state.attempted[0] & (state.board != 1)).astype(jnp.float32).flatten()
            blocked_p1 = (state.attempted[1] & (state.board != 2)).astype(jnp.float32).flatten()
            blocked = jnp.where(player_id == 0, blocked_p0, blocked_p1)
            return jnp.concatenate([own, blocked, jnp.array([state.cur_player], dtype=jnp.float32)])
        else:
            own = (state.board == (player_id + 1)).astype(jnp.float32).flatten()
            opp = (state.board == (2 - player_id)).astype(jnp.float32).flatten()
            return jnp.concatenate([own, opp, jnp.array([state.cur_player], dtype=jnp.float32)])

    def public_observation(self, state: MNKState, key: PRNGKey) -> jax.Array:
        if self.dark:
            A = self.m * self.n
            return jnp.concatenate([
                jnp.zeros(2 * A, dtype=jnp.float32),
                jnp.array([state.cur_player], dtype=jnp.float32),
            ])
        else:
            p0 = (state.board == 1).astype(jnp.float32).flatten()
            p1 = (state.board == 2).astype(jnp.float32).flatten()
            return jnp.concatenate([p0, p1, jnp.array([state.cur_player], dtype=jnp.float32)])

    def state_observation(self, state: MNKState, key: PRNGKey) -> jax.Array:
        """Full ground-truth state. Includes attempted masks for the dark variant."""
        p0 = (state.board == 1).astype(jnp.float32).flatten()
        p1 = (state.board == 2).astype(jnp.float32).flatten()
        cp = jnp.array([state.cur_player], dtype=jnp.float32)
        if self.dark:
            att_p0 = state.attempted[0].astype(jnp.float32).flatten()
            att_p1 = state.attempted[1].astype(jnp.float32).flatten()
            return jnp.concatenate([p0, p1, att_p0, att_p1, cp])
        return jnp.concatenate([p0, p1, cp])

    # ── Action legality & turn order ──────────────────────────────────────────

    def legal_actions(self, state: MNKState, player_id: jax.Array | int) -> jax.Array:
        """Active player: unattempted cells (dark) or empty cells (standard).
        Inactive player: only action 0 (dummy)."""
        is_active = jnp.int32(player_id) == state.cur_player

        if self.dark:
            not_tried_p0 = (~state.attempted[0]).flatten()
            not_tried_p1 = (~state.attempted[1]).flatten()
            active_mask = jnp.where(player_id == 0, not_tried_p0, not_tried_p1)
        else:
            active_mask = (state.board.flatten() == 0)

        inactive_mask = jnp.zeros(self.num_actions, dtype=jnp.bool_).at[0].set(True)
        return jnp.where(is_active, active_mask, inactive_mask)

    def current_player(self, state: MNKState) -> jax.Array:
        return state.cur_player

    # ── Perfect-recall representations ────────────────────────────────────────

    def information_set(
        self, state: MNKState, player_id: jax.Array | int, key: PRNGKey
    ) -> jax.Array:
        # Standard MNK: board is full history (perfect information).
        if not self.dark:
            return self.player_observation(state, jnp.int32(player_id), key)
        # Dark MNK: encode the ordered attempt sequence for perfect recall.
        # Each slot i holds (cell/A, was_blocked): -1.0 padding for unused slots.
        pid = jnp.int32(player_id)
        a = self.num_actions
        attempt_seq = jnp.where(pid == 0, state.attempt_order[0], state.attempt_order[1])

        def encode_one(cell_idx: jax.Array) -> jax.Array:
            valid = cell_idx >= 0
            safe = jnp.where(valid, cell_idx, jnp.int32(0))
            r = safe // self.n
            c = safe % self.n
            placed = state.board[r, c] == (pid + 1)
            norm = jnp.where(valid, safe.astype(jnp.float32) / a, jnp.float32(-1.0))
            blocked = jnp.where(valid, (~placed).astype(jnp.float32), jnp.float32(-1.0))
            return jnp.stack([norm, blocked])

        seq = jax.vmap(encode_one)(attempt_seq).flatten()  # (2·A,)
        return jnp.concatenate([seq, jnp.array([state.cur_player], dtype=jnp.float32)])

    def public_state(self, state: MNKState, key: PRNGKey) -> jax.Array:
        # Standard MNK: same as public_observation.
        if not self.dark:
            return self.public_observation(state, key)
        # Abrupt: every step is a turn switch, so the step count is public.
        # Non-abrupt: step count leaks how many blocks occurred (private), so
        # only the number of successfully placed stones is public.
        if self.abrupt:
            progress = state.step.astype(jnp.float32) / self.max_length
        else:
            progress = (state.board > 0).sum().astype(jnp.float32) / self.num_actions
        return jnp.array([progress, state.cur_player.astype(jnp.float32)])

    def state_representation(self, state: MNKState, key: PRNGKey) -> jax.Array:
        return self.state_observation(state, key)


class TicTacToe(MNK):
    """Tic-Tac-Toe: MNK with m=n=k=3.

    Accepts the same dark and abrupt keyword arguments as MNK.
    """

    def __init__(self, dark: bool = False, abrupt: bool = True) -> None:
        super().__init__(m=3, n=3, k=3, dark=dark, abrupt=abrupt)
