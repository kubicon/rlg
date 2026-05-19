"""Battleship for 2 simultaneous-move players.

Rules:
  - Placement phase: both players simultaneously place ships, one per step, in order.
    Ships may not be placed within the 8-neighbourhood of any existing ship.
  - Shooting phase: both players simultaneously fire at one cell per step.
  - Game ends when all ships of at least one player are sunk.
  - Draw if both players' last ships are sunk on the same step.

Action space (num_actions = 2 * N * N, shared across both phases):
  Placement action a: row = a // (2*N), col = (a // 2) % N, orient = a % 2
                       orient 0 = horizontal, orient 1 = vertical
  Shooting  action a: row = a // N, col = a % N   (only a < N*N are legal)

legal_actions respects 8-neighbourhood placement constraint and already-shot cells.

Observation sizes (float32):
  player_observation : 4*N*N + 2 + num_ships
                       own_ships(N²) | opp_hits(N²) | opp_misses(N²) | shots_on_me(N²)
                       | phase(1) | placed_frac(1) | own_ships_alive(S)
  public_observation : 4*N*N + 1
                       p0_hits(N²) | p0_misses(N²) | p1_hits(N²) | p1_misses(N²) | phase(1)
  state_observation  : 6*N*N + 1
                       p0_ships(N²) | p1_ships(N²) | p0_hits(N²) | p0_misses(N²)
                       | p1_hits(N²) | p1_misses(N²) | phase(1)
"""

from __future__ import annotations
from typing import NamedTuple, Sequence

import jax
import jax.numpy as jnp

from .base import Env, Info, PRNGKey


class BattleshipState(NamedTuple):
  ship_grids: jax.Array  # (2, N, N) int32  — 0=empty, s+1=ship-s cells
  shot_grids: (
    jax.Array
  )  # (2, N, N) bool   — shot_grids[p] = cells player p has fired at
  ships_placed: jax.Array  # (2,)      int32  — ships placed so far by each player
  ships_alive: jax.Array  # (2, S)    bool   — True if ship still has un-hit cells
  phase: jax.Array  # ()        int32  — 0=placement, 1=shooting
  done: jax.Array  # ()        bool


class Battleship(Env):
  """Battleship for 2 simultaneous-move players.

  Args:
      board_size:   side length of the square board (default: 5)
      ship_lengths: ordered sequence of ship lengths to place (default: (3, 2))
      reward_type:  'binary'     — +1 winner, 0 draw, -1 loser (default)
                    'difference' — ships sunk by me minus ships sunk on me
  """

  def __init__(
    self,
    board_size: int = 5,
    ship_lengths: Sequence[int] = (3, 2),
    reward_type: str = "binary",
  ) -> None:
    if reward_type not in ("binary", "difference"):
      raise ValueError(
        f"reward_type must be 'binary' or 'difference', got '{reward_type}'"
      )
    self.board_size = board_size
    self.ship_lengths = list(ship_lengths)
    self.num_ships = len(ship_lengths)
    self.ship_lengths_arr = jnp.array(ship_lengths, dtype=jnp.int32)
    self.reward_type = reward_type

  # ── Static properties ────────────────────────────────────────────────────

  @property
  def num_players(self) -> int:
    return 2

  @property
  def max_length(self) -> int:
    return self.num_ships + self.board_size**2

  @property
  def num_actions(self) -> int:
    return 2 * self.board_size**2

  @property
  def max_reward(self) -> float:
    if self.reward_type == "binary":
      return 1.0
    return float(self.num_ships)

  # ── State lifecycle ──────────────────────────────────────────────────────

  def init_state(self, key: PRNGKey) -> BattleshipState:
    N, S = self.board_size, self.num_ships
    return BattleshipState(
      ship_grids=jnp.zeros((2, N, N), dtype=jnp.int32),
      shot_grids=jnp.zeros((2, N, N), dtype=jnp.bool_),
      ships_placed=jnp.zeros(2, dtype=jnp.int32),
      ships_alive=jnp.ones((2, S), dtype=jnp.bool_),
      phase=jnp.int32(0),
      done=jnp.bool_(False),
    )

  # ── Internal helpers ─────────────────────────────────────────────────────

  def _exclusion_zone(self, ship_grid: jax.Array) -> jax.Array:
    """(N, N) bool: True for cells occupied by or adjacent (8-way) to any ship."""
    N = self.board_size
    padded = jnp.pad(ship_grid > 0, 1)  # (N+2, N+2), zero-padded boundaries
    excl = jnp.zeros((N, N), dtype=jnp.bool_)
    for dr in range(3):
      for dc in range(3):
        excl = excl | padded[dr : dr + N, dc : dc + N]
    return excl

  def _place_ship_on_grid(
    self,
    ship_grid: jax.Array,
    row: jax.Array,
    col: jax.Array,
    orient: jax.Array,
    ship_len: jax.Array,
    ship_idx: jax.Array,
  ) -> jax.Array:
    N = self.board_size
    r = jnp.arange(N, dtype=jnp.int32)
    c = jnp.arange(N, dtype=jnp.int32)
    rr, cc = jnp.meshgrid(r, c, indexing="ij")  # (N, N)
    h_mask = (rr == row) & (cc >= col) & (cc < col + ship_len)
    v_mask = (cc == col) & (rr >= row) & (rr < row + ship_len)
    return jnp.where(jnp.where(orient == 0, h_mask, v_mask), ship_idx + 1, ship_grid)

  def _ships_alive(
    self,
    ship_grid: jax.Array,  # (N, N) int32 — defender's grid
    opp_shots: jax.Array,  # (N, N) bool  — attacker's shots
  ) -> jax.Array:
    """(num_ships,) bool: True if ship has at least one un-hit cell."""
    return jnp.array(
      [
        ~jnp.all(jnp.where(ship_grid == (s + 1), opp_shots, True))
        for s in range(self.num_ships)
      ],
      dtype=jnp.bool_,
    )

  def _placement_mask(self, ship_grid: jax.Array, ship_len: jax.Array) -> jax.Array:
    """(num_actions,) bool: legal placement actions given current board and ship length."""
    N = self.board_size
    max_L = max(self.ship_lengths)
    n = 2 * N * N

    all_a = jnp.arange(n, dtype=jnp.int32)
    rows = all_a // (2 * N)  # (n,)
    cols = (all_a // 2) % N  # (n,)
    orients = all_a % 2  # (n,)

    pos = jnp.arange(max_L, dtype=jnp.int32)  # (max_L,)

    # Cell coordinates for each action × ship position slot
    h_r = rows[:, None]  # (n, max_L)
    h_c = cols[:, None] + pos[None, :]
    v_r = rows[:, None] + pos[None, :]
    v_c = cols[:, None]  # broadcast to (n, max_L)

    cell_r = jnp.where(orients[:, None] == 0, h_r, v_r)  # (n, max_L)
    cell_c = jnp.where(orients[:, None] == 0, h_c, v_c)

    active = pos[None, :] < ship_len  # (1, max_L) → broadcasts to (n, max_L)

    in_bounds = (cell_r >= 0) & (cell_r < N) & (cell_c >= 0) & (cell_c < N)
    safe_r = jnp.clip(cell_r, 0, N - 1)
    safe_c = jnp.clip(cell_c, 0, N - 1)

    excl = self._exclusion_zone(ship_grid)  # (N, N)
    in_excl = excl[safe_r, safe_c]  # (n, max_L)

    all_in_bounds = jnp.all(jnp.where(active, in_bounds, True), axis=1)  # (n,)
    no_conflict = ~jnp.any(active & in_excl, axis=1)  # (n,)
    return all_in_bounds & no_conflict

  # ── Phase sub-steps ──────────────────────────────────────────────────────

  def _placement_step(
    self, state: BattleshipState, actions: jax.Array
  ) -> tuple[BattleshipState, jax.Array, jax.Array]:
    N = self.board_size

    def place(p: int) -> jax.Array:
      a = actions[p]
      row = a // (2 * N)
      col = (a // 2) % N
      orient = a % 2
      idx = jnp.clip(state.ships_placed[p], 0, self.num_ships - 1)
      length = self.ship_lengths_arr[idx]
      return self._place_ship_on_grid(
        state.ship_grids[p], row, col, orient, length, idx
      )

    new_grids = jnp.stack([place(0), place(1)])
    new_placed = state.ships_placed + 1
    new_phase = jnp.where(
      jnp.all(new_placed >= self.num_ships), jnp.int32(1), jnp.int32(0)
    )

    new_state = BattleshipState(
      ship_grids=new_grids,
      shot_grids=state.shot_grids,
      ships_placed=new_placed,
      ships_alive=state.ships_alive,
      phase=new_phase,
      done=jnp.bool_(False),
    )
    return new_state, jnp.zeros(2, dtype=jnp.float32), jnp.bool_(False)

  def _shooting_step(
    self, state: BattleshipState, actions: jax.Array
  ) -> tuple[BattleshipState, jax.Array, jax.Array]:
    N = self.board_size
    r0, c0 = actions[0] // N, actions[0] % N
    r1, c1 = actions[1] // N, actions[1] % N

    new_shots = state.shot_grids.at[0, r0, c0].set(True).at[1, r1, c1].set(True)

    new_alive = jnp.stack(
      [
        self._ships_alive(state.ship_grids[0], new_shots[1]),
        self._ships_alive(state.ship_grids[1], new_shots[0]),
      ]
    )

    p0_lost = ~jnp.any(new_alive[0])
    p1_lost = ~jnp.any(new_alive[1])
    done = p0_lost | p1_lost

    if self.reward_type == "binary":
      p0_wins = p1_lost & ~p0_lost
      p1_wins = p0_lost & ~p1_lost
      term_r = jnp.stack(
        [
          jnp.where(p0_wins, 1.0, jnp.where(p1_wins, -1.0, 0.0)),
          jnp.where(p1_wins, 1.0, jnp.where(p0_wins, -1.0, 0.0)),
        ]
      )
    else:  # difference: ships sunk − ships lost
      sunk_by_p0 = jnp.sum(~new_alive[1]).astype(jnp.float32)
      sunk_by_p1 = jnp.sum(~new_alive[0]).astype(jnp.float32)
      term_r = jnp.stack([sunk_by_p0 - sunk_by_p1, sunk_by_p1 - sunk_by_p0])

    rewards = jnp.where(done, term_r, jnp.zeros(2, dtype=jnp.float32))

    new_state = BattleshipState(
      ship_grids=state.ship_grids,
      shot_grids=new_shots,
      ships_placed=state.ships_placed,
      ships_alive=new_alive,
      phase=state.phase,
      done=done,
    )
    return new_state, rewards, done

  # ── Public step ──────────────────────────────────────────────────────────

  def apply_action(
    self,
    state: BattleshipState,
    actions: jax.Array,
    key: PRNGKey,
  ) -> tuple[BattleshipState, jax.Array, jax.Array, Info]:
    s_place, r_place, _ = self._placement_step(state, actions)
    s_shoot, r_shoot, _ = self._shooting_step(state, actions)

    is_place = state.phase == 0
    next_state = jax.tree.map(lambda a, b: jnp.where(is_place, a, b), s_place, s_shoot)
    rewards = jnp.where(is_place, r_place, r_shoot)

    # Suppress rewards and keep done=True once the episode has ended
    rewards = jnp.where(state.done, jnp.zeros(2, dtype=jnp.float32), rewards)
    final_done = state.done | next_state.done
    next_state = next_state._replace(done=final_done)

    return next_state, rewards, final_done, {}

  # ── Observations ────────────────────────────────────────────────────────

  def player_observation(
    self, state: BattleshipState, player_id: jax.Array, key: PRNGKey
  ) -> jax.Array:
    """4*N²+2+S float32: own ships, hits/misses on opp, shots on me, phase, placed frac, alive."""
    p = jnp.int32(player_id)
    q = 1 - p  # opponent

    own_ships = (state.ship_grids[p] > 0).astype(jnp.float32).flatten()
    opp_hits = (
      (state.shot_grids[p] & (state.ship_grids[q] > 0)).astype(jnp.float32).flatten()
    )
    opp_misses = (
      (state.shot_grids[p] & ~(state.ship_grids[q] > 0)).astype(jnp.float32).flatten()
    )
    shots_on_me = state.shot_grids[q].astype(jnp.float32).flatten()
    phase = jnp.array([state.phase], dtype=jnp.float32)
    placed_frac = jnp.array([state.ships_placed[p] / self.num_ships], dtype=jnp.float32)
    alive = state.ships_alive[p].astype(jnp.float32)

    return jnp.concatenate(
      [own_ships, opp_hits, opp_misses, shots_on_me, phase, placed_frac, alive]
    )

  def public_observation(self, state: BattleshipState, key: PRNGKey) -> jax.Array:
    """4*N²+1 float32: both players' hit/miss grids and current phase."""
    p0_hits = (
      (state.shot_grids[0] & (state.ship_grids[1] > 0)).astype(jnp.float32).flatten()
    )
    p0_miss = (
      (state.shot_grids[0] & ~(state.ship_grids[1] > 0)).astype(jnp.float32).flatten()
    )
    p1_hits = (
      (state.shot_grids[1] & (state.ship_grids[0] > 0)).astype(jnp.float32).flatten()
    )
    p1_miss = (
      (state.shot_grids[1] & ~(state.ship_grids[0] > 0)).astype(jnp.float32).flatten()
    )
    phase = jnp.array([state.phase], dtype=jnp.float32)
    return jnp.concatenate([p0_hits, p0_miss, p1_hits, p1_miss, phase])

  def state_observation(self, state: BattleshipState, key: PRNGKey) -> jax.Array:
    """6*N²+1 float32: both ship layouts, both hit/miss grids, phase."""
    p0_ships = (state.ship_grids[0] > 0).astype(jnp.float32).flatten()
    p1_ships = (state.ship_grids[1] > 0).astype(jnp.float32).flatten()
    p0_hits = (
      (state.shot_grids[0] & (state.ship_grids[1] > 0)).astype(jnp.float32).flatten()
    )
    p0_miss = (
      (state.shot_grids[0] & ~(state.ship_grids[1] > 0)).astype(jnp.float32).flatten()
    )
    p1_hits = (
      (state.shot_grids[1] & (state.ship_grids[0] > 0)).astype(jnp.float32).flatten()
    )
    p1_miss = (
      (state.shot_grids[1] & ~(state.ship_grids[0] > 0)).astype(jnp.float32).flatten()
    )
    phase = jnp.array([state.phase], dtype=jnp.float32)
    return jnp.concatenate(
      [p0_ships, p1_ships, p0_hits, p0_miss, p1_hits, p1_miss, phase]
    )

  # ── Action legality ───────────────────────────────────────────────────────

  def legal_actions(
    self, state: BattleshipState, player_id: jax.Array | int
  ) -> jax.Array:
    """(num_actions,) bool. Placement: valid (row,col,orient) for current ship.
    Shooting: cells not yet fired at (lower N² actions only)."""
    player_id = jnp.int32(player_id)
    N = self.board_size

    # Shooting: mark unfired cells legal, upper N² always False
    shoot_flat = ~state.shot_grids[player_id].flatten()  # (N²,)
    shoot_mask = jnp.concatenate([shoot_flat, jnp.zeros(N * N, dtype=jnp.bool_)])

    # Placement: vectorised validity check for all (row, col, orient) combos
    idx = jnp.clip(state.ships_placed[player_id], 0, self.num_ships - 1)
    ship_len = self.ship_lengths_arr[idx]
    place_mask = self._placement_mask(state.ship_grids[player_id], ship_len)

    return jnp.where(state.phase == 0, place_mask, shoot_mask)

  # ── Perfect-recall representations ───────────────────────────────────────
  # NOTE: information_set is intentionally lossy — shot grids record *which*
  # cells were fired at but not the order. In Battleship the shot order carries
  # no strategic information (hits/misses are revealed immediately and the set
  # of available actions depends only on unshot cells), so this is acceptable.

  def information_set(
    self, state: BattleshipState, player_id: jax.Array | int, key: PRNGKey
  ) -> jax.Array:
    return self.player_observation(state, jnp.int32(player_id), key)

  def public_state(self, state: BattleshipState, key: PRNGKey) -> jax.Array:
    return self.public_observation(state, key)

  def state_representation(self, state: BattleshipState, key: PRNGKey) -> jax.Array:
    return self.state_observation(state, key)
