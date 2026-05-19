"""Liar's Dice for 2 sequential-move players.

Rules:
  - Each player rolls their dice privately (single chance node in init_state).
  - Players alternate: the active player either makes a bid or calls the
    previous bid a lie ("call").
  - A bid (q, f) claims at least q dice across all dice show face f.
  - Bids must be strictly increasing lexicographically: (q', f') > (q, f)
    iff q' > q, or q' == q and f' > f.
  - On "call": count all dice showing bid_face (1s are wild if wild_ones=True).
    count >= bid_quantity → bid was true, caller loses; else bidder loses.
  - Reward: ±1 (binary).

Action encoding  (num_actions = total_dice * sides + 1):
  a in [0, total_dice*sides - 1] : bid  (q = a // sides + 1, f = a % sides + 1)
  a = total_dice*sides            : call liar

Observation sizes (float32); let D = total_dice, M = max_dice, L = max_length = D*sides+1:
  player_observation : M*sides + (D+1) + (sides+1) + 1 + L*2
    own_dice_onehot(M*sides) | bid_q_onehot(D+1) | bid_f_onehot(sides+1)
    | cur_player(1) | bid_history(L*2)
  public_observation : (D+1) + (sides+1) + 1 + L*2
  state_observation  : 2*M*sides + (D+1) + (sides+1) + 1 + L*2

Perfect-recall representations alias the snapshot observations since the full
bid history is already embedded in every observation vector.
"""

from __future__ import annotations
from typing import NamedTuple

import jax
import jax.numpy as jnp

from .base import Env, Info, PRNGKey


class LiarsDiceState(NamedTuple):
    dice: jax.Array              # (2, max_dice) int32  — face values 1..sides, 0=unused
    cur_bid_quantity: jax.Array  # ()            int32  — 0 if no bid yet
    cur_bid_face: jax.Array      # ()            int32  — 0 if no bid yet
    cur_player: jax.Array        # ()            int32  — 0 or 1
    action_history: jax.Array    # (max_length,) int32  — -1=not taken
    step: jax.Array              # ()            int32
    done: jax.Array              # ()            bool


class LiarsDice(Env):
    """Liar's Dice for 2 sequential-move players.

    Args:
        dice:       int or (int, int) — dice count per player; if int, both players
                    share the same count.
        sides:      number of faces per die (default 6).
        wild_ones:  if True, face 1 counts as any face when resolving bids
                    (standard Perudo rules). Default False.
    """

    def __init__(
        self,
        dice: tuple[int, int] | int = 2,
        sides: int = 6,
        wild_ones: bool = False,
    ) -> None:
        if isinstance(dice, int):
            dice = (dice, dice)
        self.dice_per_player = tuple(dice)  # (d0, d1)
        self.sides = sides
        self.wild_ones = wild_ones
        self.total_dice = dice[0] + dice[1]
        self.max_dice = max(dice)
        self._call_action = self.total_dice * sides  # index of the "call" action

    # ── Static properties ─────────────────────────────────────────────────────

    @property
    def num_players(self) -> int:
        return 2

    @property
    def max_length(self) -> int:
        # At most total_dice*sides bids, then one call
        return self.total_dice * self.sides + 1

    @property
    def num_actions(self) -> int:
        return self.total_dice * self.sides + 1

    @property
    def max_reward(self) -> float:
        return 1.0

    # ── State lifecycle ───────────────────────────────────────────────────────

    def init_state(self, key: PRNGKey) -> LiarsDiceState:
        all_rolls = jax.random.randint(key, (2, self.max_dice), 1, self.sides + 1).astype(
            jnp.int32
        )
        d0, d1 = self.dice_per_player
        mask = jnp.array(
            [
                [1 if i < d0 else 0 for i in range(self.max_dice)],
                [1 if i < d1 else 0 for i in range(self.max_dice)],
            ],
            dtype=jnp.int32,
        )
        return LiarsDiceState(
            dice=all_rolls * mask,
            cur_bid_quantity=jnp.int32(0),
            cur_bid_face=jnp.int32(0),
            cur_player=jnp.int32(0),
            action_history=jnp.full(self.max_length, -1, dtype=jnp.int32),
            step=jnp.int32(0),
            done=jnp.bool_(False),
        )

    def apply_action(
        self,
        state: LiarsDiceState,
        actions: jax.Array,
        key: PRNGKey,
    ) -> tuple[LiarsDiceState, jax.Array, jax.Array, Info]:
        cp = state.cur_player
        action = actions[cp]
        is_call = action == self._call_action

        # Decode bid components (only meaningful when is_call is False)
        bid_q = (action // self.sides + 1).astype(jnp.int32)
        bid_f = (action % self.sides + 1).astype(jnp.int32)

        # Count dice matching the called bid
        if self.wild_ones:
            exact = jnp.sum(state.dice == state.cur_bid_face)
            wilds = jnp.sum((state.dice == 1) & (state.cur_bid_face != 1))
            matching = exact + wilds
        else:
            matching = jnp.sum(state.dice == state.cur_bid_face)

        # Caller wins if the bid was false (not enough matching dice)
        caller_reward = jnp.where(
            matching >= state.cur_bid_quantity, jnp.float32(-1.0), jnp.float32(1.0)
        )
        p0_reward = jnp.where(cp == 0, caller_reward, -caller_reward)
        terminal_rewards = jnp.stack([p0_reward, -p0_reward])

        done = is_call
        rewards = jnp.where(done, terminal_rewards, jnp.zeros(2, dtype=jnp.float32))

        new_state = LiarsDiceState(
            dice=state.dice,
            cur_bid_quantity=jnp.where(is_call, state.cur_bid_quantity, bid_q),
            cur_bid_face=jnp.where(is_call, state.cur_bid_face, bid_f),
            cur_player=jnp.where(is_call, state.cur_player, jnp.int32(1 - cp)),
            action_history=state.action_history.at[state.step].set(action),
            step=state.step + 1,
            done=done,
        )
        return new_state, rewards, done, {}

    # ── Observation helpers ───────────────────────────────────────────────────

    def _dice_onehot(self, dice_row: jax.Array) -> jax.Array:
        """(max_dice,) int32 → (max_dice * sides,) float32; 0=unused slot → zeros."""
        return jax.nn.one_hot(dice_row, self.sides + 1, dtype=jnp.float32)[:, 1:].flatten()

    def _bid_bits(self, state: LiarsDiceState) -> jax.Array:
        """One-hot current bid: (total_dice+1) + (sides+1) dims; index 0 = no bid."""
        q_bits = jax.nn.one_hot(state.cur_bid_quantity, self.total_dice + 1, dtype=jnp.float32)
        f_bits = jax.nn.one_hot(state.cur_bid_face, self.sides + 1, dtype=jnp.float32)
        return jnp.concatenate([q_bits, f_bits])

    def _action_to_bid_norm(self, action: jax.Array) -> jax.Array:
        """Map action int → (q/total_dice, f/sides); (0, 0) for call or empty slot."""
        is_bid = (action >= 0) & (action < self._call_action)
        q = jnp.where(is_bid, action // self.sides + 1, 0)
        f = jnp.where(is_bid, action % self.sides + 1, 0)
        return jnp.stack([
            q.astype(jnp.float32) / self.total_dice,
            f.astype(jnp.float32) / self.sides,
        ])

    def _common_bits(self, state: LiarsDiceState) -> jax.Array:
        """Public information: bid one-hot + cur_player + bid history."""
        bid_history = jax.vmap(self._action_to_bid_norm)(state.action_history).flatten()
        return jnp.concatenate([
            self._bid_bits(state),
            jnp.array([state.cur_player], dtype=jnp.float32),
            bid_history,
        ])

    # ── Observations ──────────────────────────────────────────────────────────

    def player_observation(
        self, state: LiarsDiceState, player_id: jax.Array, key: PRNGKey
    ) -> jax.Array:
        """own_dice_onehot(max_dice*sides) + common."""
        own_dice = self._dice_onehot(state.dice[player_id])
        return jnp.concatenate([own_dice, self._common_bits(state)])

    def public_observation(self, state: LiarsDiceState, key: PRNGKey) -> jax.Array:
        """Common public information only."""
        return self._common_bits(state)

    def state_observation(self, state: LiarsDiceState, key: PRNGKey) -> jax.Array:
        """Both players' dice + common."""
        return jnp.concatenate([
            self._dice_onehot(state.dice[0]),
            self._dice_onehot(state.dice[1]),
            self._common_bits(state),
        ])

    # ── Action legality & turn order ──────────────────────────────────────────

    def legal_actions(
        self, state: LiarsDiceState, player_id: jax.Array | int
    ) -> jax.Array:
        """Active player: valid bids + call (if a bid exists). Inactive: only action 0."""
        is_active = jnp.int32(player_id) == state.cur_player

        a = jnp.arange(self._call_action)
        q = a // self.sides + 1
        f = a % self.sides + 1
        bid_valid = (q > state.cur_bid_quantity) | (
            (q == state.cur_bid_quantity) & (f > state.cur_bid_face)
        )
        call_valid = state.cur_bid_quantity > 0

        active_mask = jnp.zeros(self.num_actions, dtype=jnp.bool_)
        active_mask = active_mask.at[: self._call_action].set(bid_valid)
        active_mask = active_mask.at[self._call_action].set(call_valid)

        inactive_mask = jnp.zeros(self.num_actions, dtype=jnp.bool_).at[0].set(True)
        return jnp.where(is_active, active_mask, inactive_mask)

    def current_player(self, state: LiarsDiceState) -> jax.Array:
        return state.cur_player

    # ── Perfect-recall representations ────────────────────────────────────────
    # The full bid history is embedded in every observation vector, so these
    # are identical to the snapshot observations.

    def information_set(
        self, state: LiarsDiceState, player_id: jax.Array | int, key: PRNGKey
    ) -> jax.Array:
        return self.player_observation(state, jnp.int32(player_id), key)

    def public_state(self, state: LiarsDiceState, key: PRNGKey) -> jax.Array:
        return self.public_observation(state, key)

    def state_representation(self, state: LiarsDiceState, key: PRNGKey) -> jax.Array:
        return self.state_observation(state, key)
