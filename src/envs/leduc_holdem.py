"""Leduc Hold'em for 2 sequential-move players.

Rules:
  - Deck: J, J, Q, Q, K, K  (ranks 0=J 1=Q 2=K, 2 copies each)
  - Each player antes 1 chip and receives 1 private card.
  - Round 1 (pre-flop): player 0 acts first; bet/raise size = 2; max 2 raises.
  - Round 2 (flop):     community card dealt; player 0 acts first; bet/raise = 4; max 2 raises.
  - Actions: 0=fold, 1=check/call, 2=bet/raise.
  - Hand ranks: pair (private rank == public rank) beats high card; higher rank breaks ties.
  - Reward: net chip profit (or binary ±1) at terminal step.

Sequential convention: the inactive player receives a legal_actions mask with only
action 0 set True. That action is ignored inside apply_action (only actions[cur_player]
is used), contributing zero policy gradient.

Observation sizes (float32):
  player_observation : 39   own_card(3) + common(36)
  public_observation : 36   common(36)
  state_observation  : 45   card0(3) + card1(3) + pending_public(3) + common(36)

common(36) = public_card_onehot(4) + pot_norm(2) + round_onehot(2)
             + call_norm(1) + raises_onehot(3) + action_history(8×3=24)

Perfect-recall representations alias the snapshot observations since the full
action history is already embedded in every observation vector.
"""

from __future__ import annotations
from typing import NamedTuple

import jax
import jax.numpy as jnp

from .base import Env, Info, PRNGKey

FOLD = 0
CALL = 1
RAISE = 2

_MAX_RAISES = 2
_MAX_STEPS = 8  # 4 actions/round × 2 rounds
_MAX_POT = 13.0  # maximum chips any player can contribute (ante + round1 + round2)
_MAX_CALL = 4.0  # maximum call_amount (round-2 bet size)


class LeducState(NamedTuple):
  private_cards: jax.Array  # (2,)       int32  — rank 0/1/2 = J/Q/K per player
  pending_public: (
    jax.Array
  )  # ()         int32  — rank of community card (revealed at flop)
  public_card: jax.Array  # ()         int32  — -1 pre-flop, then rank 0/1/2
  pot: jax.Array  # (2,)       float32 — chips contributed by each player
  round_idx: jax.Array  # ()         int32  — 0=pre-flop, 1=flop
  raises: jax.Array  # ()         int32  — raises taken in current round
  cur_player: jax.Array  # ()         int32  — 0 or 1
  call_amount: (
    jax.Array
  )  # ()         float32 — chips owed by cur_player to call (0=no bet)
  round_actions: jax.Array  # ()         int32  — actions taken so far this round
  action_history: jax.Array  # (_MAX_STEPS,) int32 — action per step, -1=not yet taken
  step: jax.Array  # ()         int32  — total steps taken
  done: jax.Array  # ()         bool


class LeducHoldem(Env):
  """Leduc Hold'em for 2 sequential-move players.

  Args:
      reward_type: 'difference' — net chip profit (default);
                   'binary'     — +1 winner, 0 draw, -1 loser.
  """

  def __init__(self, reward_type: str = "difference") -> None:
    if reward_type not in ("difference", "binary"):
      raise ValueError(
        f"reward_type must be 'difference' or 'binary', got '{reward_type}'"
      )
    self.reward_type = reward_type

  # ── Static properties ────────────────────────────────────────────────────

  @property
  def num_players(self) -> int:
    return 2

  @property
  def max_length(self) -> int:
    return _MAX_STEPS

  @property
  def num_actions(self) -> int:
    return 3  # fold | check/call | bet/raise

  @property
  def max_reward(self) -> float:
    if self.reward_type == "binary":
      return 1.0
    return float(_MAX_POT)

  # ── State lifecycle ──────────────────────────────────────────────────────

  def init_state(self, key: PRNGKey) -> LeducState:
    deck = jax.random.permutation(key, 6)  # 6 cards; rank = card_index // 2
    private_cards = (deck[:2] // 2).astype(jnp.int32)
    pending_public = (deck[2] // 2).astype(jnp.int32)
    return LeducState(
      private_cards=private_cards,
      pending_public=pending_public,
      public_card=jnp.int32(-1),
      pot=jnp.ones(2, dtype=jnp.float32),  # antes
      round_idx=jnp.int32(0),
      raises=jnp.int32(0),
      cur_player=jnp.int32(0),
      call_amount=jnp.float32(0.0),
      round_actions=jnp.int32(0),
      action_history=jnp.full(_MAX_STEPS, -1, dtype=jnp.int32),
      step=jnp.int32(0),
      done=jnp.bool_(False),
    )

  def apply_action(
    self,
    state: LeducState,
    actions: jax.Array,  # (2,) int32 — active player's action is used; other is ignored
    key: PRNGKey,
  ) -> tuple[LeducState, jax.Array, jax.Array, Info]:
    cp = state.cur_player
    action = actions[cp]

    bet_size = jnp.where(state.round_idx == 0, 2.0, 4.0)

    # Chips added to pot by the acting player this step
    added = jnp.where(
      action == CALL,
      state.call_amount,
      jnp.where(action == RAISE, state.call_amount + bet_size, 0.0),
    )
    new_pot = state.pot.at[cp].add(added)

    # What the other player owes after a bet/raise (0 after check/call/fold)
    new_call = jnp.where(action == RAISE, bet_size, 0.0)
    new_raises = state.raises + jnp.where(action == RAISE, 1, 0)

    # Conditions that end the current betting round
    is_fold = action == FOLD
    is_call = (action == CALL) & (state.call_amount > 0.0)
    is_second_check = (
      (action == CALL) & (state.call_amount == 0.0) & (state.round_actions >= 1)
    )
    round_over = is_fold | is_call | is_second_check

    # Game ends on fold or at the close of round 2
    done = is_fold | (round_over & (state.round_idx == 1))

    # Reveal community card and advance to round 2 if applicable
    move_to_r2 = round_over & ~is_fold & (state.round_idx == 0)
    new_public_card = jnp.where(move_to_r2, state.pending_public, state.public_card)
    new_round_idx = state.round_idx + jnp.where(move_to_r2, jnp.int32(1), jnp.int32(0))

    # Player 0 opens every round; within a round the other player goes next
    next_player = jnp.where(round_over, jnp.int32(0), jnp.int32(1 - cp))
    next_call = jnp.where(round_over, jnp.float32(0.0), new_call)
    next_raises = jnp.where(round_over, jnp.int32(0), new_raises)
    next_round_actions = jnp.where(round_over, jnp.int32(0), state.round_actions + 1)

    new_step = state.step + 1
    new_history = state.action_history.at[state.step].set(action)

    # ── Terminal rewards ─────────────────────────────────────────────────
    # Fold: folder loses their pot contribution; winner gains it.
    fold_winner = jnp.int32(1 - cp)
    fold_r = jnp.stack(
      [
        jnp.where(fold_winner == 0, new_pot[1], -new_pot[0]),
        jnp.where(fold_winner == 1, new_pot[0], -new_pot[1]),
      ]
    )

    # Showdown: pair (private == public) beats high card; rank breaks ties.
    p0_pair = state.private_cards[0] == new_public_card
    p1_pair = state.private_cards[1] == new_public_card
    p0_str = jnp.where(p0_pair, 3 + state.private_cards[0], state.private_cards[0])
    p1_str = jnp.where(p1_pair, 3 + state.private_cards[1], state.private_cards[1])
    p0_wins = p0_str > p1_str
    p1_wins = p1_str > p0_str
    show_r = jnp.stack(
      [
        jnp.where(p0_wins, new_pot[1], jnp.where(p1_wins, -new_pot[0], 0.0)),
        jnp.where(p1_wins, new_pot[0], jnp.where(p0_wins, -new_pot[1], 0.0)),
      ]
    )

    raw_rewards = jnp.where(is_fold, fold_r, show_r)

    if self.reward_type == "binary":
      sign = jnp.sign(raw_rewards[0])
      terminal_rewards = jnp.stack([sign, -sign])
    else:
      terminal_rewards = raw_rewards

    rewards = jnp.where(done, terminal_rewards, jnp.zeros(2, dtype=jnp.float32))

    new_state = LeducState(
      private_cards=state.private_cards,
      pending_public=state.pending_public,
      public_card=new_public_card,
      pot=new_pot,
      round_idx=new_round_idx,
      raises=next_raises,
      cur_player=next_player,
      call_amount=next_call,
      round_actions=next_round_actions,
      action_history=new_history,
      step=new_step,
      done=done,
    )
    return new_state, rewards, done, {}

  # ── Observation helpers ──────────────────────────────────────────────────

  def _public_card_bits(self, state: LeducState) -> jax.Array:
    """4-dim one-hot: index 0 = no card, 1/2/3 = J/Q/K."""
    idx = jnp.where(state.public_card >= 0, state.public_card + 1, jnp.int32(0))
    return jax.nn.one_hot(idx, 4, dtype=jnp.float32)

  def _common_bits(self, state: LeducState) -> jax.Array:
    """36-dim vector of public information."""
    return jnp.concatenate(
      [
        self._public_card_bits(state),  # 4
        state.pot / _MAX_POT,  # 2
        jax.nn.one_hot(state.round_idx, 2, dtype=jnp.float32),  # 2
        jnp.array([state.call_amount / _MAX_CALL]),  # 1
        jax.nn.one_hot(state.raises, _MAX_RAISES + 1, dtype=jnp.float32),  # 3
        jax.nn.one_hot(state.action_history, 3, dtype=jnp.float32).flatten(),  # 24
      ]
    )

  # ── Observations ────────────────────────────────────────────────────────

  def player_observation(self, state: LeducState, player_id: jax.Array) -> jax.Array:
    """39-dim: own_card_onehot(3) + common(36)."""
    own_card = jax.nn.one_hot(state.private_cards[player_id], 3, dtype=jnp.float32)
    return jnp.concatenate([own_card, self._common_bits(state)])

  def public_observation(self, state: LeducState) -> jax.Array:
    """36-dim: common public information only."""
    return self._common_bits(state)

  def state_observation(self, state: LeducState) -> jax.Array:
    """45-dim: both private cards + pending public card + common(36)."""
    card0 = jax.nn.one_hot(state.private_cards[0], 3, dtype=jnp.float32)
    card1 = jax.nn.one_hot(state.private_cards[1], 3, dtype=jnp.float32)
    pending = jax.nn.one_hot(state.pending_public, 3, dtype=jnp.float32)
    return jnp.concatenate([card0, card1, pending, self._common_bits(state)])

  # ── Action legality & turn order ────────────────────────────────────────

  def legal_actions(self, state: LeducState, player_id: jax.Array | int) -> jax.Array:
    """Active player: [fold, call, raise] with raise masked when at cap.
    Inactive player: only action 0 is legal (ignored dummy)."""
    is_active = jnp.int32(player_id) == state.cur_player
    can_raise = state.raises < _MAX_RAISES
    can_fold = state.call_amount > 0.0
    active_mask = jnp.array([can_fold, True, can_raise])
    inactive_mask = jnp.array([True, False, False])
    return jnp.where(is_active, active_mask, inactive_mask)

  def current_player(self, state: LeducState) -> jax.Array:
    return state.cur_player

  # ── Perfect-recall representations ──────────────────────────────────────
  # The full action history is embedded in every observation vector, so these
  # are identical to the snapshot observations.

  def information_set(self, state: LeducState, player_id: jax.Array | int) -> jax.Array:
    return self.player_observation(state, jnp.int32(player_id))

  def public_state(self, state: LeducState) -> jax.Array:
    return self.public_observation(state)

  def state_representation(self, state: LeducState) -> jax.Array:
    return self.state_observation(state)
