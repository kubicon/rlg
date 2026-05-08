"""MMD algorithm — PPO variant with a slowly-updated magnet policy.

Extends PPOBase with a second set of parameters (`magnet_params`) that track
the main policy via Polyak averaging after every training step:

    magnet_params ← τ · params + (1 − τ) · magnet_params

The MMD loss adds two KL terms on top of standard PPO:
  - magnet_loss  : KL(current ‖ magnet)   — pulls policy toward magnet
  - old_kl_loss  : KL(current ‖ old)      — PPO-style trust-region penalty

magnet_params are stored in TrainingState.extras['magnet_params'].
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.lax as lax
import optax

from .base import TrainingState
from .episode import collect_episodes
from .ppo import PPOBase
from ..agents.base import Agent
from ..envs.base import Env
from ..losses.mmd import mmd_loss


class MMD(PPOBase):
    """PPO with a Polyak-averaged magnet policy.

    Args:
        env, agent, rollout_len, n_epochs, batch_size, lr,
        clip_eps, vf_coef, ent_coef, gamma, gae_lambda,
        delta_clip, trace_clip: inherited from PPOBase.
        magnet_coef:     Weight of KL(current ‖ magnet) term.
        old_policy_coef: Weight of KL(current ‖ old policy) term.
        polyak_tau:      Polyak averaging rate for magnet update (closer to 1
                         = magnet tracks params faster).
    """

    def __init__(
        self,
        env:             Env,
        agent:           Agent,
        rollout_len:     int   = 128,
        n_epochs:        int   = 4,
        batch_size:      int   = 4,
        lr:              float = 3e-4,
        clip_eps:        float = 0.2,
        vf_coef:         float = 0.5,
        ent_coef:        float = 0.01,
        gamma:           float = 0.99,
        gae_lambda:      float = 0.95,
        delta_clip:      float = 1.0,
        trace_clip:      float = 1.0,
        magnet_coef:     float = 1.0,
        old_policy_coef: float = 1.0,
        polyak_tau:      float = 0.005,
    ) -> None:
        super().__init__(
            env, agent, rollout_len, n_epochs, batch_size, lr,
            clip_eps, vf_coef, ent_coef, gamma, gae_lambda,
            delta_clip, trace_clip,
        )
        self.magnet_coef     = magnet_coef
        self.old_policy_coef = old_policy_coef
        self.polyak_tau      = polyak_tau

    # ── Init ──────────────────────────────────────────────────────────────────

    def init(self, key: jax.Array) -> TrainingState:
        key, params, opt_state, env_state, agent_state = self._init_common(key)
        return TrainingState(
            params      = params,
            opt_state   = opt_state,
            env_state   = env_state,
            agent_state = agent_state,
            rng         = key,
            step        = jnp.zeros((), jnp.int32),
            extras      = {'magnet_params': params},
        )

    # ── Public step ───────────────────────────────────────────────────────────

    def step(self, state: TrainingState) -> tuple[TrainingState, dict[str, jax.Array]]:
        rng, collect_key = jax.random.split(state.rng)
        _, _, _, episodes = collect_episodes(
            self.env, self.agent, state.params, collect_key, self.batch_size)

        advantages, targets = self._compute_advantages(
            episodes.rewards, episodes.agent_output.value, episodes.dones)

        # Magnet logits are fixed throughout all epochs — precompute once.
        magnet_out    = self._eval_params(state.extras['magnet_params'], episodes)
        magnet_logits = lax.stop_gradient(magnet_out.logits)  # (B, T, P, A)

        params, opt_state = state.params, state.opt_state
        valid = self._valid_mask(episodes.dones)

        def epoch_fn(carry, _):
            params, opt_state = carry

            def total_loss(params):
                agent_out = self._eval_params(params, episodes)

                _axes    = (0, 0, 0, 0, 0, 0, 0, 0, 0, None, None, None, None, None)
                loss_P   = jax.vmap(mmd_loss, in_axes=_axes)
                loss_TP  = jax.vmap(loss_P,   in_axes=_axes)
                loss_BTP = jax.vmap(loss_TP,  in_axes=_axes)
                losses, metrics = loss_BTP(
                    agent_out.value,
                    agent_out.logits,
                    episodes.legal_actions,
                    episodes.actions,
                    episodes.agent_output.logits,
                    episodes.agent_output.value,
                    magnet_logits,
                    advantages,
                    targets,
                    self.clip_eps, self.vf_coef, self.ent_coef,
                    self.magnet_coef, self.old_policy_coef,
                )
                wmean = lambda x: self._wmean(x, valid)
                return wmean(losses), jax.tree.map(wmean, metrics)

            (_, metrics), grads = jax.value_and_grad(total_loss, has_aux=True)(params)
            updates, new_opt_state = self.optimizer.update(grads, opt_state, params)
            return (optax.apply_updates(params, updates), new_opt_state), metrics

        (params, opt_state), epoch_metrics = jax.lax.scan(
            epoch_fn, (params, opt_state), None, length=self.n_epochs)

        # Polyak update: magnet ← τ·params + (1−τ)·magnet
        magnet_params = jax.tree.map(
            lambda m, p: self.polyak_tau * p + (1.0 - self.polyak_tau) * m,
            state.extras['magnet_params'], params,
        )

        return TrainingState(
            params      = params,
            opt_state   = opt_state,
            env_state   = state.env_state,
            agent_state = state.agent_state,
            rng         = rng,
            step        = state.step + 1,
            extras      = {'magnet_params': magnet_params},
        ), jax.tree.map(jnp.mean, epoch_metrics)
