"""Regret-Matching R-NaD algorithm.

A regret-matching variant of the ``rnad`` loss path in :class:`MMD`. It reuses
MMD's target/magnet bookkeeping, the ``regularize_value`` reward transform, and
``PPOBase``'s v-trace advantage estimation unchanged. The two differences are:

  * rollout samples by regret matching with an ε exploration floor
    (:mod:`src.algorithms.rm_episode`), and
  * the loss is :func:`rm_rnad_loss` (RM strategy + squared-error NeuRD backward,
    so the regret-head outputs accumulate regret under SGD).

Network output is reinterpreted as cumulative regrets -- no new head is needed.
For the cleanest theory use a momentum-free SGD optimiser, ``n_epochs=1`` (one
regret accumulation per iteration) and no weight decay on the policy head.

Kept fully separate from the softmax MMD/R-NaD path.
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
import jax.lax as lax
import optax

from .mmd import MMD
from .types import MagnetUpdateType
from .base import TrainingState
from .rm_episode import collect_episodes_rm
from ..losses.rm_rnad import rm_rnad_loss, rm_policy


class RMRNaD(MMD):
  """R-NaD with regret matching instead of softmax.

  Args (in addition to :class:`MMD`):
      epsilon: exploration floor for ε-on-policy RM sampling
               (μ = (1-ε)·s + ε·uniform). ``0`` → pure RM on-policy.
  """

  def __init__(
    self,
    *args,
    epsilon: float = 0.1,
    schedules: dict[str, Callable[[int], float]] | None = None,
    **kwargs,
  ) -> None:
    super().__init__(*args, schedules=schedules, **kwargs)
    self.epsilon = epsilon

  def step(self, state: TrainingState) -> tuple[TrainingState, dict[str, jax.Array]]:
    def _get(name: str, default):
      s = self.schedules.get(name)
      return s(state.step) if s is not None else default

    clip_eps           = _get("clip_eps",           self.clip_eps)
    vf_coef            = _get("vf_coef",            self.vf_coef)
    ent_coef           = _get("ent_coef",           self.ent_coef)
    magnet_coef        = _get("magnet_coef",        self.magnet_coef)
    target_update_rate = _get("target_update_rate", self.target_update_rate)
    magnet_update_rate = _get("magnet_update_rate", self.magnet_update_rate)
    neurd_clip         = _get("neurd_clip",         self.neurd_clip)

    rng, collect_key = jax.random.split(state.rng)
    _, _, _, episodes = collect_episodes_rm(
      self.env, self.agent, state.params, collect_key, self.batch_size, self.epsilon
    )

    # Auxiliary networks are fixed across epochs — precompute once.
    target_out = self._eval_params(state.extras["target_params"], episodes)
    target_values = lax.stop_gradient(target_out.value)  # (B, T, P)

    magnet_out = self._eval_params(state.extras["magnet_params"], episodes)
    magnet_logits = lax.stop_gradient(magnet_out.logits)  # (B, T, P, A)

    valid = self._valid_mask(episodes.dones)

    # Value-space (dilated) regularization with RM strategies: fold the per-node
    # KL-to-magnet cost into the reward so the critic learns V_τ. Mirrors
    # MMD.step but uses RM strategies (clipped logs) instead of softmax.
    if self.regularize_value:
      legal = episodes.legal_actions
      s_mu = rm_policy(episodes.agent_output.logits, legal)
      s_ref = rm_policy(magnet_logits, legal)
      log_mu = jnp.where(legal > 0, jnp.log(jnp.clip(s_mu, 1e-9, 1.0)), 0.0)
      log_ref = jnp.where(legal > 0, jnp.log(jnp.clip(s_ref, 1e-9, 1.0)), 0.0)
      node_kl = (s_mu * (log_mu - log_ref)).sum(-1)  # (B, T, P)
      total_kl = node_kl.sum(axis=-1, keepdims=True)  # (B, T, 1)
      reg = magnet_coef * (2.0 * node_kl - total_kl)   # τ·(own − opp) per player
      rewards_eff = episodes.rewards - reg * valid[..., None]
    else:
      rewards_eff = episodes.rewards

    advantages, targets = self._compute_advantages(
      rewards_eff, target_values, episodes.dones
    )

    params, opt_state = state.params, state.opt_state

    if self.alternating:
      n_players = self.env.num_players
      active = state.step % n_players
      player_mask = jnp.zeros(n_players).at[active].set(float(n_players))

    def epoch_fn(carry, _):
      params, opt_state = carry

      def total_loss(params):
        agent_out = self._eval_params(params, episodes)

        _axes = (0,) * 9 + (None,) * 6
        loss_P = jax.vmap(rm_rnad_loss, in_axes=_axes)
        loss_TP = jax.vmap(loss_P, in_axes=_axes)
        loss_BTP = jax.vmap(loss_TP, in_axes=_axes)
        losses, metrics = loss_BTP(
          agent_out.value,
          agent_out.logits,
          episodes.legal_actions,
          episodes.actions,
          episodes.agent_output.logits,  # trajectory sampling regrets
          episodes.agent_output.value,
          magnet_logits,
          advantages,
          targets,
          clip_eps,
          vf_coef,
          ent_coef,
          magnet_coef,
          neurd_clip,
          self.epsilon,
        )
        if self.alternating:
          wmean = lambda x: self._wmean(x * player_mask, valid)
        else:
          wmean = lambda x: self._wmean(x, valid)
        return wmean(losses), jax.tree.map(wmean, metrics)

      (_, metrics), grads = jax.value_and_grad(total_loss, has_aux=True)(params)
      updates, new_opt_state = self.optimizer.update(grads, opt_state, params)
      return (optax.apply_updates(params, updates), new_opt_state), metrics

    (params, opt_state), epoch_metrics = jax.lax.scan(
      epoch_fn, (params, opt_state), None, length=self.n_epochs
    )

    target_params = optax.incremental_update(
      params, state.extras["target_params"], target_update_rate
    )
    if self.magnet_update_type == MagnetUpdateType.INCREMENTAL:
      magnet_params = optax.incremental_update(
        params, state.extras["magnet_params"], magnet_update_rate
      )
    else:
      magnet_params = optax.periodic_update(
        params, state.extras["magnet_params"], state.step, self.magnet_interval
      )

    return TrainingState(
      params=params,
      opt_state=opt_state,
      env_state=state.env_state,
      agent_state=state.agent_state,
      rng=rng,
      step=state.step + 1,
      extras={"target_params": target_params, "magnet_params": magnet_params},
    ), jax.tree.map(jnp.mean, epoch_metrics)
