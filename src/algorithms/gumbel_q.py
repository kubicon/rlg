"""GumbelQMMD — QMMD variant with a pathwise Gumbel-Softmax policy gradient.

Subclass of ``QMMD`` that reuses all of its machinery (Retrace targets,
Polyak target net, magnet reset, multi-epoch scan, alternating training) and
only swaps the policy-improvement term for the reparameterized Gumbel-Softmax
estimator in ``losses/gumbel_q.py``. The magnet KL, entropy and Q-loss are
untouched, so MMD's regularization anchor is preserved.

This is a mirror of ``QMMD.step`` with two changes:
  - the loss is always ``gumbel_q_loss`` (no loss_type branch); and
  - a per-(B, T, P) PRNG key plus the temperature ``tau`` are threaded into
    the loss for the Gumbel-Softmax draw.

The temperature is configurable via ``gumbel_tau`` and an optional
``gumbel_tau_schedule`` (a callable step -> float), kept separate from
``MMD_SCHEDULABLE`` so no existing module is modified.
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
import jax.lax as lax
import optax

from .base import TrainingState
from .episode import collect_episodes
from .mmd_q import QMMD
from .types import MagnetUpdateType
from ..losses.gumbel_q import gumbel_q_loss


class GumbelQMMD(QMMD):
  """QMMD with the policy gradient computed pathwise via Gumbel-Softmax.

  Args (in addition to all of QMMD's):
      gumbel_tau:           Gumbel-Softmax temperature (> 0). Moderate values
                            (~0.5–2) are recommended; tau -> 0 explodes the
                            gradient variance.
      gumbel_tau_schedule:  Optional callable(step) -> tau for annealing. When
                            provided it overrides ``gumbel_tau`` per step.
  """

  def __init__(
    self,
    *args,
    gumbel_tau: float = 1.0,
    gumbel_tau_schedule: Callable[[int], float] | None = None,
    **kwargs,
  ) -> None:
    super().__init__(*args, **kwargs)
    self.gumbel_tau = gumbel_tau
    self.gumbel_tau_schedule = gumbel_tau_schedule

  # ── Public step (mirror of QMMD.step with GS policy term) ──────────────────

  def step(self, state: TrainingState) -> tuple[TrainingState, dict[str, jax.Array]]:
    def _get(name: str, default):
      s = self.schedules.get(name)
      return s(state.step) if s is not None else default

    clip_eps        = _get("clip_eps",        self.clip_eps)
    vf_coef         = _get("vf_coef",         self.vf_coef)
    ent_coef        = _get("ent_coef",        self.ent_coef)
    magnet_coef     = _get("magnet_coef",     self.magnet_coef)
    old_policy_coef = _get("old_policy_coef", self.old_policy_coef)
    target_update_rate = _get("target_update_rate", self.target_update_rate)
    magnet_update_rate = _get("magnet_update_rate", self.magnet_update_rate)
    tau = (
      self.gumbel_tau_schedule(state.step)
      if self.gumbel_tau_schedule is not None
      else self.gumbel_tau
    )

    rng, collect_key, loss_key = jax.random.split(state.rng, 3)
    _, _, _, episodes = collect_episodes(
      self.env, self.agent, state.params, collect_key, self.batch_size
    )

    # ── Retrace targets: target-net Q-values, on-policy rollout policy ──────
    target_out = self._eval_params(state.extras["target_params"], episodes)
    q_targets = lax.stop_gradient(
      self._compute_q_targets(
        episodes.rewards,
        target_out.q_values,
        episodes.agent_output.logits,
        episodes.actions,
        episodes.legal_actions,
        episodes.dones,
      )
    )  # (B,T,P)

    target_q_values = lax.stop_gradient(target_out.q_values)  # (B,T,P,A)

    magnet_out = self._eval_params(state.extras["magnet_params"], episodes)
    magnet_logits = lax.stop_gradient(magnet_out.logits)  # (B,T,P,A)

    params, opt_state = state.params, state.opt_state
    valid = self._valid_mask(episodes.dones)
    B, T, P = episodes.actions.shape

    if self.alternating:
      n_players = self.env.num_players
      active = state.step % n_players
      player_mask = jnp.zeros(n_players).at[active].set(float(n_players))

    # Fresh GS keys per epoch so the relaxation is re-sampled each pass.
    epoch_keys = jax.random.split(loss_key, self.n_epochs)

    def epoch_fn(carry, epoch_key):
      params, opt_state = carry
      sample_keys = jax.random.split(epoch_key, B * T * P).reshape(B, T, P, 2)

      def total_loss(params):
        agent_out = self._eval_params(params, episodes)

        # 9 arrays + per-sample key (axis 0) + 5 scalars + tau scalar.
        _axes = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, None, None, None, None, None, None)
        loss_P = jax.vmap(gumbel_q_loss, in_axes=_axes)
        loss_TP = jax.vmap(loss_P, in_axes=_axes)
        loss_BTP = jax.vmap(loss_TP, in_axes=_axes)
        losses, metrics = loss_BTP(
          agent_out.q_values,                 # (B,T,P,A)
          agent_out.logits,                   # (B,T,P,A)
          episodes.legal_actions,             # (B,T,P,A)
          episodes.actions,                   # (B,T,P)
          episodes.agent_output.logits,       # (B,T,P,A) — sampling π
          episodes.agent_output.q_values,     # (B,T,P,A) — sampling Q
          magnet_logits,                      # (B,T,P,A)
          q_targets,                          # (B,T,P)
          target_q_values,                    # (B,T,P,A)
          sample_keys,                        # (B,T,P,2)
          clip_eps,
          vf_coef,
          ent_coef,
          magnet_coef,
          old_policy_coef,
          tau,
        )

        if self.alternating:
          wmean = lambda x: self._wmean(x * player_mask, valid)
        else:
          wmean = lambda x: self._wmean(x, valid)
        return wmean(losses), jax.tree.map(wmean, metrics)  # type: ignore

      (_, metrics), grads = jax.value_and_grad(total_loss, has_aux=True)(params)
      updates, new_opt_state = self.optimizer.update(grads, opt_state, params)
      return (optax.apply_updates(params, updates), new_opt_state), metrics

    (params, opt_state), epoch_metrics = jax.lax.scan(
      epoch_fn, (params, opt_state), epoch_keys, length=self.n_epochs
    )

    # Polyak update: target ← τ·params + (1−τ)·target
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
