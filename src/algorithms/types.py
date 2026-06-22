from enum import StrEnum


class LossType(StrEnum):
  PPO = "ppo"
  RNAD = "rnad"
  MMD = "mmd"


class MagnetUpdateType(StrEnum):
  PERIODIC = "periodic"
  INCREMENTAL = "incremental"


MMD_SCHEDULABLE = frozenset({
  "clip_eps", "vf_coef", "ent_coef",
  "magnet_coef", "old_policy_coef",
  "target_update_rate", "magnet_update_rate",
  "neurd_clip", "neurd_threshold",
  "alpha",
})
