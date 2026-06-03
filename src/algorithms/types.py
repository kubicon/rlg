from enum import StrEnum


class LossType(StrEnum):
  PPO = "ppo"
  RNAD = "rnad"
  MMD = "mmd"


class MagnetUpdateType(StrEnum):
  PERIODIC = "periodic"
  INCREMENTAL = "incremental"
