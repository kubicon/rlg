#!/usr/bin/env python
"""Compute exploitability for RM (regret-matching) checkpoints.

Identical to ``eval_checkpoints.py`` except the strategy is extracted from the
network output by REGRET MATCHING (relu, then L1-normalise) instead of softmax,
because RMRNaD interprets the policy-head output as cumulative regrets, not
logits. No exploration floor is applied here -- exploitability is measured on
the pure RM strategy (the readout policy), not the ε-on-policy behaviour.

Usage:
    python eval_checkpoints_rm.py <checkpoint_dir>
    python eval_checkpoints_rm.py public_experiments/rm_rnad_leduc
"""

from __future__ import annotations

import sys
from typing import Any

import numpy as np

import eval_checkpoints as ec
from src.tree import Strategy


def _rm_params_to_strategy(
  params: Any,
  ids: list[bytes],
  obs_batch: np.ndarray,
  mask_batch: np.ndarray,
  apply_fn,
) -> Strategy:
  """Single batched forward pass → Strategy dict, via regret matching.

  The network output is the per-action cumulative regret R. The strategy is
  ``relu(R) / Σ relu(R)`` over legal actions, falling back to uniform over legal
  actions when every regret is ≤ 0 (matching ``rm_policy`` used in training).
  """
  if not ids:
    return {}

  regrets_batch = np.asarray(apply_fn(params, obs_batch))  # (N, A)
  strategy: Strategy = {}
  for idx, iset_id in enumerate(ids):
    mask = mask_batch[idx]
    pos = np.where(mask, np.maximum(regrets_batch[idx], 0.0), 0.0)
    denom = pos.sum()
    if denom > 0:
      probs = pos / denom
    else:
      probs = mask.astype(np.float64)
      probs /= probs.sum()
    strategy[iset_id] = probs

  return strategy


def main(checkpoint_dir: str) -> None:
  # Swap only the strategy extractor; reuse all of eval_checkpoints.main.
  ec._params_to_strategy = _rm_params_to_strategy
  ec.main(checkpoint_dir)


if __name__ == "__main__":
  if len(sys.argv) != 2:
    print(__doc__)
    sys.exit(1)
  main(sys.argv[1])
