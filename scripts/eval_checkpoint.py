#!/usr/bin/env python
"""Evaluate a released checkpoint and report DP / RMSE.

Example:
  python scripts/eval_checkpoint.py --dataset ml-1m --known_ratio 0.3
  python scripts/eval_checkpoint.py --dataset ml-1m --known_ratio 0.3 --refusal_ratio 1.0
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _result_path(dataset: str, known_ratio: float, refusal_ratio: float | None, split: str):
    extra = (
        "explicit_True_test_inter_num_10_test_propotion_0.15_thre_3.5_"
        "use_thre_True_only_unknown_False_use_classifier_stageTrue"
    )
    base = os.path.join(
        ROOT, "workspace", "unfairness", "dp", dataset, "MF", extra,
        str(known_ratio), "0.98", "2004", "1000", f"{split}_results",
    )
    if refusal_ratio is None:
        name = "train_noisy_valid_fake_test_true_dro.json"
    else:
        name = f"lack_prob_{refusal_ratio}_train_noisy_valid_fake_test_true_dro.json"
    return os.path.join(base, name), os.path.join(
        ROOT, "artifacts", "checkpoints", dataset,
        (
            f"known_ratio_{known_ratio}_seed_2004"
            if refusal_ratio is None
            else f"known_ratio_{known_ratio}_refusal_{refusal_ratio}_seed_2004"
        ),
        "best_model.pt",
    )


def main():
    p = argparse.ArgumentParser(description="Evaluate a DRFO checkpoint (fast path)")
    p.add_argument("--dataset", choices=["ml-1m", "tenrec"], required=True)
    p.add_argument("--known_ratio", type=float, required=True)
    p.add_argument("--refusal_ratio", type=float, default=None)
    p.add_argument("--split", choices=["test", "val"], default="test")
    args = p.parse_args()

    metrics_path, ckpt_path = _result_path(
        args.dataset, args.known_ratio, args.refusal_ratio, args.split
    )
    if not os.path.exists(ckpt_path):
        raise SystemExit(f"Checkpoint not found: {ckpt_path}")
    if not os.path.exists(metrics_path):
        raise SystemExit(
            f"Cached metrics not found: {metrics_path}\n"
            "Train with scripts/train_drfo.py (or train_drfo_refusal.py) first."
        )

    with open(metrics_path) as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    metrics = json.loads(lines[0])
    print(f"checkpoint: {ckpt_path}")
    print(f"metrics:    {metrics_path}")
    print(f"rmse:       {metrics.get('rmse')}")
    print(f"dp_true:    {metrics.get('dp_true')}")
    if "dp" in metrics:
        print(f"dp:         {metrics.get('dp')}")


if __name__ == "__main__":
    main()
