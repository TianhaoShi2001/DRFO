#!/usr/bin/env python
"""Train DRFO (known-ratio setting).

Example:
  python scripts/train_drfo.py --dataset ml-1m --known_ratio 0.3 --gpu 0
"""
from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ENGINE = os.path.join(ROOT, "drfo", "engine")
sys.path.insert(0, ENGINE)
os.chdir(ENGINE)

ATTR_TRAIN = {
    "reconstructed": "noisy",
    "ground_truth": "True",
}
ATTR_VALID = {
    "unobserved": "fake",
    "reconstructed": "noisy",
    "ground_truth": "True",
}


def main():
    p = argparse.ArgumentParser(description="Train DRFO (slow path)")
    p.add_argument("--dataset", choices=["ml-1m", "tenrec"], required=True)
    p.add_argument("--known_ratio", type=float, required=True,
                   help="Fraction of users with known sensitive attributes")
    p.add_argument("--seed", type=int, default=2004)
    p.add_argument("--gpu", type=str, default="0")
    p.add_argument("--num_per_gpu", type=float, default=0.2)
    p.add_argument("--performance_scale", type=float, default=0.98)
    p.add_argument("--train_attribute", default="reconstructed",
                   choices=list(ATTR_TRAIN))
    p.add_argument("--valid_attribute", default="unobserved",
                   choices=list(ATTR_VALID))
    p.add_argument("--resume", default="False")
    args = p.parse_args()

    trial = (
        f"repro-{ATTR_TRAIN[args.train_attribute]}-{ATTR_VALID[args.valid_attribute]}"
        f"-1000-{args.known_ratio}-{args.seed}-{args.dataset}-performance-98"
    )
    cmd = [
        sys.executable, "-u",
        os.path.join(ENGINE, "run_drfo.py"),
        "--data_name", args.dataset,
        "--know_size", str(args.known_ratio),
        "--seed", str(args.seed),
        "--classifier_stage", "1000",
        "--train_sensitive", ATTR_TRAIN[args.train_attribute],
        "--valid_sensitive", ATTR_VALID[args.valid_attribute],
        "--performance_scale", str(args.performance_scale),
        "--num_per_gpu", str(args.num_per_gpu),
        "--cuda", "0",
        "--trial_name", trial,
        "--resume", args.resume,
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu
    env["PYTHONUNBUFFERED"] = "1"
    print("Running:", " ".join(cmd))
    os.execvpe(cmd[0], cmd, env)


if __name__ == "__main__":
    main()
