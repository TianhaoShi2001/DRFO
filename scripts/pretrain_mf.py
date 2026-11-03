#!/usr/bin/env python
"""Pre-train the MF backbone before DRFO fine-tuning.

Released checkpoints already include pretrained MF weights under
``artifacts/pretrained_mf/``. Re-run this only if you rebuild from raw data.
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ENGINE = os.path.join(ROOT, "drfo", "engine")
sys.path.insert(0, ENGINE)
os.chdir(ENGINE)

sys.argv[0] = os.path.join(ENGINE, "pretrain_mf.py")
with open(os.path.join(ENGINE, "pretrain_mf.py")) as f:
    code = compile(f.read(), os.path.join(ENGINE, "pretrain_mf.py"), "exec")
    exec(code, {"__name__": "__main__", "__file__": sys.argv[0]})
