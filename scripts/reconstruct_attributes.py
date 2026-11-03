#!/usr/bin/env python
"""Reconstruct missing sensitive attributes (SVM) and write labels / gammas.

Released labels and γ are under ``artifacts/`` for the default evaluation path.
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ENGINE = os.path.join(ROOT, "drfo", "engine")
sys.path.insert(0, ENGINE)
os.chdir(ENGINE)

sys.argv[0] = os.path.join(ENGINE, "reconstruct_sensitive.py")
with open(os.path.join(ENGINE, "reconstruct_sensitive.py")) as f:
    code = compile(f.read(), os.path.join(ENGINE, "reconstruct_sensitive.py"), "exec")
    exec(code, {"__name__": "__main__", "__file__": sys.argv[0]})
