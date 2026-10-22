"""Saving / loading trained policies and running them for the demo.

The tabular state is (progress through the queue, band just released, seat
type just released) -- all of which are *size independent* -- so one saved
table can be replayed on any aircraft preset.  Training is offline; the demo
only loads a table and runs a simulation, which takes milliseconds.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .agents.common import QTable, masked_argmax
from .config import SimConfig
from .envs import BoardingEnv
from .simulator import Simulator

POLICY_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "artifacts", "policies")


def save_policy(name: str, q: QTable, meta: Optional[Dict] = None) -> str:
    os.makedirs(POLICY_DIR, exist_ok=True)
    path = os.path.join(POLICY_DIR, name + ".json")
    with open(path, "w") as f:
        json.dump({"q": q.to_dict(), "meta": meta or {}}, f)
    return path


def load_policy(name: str) -> Tuple[Optional[QTable], Dict]:
    path = os.path.join(POLICY_DIR, name + ".json")
    if not os.path.exists(path):
        return None, {}
    with open(path) as f:
        blob = json.load(f)
    return QTable.from_dict(blob["q"]), blob.get("meta", {})


def available_policies() -> List[str]:
    if not os.path.isdir(POLICY_DIR):
        return []
    return sorted(f[:-5] for f in os.listdir(POLICY_DIR) if f.endswith(".json"))


def policy_uses_parity(cfg: SimConfig, q: QTable) -> bool:
    """Whether a saved table was trained with the row-parity action split.

    Older policies were trained with ``n_bands x 3`` actions. Their tables are
    still valid, so rather than invalidate them we infer the action space from
    the table's own width and run them in the matching mode.
    """
    return int(getattr(q, "n", 0)) >= cfg.n_bands * 3 * 2


def run_policy(cfg: SimConfig, q: QTable, log: bool = True) -> Simulator:
    """Greedy roll-out of a tabular policy; returns the finished simulator."""
    cfg = cfg.copy(use_parity=policy_uses_parity(cfg, q))
    env = BoardingEnv(cfg, log=log)
    obs, info = env.reset(seed=cfg.seed)
    done = False
    while not done:
        mask = info["action_mask"]
        if mask.sum() == 0:
            break
        a = masked_argmax(q[env.key()], mask)
        obs, r, term, trunc, info = env.step(a)
        done = term or trunc
    sim = env.sim
    while not sim.done and sim.t < cfg.max_ticks:   # drain any stragglers
        sim.tick(None)
    return sim
