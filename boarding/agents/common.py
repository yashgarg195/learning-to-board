"""Shared helpers for all agents."""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import random

import numpy as np


def masked_argmax(q: Sequence[float], mask: np.ndarray) -> int:
    best, best_v = -1, -np.inf
    for a in range(len(q)):
        if mask[a] and q[a] > best_v:
            best, best_v = a, q[a]
    if best < 0:
        valid = np.flatnonzero(mask)
        return int(valid[0]) if len(valid) else 0
    return best


def eps_greedy(q: Sequence[float], mask: np.ndarray, eps: float,
               rng: random.Random) -> int:
    valid = np.flatnonzero(mask)
    if len(valid) == 0:
        return 0
    if rng.random() < eps:
        return int(rng.choice(list(valid)))
    return masked_argmax(q, mask)


class QTable:
    """Sparse tabular Q, keyed by a discretised state."""

    def __init__(self, n_actions: int, init: float = 0.0):
        self.n = n_actions
        self.init = init
        self.q: Dict[Tuple, List[float]] = {}

    def __getitem__(self, s) -> List[float]:
        v = self.q.get(s)
        if v is None:
            v = [self.init] * self.n
            self.q[s] = v
        return v

    def __len__(self) -> int:
        return len(self.q)

    def to_dict(self):
        return {"n_actions": self.n, "init": self.init,
                "q": {repr(k): v for k, v in self.q.items()}}

    @staticmethod
    def from_dict(d) -> "QTable":
        import ast
        t = QTable(d["n_actions"], d.get("init", 0.0))
        t.q = {ast.literal_eval(k): list(v) for k, v in d["q"].items()}
        return t


def moving_average(xs: Sequence[float], w: int = 25) -> List[float]:
    if not xs:
        return []
    w = max(1, min(w, len(xs)))
    out, acc = [], 0.0
    for i, x in enumerate(xs):
        acc += x
        if i >= w:
            acc -= xs[i - w]
        out.append(acc / min(i + 1, w))
    return out
