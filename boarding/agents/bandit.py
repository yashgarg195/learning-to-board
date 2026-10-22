"""Lab 2 -- multi-armed bandits over a fixed set of boarding strategies.

Each arm is one of the baseline strategies.  Pulling an arm = running one
boarding episode with that strategy on a fresh random seed.  Reward is the
negated, scaled boarding time so that "higher is better".  We compare
epsilon-greedy and UCB1 on average reward and cumulative regret (CO2).
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import math
import random

import numpy as np

from ..config import SimConfig
from ..simulator import Simulator
from ..strategies import STRATEGIES, make_selector


def play(cfg: SimConfig, strategy: str, seed: int) -> float:
    c = cfg.copy(seed=seed)
    sim = Simulator(c, rng=random.Random(seed), log=False)
    sel = make_selector(sim, strategy)
    while not sim.done and sim.t < c.max_ticks:
        sim.tick(sel)
    return sim.stats()["ticks"]


def _reward(ticks: float, scale: float) -> float:
    return -ticks / scale


def run_bandit(cfg: SimConfig, arms: Sequence[str], n_steps: int = 400,
               algo: str = "eps", eps: float = 0.1, c: float = 2.0,
               seed: int = 0) -> Dict[str, object]:
    rng = random.Random(seed)
    k = len(arms)
    counts = [0] * k
    means = [0.0] * k
    rewards: List[float] = []
    choices: List[int] = []
    scale = float(cfg.resolved_passengers() * 4)

    # Ground truth (for regret) -- estimated from held-out seeds.
    truth = [np.mean([_reward(play(cfg, a, 10_000 + s), scale) for s in range(12)])
             for a in arms]
    best = max(truth)

    for t in range(n_steps):
        if algo == "eps":
            if rng.random() < eps or min(counts) == 0:
                a = counts.index(0) if min(counts) == 0 else rng.randrange(k)
            else:
                a = int(np.argmax(means))
        else:  # UCB1
            if min(counts) == 0:
                a = counts.index(0)
            else:
                ucb = [means[i] + c * math.sqrt(math.log(t + 1) / counts[i])
                       for i in range(k)]
                a = int(np.argmax(ucb))
        r = _reward(play(cfg, arms[a], rng.randrange(1, 10_000)), scale)
        counts[a] += 1
        means[a] += (r - means[a]) / counts[a]
        rewards.append(r)
        choices.append(a)

    regret = np.cumsum([best - truth[a] for a in choices]).tolist()
    return {
        "arms": list(arms), "counts": counts, "means": means,
        "rewards": rewards, "choices": choices, "regret": regret,
        "truth": [float(x) for x in truth], "best_arm": arms[int(np.argmax(truth))],
    }
