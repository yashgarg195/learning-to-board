"""Tabular model-free prediction and control on the real simulator.

  * Monte-Carlo prediction, first-visit vs every-visit
  * SARSA (on-policy) vs Q-learning (off-policy), plus Monte-Carlo
    control as a reference

WHY ELIGIBILITY TRACES (implementation decision, stated openly):
one-step TD cannot separate the actions of this MDP.  Releasing passenger
group A instead of group B almost always costs exactly one tick *now*; the
cost of a bad choice appears 20-40 ticks later as blocking deep in the cabin,
and the next state (progress through the queue) is almost the same whichever
group was chosen.  With lambda > 0 the credit reaches back to the decision
that caused the jam, so SARSA(lambda) learns while SARSA(0) does not.  Both
are reported in the experiments so the difference is visible.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import random

import numpy as np

from ..config import SimConfig
from ..envs import BoardingEnv, n_actions
from .common import QTable, eps_greedy, masked_argmax


def _eps_schedule(ep: int, episodes: int, lo: float, hi: float) -> float:
    return max(lo, hi - (hi - lo) * ep / max(1.0, 0.7 * episodes))


def _episode_seed(ep: int, rng: random.Random, n_seeds: int) -> int:
    """Common random numbers: reusing a fixed pool of passenger draws makes
    two candidate actions comparable on the *same* cabin, which cuts the
    variance of the learning signal dramatically."""
    if n_seeds > 0:
        return 1 + (ep % n_seeds)
    return rng.randrange(1, 10 ** 6)


# ------------------------------------------------------------ Monte-Carlo
def mc_prediction(env: BoardingEnv, episodes: int = 400, gamma: float = 1.0,
                  first_visit: bool = True, seed: int = 0) -> Dict[str, object]:
    """Monte-Carlo *prediction* of the uniform-random gate policy."""
    rng = random.Random(seed)
    returns: Dict[Tuple, List[float]] = {}
    V: Dict[Tuple, float] = {}
    ticks: List[float] = []
    for ep in range(episodes):
        obs, info = env.reset(seed=rng.randrange(1, 10 ** 6))
        traj: List[Tuple[Tuple, float]] = []
        done = False
        while not done:
            s = env.key()
            valid = np.flatnonzero(info["action_mask"])
            if len(valid) == 0:
                break
            a = int(rng.choice(list(valid)))
            obs, r, term, trunc, info = env.step(a)
            traj.append((s, r))
            done = term or trunc
        first_index: Dict[Tuple, int] = {}
        for i, (s, _) in enumerate(traj):
            first_index.setdefault(s, i)
        G = 0.0
        for t in range(len(traj) - 1, -1, -1):
            s, r = traj[t]
            G = r + gamma * G
            if first_visit and first_index[s] != t:
                continue
            returns.setdefault(s, []).append(G)
            V[s] = float(np.mean(returns[s]))
        ticks.append(float(env.sim.stats()["ticks"]))
    return {"V": V, "n_states": len(V), "ticks": ticks,
            "visits": {k: len(v) for k, v in returns.items()},
            "mode": "first_visit" if first_visit else "every_visit"}


def mc_control(env: BoardingEnv, episodes: int = 4000, gamma: float = 1.0,
               eps_start: float = 1.0, eps_end: float = 0.05,
               seed: int = 0, n_seeds: int = 64) -> Dict[str, object]:
    rng = random.Random(seed)
    q = QTable(n_actions(env.cfg))
    counts: Dict[Tuple[Tuple, int], int] = {}
    ticks: List[float] = []
    rets: List[float] = []
    for ep in range(episodes):
        eps = _eps_schedule(ep, episodes, eps_end, eps_start)
        obs, info = env.reset(seed=_episode_seed(ep, rng, n_seeds))
        traj = []
        done = False
        total = 0.0
        while not done:
            s = env.key()
            mask = info["action_mask"]
            if mask.sum() == 0:
                break
            a = eps_greedy(q[s], mask, eps, rng)
            obs, r, term, trunc, info = env.step(a)
            traj.append((s, a, r))
            total += r
            done = term or trunc
        G = 0.0
        for s, a, r in reversed(traj):
            G = r + gamma * G
            k = (s, a)
            counts[k] = counts.get(k, 0) + 1
            q[s][a] += (G - q[s][a]) / counts[k]
        ticks.append(float(env.sim.stats()["ticks"]))
        rets.append(total)
    return {"q": q, "ticks": ticks, "curve": rets, "n_states": len(q),
            "algo": "mc_control"}


# ------------------------------------------------------------- TD control
def td_control(env: BoardingEnv, episodes: int = 6000, alpha: float = 0.05,
               gamma: float = 1.0, lam: float = 0.95, eps_start: float = 1.0,
               eps_end: float = 0.05, algo: str = "sarsa", seed: int = 0,
               reward_scale: float = 0.1, n_seeds: int = 64) -> Dict[str, object]:
    """SARSA(lambda) / Watkins' Q(lambda) with replacing traces."""
    rng = random.Random(seed)
    q = QTable(n_actions(env.cfg))
    ticks: List[float] = []
    rets: List[float] = []
    for ep in range(episodes):
        eps = _eps_schedule(ep, episodes, eps_end, eps_start)
        obs, info = env.reset(seed=_episode_seed(ep, rng, n_seeds))
        trace: Dict[Tuple[Tuple, int], float] = {}
        s = env.key()
        mask = info["action_mask"]
        if mask.sum() == 0:
            continue
        a = eps_greedy(q[s], mask, eps, rng)
        done = False
        total = 0.0
        while not done:
            obs2, r, term, trunc, info = env.step(a)
            total += r
            done = term or trunc
            s2 = env.key()
            mask2 = info["action_mask"]
            rs = r * reward_scale
            cut = False
            if done or mask2.sum() == 0:
                delta = rs - q[s][a]
                a2 = 0
            else:
                a2 = eps_greedy(q[s2], mask2, eps, rng)
                a_star = masked_argmax(q[s2], mask2)
                nxt = q[s2][a2] if algo == "sarsa" else q[s2][a_star]
                delta = rs + gamma * nxt - q[s][a]
                cut = (algo != "sarsa") and (a2 != a_star)
            trace[(s, a)] = 1.0
            for key, e in list(trace.items()):
                q[key[0]][key[1]] += alpha * delta * e
                e2 = e * gamma * lam
                if e2 < 1e-3:
                    del trace[key]
                else:
                    trace[key] = e2
            if cut:
                trace.clear()
            s, a = s2, a2
        ticks.append(float(env.sim.stats()["ticks"]))
        rets.append(total)
    return {"q": q, "ticks": ticks, "curve": rets, "n_states": len(q),
            "algo": algo, "lambda": lam}


def evaluate_tabular(env: BoardingEnv, q: QTable,
                     seeds: Sequence[int]) -> List[float]:
    out = []
    for sd in seeds:
        obs, info = env.reset(seed=sd)
        done = False
        while not done:
            mask = info["action_mask"]
            if mask.sum() == 0:
                break
            a = masked_argmax(q[env.key()], mask)
            obs, r, term, trunc, info = env.step(a)
            done = term or trunc
        out.append(float(env.sim.stats()["ticks"]))
    return out
