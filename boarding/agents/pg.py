"""Unit 7 -- tabular softmax policy gradient with a *paired* baseline.

Why this agent exists
---------------------
Value-based control struggles on this MDP (see the ablation note in
``tabular.py``): every admission earns roughly the same immediate reward and
the successor state barely depends on the action, so one-step Q differences
collapse.  Eligibility traces repair that, but the learned value function is
still fighting a very noisy target -- a single episode's boarding time swings
by +-12 ticks purely because of which passengers drew carry-on bags.

Policy gradient sidesteps the value function entirely and optimises the
*ordering rule* directly.  The remaining problem is variance, and here the
simulator hands us an unusually strong tool: it is a deterministic function of
(policy, seed).  So we evaluate every policy update against a fixed pool of
seeds and use a **per-seed running baseline**.  The advantage then answers
"was this rollout better than my usual result *on this exact cabin*?", which
cancels the passenger-draw noise instead of averaging over it.

That single change is what takes the learned agent past the hand-designed
baselines.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import math
import random

import numpy as np

from ..envs import BoardingEnv, n_actions
from .common import QTable


class SoftmaxTable:
    """theta[state][action]; pi(a|s) = softmax over the *valid* actions."""

    def __init__(self, n_act: int):
        self.n_act = n_act
        self.theta: Dict[Tuple, np.ndarray] = {}

    def row(self, s: Tuple) -> np.ndarray:
        r = self.theta.get(s)
        if r is None:
            r = np.zeros(self.n_act)
            self.theta[s] = r
        return r

    def probs(self, s: Tuple, mask: np.ndarray) -> np.ndarray:
        z = np.where(mask > 0, self.row(s), -1e9)
        z = z - z.max()
        e = np.exp(z)
        return e / e.sum()

    def sample(self, s: Tuple, mask: np.ndarray,
               rng: random.Random) -> Tuple[int, np.ndarray]:
        p = self.probs(s, mask)
        u = rng.random()
        c = 0.0
        for i, pi in enumerate(p):
            c += pi
            if u <= c:
                return i, p
        return int(np.argmax(p)), p

    def greedy(self, s: Tuple, mask: np.ndarray) -> int:
        z = np.where(mask > 0, self.row(s), -1e9)
        return int(np.argmax(z))

    def to_qtable(self) -> QTable:
        """Greedy action selection over ``theta`` is identical to ``argmax`` over
        a Q-table, so the learned preferences can reuse the whole save / load /
        replay path without a second serialisation format."""
        t = QTable(self.n_act)
        t.q = {k: list(map(float, v)) for k, v in self.theta.items()}
        return t


def train_tabular_pg(env: BoardingEnv, episodes: int = 20000,
                     alpha: float = 0.25, batch: int = 16,
                     n_seeds: int = 48, entropy: float = 0.01,
                     baseline_decay: float = 0.3, seed: int = 0,
                     eval_every: int = 0,
                     eval_env: Optional[BoardingEnv] = None,
                     eval_seeds: Optional[Sequence[int]] = None,
                     on_progress: Optional[Callable[[int, int, List[Tuple[int, float]]], None]] = None
                     ) -> Dict[str, object]:
    """REINFORCE on ``env.key()`` states, batched, with a per-seed baseline.

    ``alpha`` is applied to a *normalised* advantage (units of standard
    deviations), so it is insensitive to cabin size -- a 60-passenger and a
    500-passenger cabin both need roughly the same step size.

    ``eval_every`` periodically records the *greedy* boarding time.  Policy
    gradient explores by sampling, so its raw training episodes are much slower
    than the policy it is actually learning; plotting the sampled episodes as a
    learning curve would make a working agent look flat.
    """
    rng = random.Random(seed)
    pi = SoftmaxTable(n_actions(env.cfg))
    # Per-seed running mean of the episode's boarding time.
    base: Dict[int, float] = {}
    curve: List[float] = []
    ticks: List[float] = []
    log: List[Tuple[int, float]] = []
    greedy_curve: List[Tuple[int, float]] = []

    n_batches = max(1, episodes // batch)
    for b in range(n_batches):
        rollouts = []
        advs = []
        for j in range(batch):
            sd = 1 + ((b * batch + j) % n_seeds) if n_seeds > 0 \
                else rng.randrange(1, 10 ** 6)
            obs, info = env.reset(seed=sd)
            traj: List[Tuple[Tuple, int, np.ndarray, np.ndarray]] = []
            done = False
            total = 0.0
            while not done:
                s = env.key()
                mask = info["action_mask"]
                if mask.sum() == 0:
                    break
                a, p = pi.sample(s, mask, rng)
                traj.append((s, a, p, mask))
                obs, r, term, trunc, info = env.step(a)
                total += r
                done = term or trunc
            t = float(env.sim.stats()["ticks"])
            prev = base.get(sd)
            base[sd] = t if prev is None else (1 - baseline_decay) * prev \
                + baseline_decay * t
            # Lower boarding time is better, so the advantage is negated.
            advs.append(-(t - (prev if prev is not None else t)))
            rollouts.append(traj)
            curve.append(total)
            ticks.append(t)

        adv = np.asarray(advs, dtype=np.float64)
        sd_adv = adv.std()
        if sd_adv < 1e-8:
            continue
        adv = (adv - adv.mean()) / sd_adv

        # Accumulate the batch gradient before stepping -- per-episode updates
        # would let one lucky rollout dominate.
        grad: Dict[Tuple, np.ndarray] = {}
        for traj, A in zip(rollouts, adv):
            scale = A / max(1, len(traj))
            for s, a, p, mask in traj:
                g = grad.get(s)
                if g is None:
                    g = np.zeros(pi.n_act)
                    grad[s] = g
                g -= scale * p
                g[a] += scale
                if entropy > 0.0:
                    with np.errstate(divide="ignore", invalid="ignore"):
                        logp = np.where(p > 1e-12, np.log(np.maximum(p, 1e-12)), 0.0)
                    ent_g = -p * (logp - float((p * logp).sum()))
                    g += (entropy / max(1, len(traj))) * ent_g * np.where(mask > 0, 1.0, 0.0)

        for s, g in grad.items():
            row = pi.row(s)
            row += alpha * g
            np.clip(row, -30.0, 30.0, out=row)

        if eval_every and eval_env is not None and \
                (b + 1) % max(1, eval_every // batch) == 0:
            gt = evaluate_pg(eval_env, pi,
                             eval_seeds if eval_seeds is not None
                             else VALIDATION_SEEDS)
            greedy_curve.append(((b + 1) * batch, float(np.mean(gt))))

        if on_progress is not None:
            on_progress((b + 1) * batch, n_batches * batch, greedy_curve)

    return {"policy": pi, "curve": curve, "ticks": ticks,
            "states": len(pi.theta), "log": log,
            "greedy_curve": greedy_curve}


def evaluate_pg(env: BoardingEnv, pi: SoftmaxTable,
                seeds: Sequence[int]) -> List[float]:
    out: List[float] = []
    for sd in seeds:
        obs, info = env.reset(seed=sd)
        done = False
        while not done:
            mask = info["action_mask"]
            if mask.sum() == 0:
                break
            a = pi.greedy(env.key(), mask)
            obs, r, term, trunc, info = env.step(a)
            done = term or trunc
        out.append(float(env.sim.stats()["ticks"]))
    return out


# Disjoint from both the training pool (1..n_seeds) and the test seeds (500+).
VALIDATION_SEEDS = list(range(300, 320))


def train_best_of(env: BoardingEnv, eval_env: BoardingEnv,
                  episodes: int = 12000, restarts: int = 5,
                  validation: Optional[Sequence[int]] = None,
                  **kw) -> Dict[str, object]:
    """Run several independent restarts and keep the best on a *validation*
    seed set.

    Policy gradient on this problem is genuinely multi-modal -- different
    restarts settle into different boarding schedules, and single-run results
    scatter by ~20 ticks.  Reporting one lucky seed would be dishonest, and
    reporting the mean would understate what the method can reach, so instead
    we select across restarts using seeds that appear in neither the training
    pool nor the test set.
    """
    val = list(validation if validation is not None else VALIDATION_SEEDS)
    base_seed = int(kw.pop("seed", 0))
    # Only the first restart records the greedy curve -- it is for the figure,
    # and running it on every restart would multiply the training cost.
    eval_every = int(kw.pop("eval_every", 0))
    best: Optional[Dict[str, object]] = None
    scores: List[float] = []
    for i in range(restarts):
        out = train_tabular_pg(env, episodes=episodes, seed=base_seed + i,
                               eval_every=eval_every if i == 0 else 0,
                               eval_env=eval_env, eval_seeds=val, **kw)
        score = float(np.mean(evaluate_pg(eval_env, out["policy"], val)))
        scores.append(score)
        out["validation"] = score
        out["restart"] = i
        if best is None or score < float(best["validation"]):
            best = out
    assert best is not None
    best["restart_scores"] = scores
    return best

