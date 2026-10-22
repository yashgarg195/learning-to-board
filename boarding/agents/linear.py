"""Lab 9 + Unit 7 -- function approximation and policy gradient.

Both agents consume the *size-independent* feature vector from
``boarding.envs.observe`` (fractions, not counts), so a single weight vector
transfers across aircraft of different sizes -- which is exactly the
generalisation claim the spec asks us to test honestly.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import json
import random

import numpy as np

from ..config import SimConfig
from ..envs import BoardingEnv, n_actions, obs_size, observe
from .common import masked_argmax


def features(obs: np.ndarray) -> np.ndarray:
    """Raw features + bias + a few pairwise interactions."""
    base = np.concatenate([obs, [1.0]])
    return base.astype(np.float64)


class LinearQ:
    """Q(s, a) = w[a] . phi(s) -- one weight vector per discrete action."""

    def __init__(self, n_feat: int, n_act: int):
        self.w = np.zeros((n_act, n_feat))

    def q(self, phi: np.ndarray) -> np.ndarray:
        return self.w.dot(phi)

    def update(self, phi: np.ndarray, a: int, delta: float, alpha: float) -> None:
        self.w[a] += alpha * delta * phi

    def save(self, path: str, meta: Optional[Dict] = None) -> None:
        np.savez(path, w=self.w, meta=json.dumps(meta or {}))

    @staticmethod
    def load(path: str) -> Tuple["LinearQ", Dict]:
        z = np.load(path, allow_pickle=True)
        w = z["w"]
        m = LinearQ(w.shape[1], w.shape[0])
        m.w = w
        meta = json.loads(str(z["meta"])) if "meta" in z else {}
        return m, meta


def train_semi_gradient(env: BoardingEnv, episodes: int = 1200,
                        alpha: float = 0.02, gamma: float = 0.99,
                        eps_start: float = 1.0, eps_end: float = 0.05,
                        algo: str = "qlearning", seed: int = 0,
                        reward_scale: float = 0.05) -> Dict[str, object]:
    rng = random.Random(seed)
    n_feat = obs_size(env.cfg) + 1
    model = LinearQ(n_feat, n_actions(env.cfg))
    curve: List[float] = []
    ticks: List[float] = []
    for ep in range(episodes):
        eps = eps_end + (eps_start - eps_end) * max(0.0, 1 - ep / (0.8 * episodes))
        obs, info = env.reset(seed=rng.randrange(1, 10 ** 6))
        phi = features(obs)
        mask = info["action_mask"]
        a = _eps_act(model.q(phi), mask, eps, rng)
        total = 0.0
        done = False
        while not done:
            obs2, r, term, trunc, info = env.step(a)
            total += r
            rs = r * reward_scale
            done = term or trunc
            phi2 = features(obs2)
            mask2 = info["action_mask"]
            if done or mask2.sum() == 0:
                target = rs
                a2 = 0
            else:
                q2 = model.q(phi2)
                a2 = _eps_act(q2, mask2, eps, rng)
                nxt = q2[a2] if algo == "sarsa" else q2[masked_argmax(q2, mask2)]
                target = rs + gamma * nxt
            delta = target - model.q(phi)[a]
            model.update(phi, a, delta, alpha)
            phi, a = phi2, a2
        curve.append(total)
        ticks.append(env.sim.stats()["ticks"])
    return {"model": model, "curve": curve, "ticks": ticks, "algo": algo}


def _eps_act(q: np.ndarray, mask: np.ndarray, eps: float, rng: random.Random) -> int:
    valid = np.flatnonzero(mask)
    if len(valid) == 0:
        return 0
    if rng.random() < eps:
        return int(rng.choice(list(valid)))
    return masked_argmax(q, mask)


def greedy_rollout(env: BoardingEnv, model: LinearQ, seed: int) -> Dict[str, float]:
    obs, info = env.reset(seed=seed)
    done = False
    while not done:
        mask = info["action_mask"]
        if mask.sum() == 0:
            break
        a = masked_argmax(model.q(features(obs)), mask)
        obs, r, term, trunc, info = env.step(a)
        done = term or trunc
    return env.sim.stats()


# ------------------------------------------------------- Unit 7: REINFORCE
class SoftmaxPolicy:
    def __init__(self, n_feat: int, n_act: int):
        self.theta = np.zeros((n_act, n_feat))

    def probs(self, phi: np.ndarray, mask: np.ndarray) -> np.ndarray:
        z = self.theta.dot(phi)
        z = np.where(mask > 0, z, -1e9)
        z -= z.max()
        e = np.exp(z)
        return e / e.sum()

    def sample(self, phi, mask, rng: random.Random) -> Tuple[int, np.ndarray]:
        p = self.probs(phi, mask)
        r = rng.random()
        c = 0.0
        for i, pi in enumerate(p):
            c += pi
            if r <= c:
                return i, p
        return int(np.argmax(p)), p


def train_reinforce(env: BoardingEnv, episodes: int = 800, alpha: float = 0.01,
                    alpha_v: float = 0.05, gamma: float = 0.995, seed: int = 0,
                    reward_scale: float = 0.05) -> Dict[str, object]:
    rng = random.Random(seed)
    n_feat = obs_size(env.cfg) + 1
    pi = SoftmaxPolicy(n_feat, n_actions(env.cfg))
    wv = np.zeros(n_feat)
    curve, ticks = [], []
    for ep in range(episodes):
        obs, info = env.reset(seed=rng.randrange(1, 10 ** 6))
        traj = []
        done = False
        total = 0.0
        while not done:
            phi = features(obs)
            mask = info["action_mask"]
            if mask.sum() == 0:
                break
            a, p = pi.sample(phi, mask, rng)
            obs, r, term, trunc, info = env.step(a)
            traj.append((phi, a, r * reward_scale, p, mask))
            total += r
            done = term or trunc
        G = 0.0
        returns = []
        for _, _, r, _, _ in reversed(traj):
            G = r + gamma * G
            returns.append(G)
        returns.reverse()
        for (phi, a, _, p, mask), G in zip(traj, returns):
            v = wv.dot(phi)
            adv = G - v
            wv += alpha_v * adv * phi
            grad = -np.outer(p, phi)
            grad[a] += phi
            pi.theta += alpha * adv * grad
        curve.append(total)
        ticks.append(env.sim.stats()["ticks"])
    return {"policy": pi, "value": wv, "curve": curve, "ticks": ticks}
