"""Exact dynamic programming on a tiny cabin.

The full simulator has no closed-form transition model, so DP is run on an
*exact deterministic surrogate* of a tiny 4-row x 4-seat (2-2) cabin.  This is
the spec's "DP only on the tiny cabin as a sanity check" mitigation.

Surrogate (IMPLEMENTATION DECISION, stated explicitly because it is not in the
spec):  passengers are admitted one at a time through the front door.  A
passenger admitted at time ``t`` for row ``r`` occupies the aisle at row ``r``
for ``s`` ticks (stow time, plus a shuffle penalty if the aisle neighbour on
their side is already seated).  The next passenger can therefore be admitted
after

    h = 1                  if the next row is in front of the previous row
    h = max(1, s_prev)     otherwise (they would otherwise catch the blocker)

and the episode ends ``r_last * w + s_last`` ticks after the last admission.
This reproduces the two effects that matter -- *spacing* and *seat
interference* -- while remaining a small, exactly solvable MDP.

State  : per-row occupancy (canonicalised over the two symmetric sides),
         the row of the last admitted passenger, and that passenger's service
         time (needed to make the headway Markov).
Action : (row, seat type in {window, aisle}).
Reward : -headway ticks.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import itertools
import time

import numpy as np

# Side codes: bit0 = aisle seat pending, bit1 = window seat pending
BOTH, WIN_ONLY, AISLE_ONLY, FULL = 3, 2, 1, 0
SIDE_CODES = (0, 1, 2, 3)
ROW_STATES: List[Tuple[int, int]] = sorted(
    {tuple(sorted(p)) for p in itertools.product(SIDE_CODES, repeat=2)})
ROW_INDEX = {rs: i for i, rs in enumerate(ROW_STATES)}


@dataclass(frozen=True)
class TinyMDP:
    rows: int = 4
    stow: int = 6
    shuffle: int = 4
    walk: int = 1

    @property
    def services(self) -> Tuple[int, int]:
        return (self.stow, self.stow + self.shuffle)

    @property
    def n_actions(self) -> int:
        return self.rows * 2          # (row, window|aisle)

    def start(self) -> Tuple:
        return (tuple([ROW_INDEX[(BOTH, BOTH)]] * self.rows), self.rows, 0)

    # ------------------------------------------------------------------
    def all_states(self) -> List[Tuple]:
        """Reachable states only (BFS from the start state)."""
        seen = {self.start()}
        frontier = [self.start()]
        while frontier:
            nxt = []
            for s in frontier:
                if self.is_terminal(s):
                    continue
                for a in self.actions(s):
                    ns, _ = self.step(s, a)
                    if ns not in seen:
                        seen.add(ns)
                        nxt.append(ns)
            frontier = nxt
        return sorted(seen)

    def is_terminal(self, s) -> bool:
        return all(ROW_STATES[i] == (FULL, FULL) for i in s[0])

    def actions(self, s) -> List[int]:
        acts = []
        for r in range(self.rows):
            sides = ROW_STATES[s[0][r]]
            if any(c & 2 for c in sides):
                acts.append(r * 2)        # window pending
            if any(c & 1 for c in sides):
                acts.append(r * 2 + 1)    # aisle pending
        return acts

    def step(self, s, a) -> Tuple[Tuple, float]:
        """Deterministic transition; returns (next_state, reward)."""
        rowcfg, last, sp = s
        r, kind = a // 2, a % 2           # kind 0 = window, 1 = aisle
        sides = list(ROW_STATES[rowcfg[r]])
        if kind == 0:
            # DECISION: prefer the side whose aisle seat is still free
            # (a controller would seat the easier passenger first).
            cand = [i for i, c in enumerate(sides) if c & 2]
            cand.sort(key=lambda i: 0 if sides[i] & 1 else 1)
            i = cand[0]
            blocked = not (sides[i] & 1)
            sides[i] &= ~2
            service = self.stow + (self.shuffle if blocked else 0)
        else:
            cand = [i for i, c in enumerate(sides) if c & 1]
            i = cand[0]
            sides[i] &= ~1
            service = self.stow
        new_rowcfg = list(rowcfg)
        new_rowcfg[r] = ROW_INDEX[tuple(sorted(sides))]
        s_prev = self.services[sp]
        head = 1 if last >= self.rows else (1 if r < last else max(1, s_prev))
        sp_new = 0 if service == self.stow else 1
        ns = (tuple(new_rowcfg), r, sp_new)
        cost = float(head)
        if self.is_terminal(ns):
            cost += r * self.walk + service       # last passenger must sit down
        return ns, -cost


def _index(mdp: TinyMDP, states: Sequence[Tuple]) -> Dict[Tuple, int]:
    return {s: i for i, s in enumerate(states)}


def value_iteration(mdp: TinyMDP, gamma: float = 1.0, theta: float = 1e-6,
                    max_iter: int = 500):
    states = mdp.all_states()
    idx = _index(mdp, states)
    V = np.zeros(len(states))
    terminal = np.array([mdp.is_terminal(s) for s in states])
    cache = [None if terminal[i] else
             [(a,) + mdp.step(s, a) for a in mdp.actions(s)]
             for i, s in enumerate(states)]
    t0 = time.time()
    sweeps = 0
    for _ in range(max_iter):
        delta = 0.0
        for i, s in enumerate(states):
            if terminal[i]:
                continue
            best = max(r + gamma * V[idx[ns]] for _, ns, r in cache[i])
            delta = max(delta, abs(best - V[i]))
            V[i] = best
        sweeps += 1
        if delta < theta:
            break
    policy = {}
    for i, s in enumerate(states):
        if terminal[i]:
            continue
        policy[s] = max(cache[i], key=lambda t: t[2] + gamma * V[idx[t[1]]])[0]
    return {"V": V, "states": states, "policy": policy, "sweeps": sweeps,
            "seconds": time.time() - t0, "backups": sweeps * int((~terminal).sum())}


def policy_iteration(mdp: TinyMDP, gamma: float = 1.0, theta: float = 1e-6,
                     max_iter: int = 200):
    states = mdp.all_states()
    idx = _index(mdp, states)
    terminal = np.array([mdp.is_terminal(s) for s in states])
    cache = [None if terminal[i] else
             [(a,) + mdp.step(s, a) for a in mdp.actions(s)]
             for i, s in enumerate(states)]
    policy = {s: cache[i][0][0] for i, s in enumerate(states) if not terminal[i]}
    V = np.zeros(len(states))
    t0 = time.time()
    sweeps = 0
    iters = 0
    for _ in range(max_iter):
        # policy evaluation
        while True:
            delta = 0.0
            for i, s in enumerate(states):
                if terminal[i]:
                    continue
                a = policy[s]
                ns, r = next((n, rr) for aa, n, rr in cache[i] if aa == a)
                v = r + gamma * V[idx[ns]]
                delta = max(delta, abs(v - V[i]))
                V[i] = v
            sweeps += 1
            if delta < theta:
                break
        # policy improvement
        stable = True
        for i, s in enumerate(states):
            if terminal[i]:
                continue
            best = max(cache[i], key=lambda t: t[2] + gamma * V[idx[t[1]]])[0]
            if best != policy[s]:
                policy[s] = best
                stable = False
        iters += 1
        if stable:
            break
    return {"V": V, "states": states, "policy": policy, "sweeps": sweeps,
            "iterations": iters, "seconds": time.time() - t0,
            "backups": sweeps * int((~terminal).sum())}


def rollout(mdp: TinyMDP, policy: Dict[Tuple, int]) -> Tuple[float, List[int]]:
    s = mdp.start()
    total = 0.0
    seq: List[int] = []
    while not mdp.is_terminal(s):
        a = policy[s]
        seq.append(a)
        s, r = mdp.step(s, a)
        total -= r
    return total, seq


def describe_sequence(mdp: TinyMDP, seq: Sequence[int]) -> List[str]:
    return ["row %d %s" % (a // 2 + 1, "window" if a % 2 == 0 else "aisle")
            for a in seq]
