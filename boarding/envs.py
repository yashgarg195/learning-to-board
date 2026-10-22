"""Gymnasium environment for the gate controller.

  Agent     : the gate controller
  Action    : release the next passenger from one of
              ``n_bands x 3 x parity`` groups -- row band measured from the
              door, window/middle/aisle, and the row's parity.

              The parity term exists because Steffen's advantage comes from
              interleaving *alternate rows*, so that two consecutive stowers
              never stand back to back. Without it the action space cannot
              express that schedule at all, which measurably caps the agent
              below Steffen (see `scripts/probe_rl_vs_baselines.py`). Set
              ``cfg.use_parity = False`` for the original 12-action space;
              policies record which one they were trained with.
  State     : compact, size-independent features -- how many passengers are
              still waiting per band and per seat type, aisle occupancy per
              band, how many are currently blocked, how many are seated
  Reward    : -1 per tick (+ an optional small penalty per blocking event)
  Episode   : ends when every passenger is seated
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import random

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
    _HAS_GYM = True
except Exception:                                     # pragma: no cover
    _HAS_GYM = False

    class _Shim:                                      # minimal stand-in
        pass

    gym = type("gym", (), {"Env": _Shim})             # type: ignore
    spaces = None                                     # type: ignore

from .config import SEAT_TYPES, BLOCKED, SimConfig
from .simulator import Simulator

SEAT_TYPE_INDEX = {t: i for i, t in enumerate(SEAT_TYPES)}


def n_parity(cfg: SimConfig) -> int:
    """2 when the action space distinguishes odd/even rows, else 1."""
    return 2 if getattr(cfg, "use_parity", False) else 1


def n_actions(cfg: SimConfig) -> int:
    return cfg.n_bands * len(SEAT_TYPES) * n_parity(cfg)


def action_to_group(cfg: SimConfig, a: int) -> Tuple[int, str, int]:
    """Decode an action into (band, seat type, row parity).

    Parity is 0 for both halves when the parity dimension is disabled, so
    callers can treat the three-tuple uniformly.
    """
    npar = n_parity(cfg)
    nt = len(SEAT_TYPES)
    par = a % npar if npar > 1 else 0
    rest = a // npar if npar > 1 else a
    return (rest // nt, SEAT_TYPES[rest % nt], par)


def group_to_action(cfg: SimConfig, group: Tuple[int, str, int]) -> int:
    npar = n_parity(cfg)
    base = group[0] * len(SEAT_TYPES) + SEAT_TYPE_INDEX[group[1]]
    if npar == 1:
        return base
    return base * npar + (group[2] % npar)


def passenger_group(cfg: SimConfig, p) -> Tuple[int, str, int]:
    """The action group a passenger belongs to."""
    return (p.band, p.seat.seat_type,
            p.seat.row % 2 if n_parity(cfg) > 1 else 0)


def observe(sim: Simulator, last_action: int = -1) -> np.ndarray:
    cfg = sim.cfg
    nb = cfg.n_bands
    n = float(sim.n)
    band_pending = np.zeros(nb, dtype=np.float32)
    type_pending = np.zeros(len(SEAT_TYPES), dtype=np.float32)
    for pid in sim.pending:
        p = sim.passengers[pid]
        band_pending[p.band] += 1.0
        type_pending[SEAT_TYPE_INDEX[p.seat.seat_type]] += 1.0
    band_occ = np.zeros(nb, dtype=np.float32)
    band_cap = np.zeros(nb, dtype=np.float32) + 1e-6
    in_cabin = 0
    blocked = 0
    for p in sim.passengers:
        if p.state in (1, 2, 3) and p.pos_row >= 0:
            in_cabin += 1
            band_occ[p.band] += 1.0
            if p.state == BLOCKED:
                blocked += 1
        band_cap[p.band] += 1.0
    # One-hot of the previously released group: the controller needs to know
    # what it just did in order to learn *spacing* (do not send two passengers
    # to the same row band back to back).
    last_band = np.zeros(nb + 1, dtype=np.float32)
    last_type = np.zeros(len(SEAT_TYPES) + 1, dtype=np.float32)
    last_par = np.zeros(3, dtype=np.float32)
    if last_action < 0:
        last_band[nb] = 1.0
        last_type[len(SEAT_TYPES)] = 1.0
        last_par[2] = 1.0
    else:
        lb, lt, lp = action_to_group(cfg, last_action)
        last_band[lb] = 1.0
        last_type[SEAT_TYPE_INDEX[lt]] = 1.0
        last_par[lp if n_parity(cfg) > 1 else 2] = 1.0

    feats = np.concatenate([
        band_pending / n,
        type_pending / n,
        band_occ / band_cap,
        np.array([blocked / max(1.0, in_cabin),
                  sim.n_seated / n,
                  len(sim.pending) / n], dtype=np.float32),
        last_band,
        last_type,
        last_par,
    ])
    return feats.astype(np.float32)


def obs_size(cfg: SimConfig) -> int:
    nb = cfg.n_bands
    return nb * 2 + len(SEAT_TYPES) + 3 + (nb + 1) + (len(SEAT_TYPES) + 1) + 3


PROGRESS_BINS = 12


def tabular_key(obs: np.ndarray, sim: Simulator,
                last_action: int = -1) -> Tuple:
    """Compact discrete state for the tabular agents (MC / SARSA / Q).

    DECISION: the boarding problem is essentially a *schedule* -- which group
    to call at which point of the queue -- so the tabular state is
    (progress through the queue, band just released, seat type just released,
    parity just released). Availability of a group is handled by action
    masking rather than by the state, which keeps the table small enough to
    actually be visited.

    The parity term is what lets the agent learn to *alternate* rows, which
    is the mechanism behind Steffen's advantage. It doubles the state count
    (~240 -> ~480), which is still trivially tabular.
    """
    nb = sim.cfg.n_bands
    released = sim.n - len(sim.pending)
    prog = int(min(PROGRESS_BINS - 1, released * PROGRESS_BINS / max(1, sim.n)))
    npar = n_parity(sim.cfg)
    if last_action < 0:
        if npar == 1:
            return (prog, nb, len(SEAT_TYPES))
        return (prog, nb, len(SEAT_TYPES), npar)
    lb, lt, lp = action_to_group(sim.cfg, last_action)
    if npar == 1:
        return (prog, lb, SEAT_TYPE_INDEX[lt])
    return (prog, lb, SEAT_TYPE_INDEX[lt], lp)


class BoardingEnv(gym.Env):
    """One episode = one full boarding of one aircraft configuration."""

    metadata = {"render_modes": []}

    def __init__(self, cfg: Optional[SimConfig] = None,
                 configs: Optional[List[SimConfig]] = None,
                 log: bool = False, invalid_penalty: float = 2.0):
        self.configs = configs or [cfg or SimConfig()]
        self.cfg = self.configs[0]
        self.log = log
        self.invalid_penalty = invalid_penalty
        self.sim: Optional[Simulator] = None
        self.last_action = -1
        self._rng = random.Random(0)
        if _HAS_GYM and spaces is not None:
            self.action_space = spaces.Discrete(n_actions(self.cfg))
            self.observation_space = spaces.Box(
                low=0.0, high=1.0, shape=(obs_size(self.cfg),), dtype=np.float32)

    # ------------------------------------------------------------------ API
    def reset(self, *, seed: Optional[int] = None,
              options: Optional[Dict[str, Any]] = None):
        if seed is not None:
            self._rng.seed(seed)
        cfg = self._rng.choice(self.configs)
        if seed is not None:
            cfg = cfg.copy(seed=seed)
        self.cfg = cfg
        self.sim = Simulator(cfg, rng=random.Random(cfg.seed), log=self.log)
        self.last_action = -1
        self._advance_to_decision()
        return self.obs(), self.info()

    def obs(self) -> np.ndarray:
        return observe(self.sim, self.last_action)

    def key(self) -> Tuple:
        return tabular_key(self.obs(), self.sim, self.last_action)

    def info(self) -> Dict[str, Any]:
        return {"action_mask": self.action_mask(), "ticks": self.sim.t}

    def action_mask(self) -> np.ndarray:
        mask = np.zeros(n_actions(self.cfg), dtype=np.int8)
        assert self.sim is not None
        for _, _, cands in self._free:
            for pid in cands:
                mask[group_to_action(
                    self.cfg,
                    passenger_group(self.cfg, self.sim.passengers[pid]))] = 1
        return mask

    def step(self, action: int):
        sim = self.sim
        assert sim is not None
        t0 = sim.t
        b0 = sim.blocking_events
        reward_extra = 0.0

        band, stype, par = action_to_group(self.cfg, int(action))
        npar = n_parity(self.cfg)
        chosen = None
        for deck, row, cands in self._free:
            matches = [pid for pid in cands
                       if sim.passengers[pid].band == band
                       and sim.passengers[pid].seat.seat_type == stype
                       and (npar == 1
                            or sim.passengers[pid].seat.row % 2 == par)]
            if matches:
                chosen = matches[0]
                break
        self.last_action = int(action)
        if chosen is None:
            # Invalid group: the gate stalls for a tick.
            reward_extra -= self.invalid_penalty
            sim.tick(self._noop_selector)
        else:
            sim.admit(chosen)

        self._advance_to_decision()
        terminated = sim.done
        ticks = sim.t - t0
        blocks = sim.blocking_events - b0
        reward = -float(ticks) - self.cfg.block_penalty * blocks + reward_extra
        truncated = sim.t >= self.cfg.max_ticks and not terminated
        info = self.info()
        if terminated or truncated:
            info["stats"] = sim.stats()
        return self.obs(), reward, terminated, truncated, info

    # -------------------------------------------------------------- helpers
    @staticmethod
    def _noop_selector(deck: int, row: int, cands: List[int]) -> Optional[int]:
        return None

    def _advance_to_decision(self) -> None:
        """Tick until the controller actually has a choice to make."""
        sim = self.sim
        assert sim is not None
        while True:
            self._free = sim.free_doors()
            if self._free:
                return
            if sim.done or sim.t >= self.cfg.max_ticks:
                self._free = []
                return
            sim.tick(self._noop_selector)
