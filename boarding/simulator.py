"""Tick-based, multi-aisle, multi-deck boarding simulator.

Design (spec Section 2 / Figure 4):
  * discrete ticks; each (deck, aisle, row) cell holds at most one passenger
  * passenger states: walking / stowing / blocked / seated
  * one stowing passenger blocks everyone behind them in the same aisle
  * seat interference: climbing over already-seated neighbours costs extra ticks
  * every tick is appended to a delta-encoded trajectory log for replay

The simulator is strategy-agnostic: a *selector* callback decides which waiting
passenger is admitted through a door, which is exactly the gate-controller
action of the MDP in Section 3.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import random

from .config import (BLOCKED, QUEUED, SEATED, STOWING, WALKING, Aircraft,
                     AIRCRAFT, Seat, SimConfig)


@dataclass
class Passenger:
    pid: int
    seat: Seat
    has_bag: bool
    band: int
    group: Tuple[int, str]          # (band, seat_type) -- the RL action label
    door_row: int = 0
    state: int = QUEUED
    pos_row: int = -1               # current row in the aisle (-1 = outside)
    timer: int = 0                  # remaining stow/shuffle ticks
    move_cd: int = 0                # ticks left before the next row step
    release_tick: int = -1
    enter_tick: int = -1
    seat_tick: int = -1
    blocked_ticks: int = 0
    direction: int = 1


class TrajectoryLog:
    """Delta-encoded per-tick log: small enough to ship to the browser."""

    def __init__(self) -> None:
        self.frames: List[List[List[int]]] = []
        self._last: Dict[int, Tuple[int, int, int]] = {}

    def record(self, passengers: Sequence[Passenger]) -> None:
        delta: List[List[int]] = []
        for p in passengers:
            cur = (p.state, p.pos_row, p.seat.deck)
            if self._last.get(p.pid) != cur:
                self._last[p.pid] = cur
                delta.append([p.pid, p.state, p.pos_row])
        self.frames.append(delta)

    def __len__(self) -> int:
        return len(self.frames)


class Simulator:
    def __init__(self, cfg: SimConfig, rng: Optional[random.Random] = None,
                 log: bool = True):
        self.cfg = cfg
        self.spec: Aircraft = cfg.spec()
        self.rng = rng if rng is not None else random.Random(cfg.seed)
        self.t = 0
        self.enabled_log = log
        self.log = TrajectoryLog()
        self.congestion: List[List[int]] = []

        self.passengers: List[Passenger] = self._make_passengers()
        self.n = len(self.passengers)

        # occupancy[deck][aisle][row] = pid or -1
        self.occ: List[List[List[int]]] = [
            [[-1] * d.rows for _ in range(d.n_aisles)] for d in self.spec.decks
        ]
        # seated[deck][row][block][idx] = bool, used for seat interference
        self.seated_grid: Dict[Tuple[int, int, int, int], bool] = {}

        self.doors: List[Tuple[int, int]] = self._active_doors()
        self._assign_doors()
        self.max_dist: int = max(1, max(abs(p.seat.row - p.door_row)
                                        for p in self.passengers))
        self._assign_bands()

        self.pending: List[int] = [p.pid for p in self.passengers]
        self._order_arrivals()

        self.n_seated = 0
        self.blocking_events = 0
        self.total_blocked_ticks = 0
        self.finished_tick: Optional[int] = None
        if self.enabled_log:
            self.log.record(self.passengers)
            self.congestion.append(self._census())

    # ---------------------------------------------------------------- setup
    def _make_passengers(self) -> List[Passenger]:
        seats = self.spec.seat_map()
        n = self.cfg.resolved_passengers()
        rng = random.Random(self.cfg.seed * 7919 + 13)
        if n < len(seats):
            seats = rng.sample(seats, n)
        nb = self.cfg.n_bands
        out: List[Passenger] = []
        for i, s in enumerate(sorted(seats, key=lambda x: (x.deck, x.row, x.col))):
            rows = self.spec.decks[s.deck].rows
            band = min(nb - 1, int(s.row * nb / rows))
            out.append(Passenger(
                pid=i, seat=s,
                has_bag=rng.random() < self.cfg.luggage_fraction,
                band=band, group=(band, s.seat_type),
            ))
        return out

    def _active_doors(self) -> List[Tuple[int, int]]:
        """Returns [(deck, row), ...] for the doors the gate is using."""
        doors: List[Tuple[int, int]] = []
        for di, deck in enumerate(self.spec.decks):
            front, back = deck.doors[0], deck.doors[-1]
            if self.cfg.gate == "front":
                doors.append((di, front))
            elif self.cfg.gate == "back":
                doors.append((di, back))
            else:  # "both" -- wide-bodies board through several doors
                doors.append((di, front))
                doors.append((di, back))
        return doors

    def _assign_doors(self) -> None:
        for p in self.passengers:
            cand = [d for d in self.doors if d[0] == p.seat.deck]
            if not cand:
                cand = self.doors
            # DECISION: passengers use the door closest to their seat row.
            deck, row = min(cand, key=lambda d: abs(d[1] - p.seat.row))
            p.door_row = row
            p.direction = 1 if p.seat.row >= row else -1

    def _assign_bands(self) -> None:
        """Row bands are measured as *distance from the door*.

        DECISION: using distance rather than absolute row makes a learned
        policy transfer unchanged between front-gate and back-gate boarding.
        """
        nb = self.cfg.n_bands
        for p in self.passengers:
            d = abs(p.seat.row - p.door_row)
            p.band = min(nb - 1, int((self.max_dist - d) * nb / (self.max_dist + 1)))
            p.group = (p.band, p.seat.seat_type)

    def _order_arrivals(self) -> None:
        if self.cfg.arrival == "queued":
            self.pending.sort()
        else:
            self.rng.shuffle(self.pending)

    # ------------------------------------------------------------- helpers
    def group_counts(self) -> Dict[Tuple[int, str], int]:
        counts: Dict[Tuple[int, str], int] = {}
        for pid in self.pending:
            g = self.passengers[pid].group
            counts[g] = counts.get(g, 0) + 1
        return counts

    def service_ticks(self, p: Passenger) -> int:
        base = self.cfg.stow_ticks if p.has_bag else self.cfg.no_bag_ticks
        return base + self.cfg.shuffle_ticks * self._blockers(p)

    def _blockers(self, p: Passenger) -> int:
        """Seated neighbours between the aisle and this passenger's seat."""
        s = p.seat
        blocks = self.spec.decks[s.deck].blocks
        size = blocks[s.block]
        n = 0
        for j in range(size):
            if j == s.index_in_block:
                continue
            dj = _depth(blocks, s.block, j)
            if dj < s.depth and self.seated_grid.get((s.deck, s.row, s.block, j)):
                # only neighbours on the aisle side actually get climbed over
                if (s.depth - dj) > 0 and _same_side(blocks, s.block, j, s.index_in_block):
                    n += 1
        return n

    @property
    def done(self) -> bool:
        return self.n_seated >= self.n

    # ----------------------------------------------------------- main loop
    def tick(self, selector: Optional[Callable[[int, int, List[int]], Optional[int]]] = None) -> None:
        """Advance the simulation by one tick."""
        self._service()
        self._move()
        self._admit(selector)
        self.t += 1
        if self.enabled_log:
            self.log.record(self.passengers)
            self.congestion.append(self._census())
        if self.done and self.finished_tick is None:
            self.finished_tick = self.t

    def _census(self) -> List[int]:
        """[walking, stowing, blocked, seated] at the current tick.

        Recorded per tick so the UI can plot where in time the aisle jams
        without replaying the whole trajectory client-side.
        """
        c = [0, 0, 0, 0]
        for p in self.passengers:
            if p.state == WALKING:
                c[0] += 1
            elif p.state == STOWING:
                c[1] += 1
            elif p.state == BLOCKED:
                c[2] += 1
            elif p.state == SEATED:
                c[3] += 1
        return c

    def _service(self) -> None:
        for p in self.passengers:
            if p.state == STOWING:
                p.timer -= 1
                if p.timer <= 0:
                    s = p.seat
                    self.occ[s.deck][s.aisle][p.pos_row] = -1
                    p.state = SEATED
                    p.seat_tick = self.t
                    p.pos_row = s.row
                    self.seated_grid[(s.deck, s.row, s.block, s.index_in_block)] = True
                    self.n_seated += 1

    def _move(self) -> None:
        # Front-most passengers (along their travel direction) move first so a
        # clearing aisle propagates within a single tick.
        movers = [p for p in self.passengers if p.state in (WALKING, BLOCKED)]
        movers.sort(key=lambda p: -(p.direction * p.pos_row))
        for p in movers:
            s = p.seat
            if p.pos_row == s.row:
                self._start_stow(p)
                continue
            p.move_cd -= 1
            if p.move_cd > 0:
                p.state = WALKING
                continue
            nxt = p.pos_row + p.direction
            rows = self.spec.decks[s.deck].rows
            if nxt < 0 or nxt >= rows:
                p.move_cd = 1
                continue
            if self.occ[s.deck][s.aisle][nxt] == -1:
                self.occ[s.deck][s.aisle][p.pos_row] = -1
                p.pos_row = nxt
                self.occ[s.deck][s.aisle][nxt] = p.pid
                p.move_cd = self.cfg.walk_ticks_per_row
                if p.state == BLOCKED:
                    p.state = WALKING
                if p.pos_row == s.row:
                    self._start_stow(p)
            else:
                if p.state != BLOCKED:
                    self.blocking_events += 1
                p.state = BLOCKED
                p.blocked_ticks += 1
                self.total_blocked_ticks += 1
                p.move_cd = 1

    def _start_stow(self, p: Passenger) -> None:
        if p.state == STOWING:
            return
        p.state = STOWING
        p.timer = self.service_ticks(p)

    def _admit(self, selector) -> None:
        if not self.pending:
            return
        for deck, row in self.doors:
            cands = [pid for pid in self.pending
                     if self.passengers[pid].seat.deck == deck
                     and self.passengers[pid].door_row == row
                     and self.occ[deck][self.passengers[pid].seat.aisle][row] == -1]
            if not cands:
                continue
            pid = selector(deck, row, cands) if selector else cands[0]
            if pid is None:
                continue
            self.admit(pid)

    def admit(self, pid: int) -> None:
        p = self.passengers[pid]
        s = p.seat
        self.pending.remove(pid)
        p.pos_row = p.door_row
        self.occ[s.deck][s.aisle][p.door_row] = pid
        p.state = WALKING
        p.release_tick = self.t
        p.enter_tick = self.t
        p.move_cd = self.cfg.walk_ticks_per_row
        if p.pos_row == s.row:
            self._start_stow(p)

    def free_doors(self) -> List[Tuple[int, int, List[int]]]:
        """Doors that can accept somebody right now, with their candidates."""
        out = []
        for deck, row in self.doors:
            cands = [pid for pid in self.pending
                     if self.passengers[pid].seat.deck == deck
                     and self.passengers[pid].door_row == row
                     and self.occ[deck][self.passengers[pid].seat.aisle][row] == -1]
            if cands:
                out.append((deck, row, cands))
        return out

    # --------------------------------------------------------------- stats
    def stats(self) -> Dict[str, float]:
        waits = [p.seat_tick - p.enter_tick for p in self.passengers if p.seat_tick >= 0]
        return {
            "ticks": float(self.finished_tick if self.finished_tick is not None else self.t),
            "blocking_events": float(self.blocking_events),
            "blocked_ticks": float(self.total_blocked_ticks),
            "avg_wait": float(sum(waits) / len(waits)) if waits else 0.0,
            "n_passengers": float(self.n),
            "seed": float(self.cfg.seed),
        }


def _depth(blocks: Sequence[int], b: int, i: int) -> int:
    size = blocks[b]
    if len(blocks) == 1 or b == 0:
        return size - 1 - i
    if b == len(blocks) - 1:
        return i
    half = size / 2.0
    return i if i < half else size - 1 - i


def _same_side(blocks: Sequence[int], b: int, i: int, j: int) -> bool:
    """True if seats i and j of block b are reached from the same aisle."""
    size = blocks[b]
    if len(blocks) <= 2 or b in (0, len(blocks) - 1):
        return True
    half = size / 2.0
    return (i < half) == (j < half)


def run(cfg: SimConfig, selector=None, log: bool = True,
        rng: Optional[random.Random] = None) -> Simulator:
    sim = Simulator(cfg, rng=rng, log=log)
    while not sim.done and sim.t < cfg.max_ticks:
        sim.tick(selector)
    return sim
