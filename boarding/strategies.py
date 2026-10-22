"""Baseline boarding strategies (spec Section 4).

Each strategy returns, for every passenger, a pair ``(group_index, ideal_rank)``:

  * ``group_index`` is the boarding group that must be called before the
    passenger may enter (groups are called in increasing order);
  * ``ideal_rank`` is the position inside the group under *perfect* compliance.

Arrival patterns (spec Section 4A) control *compliance*, not the strategy:

  * ``queued``  -- perfect compliance; passengers line up in the ideal order.
  * ``random``  -- the order inside a called group is the random arrival order.
  * ``groups``  -- travelling parties arrive together.  Companions inherit the
    arrival position of the first member of their party, so a family boards as
    a unit even when the strategy would have split them across the cabin.
    This is the realistic middle ground between the two extremes above.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import random

from .simulator import Passenger, Simulator

Plan = Dict[int, Tuple[int, int]]


def _dist(p: Passenger) -> int:
    """How far the seat is from the door the passenger uses."""
    return abs(p.seat.row - p.door_row)


def _rank(passengers: List[Passenger], key) -> Dict[int, int]:
    order = sorted(passengers, key=key)
    return {p.pid: i for i, p in enumerate(order)}


def strategy_random(sim: Simulator, rng: random.Random) -> Plan:
    pids = [p.pid for p in sim.passengers]
    rng.shuffle(pids)
    return {pid: (0, i) for i, pid in enumerate(pids)}


def strategy_back_to_front(sim: Simulator, rng: random.Random, zones: int = 4) -> Plan:
    plan: Plan = {}
    max_d = max(_dist(p) for p in sim.passengers) or 1
    ranks = _rank(sim.passengers, lambda p: (-_dist(p), p.seat.depth * -1))
    for p in sim.passengers:
        z = min(zones - 1, int(_dist(p) * zones / (max_d + 1)))
        plan[p.pid] = (zones - 1 - z, ranks[p.pid])   # farthest zone boards first
    return plan


def strategy_wilma(sim: Simulator, rng: random.Random) -> Plan:
    """Window / middle / aisle, each group back-to-front."""
    order = {"window": 0, "middle": 1, "aisle": 2}
    ranks = _rank(sim.passengers, lambda p: (-_dist(p),))
    return {p.pid: (order[p.seat.seat_type], ranks[p.pid]) for p in sim.passengers}


def strategy_steffen(sim: Simulator, rng: random.Random) -> Plan:
    """Modified Steffen: window/middle/aisle x alternating rows -> 6 groups.

    Inside a group the ideal order is far-to-near with sides alternating, which
    is what produces the classic spaced 'every other row' pattern.
    """
    order = {"window": 0, "middle": 1, "aisle": 2}
    plan: Plan = {}
    buckets: Dict[int, List[Passenger]] = {}
    for p in sim.passengers:
        g = order[p.seat.seat_type] * 2 + (p.seat.row % 2)
        buckets.setdefault(g, []).append(p)
    for g, members in buckets.items():
        members.sort(key=lambda p: (-_dist(p), p.seat.block))
        # alternate sides so two consecutive passengers never share a row
        side_a = [p for p in members if p.seat.block % 2 == 0]
        side_b = [p for p in members if p.seat.block % 2 == 1]
        interleaved: List[Passenger] = []
        for a, b in zip(side_a, side_b):
            interleaved.extend([a, b])
        interleaved.extend(side_a[len(side_b):])
        interleaved.extend(side_b[len(side_a):])
        for i, p in enumerate(interleaved):
            plan[p.pid] = (g, i)
    return plan


def strategy_outside_in_blocks(sim: Simulator, rng: random.Random) -> Plan:
    """Window->aisle crossed with two row zones (a practical 6-group scheme)."""
    order = {"window": 0, "middle": 1, "aisle": 2}
    max_d = max(_dist(p) for p in sim.passengers) or 1
    ranks = _rank(sim.passengers, lambda p: (-_dist(p),))
    plan: Plan = {}
    for p in sim.passengers:
        far = 0 if _dist(p) > max_d / 2 else 1
        plan[p.pid] = (order[p.seat.seat_type] * 2 + far, ranks[p.pid])
    return plan


STRATEGIES: Dict[str, Callable[[Simulator, random.Random], Plan]] = {
    "random": strategy_random,
    "back_to_front": strategy_back_to_front,
    "wilma": strategy_wilma,
    "steffen": strategy_steffen,
    "outside_in": strategy_outside_in_blocks,
}

STRATEGY_LABELS = {
    "random": "Random",
    "back_to_front": "Back-to-front",
    "wilma": "WILMA",
    "steffen": "Steffen-style",
    "outside_in": "Outside-in zones",
    "rl": "RL agent",
}


def group_arrival_ranks(sim: Simulator, rng: random.Random,
                        party_sizes: Sequence[int] = (1, 1, 2, 2, 3, 4)
                        ) -> Dict[int, int]:
    """Arrival order in which travelling parties stay together.

    DECISION (not fixed by the spec): parties are drawn from ``party_sizes``
    and formed from passengers seated in the same row block, which is how
    real bookings allocate adjacent seats.  Every member of a party inherits
    the leader's arrival position, so the party reaches the door as a unit.
    """
    by_block: Dict[Tuple[int, int, int], List[int]] = {}
    for p in sim.passengers:
        by_block.setdefault((p.seat.deck, p.seat.row, p.seat.block),
                            []).append(p.pid)

    parties: List[List[int]] = []
    for key in sorted(by_block):
        members = list(by_block[key])
        rng.shuffle(members)
        while members:
            size = min(rng.choice(list(party_sizes)), len(members))
            parties.append([members.pop() for _ in range(size)])

    rng.shuffle(parties)
    ranks: Dict[int, int] = {}
    pos = 0
    for party in parties:
        for pid in party:
            # Same base rank for the whole party; the tiny offset keeps the
            # ordering total without letting other passengers interleave.
            ranks[pid] = pos
        pos += 1
    return ranks


class PlanSelector:
    """Turns a Plan into the simulator's `selector` callback."""

    def __init__(self, sim: Simulator, plan: Plan, arrival: str,
                 arrival_rank: Optional[Dict[int, int]] = None):
        self.plan = plan
        self.arrival = arrival
        self.queued = arrival == "queued"
        if arrival_rank is not None:
            self.arrival_rank = arrival_rank
        else:
            self.arrival_rank = {pid: i for i, pid in enumerate(sim.pending)}

    def key(self, pid: int) -> Tuple[int, int]:
        g, ideal = self.plan[pid]
        return (g, ideal if self.queued else self.arrival_rank[pid])

    def __call__(self, deck: int, row: int, cands: List[int]) -> Optional[int]:
        return min(cands, key=self.key)


def make_selector(sim: Simulator, name: str,
                  rng: Optional[random.Random] = None) -> PlanSelector:
    rng = rng or random.Random(sim.cfg.seed + 101)
    plan = STRATEGIES[name](sim, rng)
    ranks = None
    if sim.cfg.arrival == "groups":
        ranks = group_arrival_ranks(sim, random.Random(sim.cfg.seed + 202))
    return PlanSelector(sim, plan, sim.cfg.arrival, ranks)
