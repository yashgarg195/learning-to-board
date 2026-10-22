"""Simulator validation against hand-computed cases.

The spec explicitly asks for this: "validate the simulator against
hand-computed cases before any RL".  Every expected number below is derived by
hand in the comments, not copied from a run.
"""
from __future__ import annotations

import os
import random
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from boarding.config import AIRCRAFT, AIRCRAFT_ORDER, SEATED, SimConfig
from boarding.simulator import Passenger, Simulator
from boarding.strategies import STRATEGIES, make_selector

BASE = SimConfig(aircraft="tiny", stow_ticks=6, no_bag_ticks=2,
                 shuffle_ticks=4, walk_ticks_per_row=1, seed=1)


def scenario(picks, cfg=BASE):
    """Build a simulator containing exactly the given (row, col, has_bag) seats."""
    sim = Simulator(cfg, rng=random.Random(0), log=False)
    seatmap = {(s.row, s.col): s for s in sim.spec.seat_map()}
    pax = []
    for i, (row, col, bag) in enumerate(picks):
        s = seatmap[(row, col)]
        pax.append(Passenger(pid=i, seat=s, has_bag=bag, band=0,
                             group=(0, s.seat_type)))
    sim.passengers = pax
    sim.n = len(pax)
    sim.occ = [[[-1] * d.rows for _ in range(d.n_aisles)] for d in sim.spec.decks]
    sim.seated_grid = {}
    sim.n_seated = 0
    sim.blocking_events = 0
    sim.total_blocked_ticks = 0
    sim.t = 0
    sim.finished_tick = None
    sim._assign_doors()
    sim.max_dist = max(1, max(abs(p.seat.row - p.door_row) for p in pax))
    sim._assign_bands()
    sim.pending = [p.pid for p in pax]
    return sim


def run(sim, cfg=BASE):
    while not sim.done and sim.t < cfg.max_ticks:
        sim.tick(lambda d, r, c: c[0])
    return sim


# ---------------------------------------------------------------------------
def test_tiny_cabin_geometry():
    deck = AIRCRAFT["tiny"].decks[0]
    seats = deck.seat_map()
    assert len(seats) == 4 * 4
    assert deck.aisle_cols() == [2]
    row0 = sorted([s for s in seats if s.row == 0], key=lambda s: s.col)
    assert [s.col for s in row0] == [0, 1, 3, 4]
    assert [s.seat_type for s in row0] == ["window", "aisle", "aisle", "window"]
    assert [s.depth for s in row0] == [1, 0, 0, 1]


def test_single_passenger_no_bag():
    # 1 tick to step aboard + 2 ticks of walking + 2 ticks to sit down.
    sim = run(scenario([(2, 1, False)]))
    assert sim.stats()["ticks"] == 1 + 2 + 2
    assert sim.blocking_events == 0


def test_single_passenger_with_bag():
    # 1 tick aboard + 3 ticks walking + 6 ticks stowing.
    sim = run(scenario([(3, 1, True)]))
    assert sim.stats()["ticks"] == 1 + 3 + 6


def test_walk_speed_scales():
    cfg = BASE.copy(walk_ticks_per_row=2)
    sim = run(scenario([(3, 1, False)], cfg), cfg)
    assert sim.stats()["ticks"] == 1 + 3 * 2 + 2


def test_seat_interference_and_blocking():
    # P0 -> row 2 aisle seat (no bag), P1 -> row 2 window seat behind it.
    # t=0 P0 enters; t=1 P0 at row 1 and P1 enters; t=2 P0 reaches row 2 and
    # starts a 2-tick sit-down, P1 reaches row 1; t=3 P1 is blocked (1 event);
    # t=4 P0 is seated and P1 steps into row 2, climbing over P0:
    # 2 (sit) + 4 (shuffle) = 6 more ticks -> finished at tick 10 + 1 = 11.
    sim = run(scenario([(2, 1, False), (2, 0, False)]))
    assert sim.blocking_events == 1
    assert sim.stats()["ticks"] == 11


def test_no_interference_when_window_boards_first():
    # Same two seats, window first: nobody climbs over anybody.
    sim = run(scenario([(2, 0, False), (2, 1, False)]))
    assert sim.stats()["ticks"] < 11


def test_door_is_blocked_by_a_stowing_passenger():
    # P0 sits in row 0 (at the door) with a bag: 6 ticks during which nobody
    # else can enter.  P1 then needs 1 tick of walking + 2 to sit = 10 total.
    sim = run(scenario([(0, 1, True), (1, 1, False)]))
    assert sim.stats()["ticks"] == 10
    assert sim.blocking_events == 0


def test_rear_gate_mirrors_front_gate():
    cfg = BASE.copy(gate="back")
    sim = run(scenario([(1, 1, False)], cfg), cfg)
    # Rear door is at row 3, the seat is in row 1: 1 aboard + 2 rows + 2 to sit.
    assert sim.stats()["ticks"] == 1 + 2 + 2


# ---------------------------------------------------------------------------
@pytest.mark.parametrize("key", AIRCRAFT_ORDER)
def test_every_preset_boards_completely(key):
    cfg = SimConfig(aircraft=key, n_passengers=min(120, AIRCRAFT[key].capacity),
                    seed=3)
    sim = Simulator(cfg, rng=random.Random(3), log=True)
    sel = make_selector(sim, "wilma")
    while not sim.done and sim.t < cfg.max_ticks:
        sim.tick(sel)
    assert sim.done, "%s never finished boarding" % key
    assert sim.n_seated == sim.n
    assert all(p.state == SEATED for p in sim.passengers)
    seats = {(p.seat.deck, p.seat.row, p.seat.col) for p in sim.passengers}
    assert len(seats) == sim.n, "duplicate seat assignment"
    assert len(sim.log.frames) == sim.t + 1


def test_aisle_capacity_is_respected():
    cfg = SimConfig(aircraft="a320", n_passengers=150, seed=7)
    sim = Simulator(cfg, rng=random.Random(7), log=False)
    sel = make_selector(sim, "random")
    while not sim.done and sim.t < cfg.max_ticks:
        sim.tick(sel)
        occupied = [p.pos_row for p in sim.passengers
                    if p.state in (1, 2, 3) and p.pos_row >= 0]
        assert len(occupied) == len(set(occupied)), "two passengers in one cell"


def test_determinism():
    a = [Simulator(SimConfig(aircraft="a320", n_passengers=100, seed=11),
                   rng=random.Random(11), log=False) for _ in range(2)]
    out = []
    for sim in a:
        sel = make_selector(sim, "steffen")
        while not sim.done and sim.t < 20000:
            sim.tick(sel)
        out.append(sim.stats())
    assert out[0] == out[1]


def test_known_strategy_ordering_under_perfect_queueing():
    """Steffen < WILMA < random < back-to-front -- the published result."""
    means = {}
    for name in STRATEGIES:
        vals = []
        for sd in range(6):
            cfg = SimConfig(aircraft="a320", n_passengers=150, seed=sd,
                            arrival="queued")
            sim = Simulator(cfg, rng=random.Random(sd), log=False)
            sel = make_selector(sim, name)
            while not sim.done and sim.t < cfg.max_ticks:
                sim.tick(sel)
            vals.append(sim.stats()["ticks"])
        means[name] = sum(vals) / len(vals)
    assert means["steffen"] < means["wilma"] < means["random"]
    assert means["random"] < means["back_to_front"]


def test_two_doors_are_faster_than_one():
    def mean(gate):
        vals = []
        for sd in range(4):
            cfg = SimConfig(aircraft="a320", n_passengers=150, seed=sd, gate=gate)
            sim = Simulator(cfg, rng=random.Random(sd), log=False)
            sel = make_selector(sim, "wilma")
            while not sim.done and sim.t < cfg.max_ticks:
                sim.tick(sel)
            vals.append(sim.stats()["ticks"])
        return sum(vals) / len(vals)
    assert mean("both") < mean("front")
