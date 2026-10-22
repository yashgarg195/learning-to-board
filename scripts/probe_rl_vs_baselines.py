"""Probe: where does the shipped agent beat the heuristics, and where not?

Runs every strategy over the same held-out seeds across a spread of cabins and
arrival patterns, so claims about the agent are checked rather than assumed.

    PYTHONPATH=. python scripts/probe_rl_vs_baselines.py
    PYTHONPATH=. python scripts/probe_rl_vs_baselines.py rl_parity_mid rl_agent_mid
"""
from __future__ import annotations

import statistics
import sys

from boarding.config import SimConfig
from boarding.policy import load_policy, run_policy
from boarding.simulator import Simulator
from boarding.strategies import make_selector

SEEDS = list(range(500, 520))
DEFAULT_POLICIES = ["rl_agent_mid", "rl_agent"]


def run_baseline(cfg, strat):
    sim = Simulator(cfg, log=False)
    sel = make_selector(sim, strat)
    while not sim.done and sim.t < cfg.max_ticks:
        sim.tick(sel)
    return sim.stats()["ticks"]


def main():
    names = sys.argv[1:] or DEFAULT_POLICIES
    policies = []
    for n in names:
        q, _meta = load_policy(n)
        if q is None:
            print("skipping %s (not found)" % n)
            continue
        policies.append((n, q))

    cases = [
        ("mid", None, "random"),
        ("mid", None, "queued"),
        ("mid", None, "groups"),
        ("atr72", None, "random"),
        ("a320", 120, "random"),
        ("a320", 180, "random"),
        ("a320", 180, "queued"),
        ("b787", None, "random"),
    ]
    baselines = ("steffen", "wilma", "outside_in")
    hdr = (f"{'cabin':>6} {'pax':>5} {'arrival':>8} | "
           + " ".join(f"{b[:8]:>9}" for b in baselines) + " | "
           + " ".join(f"{n[-9:]:>11}" for n, _ in policies))
    print(hdr)
    print("-" * len(hdr))
    for craft, pax, arrival in cases:
        res = {}
        for strat in baselines:
            res[strat] = statistics.mean(
                run_baseline(SimConfig(aircraft=craft, n_passengers=pax,
                                       arrival=arrival, seed=sd), strat)
                for sd in SEEDS)
        for name, q in policies:
            res[name] = statistics.mean(
                run_policy(SimConfig(aircraft=craft, n_passengers=pax,
                                     arrival=arrival, seed=sd), q,
                           log=False).stats()["ticks"]
                for sd in SEEDS)
        n = pax if pax else "full"
        print(f"{craft:>6} {str(n):>5} {arrival:>8} | "
              + " ".join(f"{res[b]:9.1f}" for b in baselines) + " | "
              + " ".join(f"{res[n2]:11.1f}" for n2, _ in policies))
        sys.stdout.flush()


if __name__ == "__main__":
    main()
