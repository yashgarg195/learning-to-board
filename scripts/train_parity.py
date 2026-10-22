"""Train policy-gradient agents with the row-parity action space.

The original 12-action space (band x seat type) cannot express "board every
other row", which is the mechanism behind Steffen's advantage. This script
trains agents on the 24-action space (band x seat type x row parity) and
reports them against the heuristics on held-out seeds, so the effect of the
extra action dimension is measured rather than assumed.

    PYTHONPATH=. python scripts/train_parity.py --cabin mid
    PYTHONPATH=. python scripts/train_parity.py --cabin a320 --episodes 6000
"""
from __future__ import annotations

import argparse
import statistics
import time

from boarding.agents.pg import evaluate_pg, train_best_of
from boarding.config import SimConfig
from boarding.envs import BoardingEnv, n_actions
from boarding.policy import run_policy, save_policy
from boarding.simulator import Simulator
from boarding.strategies import make_selector

EVAL_SEEDS = list(range(500, 530))
BASELINES = ("steffen", "wilma", "outside_in", "random")

CABINS = {
    "mid": dict(aircraft="mid", n_passengers=None),
    "a320": dict(aircraft="a320", n_passengers=180),
}


def baseline_ticks(cfg: SimConfig, strat: str, seeds):
    out = []
    for sd in seeds:
        c = cfg.copy(seed=sd)
        sim = Simulator(c, log=False)
        sel = make_selector(sim, strat)
        while not sim.done and sim.t < c.max_ticks:
            sim.tick(sel)
        out.append(sim.stats()["ticks"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cabin", choices=sorted(CABINS), default="mid")
    ap.add_argument("--episodes", type=int, default=12000)
    ap.add_argument("--restarts", type=int, default=6)
    ap.add_argument("--alpha", type=float, default=0.25)
    ap.add_argument("--no-parity", action="store_true",
                    help="train the 12-action control for a like-for-like A/B")
    ap.add_argument("--save-as", default=None)
    args = ap.parse_args()

    use_parity = not args.no_parity
    cfg = SimConfig(use_parity=use_parity, **CABINS[args.cabin])
    name = args.save_as or "rl_parity_{}{}".format(
        args.cabin, "" if use_parity else "_control")

    print("cabin=%s pax=%d parity=%s actions=%d episodes=%d restarts=%d"
          % (cfg.aircraft, cfg.resolved_passengers(), use_parity,
             n_actions(cfg), args.episodes, args.restarts), flush=True)

    env = BoardingEnv(cfg)
    eval_env = BoardingEnv(cfg)
    t0 = time.time()
    pg = train_best_of(env, eval_env, episodes=args.episodes,
                       restarts=args.restarts, alpha=args.alpha, batch=16,
                       n_seeds=48, entropy=0.0, seed=1)
    secs = time.time() - t0

    agent = evaluate_pg(eval_env, pg["policy"], EVAL_SEEDS)
    mean = statistics.mean(agent)
    print("  states=%d  eval=%.1f  validation=%.1f  restarts=%s  (%.0fs)"
          % (pg["states"], mean, pg["validation"],
             ["%.0f" % v for v in pg["restart_scores"]], secs), flush=True)

    rows = {"rl (this run)": mean}
    for strat in BASELINES:
        rows[strat] = statistics.mean(baseline_ticks(cfg, strat, EVAL_SEEDS))

    save_policy(name, pg["policy"].to_qtable(), {
        "algo": "tabular policy gradient (REINFORCE, paired per-seed baseline)",
        "action_space": "band x seat_type x row_parity" if use_parity
                        else "band x seat_type",
        "use_parity": use_parity,
        "episodes": args.episodes, "restarts": args.restarts,
        "aircraft": cfg.aircraft, "n_passengers": cfg.resolved_passengers(),
        "eval_seeds": [EVAL_SEEDS[0], EVAL_SEEDS[-1]],
        "eval_mid_mean" if args.cabin == "mid" else "eval_mean": mean,
        "baselines": rows,
        "states": pg["states"],
        "restart_validation_scores": pg["restart_scores"],
        "train_seconds": round(secs),
    })

    print("\n  %-16s %8s" % ("strategy", "ticks"))
    for k, v in sorted(rows.items(), key=lambda kv: kv[1]):
        print("  %-16s %8.1f" % (k, v))
    print("\nsaved as %s" % name)

    # Sanity: the saved table must reproduce the evaluation it was saved with.
    chk = statistics.mean(
        run_policy(cfg.copy(seed=sd), pg["policy"].to_qtable(),
                   log=False).stats()["ticks"] for sd in EVAL_SEEDS)
    print("round-trip check: %.1f (expected %.1f)" % (chk, mean))


if __name__ == "__main__":
    main()
