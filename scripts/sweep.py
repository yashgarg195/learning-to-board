"""Hyper-parameter sweep used to pick the shipped training settings.

    PYTHONPATH=. python scripts/sweep.py          # value-based (SARSA/MC)
    PYTHONPATH=. python scripts/sweep.py pg       # policy gradient (shipped agent)
"""
from __future__ import annotations

import itertools
import statistics
import sys

from boarding.config import SimConfig
from boarding.envs import BoardingEnv
from boarding.agents.pg import evaluate_pg, train_tabular_pg
from boarding.agents.tabular import td_control, mc_control, evaluate_tabular

EVAL = list(range(500, 530))
TRAIN = [SimConfig(aircraft="mid"), SimConfig(aircraft="mid", gate="back"),
         SimConfig(aircraft="a320", n_passengers=90), SimConfig(aircraft="atr72")]


def report(name, q, env):
    ev = evaluate_tabular(env, q, EVAL)
    print("%-42s eval %.1f +- %.1f" % (name, statistics.mean(ev),
                                       statistics.pstdev(ev)), flush=True)
    return statistics.mean(ev)


def sweep_pg():
    """The sweep that produced the shipped settings: alpha=0.25, batch=16,
    no entropy bonus.  Entropy regularisation consistently hurt -- the paired
    baseline already supplies enough signal, and the bonus blurs a schedule
    that wants to be sharp."""
    env = BoardingEnv(configs=TRAIN)
    eval_env = BoardingEnv(SimConfig(aircraft="mid"))
    best = None
    for alpha, batch, ent in itertools.product([0.25, 0.6, 1.2], [16, 32],
                                               [0.0, 0.01]):
        out = train_tabular_pg(env, episodes=12000, alpha=alpha, batch=batch,
                               n_seeds=48, entropy=ent, seed=1)
        ev = evaluate_pg(eval_env, out["policy"], EVAL)
        v = statistics.mean(ev)
        print("pg alpha=%.2f batch=%d ent=%.3f          eval %.1f +- %.1f"
              % (alpha, batch, ent, v, statistics.pstdev(ev)), flush=True)
        if best is None or v < best[0]:
            best = (v, alpha, batch, ent)
    print("BEST", best, flush=True)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "pg":
        sweep_pg()
        return
    env = BoardingEnv(configs=TRAIN)
    eval_env = BoardingEnv(SimConfig(aircraft="mid"))
    best = None
    for lam, alpha in itertools.product([0.95, 1.0], [0.3, 0.6]):
        out = td_control(env, episodes=20000, alpha=alpha, lam=lam,
                         algo="sarsa", gamma=1.0, seed=1)
        v = report("sarsa lam=%.2f alpha=%.2f" % (lam, alpha), out["q"], eval_env)
        if best is None or v < best[0]:
            best = (v, lam, alpha)
    out = mc_control(env, episodes=20000, seed=1)
    report("mc_control", out["q"], eval_env)
    print("BEST", best, flush=True)


if __name__ == "__main__":
    main()
