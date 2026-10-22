"""Offline training of the shipped policies.

    python scripts/train.py                 # everything
    python scripts/train.py --quick         # short run, for smoke tests

Produces:
    artifacts/policies/rl_agent.json        tabular SARSA(lambda) policy
    artifacts/policies/rl_linear.npz        linear function approximation
    artifacts/training.json                 learning curves + evaluations
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from boarding.agents.common import moving_average
from boarding.agents.linear import LinearQ, greedy_rollout, train_semi_gradient
from boarding.agents.pg import evaluate_pg, train_best_of
from boarding.agents.tabular import evaluate_tabular, mc_control, td_control
from boarding.config import SimConfig
from boarding.envs import BoardingEnv
from boarding.policy import POLICY_DIR, save_policy

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ART = os.path.join(ROOT, "artifacts")
EVAL_SEEDS = list(range(500, 530))

# Training is done on a *mixture* of cabins so the schedule the agent learns is
# not tied to one aircraft (the state is size independent, see boarding/envs.py).
TRAIN_CONFIGS = [
    SimConfig(aircraft="mid"),
    SimConfig(aircraft="mid", gate="back"),
    SimConfig(aircraft="a320", n_passengers=90),
    SimConfig(aircraft="atr72"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--episodes", type=int, default=0)
    args = ap.parse_args()
    episodes = args.episodes or (1500 if args.quick else 30000)
    # Policy gradient saturates if trained much past ~12k episodes with no
    # entropy bonus, so it gets its own (shorter) budget and uses restarts
    # instead of a longer single run.
    pg_episodes = 1200 if args.quick else 12000
    restarts = 2 if args.quick else 6

    os.makedirs(ART, exist_ok=True)
    os.makedirs(POLICY_DIR, exist_ok=True)
    out = {}

    env = BoardingEnv(configs=TRAIN_CONFIGS)
    eval_env = BoardingEnv(SimConfig(aircraft="mid"))

    print("SARSA(lambda) on a mixture of cabins ...", flush=True)
    t0 = time.time()
    sarsa = td_control(env, episodes=episodes, alpha=0.02, lam=1.0, gamma=1.0,
                       algo="sarsa", seed=1)
    ev = evaluate_tabular(eval_env, sarsa["q"], EVAL_SEEDS)
    print("  states=%d  eval=%.1f  (%.0fs)"
          % (sarsa["n_states"], statistics.mean(ev), time.time() - t0), flush=True)
    save_policy("rl_sarsa", sarsa["q"],
                {"algo": "sarsa(lambda=1.0)", "episodes": episodes,
                 "trained_on": [c.aircraft for c in TRAIN_CONFIGS],
                 "eval_mid_mean": statistics.mean(ev)})
    out["sarsa"] = {"ticks": sarsa["ticks"], "eval": ev,
                    "n_states": sarsa["n_states"]}

    print("Q(lambda) ...", flush=True)
    qout = td_control(env, episodes=episodes, alpha=0.02, lam=0.9, gamma=1.0,
                      algo="qlearning", seed=1)
    out["qlearning"] = {"ticks": qout["ticks"],
                        "eval": evaluate_tabular(eval_env, qout["q"], EVAL_SEEDS),
                        "n_states": qout["n_states"]}
    save_policy("rl_qlearning", qout["q"], {"algo": "q(lambda=0.9)"})
    print("  eval=%.1f" % statistics.mean(out["qlearning"]["eval"]), flush=True)

    print("one-step SARSA (ablation: why traces are needed) ...", flush=True)
    s0 = td_control(env, episodes=episodes // 2, alpha=0.05, lam=0.0, gamma=1.0,
                    algo="sarsa", seed=1)
    out["sarsa0"] = {"ticks": s0["ticks"],
                     "eval": evaluate_tabular(eval_env, s0["q"], EVAL_SEEDS)}
    print("  eval=%.1f" % statistics.mean(out["sarsa0"]["eval"]), flush=True)

    print("Monte-Carlo control ...", flush=True)
    mc = mc_control(env, episodes=episodes // 2, seed=1)
    out["mc"] = {"ticks": mc["ticks"],
                 "eval": evaluate_tabular(eval_env, mc["q"], EVAL_SEEDS)}
    save_policy("rl_mc", mc["q"], {"algo": "mc_control"})
    print("  eval=%.1f" % statistics.mean(out["mc"]["eval"]), flush=True)

    print("tabular policy gradient (shipped agent) ...", flush=True)
    t0 = time.time()
    pg = train_best_of(env, eval_env, episodes=pg_episodes, restarts=restarts,
                       alpha=0.25, batch=16, n_seeds=48, entropy=0.0, seed=1)
    pg_eval = evaluate_pg(eval_env, pg["policy"], EVAL_SEEDS)
    print("  states=%d  eval=%.1f  val=%.1f  restarts=%s  (%.0fs)"
          % (pg["states"], statistics.mean(pg_eval), pg["validation"],
             ["%.0f" % v for v in pg["restart_scores"]], time.time() - t0),
          flush=True)
    save_policy("rl_agent", pg["policy"].to_qtable(),
                {"algo": "tabular policy gradient (paired baseline)",
                 "episodes": episodes,
                 "trained_on": [c.aircraft for c in TRAIN_CONFIGS],
                 "eval_mid_mean": statistics.mean(pg_eval),
                 "restarts": restarts,
                 "restart_validation_scores": pg["restart_scores"]})
    out["pg"] = {"ticks": pg["ticks"], "eval": pg_eval, "n_states": pg["states"],
                 "restart_scores": pg["restart_scores"]}

    # The agent above is trained on a *mixture* of cabins, which costs it some
    # performance on any single one.  Train a mid-cabin specialist too so the
    # generalisation tax is measured rather than hidden.
    print("tabular policy gradient (mid-cabin specialist) ...", flush=True)
    t0 = time.time()
    pg1 = train_best_of(BoardingEnv(SimConfig(aircraft="mid")), eval_env,
                        episodes=pg_episodes, restarts=restarts, alpha=0.25,
                        batch=16, n_seeds=48, entropy=0.0, seed=1)
    pg1_eval = evaluate_pg(eval_env, pg1["policy"], EVAL_SEEDS)
    print("  states=%d  eval=%.1f  val=%.1f  restarts=%s  (%.0fs)"
          % (pg1["states"], statistics.mean(pg1_eval), pg1["validation"],
             ["%.0f" % v for v in pg1["restart_scores"]], time.time() - t0),
          flush=True)
    save_policy("rl_agent_mid", pg1["policy"].to_qtable(),
                {"algo": "tabular policy gradient (mid specialist)",
                 "episodes": episodes, "trained_on": ["mid"],
                 "eval_mid_mean": statistics.mean(pg1_eval),
                 "restart_validation_scores": pg1["restart_scores"]})
    out["pg_mid"] = {"ticks": pg1["ticks"], "eval": pg1_eval,
                     "n_states": pg1["states"],
                     "restart_scores": pg1["restart_scores"]}

    print("linear function approximation ...", flush=True)
    lin = train_semi_gradient(env, episodes=max(800, episodes // 4),
                              alpha=0.02, gamma=0.995, algo="sarsa", seed=1)
    lin_eval = [greedy_rollout(eval_env, lin["model"], sd)["ticks"]
                for sd in EVAL_SEEDS]
    lin["model"].save(os.path.join(POLICY_DIR, "rl_linear.npz"),
                      {"algo": "semi-gradient sarsa"})
    out["linear"] = {"ticks": lin["ticks"], "eval": lin_eval}
    print("  eval=%.1f" % statistics.mean(lin_eval), flush=True)

    with open(os.path.join(ART, "training.json"), "w") as f:
        json.dump(out, f)
    print("saved -> artifacts/training.json", flush=True)


if __name__ == "__main__":
    main()
