"""All offline experiments from the spec's deliverables (Section 11).

    python scripts/experiments.py [--quick]

Produces artifacts/experiments.json and artifacts/figures/*.png:
  1. mean boarding time +- std over many seeds, per strategy x gate x arrival
  2. learning curves (SARSA(0), SARSA(lambda), Q(lambda), MC, linear FA)
  3. discount-factor analysis and policy-iteration vs value-iteration cost
  4. bandit over fixed strategies: epsilon-greedy vs UCB, reward and regret
  5. transfer of a single policy to aircraft it was never trained on
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from boarding.agents.bandit import run_bandit
from boarding.agents.common import moving_average
from boarding.agents.dp import (TinyMDP, describe_sequence, policy_iteration,
                                rollout, value_iteration)
from boarding.agents.tabular import evaluate_tabular, mc_prediction
from boarding.config import AIRCRAFT, SimConfig
from boarding.envs import BoardingEnv
from boarding.policy import load_policy, run_policy
from boarding.replay import run_baseline
from boarding.strategies import STRATEGIES, STRATEGY_LABELS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ART = os.path.join(ROOT, "artifacts")
FIG = os.path.join(ART, "figures")

plt.rcParams.update({"figure.facecolor": "#0e1117", "axes.facecolor": "#161b26",
                     "savefig.facecolor": "#0e1117", "text.color": "#e6e9ef",
                     "axes.labelcolor": "#e6e9ef", "xtick.color": "#9aa4b8",
                     "ytick.color": "#9aa4b8", "axes.edgecolor": "#39415a",
                     "grid.color": "#232a38", "font.size": 9})


# ------------------------------------------------------------- 1. benchmark
def benchmark(q, seeds, aircraft="a320", n_pax=150):
    rows = []
    for gate in ("front", "back", "both"):
        for arrival in ("random", "queued"):
            for name in list(STRATEGIES.keys()) + ["rl"]:
                vals = []
                for sd in seeds:
                    cfg = SimConfig(aircraft=aircraft, n_passengers=n_pax,
                                    gate=gate, arrival=arrival, seed=sd)
                    if name == "rl":
                        if q is None:
                            continue
                        sim = run_policy(cfg, q, log=False)
                    else:
                        sim = run_baseline(cfg, name, log=False)
                    vals.append(sim.stats()["ticks"])
                if not vals:
                    continue
                rows.append({
                    "gate": gate, "arrival": arrival,
                    "strategy": STRATEGY_LABELS[name],
                    "mean": round(statistics.mean(vals), 1),
                    "std": round(statistics.pstdev(vals), 1),
                    "min": int(min(vals)), "max": int(max(vals)),
                })
    return rows


def plot_benchmark(rows):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
    for ax, arrival in zip(axes, ("random", "queued")):
        sub = [r for r in rows if r["arrival"] == arrival and r["gate"] == "front"]
        names = [r["strategy"] for r in sub]
        means = [r["mean"] for r in sub]
        stds = [r["std"] for r in sub]
        colors = ["#3ddc84" if m == min(means) else "#2f6feb" for m in means]
        ax.bar(names, means, yerr=stds, color=colors, capsize=3)
        ax.set_title("%s arrival (front door)" % arrival)
        ax.tick_params(axis="x", rotation=22)
        ax.grid(axis="y", alpha=.4)
    axes[0].set_ylabel("boarding time (ticks)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "benchmark.png"), dpi=140)
    plt.close(fig)


# --------------------------------------------------------- 2. learning curves
def plot_learning():
    path = os.path.join(ART, "training.json")
    if not os.path.exists(path):
        return
    with open(path) as f:
        tr = json.load(f)
    fig, ax = plt.subplots(figsize=(9, 3.6))
    styles = {"sarsa0": ("SARSA(0)  (one-step: does not learn)", "#ff5c5c"),
              "sarsa": ("SARSA(lambda)", "#3ddc84"),
              "qlearning": ("Q(lambda)", "#2f6feb"),
              "mc": ("Monte-Carlo control", "#ff9f45"),
              "pg": ("Policy gradient (generalist)", "#ffd166"),
              "pg_mid": ("Policy gradient (mid specialist)", "#ff6ec7"),
              "linear": ("Linear function approx.", "#b084ff")}
    for k, (lab, c) in styles.items():
        if k not in tr:
            continue
        # Policy gradient explores by *sampling*, so its raw training episodes
        # are far slower than the policy it is actually learning.  Where a
        # periodic greedy evaluation was recorded, plot that instead -- it is
        # the honest comparison against the value-based agents, which are
        # already near-greedy once epsilon has decayed.
        gc = tr[k].get("greedy_curve")
        if gc:
            xs = np.asarray([p[0] for p in gc], dtype=float)
            ys = [p[1] for p in gc]
            ax.plot(xs / xs[-1], ys, label=lab + " (greedy)", color=c,
                    lw=1.8, ls="--")
            continue
        y = moving_average(tr[k]["ticks"], 200)
        ax.plot(np.linspace(0, 1, len(y)), y, label=lab, color=c, lw=1.6)
    ax.set_xlabel("training progress (fraction of episodes)")
    ax.set_ylabel("boarding time (ticks)")
    ax.legend(frameon=False, fontsize=8, loc="upper right", ncol=2)
    ax.grid(alpha=.4)
    ax.set_title("Dashed = greedy evaluation; solid = training episodes "
                 "(still exploring, so above their final greedy score)",
                 fontsize=8, loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "learning_curves.png"), dpi=140)
    plt.close(fig)


# ------------------------------------------------- 3. discount analysis + DP
def discount_and_dp(quick=False):
    mdp = TinyMDP()
    gammas = [0.5, 0.7, 0.85, 0.95, 0.99, 1.0] if not quick else [0.7, 0.95, 1.0]
    rows, xs, ys = [], [], []
    for g in gammas:
        vi = value_iteration(mdp, gamma=g)
        cost, seq = rollout(mdp, vi["policy"])
        xs.append(g)
        ys.append(cost)
        rows.append({"gamma": g, "method": "value iteration",
                     "boarding_time": cost, "sweeps": vi["sweeps"],
                     "backups": vi["backups"], "seconds": round(vi["seconds"], 2),
                     "first_6_actions": ", ".join(describe_sequence(mdp, seq[:6]))})
    for g in ([1.0, 0.95] if not quick else [1.0]):
        pi = policy_iteration(mdp, gamma=g)
        cost, seq = rollout(mdp, pi["policy"])
        rows.append({"gamma": g, "method": "policy iteration",
                     "boarding_time": cost, "sweeps": pi["sweeps"],
                     "backups": pi["backups"], "seconds": round(pi["seconds"], 2),
                     "first_6_actions": ", ".join(describe_sequence(mdp, seq[:6]))})
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    ax.plot(xs, ys, "o-", color="#2f6feb")
    ax.set_xlabel("discount factor gamma")
    ax.set_ylabel("boarding time of the optimal policy (ticks)")
    ax.set_title("Myopic vs far-sighted control (tiny 4x4 cabin)")
    ax.grid(alpha=.4)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "discount.png"), dpi=140)
    plt.close(fig)
    return rows


# ----------------------------------------------------------------- 4. bandit
def bandits(quick=False):
    cfg = SimConfig(aircraft="a320", n_passengers=150)
    arms = list(STRATEGIES.keys())
    steps = 60 if quick else 300
    res = {}
    for algo in ("eps", "ucb"):
        res[algo] = run_bandit(cfg, arms, n_steps=steps, algo=algo, seed=3)
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.2))
    for algo, c in (("eps", "#2f6feb"), ("ucb", "#3ddc84")):
        axes[0].plot(moving_average(res[algo]["rewards"], 25), color=c,
                     label="epsilon-greedy" if algo == "eps" else "UCB1")
        axes[1].plot(res[algo]["regret"], color=c,
                     label="epsilon-greedy" if algo == "eps" else "UCB1")
    axes[0].set_title("average reward (-time / scale)")
    axes[1].set_title("cumulative regret")
    for ax in axes:
        ax.set_xlabel("pull")
        ax.grid(alpha=.4)
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "bandit.png"), dpi=140)
    plt.close(fig)
    return {k: {"counts": v["counts"], "best_arm": v["best_arm"],
                "final_regret": round(v["regret"][-1], 3),
                "arms": v["arms"]} for k, v in res.items()}


# --------------------------------------------------------------- 5. transfer
def transfer(q, seeds):
    rows = []
    if q is None:
        return rows
    for ac in ("atr72", "a320", "b787", "b777", "a380"):
        cap = AIRCRAFT[ac].capacity
        n = int(cap * 0.9)
        rl, base = [], {}
        for sd in seeds:
            cfg = SimConfig(aircraft=ac, n_passengers=n, seed=sd)
            rl.append(run_policy(cfg, q, log=False).stats()["ticks"])
            for s in STRATEGIES:
                base.setdefault(s, []).append(
                    run_baseline(cfg, s, log=False).stats()["ticks"])
        best = min(base, key=lambda s: statistics.mean(base[s]))
        rows.append({
            "aircraft": AIRCRAFT[ac].name, "passengers": n,
            "RL agent": round(statistics.mean(rl), 1),
            "best baseline": STRATEGY_LABELS[best],
            "best baseline time": round(statistics.mean(base[best]), 1),
            "random": round(statistics.mean(base["random"]), 1),
            "gap vs best baseline %": round(
                100 * (statistics.mean(rl) - statistics.mean(base[best]))
                / statistics.mean(base[best]), 1),
        })
    return rows


# ------------------------------------------------------------------- Lab 6
def mc_first_vs_every(quick=False):
    env = BoardingEnv(SimConfig(aircraft="tiny"))
    eps = 120 if quick else 600
    out = []
    for fv in (True, False):
        r = mc_prediction(env, episodes=eps, first_visit=fv, seed=5)
        vs = list(r["V"].values())
        out.append({"mode": r["mode"], "states_visited": r["n_states"],
                    "mean_V": round(float(np.mean(vs)), 2),
                    "min_V": round(float(np.min(vs)), 2),
                    "max_V": round(float(np.max(vs)), 2),
                    "total_updates": int(sum(r["visits"].values()))})
    return out


def generalisation_tax(seeds):
    """What does training on a mixture of cabins cost on a single cabin?

    The spec asks for transfer to be reported honestly, which includes the
    price paid for it: the generalist is compared against a specialist trained
    on ``mid`` alone, plus every baseline, on the same seeds.
    """
    gen, _ = load_policy("rl_agent")
    spec, _ = load_policy("rl_agent_mid")
    rows = []
    for name, q in (("RL generalist (mixture-trained)", gen),
                    ("RL specialist (mid only)", spec)):
        if q is None:
            continue
        ts = [run_policy(SimConfig(aircraft="mid", seed=sd), q,
                         log=False).stats()["ticks"] for sd in seeds]
        rows.append({"policy": name, "mean": round(statistics.mean(ts), 1),
                     "stdev": round(statistics.pstdev(ts), 1)})
    for s in STRATEGIES:
        ts = [run_baseline(SimConfig(aircraft="mid", seed=sd), s,
                           log=False).stats()["ticks"] for sd in seeds]
        rows.append({"policy": STRATEGY_LABELS[s],
                     "mean": round(statistics.mean(ts), 1),
                     "stdev": round(statistics.pstdev(ts), 1)})
    return sorted(rows, key=lambda r: r["mean"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    os.makedirs(FIG, exist_ok=True)
    seeds = list(range(8)) if args.quick else list(range(40))
    q, meta = load_policy("rl_agent")

    print("1/5 benchmark ...", flush=True)
    bench = benchmark(q, seeds)
    plot_benchmark(bench)
    print("2/5 learning curves ...", flush=True)
    plot_learning()
    print("3/5 discount + DP ...", flush=True)
    dp = discount_and_dp(args.quick)
    print("4/5 bandits ...", flush=True)
    bandit = bandits(args.quick)
    print("5/5 transfer + MC prediction ...", flush=True)
    tr = transfer(q, seeds[:12])
    gtax = generalisation_tax(seeds)
    mc = mc_first_vs_every(args.quick)

    blob = {"benchmark": bench, "dp": dp, "bandit": bandit, "transfer": tr,
            "generalisation_tax": gtax,
            "mc_prediction": mc, "policy_meta": meta, "seeds": len(seeds)}
    with open(os.path.join(ART, "experiments.json"), "w") as f:
        json.dump(blob, f, indent=1)
    print("saved -> artifacts/experiments.json", flush=True)


if __name__ == "__main__":
    main()
