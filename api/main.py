"""FastAPI application for the Smart Airplane Boarding Optimizer.

The Python simulator is the single source of truth: every endpoint here runs
the real engine in ``boarding/`` and returns its recorded trajectory.  The
frontend replays those frames; it never invents simulation events.
"""
from __future__ import annotations

import os
import statistics
from typing import Dict, List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from boarding.agents.pg import (VALIDATION_SEEDS, evaluate_pg,
                                train_tabular_pg)
from boarding.config import (BLOCKED, QUEUED, SEATED, STOWING, WALKING,
                             SimConfig)
from boarding.envs import BoardingEnv
from boarding.policy import (available_policies, load_policy, run_policy,
                             save_policy)
from boarding.replay import build_replay, run_baseline
from boarding.strategies import STRATEGY_LABELS

from .jobs import REGISTRY, Job
from .schemas import (CompareRequest, CompareResponse, CompareRow,
                      ErrorResponse, OptionsResponse, PolicyOut,
                      SimConfigIn, SimulateRequest, SimulateResponse,
                      TrainJob, TrainRequest, aircraft_catalogue,
                      strategy_catalogue)

app = FastAPI(
    title="Smart Airplane Boarding Optimizer API",
    version="1.0.0",
    description=(
        "Runs the boarding simulator, compares strategies and trains the "
        "tabular policy-gradient agent. All timings are in simulator ticks."
    ),
)

ALLOWED_ORIGINS = os.environ.get(
    "BOARDING_CORS_ORIGINS",
    "http://localhost:5173,http://127.0.0.1:5173,"
    "http://localhost:4173,http://127.0.0.1:4173",
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in ALLOWED_ORIGINS if o.strip()],
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


def _error(code: str, message: str, detail=None, status: int = 400):
    return JSONResponse(
        status_code=status,
        content={"error": {"code": code, "message": message,
                           "detail": detail}},
    )


@app.exception_handler(RequestValidationError)
async def _validation_handler(request: Request, exc: RequestValidationError):
    return _error("invalid_request",
                  "One or more configuration fields are invalid.",
                  [{"field": ".".join(str(x) for x in e["loc"][1:]),
                    "problem": e["msg"]} for e in exc.errors()],
                  status=422)


@app.exception_handler(HTTPException)
async def _http_handler(request: Request, exc: HTTPException):
    code = exc.detail.get("code") if isinstance(exc.detail, dict) else "error"
    msg = exc.detail.get("message") if isinstance(exc.detail, dict) \
        else str(exc.detail)
    return _error(code or "error", msg or "Request failed",
                  status=exc.status_code)


def to_sim_config(c: SimConfigIn) -> SimConfig:
    return SimConfig(
        aircraft=c.aircraft,
        n_passengers=c.n_passengers,
        gate=c.gate,
        arrival=c.arrival,
        luggage_fraction=c.luggage_fraction,
        stow_ticks=c.stow_ticks,
        no_bag_ticks=c.no_bag_ticks,
        shuffle_ticks=c.shuffle_ticks,
        walk_ticks_per_row=c.walk_ticks_per_row,
        seed=c.seed,
    )


def _run(cfg: SimConfig, strategy: str, policy: Optional[str], log: bool):
    """Run one simulation with either a baseline or a saved RL policy."""
    if strategy != "rl":
        return run_baseline(cfg, strategy, log=log)
    if not policy:
        raise HTTPException(status_code=400, detail={
            "code": "policy_required",
            "message": "strategy='rl' needs a policy name. Train one first "
                       "or pick a saved policy."})
    q, _meta = load_policy(policy)
    if q is None:
        raise HTTPException(status_code=404, detail={
            "code": "policy_not_found",
            "message": "No saved policy named '{}'.".format(policy)})
    return run_policy(cfg, q, log=log)


@app.get("/api/health")
def health() -> Dict:
    return {"status": "ok", "service": "boarding-api", "version": app.version}


@app.get("/api/options", response_model=OptionsResponse)
def options() -> Dict:
    pols = []
    for name in available_policies():
        _q, meta = load_policy(name)
        pols.append(PolicyOut(name=name, meta=meta or {}))
    return {
        "aircraft": aircraft_catalogue(),
        "gates": ["front", "back", "both"],
        "arrivals": [
            {"value": "random", "label": "Random",
             "description": "Passengers show up in any order within their "
                            "called group."},
            {"value": "queued", "label": "Queued",
             "description": "Perfect compliance: everyone lines up in the "
                            "strategy's ideal order."},
            {"value": "groups", "label": "Travel parties",
             "description": "Families and companions arrive together, so a "
                            "party boards as one unit."},
        ],
        "strategies": strategy_catalogue(),
        "policies": pols,
        "states": {"QUEUED": QUEUED, "WALKING": WALKING, "STOWING": STOWING,
                   "BLOCKED": BLOCKED, "SEATED": SEATED},
    }


@app.post("/api/simulate", response_model=SimulateResponse)
def simulate(req: SimulateRequest) -> Dict:
    cfg = to_sim_config(req.config)
    sim = _run(cfg, req.strategy, req.policy, log=True)
    payload = build_replay(sim)
    payload["congestion"] = sim.congestion
    payload["strategy"] = req.strategy
    payload["policy"] = req.policy
    payload["config"] = req.config
    return payload


@app.post("/api/compare", response_model=CompareResponse)
def compare(req: CompareRequest) -> Dict:
    seeds = [req.config.seed + i for i in range(req.trials)]
    rows: List[CompareRow] = []
    for strat in req.strategies:
        ticks: List[float] = []
        blocked: List[float] = []
        waits: List[float] = []
        for sd in seeds:
            cfg = to_sim_config(req.config.model_copy(update={"seed": sd}))
            sim = _run(cfg, strat, req.policy, log=False)
            st = sim.stats()
            ticks.append(st["ticks"])
            blocked.append(st["blocked_ticks"])
            waits.append(st["avg_wait"])
        rows.append(CompareRow(
            strategy=strat,
            label=STRATEGY_LABELS[strat],
            mean_ticks=float(statistics.mean(ticks)),
            std_ticks=float(statistics.pstdev(ticks)) if len(ticks) > 1 else 0.0,
            min_ticks=float(min(ticks)),
            max_ticks=float(max(ticks)),
            mean_blocked_ticks=float(statistics.mean(blocked)),
            mean_avg_wait=float(statistics.mean(waits)),
            trials=len(seeds),
            per_seed=[float(t) for t in ticks],
        ))
    rows.sort(key=lambda r: r.mean_ticks)
    return {"seeds": seeds, "rows": rows, "config": req.config}


def _train_fn(req: TrainRequest, job: Job):
    cfg = to_sim_config(req.config)
    env = BoardingEnv(cfg, log=False)
    eval_env = BoardingEnv(cfg, log=False)
    eval_seeds = list(VALIDATION_SEEDS)

    best = None
    per_restart = []
    for r in range(req.restarts):
        out = train_tabular_pg(
            env, episodes=req.episodes, alpha=req.alpha, batch=req.batch,
            entropy=0.0, seed=r, eval_every=max(req.batch, req.episodes // 12),
            eval_env=eval_env, eval_seeds=eval_seeds,
            on_progress=lambda d, t, c, _r=r: job.progress(
                _r * req.episodes + d, req.restarts * req.episodes, c),
        )
        score = float(np.mean(evaluate_pg(eval_env, out["policy"], eval_seeds)))
        per_restart.append({"restart": r, "validation_ticks": score,
                            "states": out["states"]})
        if best is None or score < best[0]:
            best = (score, out)

    assert best is not None
    score, out = best
    pi = out["policy"]
    q = pi.to_qtable()

    baseline = {}
    for strat in ("random", "wilma", "steffen"):
        tk = [run_baseline(to_sim_config(
            req.config.model_copy(update={"seed": sd})), strat,
            log=False).stats()["ticks"] for sd in eval_seeds[:10]]
        baseline[strat] = float(statistics.mean(tk))

    rl_ticks = [run_policy(to_sim_config(
        req.config.model_copy(update={"seed": sd})), q,
        log=False).stats()["ticks"] for sd in eval_seeds[:10]]

    meta = {
        "algo": "tabular policy gradient (REINFORCE, paired per-seed baseline)",
        "episodes": req.episodes, "restarts": req.restarts,
        "alpha": req.alpha, "gamma": req.gamma, "batch": req.batch,
        "aircraft": req.config.aircraft, "arrival": req.config.arrival,
        "gate": req.config.gate,
        "n_passengers": req.config.n_passengers,
        "validation_ticks": score,
        "validation_seeds": [eval_seeds[0], eval_seeds[-1]],
        "states": out["states"],
    }
    saved = None
    if req.save_as:
        save_policy(req.save_as, q, meta)
        saved = req.save_as

    return {
        "saved_as": saved,
        "validation_ticks": score,
        "eval_mean_ticks": float(statistics.mean(rl_ticks)),
        "baseline_ticks": baseline,
        "states": out["states"],
        "per_restart": per_restart,
        "greedy_curve": [[float(a), float(b)] for a, b in out["greedy_curve"]],
        "meta": meta,
    }


@app.post("/api/train", response_model=TrainJob, status_code=202)
def train(req: TrainRequest) -> Dict:
    job = REGISTRY.create(req.episodes * req.restarts)
    REGISTRY.run(job, lambda j: _train_fn(req, j))
    return job.snapshot()


@app.get("/api/train/{job_id}", response_model=TrainJob)
def train_status(job_id: str) -> Dict:
    job = REGISTRY.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail={
            "code": "job_not_found",
            "message": "No training job '{}'. Jobs are kept in memory and "
                       "are lost when the API restarts.".format(job_id)})
    return job.snapshot()


@app.get("/api/train", response_model=List[TrainJob])
def train_list() -> List[Dict]:
    return [j.snapshot() for j in REGISTRY.list()]


@app.get("/api/policies", response_model=List[PolicyOut])
def policies() -> List[Dict]:
    out = []
    for name in available_policies():
        _q, meta = load_policy(name)
        out.append({"name": name, "meta": meta or {}})
    return out
