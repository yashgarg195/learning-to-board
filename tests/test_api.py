"""API contract tests: validation, errors, trajectory serialization.

These run the real simulator through the real app, so a regression in the
engine shows up here too.  Scenarios use the ``tiny`` cabin (4 rows, 16 seats)
so the whole file finishes in seconds.
"""
from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from api.main import app
from boarding.config import BLOCKED, QUEUED, SEATED, STOWING, WALKING

client = TestClient(app)

TINY = {"aircraft": "tiny", "seed": 1}


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_options_lists_only_supported_values():
    o = client.get("/api/options").json()
    assert {a["value"] for a in o["aircraft"]} == {
        "tiny", "mid", "atr72", "a320", "b787", "b777", "a380"}
    assert {a["value"] for a in o["arrivals"]} == {
        "random", "queued", "groups"}
    assert o["gates"] == ["front", "back", "both"]
    assert o["states"] == {"QUEUED": QUEUED, "WALKING": WALKING,
                           "STOWING": STOWING, "BLOCKED": BLOCKED,
                           "SEATED": SEATED}
    # Every advertised aircraft reports a real seat count.
    assert all(a["seats"] > 0 for a in o["aircraft"])


def test_options_strategies_all_runnable():
    o = client.get("/api/options").json()
    for s in o["strategies"]:
        if s["value"] == "rl":
            continue
        r = client.post("/api/simulate",
                        json={"config": TINY, "strategy": s["value"]})
        assert r.status_code == 200, s["value"]


def test_simulate_trajectory_is_consistent():
    r = client.post("/api/simulate", json={"config": TINY, "strategy": "wilma"})
    assert r.status_code == 200
    j = r.json()
    n = len(j["passengers"])
    assert n == 16
    # One congestion sample per recorded frame.
    assert len(j["congestion"]) == len(j["frames"])
    # The clock and the trajectory agree.
    assert j["stats"]["ticks"] == pytest.approx(len(j["frames"]) - 1)
    # Replaying the deltas ends with everyone seated.
    state = [QUEUED] * n
    for frame in j["frames"]:
        for pid, st, _pos in frame:
            state[pid] = st
    assert state == [SEATED] * n
    # The final census matches.
    assert j["congestion"][-1] == [0, 0, 0, n]


def test_simulate_deltas_only_contain_changes():
    j = client.post("/api/simulate",
                    json={"config": TINY, "strategy": "random"}).json()
    seen = {}
    for frame in j["frames"][1:]:
        for pid, st, pos in frame:
            assert seen.get(pid) != (st, pos), "unchanged passenger re-sent"
            seen[pid] = (st, pos)


def test_simulate_is_reproducible_and_seed_sensitive():
    body = {"config": dict(TINY, aircraft="mid"), "strategy": "steffen"}
    a = client.post("/api/simulate", json=body).json()
    b = client.post("/api/simulate", json=body).json()
    assert a["stats"] == b["stats"]
    assert a["frames"] == b["frames"]

    other = dict(body, config=dict(TINY, aircraft="mid", seed=2))
    c = client.post("/api/simulate", json=other).json()
    assert c["stats"]["seed"] == 2.0


@pytest.mark.parametrize("arrival", ["random", "queued", "groups"])
def test_every_arrival_pattern_actually_runs(arrival):
    j = client.post("/api/simulate", json={
        "config": dict(TINY, aircraft="mid", arrival=arrival),
        "strategy": "wilma"}).json()
    assert j["stats"]["ticks"] > 0
    assert j["config"]["arrival"] == arrival


@pytest.mark.parametrize("gate", ["front", "back", "both"])
def test_every_gate_actually_runs(gate):
    j = client.post("/api/simulate", json={
        "config": dict(TINY, gate=gate), "strategy": "wilma"}).json()
    assert j["stats"]["ticks"] > 0
    doors = j["decks"][0]["doors"]
    assert len(doors) == (2 if gate == "both" else 1)


def test_two_deck_aircraft_returns_two_decks():
    j = client.post("/api/simulate", json={
        "config": {"aircraft": "a380", "n_passengers": 40, "seed": 5},
        "strategy": "wilma"}).json()
    assert len(j["decks"]) == 2
    assert {p["d"] for p in j["passengers"]} <= {0, 1}


def test_luggage_fraction_changes_boarding_time():
    def ticks(frac):
        return client.post("/api/simulate", json={
            "config": dict(TINY, aircraft="mid", luggage_fraction=frac),
            "strategy": "wilma"}).json()["stats"]["ticks"]

    assert ticks(0.0) < ticks(1.0)


def test_invalid_config_returns_structured_422():
    r = client.post("/api/simulate",
                    json={"config": {"aircraft": "concorde"}})
    assert r.status_code == 422
    err = r.json()["error"]
    assert err["code"] == "invalid_request"
    assert err["detail"][0]["field"] == "config.aircraft"


def test_out_of_range_values_rejected():
    for bad in ({"luggage_fraction": 1.5}, {"n_passengers": 0},
                {"seed": -1}, {"walk_ticks_per_row": 0}):
        r = client.post("/api/simulate", json={"config": dict(TINY, **bad)})
        assert r.status_code == 422, bad


def test_unknown_field_rejected():
    r = client.post("/api/simulate",
                    json={"config": dict(TINY, turbulence=3)})
    assert r.status_code == 422


def test_rl_without_policy_is_a_clear_error():
    r = client.post("/api/simulate", json={"config": TINY, "strategy": "rl"})
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "policy_required"


def test_missing_policy_is_404():
    r = client.post("/api/simulate",
                    json={"config": TINY, "strategy": "rl",
                          "policy": "no_such_policy"})
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "policy_not_found"


def test_compare_uses_the_same_seeds_for_every_strategy():
    r = client.post("/api/compare", json={
        "config": dict(TINY, aircraft="mid"),
        "strategies": ["random", "wilma", "steffen"], "trials": 3})
    assert r.status_code == 200
    j = r.json()
    assert j["seeds"] == [1, 2, 3]
    assert all(len(row["per_seed"]) == 3 for row in j["rows"])
    # Sorted best-first, and a sane ordering on a mid cabin.
    means = [row["mean_ticks"] for row in j["rows"]]
    assert means == sorted(means)
    by = {row["strategy"]: row["mean_ticks"] for row in j["rows"]}
    assert by["wilma"] < by["random"]


def test_compare_rejects_empty_strategy_list():
    r = client.post("/api/compare",
                    json={"config": TINY, "strategies": [], "trials": 2})
    assert r.status_code == 422


def test_policies_endpoint_shape():
    r = client.get("/api/policies")
    assert r.status_code == 200
    for p in r.json():
        assert isinstance(p["name"], str) and isinstance(p["meta"], dict)


def test_training_job_runs_and_reports_real_progress():
    r = client.post("/api/train", json={
        "config": {"aircraft": "tiny", "seed": 3},
        "episodes": 100, "restarts": 1, "batch": 10})
    assert r.status_code == 202
    job_id = r.json()["job_id"]
    assert r.json()["status"] in ("queued", "running")

    for _ in range(120):
        s = client.get("/api/train/" + job_id).json()
        if s["status"] in ("done", "error"):
            break
        time.sleep(0.5)

    assert s["status"] == "done", s["message"]
    assert s["episodes_done"] == s["episodes_total"] == 100
    res = s["result"]
    # Real measured numbers, not placeholders.
    assert res["validation_ticks"] > 0
    assert res["states"] > 0
    assert set(res["baseline_ticks"]) == {"random", "wilma", "steffen"}
    assert len(res["greedy_curve"]) > 0


def test_unknown_job_is_404():
    r = client.get("/api/train/deadbeef")
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "job_not_found"


def test_train_rejects_bad_policy_name():
    r = client.post("/api/train", json={
        "config": TINY, "episodes": 100, "save_as": "../escape"})
    assert r.status_code == 422
