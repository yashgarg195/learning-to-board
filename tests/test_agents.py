"""Environment / agent smoke tests."""
from __future__ import annotations

import os
import random
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from boarding.agents.dp import (TinyMDP, describe_sequence, policy_iteration,
                                rollout, value_iteration)
from boarding.agents.common import masked_argmax
from boarding.agents.linear import features, train_semi_gradient
from boarding.agents.pg import evaluate_pg, train_tabular_pg
from boarding.agents.tabular import evaluate_tabular, mc_control, td_control
from boarding.config import SimConfig
from boarding.envs import (BoardingEnv, action_to_group, group_to_action,
                           n_actions, obs_size)
from boarding.replay import build_replay, run_baseline


def test_action_group_roundtrip():
    cfg = SimConfig()
    for a in range(n_actions(cfg)):
        assert group_to_action(cfg, action_to_group(cfg, a)) == a


def test_env_runs_to_completion_and_masks_are_honest():
    env = BoardingEnv(SimConfig(aircraft="mid"))
    obs, info = env.reset(seed=1)
    assert obs.shape == (obs_size(env.cfg),)
    rng = random.Random(0)
    steps = 0
    done = False
    while not done:
        mask = info["action_mask"]
        assert mask.sum() > 0
        a = int(rng.choice(list(np.flatnonzero(mask))))
        obs, r, term, trunc, info = env.step(a)
        assert r <= 0
        steps += 1
        done = term or trunc
    assert env.sim.done
    assert steps == env.sim.n       # exactly one decision per passenger


def test_env_reward_equals_negative_boarding_time():
    env = BoardingEnv(SimConfig(aircraft="atr72"))
    obs, info = env.reset(seed=4)
    total = 0.0
    done = False
    while not done:
        a = int(np.flatnonzero(info["action_mask"])[0])
        obs, r, term, trunc, info = env.step(a)
        total += r
        done = term or trunc
    # -1 per tick, plus the single tick consumed before the first decision.
    assert abs(total) == pytest.approx(env.sim.stats()["ticks"], abs=1.0)


def test_observation_is_size_independent():
    a = BoardingEnv(SimConfig(aircraft="atr72"))
    b = BoardingEnv(SimConfig(aircraft="b777"))
    oa, _ = a.reset(seed=1)
    ob, _ = b.reset(seed=1)
    assert oa.shape == ob.shape
    assert a.key()[1:] == b.key()[1:]


def test_tiny_dp_recovers_window_first_back_to_front():
    mdp = TinyMDP(rows=3)
    vi = value_iteration(mdp, gamma=1.0)
    pi = policy_iteration(mdp, gamma=1.0)
    cv, seq = rollout(mdp, vi["policy"])
    cp, _ = rollout(mdp, pi["policy"])
    assert cv == cp, "PI and VI must agree on the optimal value"
    desc = describe_sequence(mdp, seq)
    # every window seat is called before any aisle seat
    first_aisle = next(i for i, d in enumerate(desc) if "aisle" in d)
    assert all("window" in d for d in desc[:first_aisle])
    assert first_aisle == mdp.rows * 2


def test_dp_discount_changes_the_policy():
    mdp = TinyMDP(rows=3)
    far = rollout(mdp, value_iteration(mdp, gamma=1.0)["policy"])[0]
    myopic = rollout(mdp, value_iteration(mdp, gamma=0.3)["policy"])[0]
    assert far <= myopic


def test_training_runs_and_improves_on_random():
    env = BoardingEnv(SimConfig(aircraft="atr72"))
    out = td_control(env, episodes=600, alpha=0.5, lam=1.0, algo="sarsa", seed=2)
    trained = evaluate_tabular(env, out["q"], range(900, 910))
    rng = random.Random(1)
    rand = []
    for sd in range(900, 910):
        obs, info = env.reset(seed=sd)
        done = False
        while not done:
            a = int(rng.choice(list(np.flatnonzero(info["action_mask"]))))
            obs, r, term, trunc, info = env.step(a)
            done = term or trunc
        rand.append(env.sim.stats()["ticks"])
    assert np.mean(trained) < np.mean(rand)


def test_linear_agent_trains():
    env = BoardingEnv(SimConfig(aircraft="atr72"))
    out = train_semi_gradient(env, episodes=120, seed=1)
    assert np.isfinite(out["model"].w).all()
    assert features(env.reset(seed=1)[0]).shape[0] == obs_size(env.cfg) + 1


def test_replay_payload_is_consistent():
    cfg = SimConfig(aircraft="a320", n_passengers=60, seed=5)
    sim = run_baseline(cfg, "wilma")
    pay = build_replay(sim)
    assert len(pay["frames"]) == sim.t + 1
    state = {}
    for frame in pay["frames"]:
        for pid, st, row in frame:
            state[pid] = st
    assert len(state) == sim.n
    assert all(v == 4 for v in state.values()), "everyone must end up seated"
    assert len(pay["passengers"]) == sim.n


def test_a380_has_two_decks_in_the_payload():
    cfg = SimConfig(aircraft="a380", n_passengers=200, seed=2)
    pay = build_replay(run_baseline(cfg, "steffen"))
    assert len(pay["decks"]) == 2
    assert {p["d"] for p in pay["passengers"]} == {0, 1}


def test_policy_gradient_beats_random_and_exports_a_qtable():
    env = BoardingEnv(SimConfig(aircraft="atr72"))
    out = train_tabular_pg(env, episodes=1200, alpha=0.25, batch=16,
                           n_seeds=24, entropy=0.0, seed=3)
    seeds = list(range(900, 910))
    trained = evaluate_pg(env, out["policy"], seeds)

    rng = random.Random(7)
    rand = []
    for sd in seeds:
        obs, info = env.reset(seed=sd)
        done = False
        while not done:
            a = int(rng.choice(list(np.flatnonzero(info["action_mask"]))))
            obs, r, term, trunc, info = env.step(a)
            done = term or trunc
        rand.append(env.sim.stats()["ticks"])
    assert np.mean(trained) < np.mean(rand)

    # The exported table must reproduce the softmax policy's greedy choices,
    # which is what lets the demo reuse the Q-table replay path.
    q = out["policy"].to_qtable()
    obs, info = env.reset(seed=901)
    mask = info["action_mask"]
    assert masked_argmax(q[env.key()], mask) == out["policy"].greedy(env.key(), mask)


def test_parity_action_space_round_trips():
    """Every action decodes to a (band, seat type, parity) and back."""
    from boarding.envs import (action_to_group, group_to_action, n_actions,
                               n_parity)
    cfg = SimConfig(aircraft="mid", use_parity=True)
    assert n_parity(cfg) == 2
    assert n_actions(cfg) == cfg.n_bands * 3 * 2
    seen = set()
    for a in range(n_actions(cfg)):
        g = action_to_group(cfg, a)
        assert 0 <= g[0] < cfg.n_bands and g[2] in (0, 1)
        assert group_to_action(cfg, g) == a
        seen.add(g)
    assert len(seen) == n_actions(cfg)


def test_parity_can_be_disabled_for_legacy_policies():
    from boarding.envs import action_to_group, group_to_action, n_actions
    cfg = SimConfig(aircraft="mid", use_parity=False)
    assert n_actions(cfg) == cfg.n_bands * 3
    for a in range(n_actions(cfg)):
        g = action_to_group(cfg, a)
        assert g[2] == 0
        assert group_to_action(cfg, g) == a


def test_parity_actions_select_the_right_row_parity():
    """A parity action must only ever release a passenger from that parity."""
    from boarding.envs import BoardingEnv, action_to_group
    env = BoardingEnv(SimConfig(aircraft="mid", use_parity=True, seed=3))
    env.reset(seed=3)
    for _ in range(40):
        mask = env.info()["action_mask"]
        if mask.sum() == 0:
            break
        a = int(np.argmax(mask))
        _band, _st, par = action_to_group(env.cfg, a)
        before = set(env.sim.pending)
        _o, _r, term, trunc, _i = env.step(a)
        released = before - set(env.sim.pending)
        for pid in released:
            p = env.sim.passengers[pid]
            assert p.seat.row % 2 == par
            assert p.seat.seat_type == _st
            assert p.band == _band
        if term or trunc:
            break


def test_legacy_policy_still_loads_and_runs():
    """The shipped 12-action policies must survive the action-space change."""
    from boarding.policy import load_policy, policy_uses_parity, run_policy
    q, _meta = load_policy("rl_agent_mid")
    if q is None:
        pytest.skip("shipped policy not present")
    cfg = SimConfig(aircraft="mid", seed=500)
    assert policy_uses_parity(cfg, q) is False
    sim = run_policy(cfg, q, log=False)
    assert sim.done
    assert sim.stats()["ticks"] > 0
