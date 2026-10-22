# Implementation Plan — Smart Airplane Boarding Optimizer

Target: a working React + FastAPI prototype where a user configures a boarding
scenario, runs a real simulation, replays its trajectory, compares strategies, and
trains/loads a genuine RL policy.

The Python simulation and RL engine already exists, is validated against hand-computed
cases, and is reused as-is. The work is the service layer and the web client.

---

## Status legend

`[x]` done and verified · `[~]` partial, limitation documented · `[ ]` not started

---

## Phase 0 — Inherited (pre-existing, verified)

- [x] Tick-based simulator with aisle occupancy, luggage stow, seat-interference
      shuffle, multi-door and multi-deck support — `boarding/simulator.py`
- [x] Aircraft presets: ATR 72, A320, 787, 777, A380 (2 decks), plus `mid`/`tiny`
      research cabins — `boarding/config.py`
- [x] Baselines: random, back-to-front, WILMA, Steffen-style, outside-in
- [x] Gymnasium environment with compact size-independent state + action masking
- [x] RL ladder: bandits, exact DP, MC, SARSA(λ)/Q(λ), linear FA, policy gradient
- [x] 30 passing tests, incl. 19 hand-computed simulator scenarios
- [x] Trained policies in `artifacts/policies/`

**Acceptance:** `pytest tests -q` passes; baselines reproduce published ordering
(Steffen ≪ WILMA < random < back-to-front).

## Phase 1 — Engine gaps required by this spec

- [x] `arrival="groups"` pattern (spec §4A lists random / queued / **groups**;
      engine had only random and queued)
- [x] Per-tick congestion series exposed for the frontend trace

**Acceptance:** new arrival mode changes measured boarding time; covered by a test.

## Phase 2 — FastAPI backend

- [x] `api/schemas.py` — Pydantic request/response models, enum-constrained options
- [x] `GET /api/health` — liveness + engine version
- [x] `GET /api/options` — aircraft presets, capacities, strategies, arrival modes,
      gates, and which combinations are supported (drives the UI; no dead controls)
- [x] `POST /api/simulate` — run one scenario, return trajectory + stats
- [x] `POST /api/compare` — several strategies, same manifest/seed, N trials,
      returns mean and standard deviation per strategy
- [x] `POST /api/train` — start a background training job
- [x] `GET /api/train/{job_id}` — real status, progress, curve, error
- [~] `GET /api/policies` — list saved policies with metadata. **Adjusted:** no
      `activate` endpoint; `POST /api/simulate` and `/api/compare` take a `policy`
      name directly, so there is no server-side "active policy" to mutate.
- [x] Structured error envelope + 422 validation; CORS for the Vite dev origin

**Acceptance:** every endpoint exercised by `tests/test_api.py`; training does not
block the event loop; invalid configs return a structured 422.

## Phase 3 — Frontend

- [x] Vite + React + TypeScript + Tailwind, Node pinned via Volta
- [x] Typed API client mirroring the Pydantic schemas
- [x] Configuration panel — every control bound to a real engine parameter
- [x] SVG cabin visualization — seats by type, aisle, doors, passenger states,
      multi-deck, congestion
- [x] Trajectory playback — play / pause / reset / speed / scrub, replaying backend
      frames without re-running the simulation
- [x] Metrics with documented units and definitions
- [x] Strategy comparison with Recharts, showing variability not just means
- [x] Training panel with genuine job status and an empty state when untrained
- [x] Loading / empty / error / success states; keyboard focus; responsive

**Acceptance:** `tsc --noEmit` clean, `vite build` succeeds, vitest passes, and the
replayed animation is driven only by backend frames.

## Phase 4 — Verification

- [x] Backend and frontend both start from documented commands
- [x] A scenario is configured, run, replayed, and compared in the real UI
- [x] A training job runs to completion and its policy is used for a simulation
- [x] Production build verified

## Phase 5 — Documentation

- [x] `README.md` rewritten for the React + FastAPI stack
- [x] `docs/api.md` endpoint reference
- [x] `docs/skills-used.md` honest record
- [x] `config.example.json`, `.env.example`, `.gitignore`, dependency files

---

## Verified results

| Check | Command | Result |
| --- | --- | --- |
| Python suite | `PYTHONPATH=. .venv/bin/python -m pytest -q` | **55 passed** (19 simulator + 11 agent + 25 API) |
| Frontend tests | `cd frontend && npm test` | **13 passed** (7 trajectory + 6 App integration) |
| Type check | `npm run typecheck` | clean |
| Production build | `npm run build` | succeeds (~550 kB JS, 161 kB gzip; Recharts dominates) |
| Browser end-to-end | `node e2e-check.mjs` | configure → run (364 ticks) → step → 4× play advanced exactly 50 ticks in 1.5 s → scrub to 50 % → compare 4 strategies → train 400 episodes → replay the trained policy |

Two real bugs were found by that end-to-end run and fixed: `/api/options` reported
A320 as 210 seats (it counted aisle columns) and the cabin SVG clipped at the right
edge.

---

## Known limitations

- Training jobs live in an in-process registry (last 32) and are lost when the API
  restarts. Saved policies persist on disk.
- A short in-UI training run (a few hundred episodes) loses to WILMA and Steffen.
  This is reported in the UI rather than hidden; the shipped policies were trained
  offline with `scripts/train.py`.
- Recharts 2.x emits React `defaultProps` deprecation warnings in the console.
  Upstream, not project code.
- Steffen is implemented as the documented practical interpretation, not the
  strict "perfect" variant.



No database, message broker, auth, deployment pipeline, or deep-learning framework.
Training jobs are in-process and single-node; that is sufficient for a prototype and
is documented as a limitation rather than hidden.

---

## Key design constraints

1. **The Python engine is the single source of truth.** The frontend replays frames;
   it never invents passenger movement.
2. **No dead controls.** `GET /api/options` declares what is supported, and the UI
   disables anything that is not.
3. **No fabricated RL results.** If no policy is trained, the UI says so and offers
   the training workflow.
4. **Trajectory-based playback.** One simulation produces one trajectory, replayable
   at any speed without recomputation.
