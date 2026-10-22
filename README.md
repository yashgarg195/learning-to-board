# Learning to Board

**An RL agent for optimal airplane boarding order, with a configurable simulator and visual replay.**

Everything described below is implemented and runnable: a validated tick-based
boarding simulator, five baseline strategies, a Gymnasium environment, a full ladder
of RL methods (bandits → DP → MC → TD → function approximation → policy gradient), a
**FastAPI** backend, and a **React + TypeScript** front end that replays recorded
trajectories in an SVG cabin.

Passengers keep their assigned seats. The object of optimisation is the **boarding
order**, not the seat assignment.

---

## Quickstart

Two processes: a Python API and a Vite dev server. Requires **Python 3.9+** (the
codebase is deliberately 3.9-compatible) and **Node 20+**.

```bash
cd learning-to-board

# --- backend ---------------------------------------------------------------
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
PYTHONPATH=. .venv/bin/uvicorn api.main:app --reload --port 8000

# --- frontend (second terminal) --------------------------------------------
cd frontend
npm install
npm run dev            # http://localhost:5173
```

Open <http://localhost:5173>. The Vite dev server proxies `/api` to port 8000, so
no CORS configuration is needed in development. Interactive API docs are at
<http://127.0.0.1:8000/docs>; the full reference is [`docs/api.md`](docs/api.md).

Pre-trained policies ship in `artifacts/policies/`, so the RL strategy works
immediately without training anything.

### Tests and checks

```bash
# Python: 19 simulator + 15 agent + 32 API tests
PYTHONPATH=. .venv/bin/python -m pytest -q

# Frontend: unit + integration tests, strict type check, production build
cd frontend && npm test && npm run typecheck && npm run build
```

### Offline training and figures

Training from the UI is deliberately small-budget. The shipped policies were
produced offline:

```bash
PYTHONPATH=. .venv/bin/python scripts/train.py --quick     # ~2 min smoke run
PYTHONPATH=. .venv/bin/python scripts/train.py             # full run
PYTHONPATH=. .venv/bin/python scripts/experiments.py       # all deliverables + figures

# the shipped agent (24-action, row parity) -- this is rl_parity_mid
PYTHONPATH=. .venv/bin/python scripts/train_parity.py \
  --cabin mid --episodes 10000 --restarts 3 --save-as rl_parity_mid

# score any saved policy against every baseline, across cabins
PYTHONPATH=. .venv/bin/python scripts/probe_rl_vs_baselines.py rl_parity_mid
```

### Presentation reference

`presentation/boarding-technical-reference.pdf` is a 13-page technical reference
covering the simulator, the MDP, training, evaluation, measured results, negative
results, the API and the test suite. It is **generated, not written**: every number in
it is either read from the code or measured by running the simulator at build time, so
it cannot drift from the implementation.

```bash
.venv/bin/pip install -r requirements-docs.txt
PYTHONPATH=. .venv/bin/python presentation/build_reference.py
```

### Configuration

Copy `.env.example` if you need to change the API origin or the CORS allow-list;
neither is required for the default local setup. `config.example.json` documents
every simulation parameter and is a valid `POST /api/simulate` body.

---

## Repository map

| Path | What it is |
| --- | --- |
| `boarding/config.py` | Aircraft presets, seat geometry, `SimConfig` |
| `boarding/simulator.py` | Tick-based boarding simulator + delta-encoded trajectory log |
| `boarding/strategies.py` | random, back-to-front, WILMA, Steffen, outside-in |
| `boarding/envs.py` | `BoardingEnv` (Gymnasium), observation + tabular state, action masking |
| `boarding/agents/bandit.py` | ε-greedy and UCB1 over fixed strategies, with regret |
| `boarding/agents/dp.py` | Exact MDP on a tiny cabin, policy & value iteration |
| `boarding/agents/tabular.py` | MC prediction/control, SARSA(λ), Watkins' Q(λ) |
| `boarding/agents/linear.py` | Semi-gradient SARSA/Q over size-independent features |
| `boarding/agents/pg.py` | Tabular policy gradient, paired baseline, restart selection (**shipped agent**) |
| `boarding/replay.py`, `boarding/policy.py` | Replay payloads, policy save/load/rollout |
| `api/main.py` | FastAPI app — simulate, compare, train, policies |
| `api/schemas.py` | Pydantic models; the single source of valid options |
| `api/jobs.py` | In-process background training jobs |
| `frontend/src/lib/trajectory.ts` | Expands delta-encoded frames into per-tick state |
| `frontend/src/components/CabinPlan.tsx` | SVG cabin plan view |
| `frontend/src/components/CongestionTrace.tsx` | Aisle-activity strip chart + timeline scrubber |
| `scripts/` | `train.py`, `train_parity.py`, `probe_rl_vs_baselines.py`, `experiments.py`, `sweep.py` |
| `presentation/build_reference.py` | Generates the technical-reference PDF, measuring every number live |
| `tests/` | 19 simulator tests (hand-computed), 15 agent tests, 32 API tests |

---

## The simulator

A cell is `(deck, aisle, row)` and holds at most one passenger. Each tick:

1. **service** — passengers stowing decrement their timer; at zero they take their seat;
2. **move** — walking passengers advance one cell toward their row if it is free;
3. **admit** — the gate controller may release one waiting passenger per free door.

Movement is resolved front-most-first along the direction of travel, so a cleared cell
propagates backwards through the queue within the same tick.

**Service time** = `stow_ticks` (or `no_bag_ticks` for a passenger without a bag) +
`shuffle_ticks × blockers`, where `blockers` counts already-seated neighbours between the
passenger and the aisle on the same side. This is the seat-interference term that makes
window-first strategies pay off.

### Validation before any RL

The spec lists *"simulator bugs look like learning failures"* as a risk, so the simulator
is pinned by 19 hand-computed tests (`tests/test_simulator.py`) covering a lone
passenger, pure queueing, blocking cascades, shuffle costs, multi-door and multi-deck
cases.

One non-obvious invariant those tests exposed: **entering the aircraft costs one tick**,
because `admit` runs *after* `move` within a tick. A lone passenger therefore takes
`1 + rows × walk_ticks_per_row + service` ticks, not `rows × walk + service`.

The second check is external: the baselines reproduce the published ordering from the
boarding literature (A320, 150 passengers, queued arrival, front door):

| Strategy | Ticks |
| --- | --- |
| Steffen (modified) | 208 |
| WILMA (window-middle-aisle) | 388 |
| Random | 418 |
| Back-to-front | 609 |

Steffen ≪ WILMA < random < back-to-front is exactly the result the literature reports,
including the counter-intuitive one — back-to-front is *worse* than random. Boarding
through both doors is roughly 40% faster across the board.

---

## The MDP

Following §3 of the proposal:

| Element | Implementation |
| --- | --- |
| **Agent** | Gate controller. One `step` = one admission decision. |
| **Action** | `(row band, seat type, row parity)` → `n_bands × 3 × 2 = 24` discrete actions. Groups, never individuals. |
| **State** | Size-independent features (see below). |
| **Reward** | `−1` per tick, plus an optional `block_penalty` per blocking event. |
| **Episode end** | All passengers seated. |
| **Discount** | Swept in `scripts/experiments.py`; see the discount analysis below. |

Two details are worth calling out because they drove the results.

**Bands are measured as distance from the door**, not absolute row number. Band 0 is
always the farthest band from whichever door the passenger uses. A policy learned on a
front-gate cabin therefore transfers unchanged to a back-gate cabin, and to both-door
boarding where the two streams have opposite orientations.

**Group availability is handled by action masking, not by the state.** Encoding "which
groups still have passengers" as state bits blew the table up to ~19,000 states with
about 12 visits each — worse than random. Masking keeps the tabular state at
`(progress bin, last band, last seat type, last parity)` ≈ 290 visited states, which is
actually learnable.

**Row parity is part of the action.** The original action space was `(band, seat type)`
only — 12 actions. That space *cannot express Steffen's schedule*, whose entire
mechanism is alternating odd and even rows so that two consecutive stowers never stand
back to back. The measured consequence is in the table below: with 12 actions the agent
matched WILMA but was structurally capped below Steffen. Adding the parity bit doubles
the action space to 24 and the state count to ~290 — still trivially tabular — and gives
the agent the vocabulary to discover interleaving on its own.

Policies trained before this change are still loadable: `policy_uses_parity()` infers
the action space from the saved table's own width and runs it in the matching mode, so
`rl_agent` and `rl_agent_mid` reproduce their original numbers exactly.

The function-approximation agent sees the richer 23-dimensional observation from
`observe()`: band-pending fractions, seat-type-pending fractions, band aisle occupancy,
blocked/seated/pending scalars, and one-hot encodings of the last action. All of these
are *fractions*, so one weight vector applies to a 48-seat ATR-72 and a 500-seat A380
alike.

---

## Results

### Exact DP on the tiny cabin

The real simulator has no transition model, so — as the proposal recommends — DP runs on
an **exact deterministic surrogate** of a 4-row, 2-2 cabin
(`boarding/agents/dp.py`), with 48,001 reachable states.

| Method | Sweeps | Time | Optimal boarding time |
| --- | --- | --- | --- |
| Value iteration | 2 | 0.19 s | **37** |
| Policy iteration | 19 (10 improvements) | 1.30 s | **37** |

Both converge to the same optimum, and value iteration is roughly 7× cheaper here
because the horizon is short.

The headline result is *what* DP found. The optimal order is **all window seats
back-to-front, then all aisle seats back-to-front** (`row 4 window, row 3 window, row 2
window, row 1 window, row 4 aisle, ...`) — that is, WILMA's seat ordering combined with
back-to-front spacing, rediscovered from scratch with no hand-designed heuristic. This
is precisely the "rediscovering similar patterns" success criterion from §8 of the
proposal, and it holds for every discount factor tested.

### Learning on the mid-size cabin

Mid cabin (10 rows, 3-3, 60 passengers), front gate, evaluated greedily on seeds 500–519
which are disjoint from the training seed pool.

All agents were trained on the same mixture of cabins unless marked otherwise.

| Policy | Mean ticks | Notes |
| --- | --- | --- |
| Linear function approximation | 228.2 | 23 size-independent features |
| Watkins' Q(λ=0.9) | 207.5 | traces cut by every exploratory action |
| One-step SARSA(0) | 207.8 | **deliberate ablation — does not learn** |
| Monte-Carlo control | 175.7 | full-episode returns do carry the credit |
| SARSA(λ=1.0) | 163.6 | 145 states, 24k episodes |
| **Policy gradient, generalist** | **164.8 ± 9.1** | shipped as `rl_agent` |
| **Policy gradient, mid specialist** | **157.1 ± 9.3** | shipped as `rl_agent_mid` |
| **Policy gradient + row parity** | **154.0 ± 7.9** | shipped as `rl_parity_mid` — the current agent |

Adding **row parity** to the action space (24 actions instead of 12) is what finally
pushed the agent past the heuristics. Parity is what lets the policy express "do not
release two adjacent rows back to back", which is the mechanism behind Steffen-style
boarding. Without it the agent can pick a distance band and a seat type but cannot
alternate.

Head-to-head on the mid cabin over the **30 held-out evaluation seeds (500–529)**,
measured by `presentation/build_reference.py`:

| Policy | Mean ticks | Std | Best | Worst |
| --- | --- | --- | --- | --- |
| **`rl_parity_mid`** | **154.0** | **7.9** | 140 | 169 |
| `rl_agent_mid` | 157.1 | 9.3 | 135 | 176 |
| WILMA | 159.0 | 10.9 | 138 | 181 |
| Steffen-style | 159.1 | 10.1 | 142 | 184 |
| Outside-in zones | 161.0 | 11.7 | 143 | 183 |
| `rl_agent` (generalist) | 164.8 | 9.1 | 143 | 182 |
| Random | 221.5 | 18.7 | 178 | 250 |
| Back-to-front | 265.3 | 15.4 | 237 | 294 |

On this cabin the parity agent is **5.1 ticks (3.2%) ahead of Steffen**, with the lowest
standard deviation of any policy tested. That gap is larger than the earlier 1-tick tie,
but it is still only about half a standard deviation, so the honest phrasing is
"consistently ahead on this cabin", not "solved boarding".

### The generalisation gap (negative result)

The same policy on the **A320**, which it was never trained on, over the same 30 seeds:

| Policy | Mean ticks | Std |
| --- | --- | --- |
| Steffen-style | 379.3 | 13.8 |
| WILMA | 383.7 | 13.7 |
| Outside-in zones | 386.9 | 12.7 |
| `rl_agent_mid` | 392.1 | 15.7 |
| `rl_parity_mid` | 394.0 | 13.6 |
| `rl_agent` | 407.0 | 14.1 |
| Random | 493.5 | 26.7 |
| Back-to-front | 615.6 | 30.5 |

**The agent loses to every hand-designed heuristic off-distribution.** Training a parity
agent directly on the A320 did not fix it:

| Budget | Eval ticks | vs Steffen (379.3) |
| --- | --- | --- |
| 6,000 episodes | 426.5 | −47.2 |
| 24,000 episodes | 421.0 | −41.7 |

Quadrupling the sample budget recovered ~5 ticks of a ~47-tick gap, which rules out
undertraining and points at the **state representation**: distance bands and row parity
mean different things on a 30-row cabin than on a 10-row one. Neither A320 policy is
shipped — both were deleted rather than left in the dropdown.

Regenerate everything with `scripts/train.py` then `scripts/experiments.py`; the tables
land in `artifacts/training.json` and `artifacts/experiments.json`. The head-to-head
tables above come from `presentation/build_reference.py`, which measures them live.

### Reading the learning-curve figure

`artifacts/figures/learning_curves.png` plots the value-based agents as their raw
training episodes (solid) and the policy-gradient agents as periodic **greedy**
evaluations (dashed). The distinction matters: policy gradient explores by *sampling*
from the softmax, so its training episodes run ~50 ticks slower than the policy it is
actually learning, and plotting them would make a working agent look flat.

The solid curves are therefore an upper bound on their own agents too — SARSA(λ) ends
the plot around 183 while its greedy evaluation is 163.6, the difference being residual
ε-exploration. Compare *final greedy numbers* from the table above, not curve endpoints.

### Restarts, not lucky seeds

Policy gradient on this problem is genuinely multi-modal: independent restarts settle
into different boarding schedules and scatter by roughly 20 ticks. The validation scores
across five restarts of the specialist were **190, 211, 165, 161, 172**.

Reporting the best of those as if it were a typical run would be dishonest, and
reporting the mean would understate what the method reaches. So `train_best_of` runs
several restarts and selects using a **validation seed set (300–319) that appears in
neither the training pool (1–48) nor the test set (500–529)**. Selection generalises
cleanly — the restart with validation 160.9 scored 157.1 on the held-out test seeds.

Training longer is *not* a substitute: with no entropy bonus the softmax saturates, and
24k episodes scored worse (172.8) than 12k. Restarts buy what a longer run does not.

### Why one-step TD fails here (an instructive negative result)

`scripts/train.py` deliberately trains a one-step SARSA(0) ablation, and it does not
learn. The reason is structural rather than a tuning problem:

* the immediate reward is ≈ `−1` for *every* legal action, and
* the successor state is almost independent of which group was released.

So `Q(s, a₁) − Q(s, a₂)` collapses toward zero and the greedy policy is arbitrary. The
cost of releasing the wrong group is only paid 20–40 ticks later, when the aisle jams.

Three fixes were tried, in order:

1. **Eligibility traces.** SARSA(λ≈0.95–1.0) works, because the trace carries credit back
   across the whole delay. Watkins' Q(λ) does *not* — its traces get cut by every
   exploratory action, which on a 60-step episode means almost immediately.
2. **Monte-Carlo control.** Also works; the full episode return carries the credit.
3. **Policy gradient with a paired baseline.** The shipped agent. Best on a single
   cabin; roughly level with SARSA(λ) when both are trained across the mixture.

That last one is the main engineering contribution, so it deserves a paragraph.

### Paired baselines (implementation decision)

The simulator is a *deterministic* function of (policy, seed): the randomness lives
entirely in which passengers draw carry-on bags and in arrival jitter. A single episode's
boarding time swings by ±12 ticks for reasons that have nothing to do with the policy,
which dwarfs the ~8-tick gap between a good policy and a great one.

`boarding/agents/pg.py` exploits this by keeping a **per-seed running mean** of boarding
time and using it as the REINFORCE baseline. The advantage then answers *"was this
rollout better than my usual result on this exact cabin?"*, which cancels the
passenger-draw noise instead of averaging over it. Combined with batched gradient
accumulation and advantage normalisation, this is what moved the learned agent from
~180 ticks (behind WILMA) to ~155 (ahead of everything).

Hyperparameters were chosen by the sweep in `scripts/sweep.py pg`: α = 0.25, batch = 16,
48-seed pool, no entropy bonus. Entropy regularisation consistently *hurt* (169.7 vs
154.6 in the sweep) — the action space is small enough that the paired baseline already
supplies enough signal, and the bonus just blurs a schedule that wants to be sharp.

### Bandits, discount, and transfer

### Bandits over fixed strategies

ε-greedy and UCB1 choose among the five fixed strategies over 300 pulls, with cumulative
regret plotted in `artifacts/figures/bandit.png`.

| Algorithm | Pull counts (random / b2f / WILMA / Steffen / outside-in) | Best arm | Final regret |
| --- | --- | --- | --- |
| ε-greedy | 8 / 4 / 4 / **277** / 7 | WILMA | **2.76** |
| UCB1 | 46 / 35 / 72 / 72 / 75 | WILMA | 16.33 |

Both identify the same best arm, but ε-greedy wins decisively on regret here, which is
the opposite of the usual textbook ordering. The reason is the gap structure: WILMA,
Steffen and outside-in are within noise of each other, so UCB1's confidence bounds stay
overlapping and it keeps splitting pulls three ways for the whole horizon. ε-greedy
commits early, and because the top three arms are nearly equivalent, committing to a
*near*-best arm costs almost nothing. UCB1 would win back the difference over a longer
horizon or with better-separated arms.

### Discount analysis

Swept over γ ∈ {0.5, 0.7, 0.85, 0.95, 0.99, 1.0} on the tiny cabin. **Every value
returns the same optimal policy and the same boarding time of 37 ticks.**

This is a negative result, and it is reported as one rather than dressed up. The tiny
cabin's horizon is only ~37 steps, so even γ = 0.5 still weighs the seat-interference
cost that arrives 5–10 steps later heavily enough to make the correct choice. The
myopic-vs-farsighted distinction simply does not bite at this
scale — it reappears on the mid cabin, where the delay between releasing a group and
the resulting aisle jam is 20–40 ticks and low γ does fail.

### Monte-Carlo prediction: first-visit vs every-visit

| Mode | States visited | Mean V | Total updates |
| --- | --- | --- | --- |
| First-visit | 97 | −40.72 | 9,434 |
| Every-visit | 97 | −40.71 | 9,600 |

The estimates agree to two decimal places. Every-visit makes ~2% more updates for no
measurable gain, because the boarding MDP rarely revisits a state within an episode —
progress through the queue is monotone, so most states are visited exactly once anyway
and the two estimators nearly coincide by construction.

### Transfer to unseen layouts

The generalist (trained on `mid` front, `mid` back, A320 with 90 passengers, and
ATR-72) evaluated unchanged on five presets at 90% load, 12 seeds each:

| Aircraft | Pax | RL agent | Best baseline | Gap |
| --- | --- | --- | --- | --- |
| ATR 72 (2-2) | 64 | 159.7 | 153.8 (outside-in) | +3.8% |
| A320 / 737 (3-3) | 162 | 365.8 | 348.1 (WILMA) | +5.1% |
| Boeing 787 (3-3-3) | 291 | 381.4 | 349.8 (outside-in) | +9.0% |
| Boeing 777 (3-4-3) | 378 | 433.2 | 430.8 (outside-in) | +0.6% |
| **Airbus A380 (2 decks)** | 612 | **437.9** | 450.3 (outside-in) | **−2.8%** |

Read honestly, this is mixed and the failures matter as much as the win:

* It **beats every baseline on the A380** — the largest, two-deck layout, furthest from
  anything it trained on. That is the strongest evidence that the size-independent
  state encoding does what it was designed to do.
* It is **worst on the 787** (+9%), a twin-aisle layout whose 3-3-3 block geometry has
  no analogue in the training mixture — the agent never saw a centre block with no
  window seat during training.
* It beats random everywhere, so the learned schedule is never nonsense.

The pattern is consistent: transfer tracks whether the *seat-block geometry* resembles
training, not whether the aircraft size does.

---

## The interface

A React + TypeScript single-page app over the FastAPI backend. It implements the
Figure 7 mockup — configuration on the left, replay and comparison on the right —
and nothing in it is simulated client-side.

**Trajectory architecture.** `POST /api/simulate` returns the complete recorded
trajectory as delta-encoded frames. The frontend expands those deltas once into
per-tick cabin states, then replays them. Changing speed or dragging the scrubber
never re-runs the simulation, and the browser cannot invent a passenger position
the Python engine did not produce. See [`docs/api.md`](docs/api.md) for the frame
format.

* **Cabin plan (SVG).** Rows run nose-to-tail left-to-right, seat columns top to
  bottom, with hatched aisle lanes and magenta door markers. Seats shade as they
  fill; passengers in the aisle are colour-coded by state — walking (blue), stowing
  (amber), blocked (red), seated (tan). Multi-deck aircraft render both decks, and a
  gate stack shows who has not boarded yet.
* **Aisle-activity trace.** A strip chart of walking / stowing / blocked over time,
  scaled to the peak number of *active* passengers rather than the whole manifest —
  on a 180-seat cabin only ~20 people move at once, so stacking the seated majority
  would flatten the signal this chart exists to show. A ribbon underneath tracks the
  seated share. Click or drag it to scrub; it is the timeline.
* **Playback.** Play / pause, single-tick step, reset, replay, 0.5×–8× speed, and a
  slider. Space toggles playback; the trace accepts arrow keys, Home and End.
* **Metrics.** Total boarding time, elapsed tick, seated / remaining, average wait
  and aisle-blocking statistics, each with a tooltip defining it. Units are stated
  explicitly — all times are simulator ticks.
* **Comparison.** Any subset of strategies over the same seed sequence, with a bar
  chart carrying ±1 standard-deviation error bars and a table of mean / std / best /
  worst / average wait.
* **Training.** Start a background run, watch genuine progress and a genuine greedy
  evaluation curve, then see the trained policy benchmarked against three baselines
  on the same seeds. A trained policy is saved and becomes selectable as a strategy.
  The panel states the state / action / reward / termination definitions inline.

**States are honest.** Loading, empty, error and "no agent trained yet" states are
distinct and actionable; backend validation errors are rendered field by field. The
RL strategy stays disabled, with an explanation, until a policy exists. No chart
shows placeholder data, and a short in-UI training run will usually *lose* to
WILMA — the UI reports that rather than hiding it.

**Two kinds of saved policy.** `artifacts/policies/` holds the trained agents
(`rl_parity_mid`, `rl_agent_mid`, `rl_agent`) *and* the algorithm-comparison artifacts
(`rl_mc`, `rl_sarsa`, `rl_qlearning`, `rl_linear`) that exist to demonstrate which
algorithms fail here. The latter are genuinely slower than the heuristics — that is
the finding they encode — so they are **excluded from the boarding dropdown
entirely** and kept only as report evidence. `load_policy` still accepts them by
name for scripted runs. The dropdown auto-selects the strongest trained agent, so a
known-weak artifact is never mistaken for "the RL agent".

**Options withheld from the UI.** Three engine capabilities are deliberately not
offered as product options, because showing them would mislead:

| Hidden | Still works via | Why |
|---|---|---|
| `aircraft: "tiny"` | API, `pytest`, `scripts/` | A 4-row fixture for hand-verifiable tests, not a cabin worth demonstrating. |
| `arrival: "queued"` | API, `scripts/experiments.py` | `BoardingEnv` never reads `cfg.arrival`. Under perfect compliance the fixed strategies get passenger-by-passenger sequencing while the agent still releases coarse groups, so any comparison on it flatters the heuristics — Steffen most of all. |
| `TrainRequest.gamma` | removed outright | It was never forwarded to the trainer but *was* written into saved metadata, so it could record a discount factor that never applied. The trainer is undiscounted by construction (γ = 1). |

The `mid` cabin is grouped separately in the dropdown as a **benchmark cabin**
rather than an aircraft, because it is the cabin the agent is trained and
evaluated on and should not be read as a real type.

**Design.** A deliberately non-generic "sectional chart" direction: chart-paper
ground, chart-ink text, airspace magenta for the primary/RL accent, Class-B blue for
baselines and the aisle, terrain tan for seated, caution amber for stowing. Archivo
for display, IBM Plex Sans for body, IBM Plex Mono for data. Responsive down to
tablet, visible keyboard focus throughout, `prefers-reduced-motion` respected.

Presets: ATR-72 (2-2, single aisle), A320 (3-3), B787 and B777 (twin aisle), A380
(two decks), plus `mid` and `tiny` cabins used for the tabular and DP work.

---

## Spec requirement vs implementation decision

Everything the specification asks for is implemented. Where the specification was
silent, the choice made here is listed as a decision.

### From the specification

| Requirement | Where |
| --- | --- |
| Configurable tick-based simulator | `boarding/simulator.py` |
| Passengers keep assigned seats; order is optimised | enforced by construction |
| Gate-controller agent, group actions, −1/tick reward | `boarding/envs.py` |
| Optional blocking penalty | `SimConfig.block_penalty` |
| Size-independent state for generalisation | `observe()` / `tabular_key()` |
| Baselines: random, back-to-front, WILMA, Steffen | `boarding/strategies.py` |
| Aircraft presets, aisles, seat blocks, doors, decks | `boarding/config.py` |
| Gymnasium environment | `BoardingEnv` |
| ε-greedy + UCB bandits over strategies, with regret | `agents/bandit.py` |
| Exact PI/VI on a ~4×4 cabin + discount analysis | `agents/dp.py` |
| First-visit vs every-visit MC prediction | `agents/tabular.py` |
| Q-learning vs SARSA on a mid cabin | `agents/tabular.py` |
| Linear function approximation | `agents/linear.py` |
| REINFORCE with baseline | `agents/linear.py`, `agents/pg.py` |
| Validate simulator against hand-computed cases | `tests/test_simulator.py` |
| Mean boarding time + variance across gates/arrivals | `scripts/experiments.py` |
| Learning curves, discount analysis, PI vs VI | `scripts/experiments.py` |
| Transfer to an unseen layout | `scripts/experiments.py` |
| Web UI: replay controls, colour coding, fixed seed | `frontend/` + `api/` |

### Implementation decisions (not specified)

| Decision | Rationale |
| --- | --- |
| Bands measured as **distance from the door** | Makes one policy valid for front, back and both-door boarding. |
| Group availability via **action masking**, not state bits | Explicit state bits produced ~19k states and learned worse than random. |
| Tabular state = `(progress bin, last band, last seat type)` | Smallest state that still expresses a schedule; ~240 states. |
| **Paired per-seed baseline** for policy gradient | Cancels passenger-draw noise, which otherwise dwarfs the policy signal. |
| **Restart selection on a held-out validation seed set** | Policy gradient here is multi-modal; selecting on seeds disjoint from train *and* test keeps the reported number honest. |
| Ship both a generalist (`rl_agent`) and a specialist (`rl_agent_mid`) | Makes the cost of generalisation measurable instead of implicit. |
| Common random numbers (fixed seed pool) in all tabular training | Variance reduction; eval seeds are disjoint from the training pool. |
| DP on an **exact deterministic surrogate**, not the simulator | The simulator exposes no transition model; the proposal scopes DP as a sanity check anyway. |
| Centre seat blocks (e.g. 3-**4**-3) have no window seats | Geometrically true; the block splits down the middle, each half served by its own aisle. |
| Delta-encoded trajectory log | Keeps A380 replays small enough to ship to the browser. |
| **React + FastAPI** instead of a single Python UI process | The simulator stays the sole source of truth; the browser only replays recorded trajectories, which also makes speed changes and scrubbing free. |
| Training jobs held **in process**, not in a broker | A single-user research prototype; a queue would be infrastructure with no requirement behind it. Documented limitation: jobs are lost on API restart (saved policies are not). |
| Congestion trace scaled to **active** passengers, not the manifest | On a 180-seat cabin ~20 people move at once; stacking the seated majority flattens exactly the signal the chart exists to show. |
| Outside-in added as a fifth baseline | Standard in the literature and a useful stress test for the seat-interference model. |
| Python 3.9 compatibility throughout | Only interpreter available on the target machine. |

---

## Honest assessment

Success here means *matching good baselines and rediscovering similar patterns, not a
large margin* — the ceiling above a well-tuned heuristic is genuinely low. It is worth
being precise about what was and was not achieved:

**Achieved.**

* The parity agent (**154.0 ± 7.9** over 30 held-out seeds) **beats every hand-designed
  strategy on the cabin it was trained on** — Steffen 159.1, WILMA 159.0, outside-in
  161.0 — and does so with the lowest variance of any policy tested. It is far ahead of
  random (221.5) and back-to-front (265.3).
* The win came from a **representation change, not more compute**: adding row parity to
  the action space is what let the policy express alternation, the mechanism behind
  Steffen-style boarding.
* Exact DP independently rediscovers **window-first back-to-front**, with no heuristic
  supplied.
* The generalist **beats every baseline on the unseen A380** (437.9 vs 450.3), which is
  real evidence for the size-independent state design.

**Where it falls short, stated plainly.**

* **It wins only on the cabin it was trained on.** `rl_parity_mid` is 5.1 ticks ahead of
  Steffen on `mid` but 14.7 ticks *behind* Steffen on the A320, losing to every
  hand-designed heuristic off-distribution. Training directly on the A320 at 6k and 24k
  episodes produced 426.5 and 421.0 ticks against Steffen's 379.3, so the problem is the
  band/parity state representation rather than the sample budget. Neither A320 policy is
  shipped.
* **The margin is under one standard deviation.** 154.0 ± 7.9 against Steffen's
  159.1 ± 10.1 is a consistent win on 30 held-out seeds, not a decisive one. The correct
  claim is "consistently ahead on this cabin", not that boarding is solved.
* **Generalisation costs about 11 ticks.** The generalist `rl_agent` scores 164.8 on
  `mid` against the parity specialist's 154.0. The specialist is not passed off as a
  general result to hide this; all three agents ship and all three are measured.
* **The gate-controller action space cannot express Steffen's side interleaving.** Row
  parity gets part of the way there, which is why it helped, but the agent still cannot
  address the two sides of the aisle separately. Fixing this needs per-side actions — a
  clean extension, not a retrofit.
* **The method needed restart selection.** A single policy-gradient run is a coin flip
  between ~160 and ~190 validation ticks; without best-of-N selection on a held-out
  validation pool the result is not reproducible.
* **One-step SARSA(0), Q(λ) and linear FA all failed** (207.8 / 207.5 / 228.2). They are
  trained, plotted and kept rather than quietly dropped.
* **The discount-factor analysis found nothing on the tiny cabin.** Reported as a
  negative result with the reason, rather than padded.
* **Ticks are not seconds.** The timing constants were never calibrated against real
  boarding data, so every comparison here is relative.

---

## Tests

```bash
PYTHONPATH=. .venv/bin/python -m pytest -q
```

**66 Python tests**, all passing:

* **19 simulator tests** with hand-computed expectations — a
  lone passenger's exact boarding time, aisle blocking, luggage delays, seat-shuffle
  costs, determinism under a fixed seed.
* **15 agent tests** — env contract, action masking, DP optimality on the tiny
  cabin, tabular convergence, policy round-tripping, replay payload integrity.
* **32 API tests** — every aircraft / gate / arrival / strategy option, trajectory
  consistency (`stats.ticks == len(frames) - 1`, deltas contain only changes, the
  replay ends with everyone seated), reproducibility under a fixed seed, seed
  pairing in `/api/compare`, structured validation errors, and the full background
  training-job lifecycle.

**13 frontend tests** — `cd frontend && npm test`:

* Delta expansion — carry-forward, immutability of earlier ticks, full-width frames.
* Congestion series — bands non-negative and summing to the manifest.
* App integration against a mocked API — unreachable backend, empty state, RL
  disabled without a policy, metrics rendered from the response, scrubbing without
  extra network calls, structured validation errors surfaced to the user.

Plus `npm run typecheck` (strict TypeScript, clean) and `npm run build`.

`frontend/e2e-check.mjs` is a manual Playwright script that drives the real stack —
configure → run → step → play → scrub → compare → train → replay the trained policy.
It needs both servers running and a Chromium binary:

```bash
cd frontend && CHROME_PATH="/path/to/chromium" node e2e-check.mjs
```
