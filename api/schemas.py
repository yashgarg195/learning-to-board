"""Pydantic request/response models for the boarding API.

Every option the UI can offer is enumerated here, so an unsupported aircraft,
gate or arrival pattern is rejected with a 422 instead of silently doing
nothing.  The ``/api/options`` endpoint is generated from the same literals,
which is what keeps the configuration panel honest.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field
from typing_extensions import Literal

from boarding.config import AIRCRAFT
from boarding.strategies import STRATEGY_LABELS

AircraftName = Literal["tiny", "mid", "atr72", "a320", "b787", "b777", "a380"]
Gate = Literal["front", "back", "both"]
Arrival = Literal["random", "queued", "groups"]
Strategy = Literal["random", "back_to_front", "wilma", "steffen",
                   "outside_in", "rl"]

#: Accepted by the engine and the API, but not offered in the UI.
#:
#: ``tiny`` is a hand-verifiable test fixture. ``queued`` is withheld because
#: ``BoardingEnv`` never reads ``cfg.arrival``: under perfect compliance the
#: fixed strategies get passenger-by-passenger sequencing while the RL agent
#: still releases coarse groups, so any comparison on it flatters the
#: heuristics. ``scripts/experiments.py`` and the test suite still use both.
HIDDEN_AIRCRAFT = frozenset({"tiny"})
HIDDEN_ARRIVALS = frozenset({"queued"})

#: Cabins that exist to benchmark the agent rather than model a real type.
BENCHMARK_CABINS = frozenset({"mid"})


class SimConfigIn(BaseModel):
    """The simulation knobs exposed to the UI (spec Section 4A)."""

    aircraft: AircraftName = "a320"
    n_passengers: Optional[int] = Field(
        default=None, ge=1, le=900,
        description="Number of passengers; null means a full cabin.")
    gate: Gate = "front"
    arrival: Arrival = "random"
    luggage_fraction: float = Field(default=0.6, ge=0.0, le=1.0)
    stow_ticks: int = Field(default=6, ge=1, le=60)
    no_bag_ticks: int = Field(default=2, ge=1, le=60)
    shuffle_ticks: int = Field(default=4, ge=0, le=60)
    walk_ticks_per_row: int = Field(default=1, ge=1, le=10)
    seed: int = Field(default=42, ge=0, le=2**31 - 1)

    model_config = {"extra": "forbid"}


class SimulateRequest(BaseModel):
    config: SimConfigIn = Field(default_factory=SimConfigIn)
    strategy: Strategy = "wilma"
    policy: Optional[str] = Field(
        default=None,
        description="Saved policy name; required when strategy is 'rl'.")

    model_config = {"extra": "forbid"}


class DeckOut(BaseModel):
    name: str
    rows: int
    cols: int
    aisleCols: List[int]
    blocks: List[int]
    doors: List[int]


class PassengerOut(BaseModel):
    d: int
    r: int
    c: int
    ac: int
    t: Literal["window", "middle", "aisle"]
    bag: int
    id: str


class StatsOut(BaseModel):
    ticks: float
    blocking_events: float
    blocked_ticks: float
    avg_wait: float
    n_passengers: float
    seed: float


class SimulateResponse(BaseModel):
    decks: List[DeckOut]
    passengers: List[PassengerOut]
    frames: List[List[List[int]]] = Field(
        description="Delta-encoded per tick: [passengerIndex, state, posRow].")
    congestion: List[List[int]] = Field(
        description="Per tick census [walking, stowing, blocked, seated].")
    stats: StatsOut
    strategy: Strategy
    policy: Optional[str] = None
    config: SimConfigIn


class CompareRequest(BaseModel):
    config: SimConfigIn = Field(default_factory=SimConfigIn)
    strategies: List[Strategy] = Field(
        default=["random", "back_to_front", "wilma", "steffen"],
        min_length=1, max_length=6)
    trials: int = Field(default=5, ge=1, le=40)
    policy: Optional[str] = None

    model_config = {"extra": "forbid"}


class CompareRow(BaseModel):
    strategy: Strategy
    label: str
    mean_ticks: float
    std_ticks: float
    min_ticks: float
    max_ticks: float
    mean_blocked_ticks: float
    mean_avg_wait: float
    trials: int
    per_seed: List[float]


class CompareResponse(BaseModel):
    seeds: List[int]
    rows: List[CompareRow]
    config: SimConfigIn


class TrainRequest(BaseModel):
    config: SimConfigIn = Field(default_factory=SimConfigIn)
    episodes: int = Field(default=1500, ge=100, le=40000)
    restarts: int = Field(default=1, ge=1, le=8)
    alpha: float = Field(default=0.25, gt=0.0, le=2.0)
    # No gamma: the reward is -1 per tick and the objective is *total*
    # boarding time, so every step in an episode is worth the same and the
    # trainer is undiscounted by construction (gamma = 1). The field used to
    # exist here, was never forwarded to the trainer, and was written into
    # the saved metadata -- which could record a discount that never applied.
    batch: int = Field(default=16, ge=1, le=128)
    save_as: Optional[str] = Field(
        default=None, pattern=r"^[A-Za-z0-9_\-]{1,40}$")

    model_config = {"extra": "forbid"}


class TrainJob(BaseModel):
    job_id: str
    status: Literal["queued", "running", "done", "error"]
    created_at: float
    episodes_done: int
    episodes_total: int
    message: Optional[str] = None
    curve: List[List[float]] = Field(
        default_factory=list,
        description="[episode, greedy mean ticks] evaluation points.")
    result: Optional[Dict] = None


class PolicyOut(BaseModel):
    name: str
    meta: Dict
    label: str = ""
    kind: Literal["agent", "reference", "user"] = "agent"
    note: str = ""


class OptionsResponse(BaseModel):
    aircraft: List[Dict]
    gates: List[str]
    arrivals: List[Dict]
    strategies: List[Dict]
    policies: List[PolicyOut]
    states: Dict[str, int]


class ErrorBody(BaseModel):
    code: str
    message: str
    detail: Optional[object] = None


class ErrorResponse(BaseModel):
    error: ErrorBody


def aircraft_catalogue() -> List[Dict]:
    """Cabins offered in the UI.

    ``tiny`` is deliberately excluded: it is a 4-row fixture that exists so
    the simulator's behaviour can be worked out by hand in the tests, not a
    cabin anyone would demo. The engine and the API still accept it.
    """
    out = []
    for key, spec in AIRCRAFT.items():
        if key not in AircraftName.__args__:  # type: ignore[attr-defined]
            continue
        if key in HIDDEN_AIRCRAFT:
            continue
        seats = sum(d.rows * sum(d.blocks) for d in spec.decks)
        out.append({
            "value": key,
            "label": spec.name,
            "decks": len(spec.decks),
            "rows": spec.decks[0].rows,
            "seats": seats,
            "layout": "-".join(str(b) for b in spec.decks[0].blocks),
            "kind": "benchmark" if key in BENCHMARK_CABINS else "aircraft",
        })
    return out


def strategy_catalogue() -> List[Dict]:
    descriptions = {
        "random": "Free-for-all: passengers board in arrival order.",
        "back_to_front": "Rear zones called first, four zones.",
        "wilma": "All windows, then all middles, then all aisles.",
        "steffen": "Alternating rows, outside-in, so neighbours never collide.",
        "outside_in": "Outside-in within back-to-front zones.",
        "rl": "A trained tabular policy chosen by the agent at each release.",
    }
    return [{"value": k, "label": STRATEGY_LABELS[k],
             "description": descriptions[k]}
            for k in Strategy.__args__]  # type: ignore[attr-defined]
