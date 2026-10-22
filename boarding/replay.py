"""Turn a finished simulation into a compact replay payload for the UI."""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import random

from .config import SimConfig
from .simulator import Simulator
from .strategies import STRATEGIES, make_selector


def layout_payload(sim: Simulator) -> List[Dict]:
    out = []
    for di, deck in enumerate(sim.spec.decks):
        doors = sorted({row for d, row in sim.doors if d == di})
        out.append({
            "name": deck.name,
            "rows": deck.rows,
            "cols": deck.n_cols,
            "aisleCols": deck.aisle_cols(),
            "blocks": list(deck.blocks),
            "doors": doors,
        })
    return out


def build_replay(sim: Simulator) -> Dict:
    pax = []
    for p in sim.passengers:
        pax.append({
            "d": p.seat.deck, "r": p.seat.row, "c": p.seat.col,
            "ac": sim.spec.decks[p.seat.deck].aisle_cols()[p.seat.aisle],
            "t": p.seat.seat_type, "bag": 1 if p.has_bag else 0,
            "id": p.seat.seat_id,
        })
    return {
        "decks": layout_payload(sim),
        "passengers": pax,
        "frames": sim.log.frames,
        "stats": sim.stats(),
    }


def run_baseline(cfg: SimConfig, strategy: str, log: bool = True) -> Simulator:
    sim = Simulator(cfg, rng=random.Random(cfg.seed), log=log)
    sel = make_selector(sim, strategy)
    while not sim.done and sim.t < cfg.max_ticks:
        sim.tick(sel)
    return sim
