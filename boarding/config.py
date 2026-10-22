"""Aircraft layouts and simulation configuration.

Spec requirements implemented here (Section 6 of the project overview):
  ATR 72 (2-2, 1 aisle), A320/737 (3-3, 1 aisle), 787 (3-3-3, 2 aisles),
  777 (3-4-3, 2 aisles), A380 (upper 2-4-2 + main 3-4-3, two decks).

Implementation decisions (not fixed by the spec) are marked with `DECISION:`.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Sequence, Tuple

SEAT_LETTERS = "ABCDEFGHJK"  # 'I' skipped, as in real seat maps

# Passenger runtime states (also used as colour codes by the frontend).
QUEUED = 0
WALKING = 1
STOWING = 2
BLOCKED = 3
SEATED = 4

STATE_NAMES = {
    QUEUED: "queued",
    WALKING: "walking",
    STOWING: "stowing",
    BLOCKED: "blocked",
    SEATED: "seated",
}


@dataclass(frozen=True)
class Seat:
    deck: int
    row: int           # 0-based, 0 = front-most row
    block: int         # seat block index (left -> right)
    index_in_block: int
    col: int           # rendering column (seats and aisles interleaved)
    aisle: int         # which aisle this seat is reached from
    depth: int         # seats to climb over to reach the aisle (0 = aisle seat)
    seat_type: str     # "aisle" | "middle" | "window"
    letter: str

    @property
    def seat_id(self) -> str:
        return "%d%s" % (self.row + 1, self.letter)


@dataclass(frozen=True)
class Deck:
    name: str
    rows: int
    blocks: Tuple[int, ...]
    # Door positions expressed as row indices; "front" doors first.
    doors: Tuple[int, ...] = ()

    @property
    def n_aisles(self) -> int:
        return max(1, len(self.blocks) - 1)

    @property
    def n_cols(self) -> int:
        return sum(self.blocks) + self.n_aisles

    @property
    def seats_per_row(self) -> int:
        return sum(self.blocks)

    @property
    def capacity(self) -> int:
        return self.rows * self.seats_per_row

    def aisle_cols(self) -> List[int]:
        cols: List[int] = []
        c = 0
        for b, size in enumerate(self.blocks):
            c += size
            if b < len(self.blocks) - 1:
                cols.append(c)
                c += 1
        if not cols:  # single block (degenerate) -> aisle on the right
            cols.append(sum(self.blocks))
        return cols

    def seat_map(self) -> List[Seat]:
        """All seats of this deck with geometry + interference metadata."""
        seats: List[Seat] = []
        n_aisles = self.n_aisles
        for row in range(self.rows):
            col = 0
            letter_i = 0
            for b, size in enumerate(self.blocks):
                for i in range(size):
                    # Which aisle serves this seat, and how deep is it?
                    if len(self.blocks) == 1:
                        aisle, depth = 0, (size - 1 - i)
                    elif b == 0:
                        aisle, depth = 0, (size - 1 - i)        # aisle on the right
                    elif b == len(self.blocks) - 1:
                        aisle, depth = n_aisles - 1, i           # aisle on the left
                    else:
                        # DECISION: centre blocks are split down the middle;
                        # the left half uses the left aisle, the right half the right one.
                        half = size / 2.0
                        if i < half:
                            aisle, depth = b - 1, i
                        else:
                            aisle, depth = b, (size - 1 - i)
                    max_depth = max(
                        depth_of(self.blocks, b, j, n_aisles) for j in range(size)
                    )
                    if depth == 0:
                        stype = "aisle"
                    elif depth == max_depth and b in (0, len(self.blocks) - 1):
                        stype = "window"
                    else:
                        stype = "middle"
                    seats.append(
                        Seat(
                            deck=0, row=row, block=b, index_in_block=i, col=col,
                            aisle=aisle, depth=depth, seat_type=stype,
                            letter=SEAT_LETTERS[letter_i % len(SEAT_LETTERS)],
                        )
                    )
                    col += 1
                    letter_i += 1
                if b < len(self.blocks) - 1:
                    col += 1  # skip the aisle column
        return seats


def depth_of(blocks: Sequence[int], b: int, i: int, n_aisles: int) -> int:
    size = blocks[b]
    if len(blocks) == 1:
        return size - 1 - i
    if b == 0:
        return size - 1 - i
    if b == len(blocks) - 1:
        return i
    half = size / 2.0
    return i if i < half else size - 1 - i


@dataclass(frozen=True)
class Aircraft:
    key: str
    name: str
    decks: Tuple[Deck, ...]

    @property
    def capacity(self) -> int:
        return sum(d.capacity for d in self.decks)

    def seat_map(self) -> List[Seat]:
        seats: List[Seat] = []
        for di, deck in enumerate(self.decks):
            for s in deck.seat_map():
                seats.append(
                    Seat(deck=di, row=s.row, block=s.block,
                         index_in_block=s.index_in_block, col=s.col, aisle=s.aisle,
                         depth=s.depth, seat_type=s.seat_type, letter=s.letter)
                )
        return seats

    def describe(self) -> str:
        parts = ["-".join(str(b) for b in d.blocks) for d in self.decks]
        return " / ".join(parts)


def _deck(name: str, rows: int, blocks: Tuple[int, ...]) -> Deck:
    # DECISION: doors sit at the first and last row of the cabin.
    return Deck(name=name, rows=rows, blocks=blocks, doors=(0, rows - 1))


AIRCRAFT: Dict[str, Aircraft] = {
    "atr72": Aircraft("atr72", "ATR 72 (2-2, regional)",
                      (_deck("main", 18, (2, 2)),)),
    "a320": Aircraft("a320", "A320 / 737 (3-3)",
                     (_deck("main", 30, (3, 3)),)),
    "b787": Aircraft("b787", "Boeing 787 (3-3-3)",
                     (_deck("main", 36, (3, 3, 3)),)),
    "b777": Aircraft("b777", "Boeing 777 (3-4-3)",
                     (_deck("main", 42, (3, 4, 3)),)),
    "a380": Aircraft("a380", "Airbus A380 (2 decks)",
                     (_deck("main", 44, (3, 4, 3)),
                      _deck("upper", 30, (2, 4, 2)))),
    # Tiny cabins used for exact DP / tabular work (Labs 3-6).
    "tiny": Aircraft("tiny", "Tiny cabin (4 rows, 2-2)",
                     (_deck("main", 4, (2, 2)),)),
    "mid": Aircraft("mid", "Mid cabin (10 rows, 3-3)",
                    (_deck("main", 10, (3, 3)),)),
}

AIRCRAFT_ORDER = ["atr72", "a320", "b787", "b777", "a380", "mid", "tiny"]


@dataclass
class SimConfig:
    """Everything the simulator needs. Mirrors Section 3 'environment parameters'."""
    aircraft: str = "a320"
    n_passengers: Optional[int] = None     # None -> full cabin
    gate: str = "front"                    # "front" | "back" | "both"
    arrival: str = "random"                # "random" | "queued" | "groups"
    luggage_fraction: float = 0.7
    stow_ticks: int = 6                    # time to stow a bag
    no_bag_ticks: int = 2                  # time to sit down without a bag
    shuffle_ticks: int = 4                 # per seat climbed over
    walk_ticks_per_row: int = 1
    seed: int = 42
    n_bands: int = 4                       # row bands used by actions/features
    block_penalty: float = 0.0             # optional reward shaping (Section 3)
    max_ticks: int = 20000

    def spec(self) -> Aircraft:
        return AIRCRAFT[self.aircraft]

    def resolved_passengers(self) -> int:
        cap = self.spec().capacity
        if self.n_passengers is None:
            return cap
        return max(1, min(int(self.n_passengers), cap))

    def copy(self, **kw) -> "SimConfig":
        d = asdict(self)
        d.update(kw)
        return SimConfig(**d)


SEAT_TYPES = ("window", "middle", "aisle")
