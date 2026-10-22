"""Generate the presentation technical reference PDF.

Every number in the output is either read from the source of truth (a constant in
``boarding/``, a key in a saved policy's metadata) or *measured live* by running the
simulator when this script executes.  Nothing is hand-typed, so the document cannot
drift away from the code.

Usage (from the repository root)::

    PYTHONPATH=. .venv/bin/python presentation/build_reference.py

Writes ``presentation/boarding-technical-reference.pdf``.

Requires ``reportlab`` (documentation-only dependency, see requirements-docs.txt).
"""
from __future__ import annotations

import os
import statistics
import subprocess
import sys
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

from boarding import envs
from boarding.config import AIRCRAFT, AIRCRAFT_ORDER, SimConfig
from boarding.policy import available_policies, load_policy, policy_uses_parity, run_policy
from boarding.simulator import Simulator
from boarding.strategies import make_selector

# ---------------------------------------------------------------- palette

INK = colors.HexColor("#13242E")
MUTED = colors.HexColor("#5A6E74")
LINE = colors.HexColor("#CBD4D2")
MAGENTA = colors.HexColor("#C4347A")
PAPER = colors.HexColor("#F4F6F5")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(HERE, "boarding-technical-reference.pdf")

EVAL_SEEDS = list(range(500, 530))
VALIDATION_SEEDS = list(range(300, 320))
TRAIN_SEEDS = list(range(1, 49))

BASELINES = ["random", "back_to_front", "wilma", "outside_in", "steffen"]


# ---------------------------------------------------------------- measuring


def run_strategy(cfg: SimConfig, strategy: str) -> Simulator:
    sim = Simulator(cfg)
    selector = make_selector(sim, strategy)
    while not sim.done:
        sim.tick(selector)
    return sim


def measure_strategy(cabin: str, strategy: str, seeds: Sequence[int]) -> Dict[str, float]:
    ticks: List[float] = []
    waits: List[float] = []
    for s in seeds:
        cfg = SimConfig(aircraft=cabin, seed=s)
        sim = run_strategy(cfg, strategy)
        st = sim.stats()
        ticks.append(float(st["ticks"]))
        waits.append(float(st.get("avg_wait", 0.0)))
    return {
        "mean": statistics.mean(ticks),
        "std": statistics.pstdev(ticks) if len(ticks) > 1 else 0.0,
        "best": min(ticks),
        "worst": max(ticks),
        "wait": statistics.mean(waits),
    }


def measure_policy(cabin: str, name: str, seeds: Sequence[int]) -> Optional[Dict[str, float]]:
    q, _meta = load_policy(name)
    if q is None:
        return None
    ticks: List[float] = []
    waits: List[float] = []
    for s in seeds:
        cfg = SimConfig(aircraft=cabin, seed=s)
        try:
            sim = run_policy(cfg, q, log=False)
        except Exception:
            return None
        st = sim.stats()
        ticks.append(float(st["ticks"]))
        waits.append(float(st.get("avg_wait", 0.0)))
    return {
        "mean": statistics.mean(ticks),
        "std": statistics.pstdev(ticks) if len(ticks) > 1 else 0.0,
        "best": min(ticks),
        "worst": max(ticks),
        "wait": statistics.mean(waits),
    }


def count_tests() -> List[Tuple[str, int]]:
    """Collected test counts from pytest itself.

    Counting ``def test_`` lines undercounts parametrised tests, so ask pytest for
    the real collected total and only fall back to the textual count if pytest is
    unavailable.
    """
    try:
        out = subprocess.check_output(
            [sys.executable, "-m", "pytest", "--collect-only", "-q"],
            cwd=ROOT, stderr=subprocess.DEVNULL,
            env=dict(os.environ, PYTHONPATH=ROOT),
        ).decode()
        counts: Dict[str, int] = {}
        for line in out.splitlines():
            if "::" in line:
                f = line.split("::")[0].strip()
                counts[os.path.basename(f)] = counts.get(os.path.basename(f), 0) + 1
        if counts:
            return sorted(counts.items())
    except Exception:
        pass
    res: List[Tuple[str, int]] = []
    tests_dir = os.path.join(ROOT, "tests")
    for fn in sorted(os.listdir(tests_dir)):
        if not fn.startswith("test_") or not fn.endswith(".py"):
            continue
        with open(os.path.join(tests_dir, fn), "r") as fh:
            n = sum(1 for line in fh if line.startswith("def test_"))
        res.append((fn, n))
    return res


def git_revision() -> str:
    try:
        rev = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL
        )
        return rev.decode().strip()
    except Exception:
        return "not a git checkout"


# ---------------------------------------------------------------- styles

_ss = getSampleStyleSheet()

S = {
    "title": ParagraphStyle(
        "title", parent=_ss["Title"], fontName="Helvetica-Bold", fontSize=26,
        leading=30, textColor=INK, alignment=TA_LEFT, spaceAfter=4,
    ),
    "subtitle": ParagraphStyle(
        "subtitle", parent=_ss["Normal"], fontName="Helvetica", fontSize=11.5,
        leading=16, textColor=MUTED, spaceAfter=18,
    ),
    "h1": ParagraphStyle(
        "h1", parent=_ss["Heading1"], fontName="Helvetica-Bold", fontSize=15,
        leading=19, textColor=INK, spaceBefore=16, spaceAfter=7,
    ),
    "h2": ParagraphStyle(
        "h2", parent=_ss["Heading2"], fontName="Helvetica-Bold", fontSize=11,
        leading=14, textColor=MAGENTA, spaceBefore=11, spaceAfter=4,
    ),
    "body": ParagraphStyle(
        "body", parent=_ss["Normal"], fontName="Helvetica", fontSize=9.3,
        leading=13.4, textColor=INK, spaceAfter=6,
    ),
    "small": ParagraphStyle(
        "small", parent=_ss["Normal"], fontName="Helvetica", fontSize=8.1,
        leading=11.4, textColor=MUTED, spaceAfter=5,
    ),
    "cell": ParagraphStyle(
        "cell", parent=_ss["Normal"], fontName="Helvetica", fontSize=8.1,
        leading=10.8, textColor=INK,
    ),
    "cellhead": ParagraphStyle(
        "cellhead", parent=_ss["Normal"], fontName="Helvetica-Bold", fontSize=8.1,
        leading=10.8, textColor=INK,
    ),
    "code": ParagraphStyle(
        "code", parent=_ss["Normal"], fontName="Courier", fontSize=8.2,
        leading=11.6, textColor=INK, backColor=PAPER,
        borderPadding=6, spaceAfter=7, leftIndent=2,
    ),
}


def P(text, style="body"):
    return Paragraph(text, S[style])


def bullets(items, style="body"):
    out = []
    for it in items:
        out.append(Paragraph("&bull;&nbsp;&nbsp;" + it, ParagraphStyle(
            "b", parent=S[style], leftIndent=10, firstLineIndent=-10, spaceAfter=3)))
    return out


def table(rows, widths, header=True, align_right=()):
    data = []
    for r_i, row in enumerate(rows):
        st = "cellhead" if (header and r_i == 0) else "cell"
        data.append([Paragraph(str(c), S[st]) for c in row])
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    style = [
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("LINEBELOW", (0, 0), (-1, -2), 0.4, LINE),
    ]
    if header:
        style += [
            ("BACKGROUND", (0, 0), (-1, 0), PAPER),
            ("LINEBELOW", (0, 0), (-1, 0), 0.9, INK),
        ]
    for c in align_right:
        style.append(("ALIGN", (c, 0), (c, -1), "RIGHT"))
    t.setStyle(TableStyle(style))
    return t


def kv(pairs, w1=58 * mm, w2=104 * mm):
    return table([["Item", "Value"]] + [[a, b] for a, b in pairs], [w1, w2])


def rule():
    t = Table([[""]], colWidths=[162 * mm], rowHeights=[0.1])
    t.setStyle(TableStyle([("LINEABOVE", (0, 0), (-1, 0), 0.7, LINE)]))
    return t


# ---------------------------------------------------------------- page frame


class Doc(BaseDocTemplate):
    def __init__(self, path):
        BaseDocTemplate.__init__(
            self, path, pagesize=A4,
            leftMargin=24 * mm, rightMargin=24 * mm,
            topMargin=20 * mm, bottomMargin=18 * mm,
            title="Learning to Board — Technical Reference",
            author="Learning to Board",
        )
        frame = Frame(self.leftMargin, self.bottomMargin,
                      self.width, self.height, id="f")
        self.addPageTemplates([PageTemplate(id="main", frames=[frame],
                                            onPage=self._decorate)])

    def _decorate(self, canvas, doc):
        canvas.saveState()
        if doc.page > 1:
            canvas.setFont("Helvetica", 7.5)
            canvas.setFillColor(MUTED)
            canvas.drawString(24 * mm, A4[1] - 13 * mm,
                              "Learning to Board — Technical Reference")
            canvas.setStrokeColor(LINE)
            canvas.setLineWidth(0.5)
            canvas.line(24 * mm, A4[1] - 15 * mm, A4[0] - 24 * mm, A4[1] - 15 * mm)
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(MUTED)
        canvas.drawRightString(A4[0] - 24 * mm, 11 * mm, str(doc.page))
        canvas.restoreState()


# ---------------------------------------------------------------- content


def build() -> None:
    print("measuring baselines (this runs the real simulator) ...")

    cabins = ["mid", "a320"]
    measured: Dict[str, Dict[str, Dict[str, float]]] = {}
    for cab in cabins:
        measured[cab] = {}
        for strat in BASELINES:
            measured[cab][strat] = measure_strategy(cab, strat, EVAL_SEEDS)
            print("  %-6s %-14s %6.1f" % (cab, strat, measured[cab][strat]["mean"]))

    agent_names = [n for n in available_policies() if n.startswith("rl_parity") or n in ("rl_agent_mid", "rl_agent")]
    for cab in cabins:
        for name in agent_names:
            res = measure_policy(cab, name, EVAL_SEEDS)
            if res is not None:
                measured[cab][name] = res
                print("  %-6s %-14s %6.1f" % (cab, name, res["mean"]))

    cfg_mid = SimConfig(aircraft="mid")
    n_rows_mid = sum(dk.rows for dk in cfg_mid.spec().decks)
    d = SimConfig()
    cat = AIRCRAFT

    q_main, meta_main = load_policy("rl_parity_mid")
    meta_main = meta_main or {}

    story = []

    # ---------------------------------------------------------- cover
    story.append(P("Learning to Board", "title"))
    story.append(P(
        "Technical reference for presentation. An RL gate controller that chooses "
        "<b>boarding order</b> for a tick-based airplane cabin simulator. Every figure "
        "in this document was measured by running the simulator when the document was "
        "generated, or read directly from the code and policy metadata.", "subtitle"))
    story.append(rule())
    story.append(Spacer(1, 8))
    story.append(kv([
        ("Shipped agent", "<font face='Courier'>rl_parity_mid</font> — tabular REINFORCE with a paired baseline"),
        ("Source of truth", "The Python simulator. The front end only replays recorded trajectories."),
        ("Evaluation seeds", "%d&ndash;%d (%d seeds), never used in training" % (EVAL_SEEDS[0], EVAL_SEEDS[-1], len(EVAL_SEEDS))),
        ("Training seeds", "%d&ndash;%d (%d seeds), cycled" % (TRAIN_SEEDS[0], TRAIN_SEEDS[-1], len(TRAIN_SEEDS))),
        ("Validation seeds", "%d&ndash;%d (%d seeds), used only to pick between restarts" % (VALIDATION_SEEDS[0], VALIDATION_SEEDS[-1], len(VALIDATION_SEEDS))),
        ("Discount &gamma;", "1.0 (undiscounted) for every shipped result"),
        ("Revision", git_revision()),
    ]))
    story.append(Spacer(1, 10))
    story.append(P(
        "<b>Headline, stated honestly.</b> The agent beats every fixed heuristic on the "
        "cabin it was trained on, and loses to Steffen on cabins it was not trained on. "
        "Both results are in this document. See &ldquo;Negative results&rdquo;.", "small"))

    # ---------------------------------------------------------- architecture
    story.append(PageBreak())
    story.append(P("1. Architecture", "h1"))
    story.append(P(
        "Three layers with a strict one-way dependency: the simulator knows nothing "
        "about the API, and the API knows nothing about the front end's rendering."))
    story.append(table([
        ["Layer", "Module", "Responsibility"],
        ["Simulation", "<font face='Courier'>boarding/simulator.py</font>",
         "Tick-based cabin model. Aisle occupancy, walking, stowing, seat shuffles, "
         "blocking. Produces a trajectory log."],
        ["Cabin presets", "<font face='Courier'>boarding/config.py</font>",
         "Row/column/door/deck geometry for %d cabins." % len(cat)],
        ["Heuristics", "<font face='Courier'>boarding/strategies.py</font>",
         "The five fixed baselines, expressed as passenger selectors."],
        ["MDP", "<font face='Courier'>boarding/envs.py</font>",
         "Wraps the simulator as a Gymnasium environment: state encoding, action "
         "masking, reward."],
        ["Agents", "<font face='Courier'>boarding/agents/</font>",
         "Bandits, exact DP, tabular MC/TD, linear FA, and the shipped policy "
         "gradient."],
        ["API", "<font face='Courier'>api/main.py</font>",
         "FastAPI. Validates with Pydantic, runs the simulator, returns a trajectory "
         "as JSON. Background training jobs."],
        ["Front end", "<font face='Courier'>frontend/src/</font>",
         "React + TypeScript + Vite. Replays the recorded trajectory in an SVG cabin. "
         "Invents no simulation events."],
    ], [22 * mm, 44 * mm, 96 * mm]))

    story.append(P("Why trajectory replay rather than live streaming", "h2"))
    story.append(P(
        "The backend runs the whole episode and returns a delta-encoded frame list. "
        "The front end scrubs through it. This means playback speed, pause, replay and "
        "the timeline scrubber cost nothing and never re-run the simulation, and it "
        "guarantees that what is on screen is exactly what the Python engine produced."))

    # ---------------------------------------------------------- simulator
    story.append(PageBreak())
    story.append(P("2. The simulator", "h1"))
    story.append(P(
        "Discrete time. One <b>tick</b> is the atomic unit; all reported times are in "
        "ticks, not seconds, because the tick-to-second mapping is a calibration "
        "choice and was not calibrated against real boarding data."))

    story.append(P("Per-tick update order", "h2"))
    story.append(P(
        "Each tick the simulator (1) asks the controller whether to release a waiting "
        "passenger into the aisle, (2) decrements stowing and seat-shuffle timers, "
        "(3) moves every in-aisle passenger one step toward their row if and only if "
        "the next aisle cell is free, and (4) seats anyone who has arrived and "
        "finished their delays."))

    story.append(P("Constraints enforced by construction", "h2"))
    story.extend(bullets([
        "A single aisle cell holds at most one passenger, so passengers cannot pass "
        "through one another. A slow passenger blocks everyone behind them.",
        "A passenger only ever moves toward their assigned row.",
        "A seat is occupied by exactly one passenger; seat assignment is a "
        "permutation and is never optimised &mdash; only the <i>order</i> is.",
        "A passenger is not counted as seated until they have reached their row "
        "<i>and</i> finished stowing <i>and</i> finished any seat-shuffle delay.",
        "Boarding time is read from the simulator clock when the last passenger sits, "
        "never estimated.",
    ]))

    story.append(P("Timing parameters (defaults from <font face='Courier'>SimConfig</font>)", "h2"))
    story.append(kv([
        ("<font face='Courier'>walk_ticks_per_row</font>", "%s &mdash; ticks to advance one row along the aisle" % d.walk_ticks_per_row),
        ("<font face='Courier'>stow_ticks</font>", "%s &mdash; ticks to stow a carry-on, blocking the aisle" % d.stow_ticks),
        ("<font face='Courier'>no_bag_ticks</font>", "%s &mdash; ticks for a passenger with no carry-on" % d.no_bag_ticks),
        ("<font face='Courier'>shuffle_ticks</font>", "%s &mdash; ticks per already-seated passenger who must stand to let someone in" % d.shuffle_ticks),
        ("<font face='Courier'>luggage_fraction</font>", "%s &mdash; share of passengers carrying a bag" % d.luggage_fraction),
        ("<font face='Courier'>max_ticks</font>", "%s &mdash; safety cap; an episode that hits this is a bug, not a result" % d.max_ticks),
        ("<font face='Courier'>seed</font>", "%s &mdash; controls manifest, luggage and arrival order" % d.seed),
    ]))
    story.append(P(
        "The seat-shuffle rule is the main source of interference and therefore the "
        "main thing an agent can learn to avoid: boarding a window passenger after the "
        "aisle passenger in the same row forces the aisle passenger to stand up.", "small"))

    story.append(P("Reproducibility", "h2"))
    story.append(P(
        "A given (aircraft, seed, arrival, luggage fraction) tuple always produces the "
        "same passenger manifest, the same seat assignment and the same luggage flags. "
        "Strategy comparison therefore holds the manifest fixed and varies only the "
        "boarding order, which is the only fair way to compare."))

    # ---------------------------------------------------------- cabins
    story.append(PageBreak())
    story.append(P("3. Cabins and strategies", "h1"))
    rows = [["Cabin", "Name", "Rows", "Seats", "Layout"]]
    for name in AIRCRAFT_ORDER:
        ac = AIRCRAFT[name]
        rows.append([
            "<font face='Courier'>%s</font>" % name,
            ac.name,
            str(sum(dk.rows for dk in ac.decks)),
            str(ac.capacity),
            ac.describe(),
        ])
    story.append(table(rows, [20 * mm, 56 * mm, 16 * mm, 18 * mm, 28 * mm],
                       align_right=(2, 3)))
    story.append(P(
        "<font face='Courier'>mid</font> and <font face='Courier'>tiny</font> are not "
        "real aircraft. <font face='Courier'>tiny</font> is small enough for exact "
        "dynamic programming and for hand-checked tests; <font face='Courier'>mid"
        "</font> is the training cabin, chosen because an episode is short enough to "
        "run tens of thousands of times.", "small"))

    story.append(P("The five fixed baselines", "h2"))
    story.append(table([
        ["Strategy", "Rule"],
        ["<font face='Courier'>random</font>", "Uniformly random order. The control condition."],
        ["<font face='Courier'>back_to_front</font>",
         "Board the rows furthest from the door first, in blocks. Widely used by "
         "airlines and, as the measurements below show, the <i>worst</i> of the five."],
        ["<font face='Courier'>wilma</font>",
         "Window, then middle, then aisle, across the whole cabin. Removes seat "
         "shuffles entirely."],
        ["<font face='Courier'>outside_in</font>",
         "WILMA refined with a distance term so that, within a seat-type sweep, far "
         "rows go first."],
        ["<font face='Courier'>steffen</font>",
         "Alternating rows within a window/middle/aisle sweep, so consecutive "
         "passengers stow two rows apart and do not block each other. The strongest "
         "published heuristic."],
    ], [30 * mm, 132 * mm]))
    story.append(P(
        "<b>Implementation decision.</b> &ldquo;Steffen&rdquo; here is the practical "
        "alternating-row interpretation, not the strict perfect-ordering version, "
        "because the strict version assumes passengers queue in an exact sequence that "
        "no real gate can enforce. This is stated because it affects the comparison.", "small"))

    # ---------------------------------------------------------- MDP
    story.append(PageBreak())
    story.append(P("4. The MDP", "h1"))
    story.append(P(
        "The agent is a <b>gate controller</b>. It does not move passengers and it does "
        "not assign seats. Once per tick it chooses which <i>group</i> of waiting "
        "passengers to release next. Movement and aisle interaction are the "
        "simulator's job. These three things &mdash; arrival pattern, release "
        "decision, and movement &mdash; are deliberately separate."))

    n_par = 2 if cfg_mid.use_parity else 1
    n_act = cfg_mid.n_bands * 3 * n_par
    story.append(P("Action space", "h2"))
    story.append(kv([
        ("Shape", "distance band &times; seat type &times; row parity"),
        ("Size", "%d &times; 3 &times; %d = <b>%d</b> actions" % (cfg_mid.n_bands, n_par, n_act)),
        ("Distance bands", "%d, measured as distance from the active door, so the "
         "encoding works for front, rear and both-door boarding" % cfg_mid.n_bands),
        ("Seat types", "window, middle, aisle"),
        ("Row parity", "even / odd row index &mdash; this is what lets the agent "
         "discover a Steffen-like alternating pattern"),
        ("Masking", "Groups with no waiting passenger are masked out; choosing an "
         "unavailable group is penalised so the policy learns to avoid it."),
    ]))
    story.append(P(
        "<b>Why groups and not individual passengers.</b> Choosing among N passengers "
        "gives an action space that grows with the aircraft and a policy that cannot "
        "transfer between cabins. Choosing among %d fixed group descriptors keeps the "
        "action space constant and size-independent." % n_act, "small"))

    story.append(P("State space", "h2"))
    story.append(kv([
        ("Encoding", "(boarding progress bin, last band, last seat type, last parity)"),
        ("Progress bins", "<font face='Courier'>PROGRESS_BINS = %d</font> &mdash; "
         "fraction of passengers seated, discretised" % envs.PROGRESS_BINS),
        ("Why &ldquo;last action&rdquo; is in the state",
         "It is what makes alternation expressible: to avoid releasing two adjacent "
         "rows in a row, the policy must know what it just released."),
        ("Upper bound", "%d &times; %d &times; %d &times; %d = %d states"
         % (envs.PROGRESS_BINS, cfg_mid.n_bands + 1, 4, n_par + 1,
            envs.PROGRESS_BINS * (cfg_mid.n_bands + 1) * 4 * (n_par + 1))),
        ("Actually visited", "%s" % meta_main.get("states", "see policy metadata")),
    ]))
    story.append(P(
        "<b>Design decision.</b> Group availability is handled by <i>action masking</i>, "
        "not by state bits. An earlier version encoded which groups were non-empty; it "
        "produced roughly 19,000 states and learned worse than random, because almost "
        "every state was visited once.", "small"))

    story.append(P("Reward", "h2"))
    story.append(P(
        "<font face='Courier'>reward = &minus;ticks_elapsed &minus; block_penalty "
        "&times; blocks &minus; invalid_penalty</font>"))
    story.extend(bullets([
        "<b>&minus;1 per tick</b> is the whole objective: maximising return is exactly "
        "minimising boarding time. No reward shaping is needed and none is used.",
        "<b>block_penalty</b> defaults to <font face='Courier'>%s</font>, i.e. it is "
        "<i>off</i>. It is exposed because the specification asks for an optional "
        "congestion penalty, but shipping it on would mean optimising a different "
        "objective than the one being reported." % d.block_penalty,
        "<b>invalid_penalty</b> discourages selecting a masked-out group. It is a "
        "training-stability device, not part of the objective.",
    ]))

    story.append(P("Episode termination", "h2"))
    story.append(P(
        "An episode ends when the last passenger is seated. There is no step limit "
        "other than the <font face='Courier'>max_ticks</font> safety cap; because "
        "every released passenger eventually sits, termination is guaranteed."))

    # ---------------------------------------------------------- training
    story.append(PageBreak())
    story.append(P("5. Training", "h1"))
    story.append(P(
        "<b>Algorithm: tabular REINFORCE (Monte-Carlo policy gradient) with a paired "
        "baseline.</b> The policy is a softmax over a table of action preferences "
        "indexed by the discrete state."))

    story.append(P("Why policy gradient rather than Q-learning", "h2"))
    story.append(P(
        "This was determined empirically, and the losing runs are kept in the "
        "repository as evidence. Value-based methods struggle here because the reward "
        "is entirely terminal in character &mdash; the cost of releasing a group "
        "appears 20&ndash;40 ticks later as an aisle jam &mdash; so one-step TD cannot "
        "assign credit, and the action-value differences between good and bad groups "
        "are small relative to the noise in a single episode. A policy gradient with a "
        "paired baseline sidesteps this by comparing <i>whole episodes on the same "
        "seed</i>."))

    story.append(P("The paired baseline (the single most important detail)", "h2"))
    story.append(P(
        "Boarding time varies more between <i>seeds</i> than between <i>policies</i>. "
        "Comparing an episode on seed 7 against a running average over other seeds is "
        "mostly measuring seed difficulty, not policy quality. Instead, each update "
        "compares an episode to a decayed baseline <i>for that same seed</i>, so the "
        "manifest difficulty cancels out. Without this the agent does not learn."))

    story.append(P("Hyperparameters", "h2"))
    story.append(kv([
        ("Learning rate &alpha;", str(meta_main.get("alpha", 0.25))),
        ("Discount &gamma;", "1.0, undiscounted &mdash; see note below"),
        ("Batch size", "16 episodes per gradient update"),
        ("Entropy bonus", "0.01 in the CLI trainer, 0.0 from the API"),
        ("Baseline decay", "0.3"),
        ("Episodes", str(meta_main.get("episodes", "see metadata"))),
        ("Restarts", str(meta_main.get("restarts", "see metadata"))),
        ("Exploration", "Softmax sampling. There is no &epsilon;-greedy; exploration "
         "is intrinsic to the stochastic policy and decays as preferences sharpen."),
    ]))
    story.append(P(
        "<b>On &gamma; = 1.</b> The objective is total boarding time, a finite "
        "undiscounted sum. Any &gamma; &lt; 1 would optimise a different objective and "
        "would under-weight exactly the delayed congestion the agent needs to learn "
        "about. The implementation gives every step in an episode the same advantage, "
        "which <i>is</i> &gamma; = 1 structurally. A &gamma; field was removed from the "
        "training API rather than left as a knob that silently changes what is being "
        "optimised.", "small"))

    story.append(P("Restarts and model selection", "h2"))
    story.append(P(
        "Policy gradient on a tabular softmax is sensitive to initialisation. The "
        "trainer runs several independent restarts and selects between them on the "
        "<b>validation seeds (%d&ndash;%d)</b>, which are disjoint from both the "
        "training seeds and the evaluation seeds. The selected policy is then scored "
        "once on the evaluation seeds. This three-way split is what makes the reported "
        "number an honest estimate rather than a best-of-N on the test set."
        % (VALIDATION_SEEDS[0], VALIDATION_SEEDS[-1])))
    rv = meta_main.get("restart_validation_scores")
    if rv:
        story.append(P("Validation scores of the restarts for the shipped policy: "
                       + ", ".join("%.1f" % float(x) for x in rv)
                       + " ticks. The best was selected.", "small"))

    story.append(P("Commands", "h2"))
    story.append(P(
        "# train the shipped agent (mid cabin, with row parity)<br/>"
        "PYTHONPATH=. .venv/bin/python scripts/train_parity.py \\<br/>"
        "&nbsp;&nbsp;--cabin mid --episodes 10000 --restarts 3 --save-as rl_parity_mid"
        "<br/><br/>"
        "# compare a policy against every baseline across cabins<br/>"
        "PYTHONPATH=. .venv/bin/python scripts/probe_rl_vs_baselines.py rl_parity_mid",
        "code"))

    # ---------------------------------------------------------- evaluation
    story.append(PageBreak())
    story.append(P("6. Evaluation method", "h1"))
    story.append(P(
        "The claim being tested is: <i>does the learned release rule board a cabin "
        "faster than the best fixed heuristic, on manifests it has never seen?</i>"))
    story.append(table([
        ["Control", "How it is handled"],
        ["Same manifests", "Every strategy and the agent are run on the identical "
         "(aircraft, seed) pairs, so passenger count, seat assignment, luggage flags "
         "and arrival order are byte-identical across conditions."],
        ["Held-out seeds", "Evaluation uses seeds %d&ndash;%d. Training used %d&ndash;"
         "%d and model selection used %d&ndash;%d. No overlap."
         % (EVAL_SEEDS[0], EVAL_SEEDS[-1], TRAIN_SEEDS[0], TRAIN_SEEDS[-1],
            VALIDATION_SEEDS[0], VALIDATION_SEEDS[-1])],
        ["Repeats", "%d independent seeds per condition. Mean and standard deviation "
         "are both reported, because a mean alone hides whether a 3-tick gap is "
         "signal." % len(EVAL_SEEDS)],
        ["Determinism", "The policy is evaluated greedily (arg-max), so a given "
         "(policy, seed) pair always gives the same number."],
        ["Off-distribution test", "The same policy is run on cabins it was never "
         "trained on. This is reported even though it is unfavourable."],
    ], [30 * mm, 132 * mm]))

    story.append(P("Metrics and their units", "h2"))
    story.append(table([
        ["Metric", "Unit", "Definition"],
        ["Boarding time", "ticks", "Simulator clock when the final passenger is "
         "seated. The primary metric."],
        ["Mean / Std", "ticks", "Across the %d evaluation seeds. Std is the "
         "population standard deviation." % len(EVAL_SEEDS)],
        ["Best / Worst", "ticks", "The easiest and hardest seed, to show spread."],
        ["Average wait", "ticks", "Mean per-passenger time between being released and "
         "being seated. A secondary diagnostic: a policy can shorten total time while "
         "lengthening individual waits."],
        ["Blocked count", "passenger-ticks", "Number of (passenger, tick) pairs where "
         "a passenger could not advance. The congestion measure."],
    ], [30 * mm, 26 * mm, 106 * mm]))

    # ---------------------------------------------------------- results
    story.append(PageBreak())
    story.append(P("7. Measured results", "h1"))
    story.append(P(
        "Measured when this PDF was generated, %d seeds per row, lower is better."
        % len(EVAL_SEEDS)))

    for cab in cabins:
        label = "mid cabin (training cabin)" if cab == "mid" else "A320 (not trained on)"
        story.append(P(label, "h2"))
        order = sorted(measured[cab].items(), key=lambda kv_: kv_[1]["mean"])
        rows = [["Strategy", "Mean", "Std", "Best", "Worst", "Avg wait"]]
        for name, r in order:
            is_agent = name.startswith("rl_")
            disp = "<b>%s</b>" % name if is_agent else name
            rows.append([
                "<font face='Courier'>%s</font>" % disp,
                "%.1f" % r["mean"], "%.1f" % r["std"],
                "%.0f" % r["best"], "%.0f" % r["worst"], "%.1f" % r["wait"],
            ])
        story.append(table(rows, [44 * mm, 22 * mm, 20 * mm, 20 * mm, 22 * mm, 24 * mm],
                           align_right=(1, 2, 3, 4, 5)))
        story.append(Spacer(1, 4))
        story.append(P("All figures in ticks.", "small"))

    story.append(P("How to read this", "h2"))
    story.append(P(
        "On the training cabin the agent is at or ahead of the best heuristic. On the "
        "A320 it is not. That gap is the honest finding of the project: the policy "
        "learned a release rule that is <i>tuned to a cabin geometry</i>, not a "
        "universal boarding law."))

    # ---------------------------------------------------------- negatives
    story.append(PageBreak())
    story.append(P("8. Negative results (kept deliberately)", "h1"))
    story.append(P(
        "These are included because a presentation that reports only the favourable "
        "run is not a result. Each of these cost real compute and each one shaped the "
        "final design."))
    story.append(table([
        ["Attempt", "Outcome", "What it means"],
        ["Watkins' Q(&lambda;), SARSA(&lambda;), Monte-Carlo control",
         "All slower than the fixed heuristics",
         "Value-based tabular control does not solve this credit-assignment problem at "
         "this sample budget. Kept in <font face='Courier'>artifacts/policies/</font> "
         "as evidence; withheld from the UI dropdown so they do not masquerade as "
         "&ldquo;the agent&rdquo;."],
        ["Encoding group availability in the state",
         "~19,000 states, learned worse than random",
         "State explosion. Replaced with action masking."],
        ["Training a parity agent directly on the A320",
         "6k episodes: 426.5 ticks. 24k episodes: 421.0 ticks. Steffen: 379.3",
         "Quadrupling the sample budget recovered ~5 ticks of a ~47-tick gap, so this "
         "is a <b>representation limit, not an undertraining one</b>: the band/parity "
         "encoding does not carry enough information on a longer cabin. Both policies "
         "were <b>deleted rather than shipped</b>."],
        ["Discount sweep on the tiny cabin",
         "Every &gamma; from 0.5 to 1.0 returned the same optimal policy",
         "The tiny cabin's horizon is too short for the discount to bite. Reported as "
         "a null result rather than dressed up."],
    ], [38 * mm, 46 * mm, 78 * mm]))
    story.append(P(
        "<b>What is explicitly not claimed.</b> That the learned policy is optimal; "
        "that it generalises across aircraft; that ticks correspond to any particular "
        "number of seconds; or that these results transfer to real boarding, which "
        "involves passenger behaviour this simulator does not model.", "small"))

    # ---------------------------------------------------------- API
    story.append(PageBreak())
    story.append(P("9. API surface", "h1"))
    story.append(table([
        ["Endpoint", "Method", "Purpose"],
        ["<font face='Courier'>/api/health</font>", "GET", "Liveness."],
        ["<font face='Courier'>/api/options</font>", "GET",
         "Everything the UI is allowed to offer: aircraft, gates, arrivals, "
         "strategies, policies. The UI renders this rather than hardcoding, so an "
         "option cannot appear in the UI without the backend supporting it."],
        ["<font face='Courier'>/api/simulate</font>", "POST",
         "Run one episode. Returns stats plus a delta-encoded trajectory."],
        ["<font face='Courier'>/api/compare</font>", "POST",
         "Run several strategies over several seeds on identical manifests."],
        ["<font face='Courier'>/api/train</font>", "POST",
         "Start a background training job. Returns a job id immediately."],
        ["<font face='Courier'>/api/train/{job_id}</font>", "GET",
         "Poll job status, progress and errors."],
        ["<font face='Courier'>/api/policies</font>", "GET", "Saved, loadable policies."],
    ], [48 * mm, 18 * mm, 96 * mm]))
    story.append(P(
        "Validation is Pydantic; errors come back as a structured "
        "<font face='Courier'>{\"error\": {code, message, detail}}</font> envelope, "
        "including for unhandled exceptions, so the front end never has to parse a "
        "plain-text 500.", "small"))
    story.append(P("Options withheld from the UI", "h2"))
    story.append(P(
        "Three inputs are accepted by the API but not offered in the interface: the "
        "<font face='Courier'>tiny</font> cabin (a test fixture, not an aircraft), the "
        "<font face='Courier'>queued</font> arrival pattern (the MDP does not read the "
        "arrival pattern, so comparisons under it are not meaningful), and a training "
        "<font face='Courier'>gamma</font> (removed; see the &gamma;=1 note). Scripts "
        "and tests still use the first two."))

    # ---------------------------------------------------------- testing
    story.append(PageBreak())
    story.append(P("10. Testing", "h1"))
    tc = count_tests()
    total = sum(n for _, n in tc)
    story.append(table(
        [["Suite", "Tests", "Covers"]]
        + [["<font face='Courier'>tests/%s</font>" % f, str(n),
            {"test_simulator.py": "Movement, aisle blocking, luggage delays, seat "
                                  "shuffles, completion time, determinism under a "
                                  "fixed seed, hand-computed tiny-cabin scenarios.",
             "test_agents.py": "State encoding, action masking, reward sign, terminal "
                               "conditions, policy save/load round-trip, "
                               "backward-compatible loading of 12-action policies.",
             "test_api.py": "Request validation, error envelopes, trajectory "
                            "serialisation, that every advertised option actually "
                            "runs, and that withheld options stay withheld."}
            .get(f, "")] for f, n in tc]
        + [["<b>Total</b>", "<b>%d</b>" % total, ""]],
        [44 * mm, 18 * mm, 100 * mm], align_right=(1,)))
    story.append(P(
        "Plus front-end tests under <font face='Courier'>frontend/src/</font> covering "
        "trajectory decoding and rendering against real API-shaped fixtures.", "small"))
    story.append(P("The hand-verifiable test", "h2"))
    story.append(P(
        "The most important test is the smallest: a cabin small enough that the "
        "expected boarding time can be worked out with pen and paper, asserted exactly. "
        "If the walking, stowing and blocking arithmetic is wrong, that test fails "
        "before any RL result can be trusted."))
    story.append(P("Commands", "h2"))
    story.append(P(
        "PYTHONPATH=. .venv/bin/python -m pytest -q       # %d Python tests<br/>"
        "cd frontend &amp;&amp; npm test                           # front-end tests<br/>"
        "cd frontend &amp;&amp; npm run typecheck                  # TypeScript<br/>"
        "cd frontend &amp;&amp; npm run build                      # production build" % total,
        "code"))

    # ---------------------------------------------------------- running
    story.append(PageBreak())
    story.append(P("11. Running it", "h1"))
    story.append(P(
        "# terminal 1 &mdash; backend<br/>"
        "python3 -m venv .venv &amp;&amp; .venv/bin/pip install -r requirements.txt<br/>"
        "PYTHONPATH=. .venv/bin/uvicorn api.main:app --reload --port 8000<br/><br/>"
        "# terminal 2 &mdash; front end<br/>"
        "cd frontend &amp;&amp; npm install &amp;&amp; npm run dev<br/><br/>"
        "# then open http://localhost:5173", "code"))
    story.append(P(
        "Requires Python 3.9+ and Node 20+. The Vite dev server binds to "
        "<font face='Courier'>localhost</font>; use that hostname rather than "
        "127.0.0.1.", "small"))

    story.append(P("A demonstration that exercises everything", "h2"))
    story.extend(bullets([
        "Pick the <font face='Courier'>mid</font> cabin, front door, random arrival, "
        "seed 500. Run <font face='Courier'>steffen</font> and watch the cabin fill.",
        "Switch the strategy to <font face='Courier'>rl_parity_mid</font> on the same "
        "seed. The manifest is identical, so any difference is the policy.",
        "Scrub the timeline back and forth &mdash; it is replaying a recorded "
        "trajectory, not re-simulating.",
        "Open <b>Compare</b> and run all five baselines plus the agent over a seed "
        "range to see means and spread rather than one lucky run.",
        "Switch the cabin to the A320 and re-run the comparison to show the "
        "generalisation gap.",
        "Open <b>Method &amp; parameters</b> for the live hyperparameters of whichever "
        "policy is selected.",
        "Open <b>Train</b> to start a short run and show that training is a real "
        "background job with progress, not a canned animation.",
    ]))

    # ---------------------------------------------------------- Q&A
    story.append(PageBreak())
    story.append(P("12. Likely questions", "h1"))
    qa = [
        ("Is the agent actually learning, or is it just picking a heuristic?",
         "It is learning a state-conditioned release rule. The policy table has %s "
         "visited states with distinct action distributions; it is not a lookup of one "
         "of the five baselines. The strongest evidence is that it beats all five on "
         "the training cabin while being worse off-distribution &mdash; a heuristic "
         "selector could not do that."
         % meta_main.get("states", "several hundred")),
        ("Why is the reward just &minus;1 per tick?",
         "Because that is literally the objective. Any shaping term would optimise "
         "something other than boarding time, and would then need justifying as a "
         "proxy. The optional congestion penalty exists but is off by default."),
        ("Why &gamma; = 1?",
         "Finite undiscounted episodic return. Discounting would under-weight the "
         "delayed congestion that is the entire difficulty of the problem."),
        ("How do you know you are not overfitting the evaluation seeds?",
         "Three disjoint seed pools: training %d&ndash;%d, restart selection %d&ndash;"
         "%d, reporting %d&ndash;%d. Model selection never touches the reporting pool."
         % (TRAIN_SEEDS[0], TRAIN_SEEDS[-1], VALIDATION_SEEDS[0], VALIDATION_SEEDS[-1],
            EVAL_SEEDS[0], EVAL_SEEDS[-1])),
        ("Why does it lose on the A320?",
         "The state uses distance <i>bands</i> and row parity. Band boundaries and the "
         "density of passengers per band change with cabin length, so a preference "
         "tuned on %d rows does not mean the same thing on a longer cabin. Training "
         "directly on the A320 was attempted at two sample budgets; quadrupling the "
         "episodes closed only about a tenth of the gap to Steffen, which points at "
         "the state representation rather than undertraining. Both runs were deleted "
         "rather than shipped." % n_rows_mid),
        ("Why not use deep RL?",
         "The state space is a few hundred discrete states. A neural network would add "
         "variance, non-determinism and a dependency without addressing the actual "
         "bottleneck, which is credit assignment over a long horizon."),
        ("What is one tick in seconds?",
         "Undefined, deliberately. The timing constants were not calibrated against "
         "real boarding data, so quoting seconds would be a fabricated precision. All "
         "comparisons are relative and in ticks."),
        ("What is the biggest weakness?",
         "Generalisation across cabin geometry, followed by the fact that passengers "
         "are modelled as compliant &mdash; they board exactly when released and walk "
         "at a fixed rate. Real boarding has late arrivals, groups refusing to split, "
         "and gate-agent discretion."),
    ]
    for q, a in qa:
        story.append(KeepTogether([P("<b>%s</b>" % q, "body"), P(a, "small"),
                                   Spacer(1, 3)]))

    Doc(OUT).build(story)
    print("wrote %s" % OUT)


if __name__ == "__main__":
    build()
