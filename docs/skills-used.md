# Skills: what was found, what was used

An honest record. Nothing in this file claims a skill was used unless its file was
actually read from a verified local source and its guidance visibly changed the work.

## Governing constraint

This machine runs under an enterprise Copilot CLI policy, surfaced in-session:

> Use Copilot CLI responsibly: Only install approved/verified plugins, hooks, or MCP
> integrations. Do not install or run unreviewed third-party plugins/hooks/scripts
> from the internet.

Eight of the ten requested skills are community packages distributed through
third-party GitHub repositories and install scripts. Fetching and executing those
would violate the policy directly, so **they were not installed**. The request also
asked me to inspect installation scripts and prefer official sources before running
anything; under this policy the correct outcome is to decline, document, and move on.

## What is actually registered on this machine

Marketplaces configured in `~/.copilot/settings.json` and `~/.claude/plugins/`:

| Marketplace | Source | Trust |
| --- | --- | --- |
| `linkedin-security` | `linkedin-managed/.github-private` | Enterprise-approved |
| `linkedin-plugins` | `linkedin-multiproduct/li-productivity-agents` | Enterprise-approved |
| `claude-plugins-official` | `anthropics/claude-plugins-official` | **Official Anthropic** |

Locally available skills were enumerated with
`find ~/.claude/plugins/marketplaces -name SKILL.md` and `ls ~/.copilot/skills`.

## Per-skill outcome

| # | Requested | Found? | Action | Effect on this project |
| --- | --- | --- | --- | --- |
| 1 | **Taste** | No | Not installed — no official or enterprise-approved source. Name is ambiguous; several unrelated community repos use it. | None. Its intent (visual quality, design judgement) is covered by the official `frontend-design` skill below. |
| 2 | **Agent Skills** | Partially | Not a single installable package. This is Anthropic's open **Agent Skills** *format* (`SKILL.md` + frontmatter), which is how every skill here is already packaged. | Used as a **format**: skills were discovered and read via the `SKILL.md` convention. Nothing to install. |
| 3 | **Caveman** | No | Not installed — community skill, no verified source. | None. Its stated intent (terse output, less complexity) overlaps the CLI's existing brevity instructions, which were already in force. |
| 4 | **Graphify** | No | Not installed — community skill, no verified source. | None. Architecture understanding was done directly by reading the module graph; the result is the repository map in `README.md`. |
| 5 | **UI UX Pro Max** | No | Not installed — community skill, no verified source. | None directly. Interface design was driven by the official `frontend-design` skill. |
| 6 | **Ponytail** | No | Not identifiable with confidence. Searching the registered marketplaces returned nothing; the name does not map to a known official skill. | None. Documented rather than guessed at. |
| 7 | **Anthropic Skills** | **Yes** | Already registered locally as the `claude-plugins-official` marketplace (`anthropics/claude-plugins-official`). Read directly from disk; nothing was newly fetched or executed. | **Used — see below.** |
| 8 | **Karpathy Skills** | No | Not installed — community skill, no verified source. | None as a package. |
| 9 | **Matt Pocock Skills** | No | Not installed — community skill, no verified source. | None as a package. Its subject (TypeScript practice) is addressed directly: `strict` mode, no `any` in app code, discriminated unions for async state, and types generated to mirror the Pydantic schemas. |
| 10 | **Superpowers** | No | Not installed — community skill, no verified source. | None. Planning/verification structure comes from `IMPLEMENTATION_PLAN.md` and the test suite. |

**Score: 1 of 10 installed and genuinely used, 1 applied as a format, 8 declined on
policy grounds.**

## The skill that was actually used

### `frontend-design` — `anthropics/claude-plugins-official`

Read from
`~/.claude/plugins/marketplaces/claude-plugins-official/plugins/frontend-design/skills/frontend-design/SKILL.md`.

Its guidance changed concrete decisions:

1. **It names the three looks AI design clusters into** — cream/serif/terracotta;
   near-black with one acid accent; broadsheet hairline columns — and says not to
   spend a free axis on them. My first instinct was the near-black-plus-neon ops
   console, which is squarely default #2. **Rejected and replaced** with an
   aeronautical-sectional-chart direction (pale chart paper, airspace magenta, Class-B
   blue, terrain tan) drawn from the subject's own vernacular.
2. **"Ground it in the subject."** Seats, aisles and traffic are drawn with chart
   conventions rather than generic dashboard rectangles; the legend reads like a chart
   key.
3. **"Typography carries the personality."** Rejected Inter-everywhere. The pairing is
   Archivo (display) + IBM Plex Sans (body) + IBM Plex Mono (data/seat IDs), chosen for
   an instrument-panel feel.
4. **"Spend your boldness in one place."** The signature element is the live
   **congestion trace** beneath the cabin — a flight-data-recorder-style strip chart,
   time-synced to the scrubber, showing where in time the aisle jams. Everything else
   is deliberately quiet.
5. **"Structure is information."** No decorative `01 / 02 / 03` numbering. Row numbers
   and seat letters are real identifiers, and the trace's x-axis is real simulation
   time.
6. **"Avoid a page made entirely of metric cards."** The cabin and trace dominate the
   layout; metrics are a compact instrument strip, not a grid of tiles.
7. **Writing guidance** shaped the copy: active voice, consistent verbs between control
   and result ("Run" → "Running"), empty states written as invitations
   ("No policy trained yet — train one to compare it against the baselines"), and
   errors that state what failed and what to do.

## How to verify this record

```bash
# Marketplaces registered on this machine
cat ~/.copilot/settings.json | head -30
cat ~/.claude/plugins/known_marketplaces.json

# Every locally available skill
find ~/.claude/plugins/marketplaces -name SKILL.md
ls ~/.copilot/skills

# The one skill that influenced the design
cat ~/.claude/plugins/marketplaces/claude-plugins-official/plugins/frontend-design/skills/frontend-design/SKILL.md
```

If you want the eight declined skills installed, that is a decision for you to make
against your organisation's policy — tell me which sources you consider approved and
I will install them from those specific sources.
