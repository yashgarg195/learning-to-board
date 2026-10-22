import { useMemo } from "react";
import type { Deck, PassengerSpec } from "../api/types";
import { BLOCKED, QUEUED, SEATED, STOWING, WALKING } from "../api/types";
import type { PassengerFrame } from "../lib/trajectory";

const CELL = 19;
const GAP = 2;
const PAD = 16;
const QUEUE_W = 54;
const DECK_TOP = 26;   // headroom for the deck label above the door markers

interface Props {
  decks: Deck[];
  passengers: PassengerSpec[];
  frame: PassengerFrame[] | undefined;
  highlight: number | null;
  onHighlight: (i: number | null) => void;
}

function seatFill(state: number | undefined): string {
  switch (state) {
    case SEATED: return "#D8C9A8";
    case STOWING: return "#F7E6CE";
    default: return "#FFFFFF";
  }
}

function paxFill(state: number): string {
  switch (state) {
    case WALKING: return "#2E6DA4";
    case STOWING: return "#E0902F";
    case BLOCKED: return "#B4442E";
    case SEATED: return "#A8966D";
    default: return "#9FB0AE";
  }
}

/**
 * Cabin plan view: rows run left-to-right (nose at the left), seat columns
 * top-to-bottom. Every glyph is driven by the backend frame — nothing here
 * advances or invents a passenger position.
 */
export function CabinPlan({
  decks, passengers, frame, highlight, onHighlight,
}: Props) {
  const byDeck = useMemo(() => {
    const m: number[][] = decks.map(() => []);
    passengers.forEach((p, i) => m[p.d]?.push(i));
    return m;
  }, [decks, passengers]);

  const maxRows = Math.max(...decks.map((d) => d.rows));
  const width = PAD * 2 + QUEUE_W + maxRows * (CELL + GAP) + 10;
  const heights = decks.map((d) => d.cols * (CELL + GAP));
  const height = PAD * 2 + heights.reduce((a, b) => a + b, 0)
    + (decks.length - 1) * 26 + decks.length * DECK_TOP;

  let yCursor = PAD;

  return (
    <svg
      viewBox={`0 0 ${width} ${height}`}
      className="w-full h-auto"
      role="img"
      aria-label="Aircraft cabin plan showing seats and boarding passengers"
    >
      <defs>
        <pattern id="hatch" width="4" height="4" patternUnits="userSpaceOnUse"
          patternTransform="rotate(45)">
          <rect width="4" height="4" fill="#EDF0EE" />
          <line x1="0" y1="0" x2="0" y2="4" stroke="#CBD4D2" strokeWidth="1.5" />
        </pattern>
      </defs>

      {decks.map((deck, di) => {
        const top = yCursor;
        const cabinX = PAD + QUEUE_W;
        yCursor += DECK_TOP + heights[di] + 26;
        const aisle = new Set(deck.aisleCols);
        const indices = byDeck[di] ?? [];

        // Index seats by grid cell so a seat can find its occupant's state.
        const occupant = new Map<string, number>();
        indices.forEach((i) => {
          const p = passengers[i];
          occupant.set(`${p.r}:${p.c}`, i);
        });

        return (
          <g key={deck.name}>
            <text x={cabinX} y={top + 10} className="chart-label"
              fill="#5A6E74" fontSize="10" fontFamily="IBM Plex Mono"
              letterSpacing="1.2">
              {deck.name.toUpperCase()} DECK · {deck.rows} ROWS ·{" "}
              {deck.blocks.join("-")}
            </text>

            {/* Fuselage outline */}
            <rect
              x={cabinX - 6} y={top + DECK_TOP - 6}
              width={deck.rows * (CELL + GAP) + 12}
              height={heights[di] + 12}
              rx="10" fill="#FFFFFF" stroke="#9FB0AE" strokeWidth="1"
            />

            {/* Aisle lanes */}
            {deck.aisleCols.map((c) => (
              <rect
                key={`aisle-${c}`}
                x={cabinX} y={top + DECK_TOP + c * (CELL + GAP)}
                width={deck.rows * (CELL + GAP) - GAP} height={CELL}
                fill="url(#hatch)"
              />
            ))}

            {/* Doors */}
            {deck.doors.map((r) => (
              <g key={`door-${r}`}>
                <rect
                  x={cabinX + r * (CELL + GAP) - 1} y={top + DECK_TOP - 10}
                  width={CELL + 2} height="5" fill="#C4347A" rx="1"
                />
                <rect
                  x={cabinX + r * (CELL + GAP) - 1}
                  y={top + DECK_TOP + heights[di] - GAP + 5}
                  width={CELL + 2} height="5" fill="#C4347A" rx="1"
                />
              </g>
            ))}

            {/* Seats */}
            {Array.from({ length: deck.rows }).map((_, r) =>
              Array.from({ length: deck.cols }).map((__, c) => {
                if (aisle.has(c)) return null;
                const i = occupant.get(`${r}:${c}`);
                const st = i === undefined ? undefined : frame?.[i]?.state;
                const isHi = i !== undefined && i === highlight;
                return (
                  <rect
                    key={`s-${r}-${c}`}
                    x={cabinX + r * (CELL + GAP)}
                    y={top + DECK_TOP + c * (CELL + GAP)}
                    width={CELL} height={CELL} rx="2.5"
                    fill={i === undefined ? "#F1F3F2" : seatFill(st)}
                    stroke={isHi ? "#C4347A" : "#CBD4D2"}
                    strokeWidth={isHi ? 1.8 : 0.8}
                  />
                );
              }),
            )}

            {/* Passengers that are not yet seated */}
            {indices.map((i) => {
              const f = frame?.[i];
              if (!f || f.state === SEATED || f.state === QUEUED) return null;
              const p = passengers[i];
              const cx = cabinX + f.posRow * (CELL + GAP) + CELL / 2;
              const cy = top + DECK_TOP + p.ac * (CELL + GAP) + CELL / 2;
              if (f.posRow < 0) return null;
              return (
                <g key={`p-${i}`}
                  onMouseEnter={() => onHighlight(i)}
                  onMouseLeave={() => onHighlight(null)}>
                  <circle
                    cx={cx} cy={cy} r={CELL / 2 - 3.2}
                    fill={paxFill(f.state)}
                    stroke={highlight === i ? "#13242E" : "none"}
                    strokeWidth="1.4"
                  />
                  {p.bag === 1 && f.state === STOWING && (
                    <rect
                      x={cx - 2.2} y={cy - 7.5} width="4.4" height="3.4"
                      fill="#13242E" rx="0.6"
                    />
                  )}
                </g>
              );
            })}

            {/* Gate queue: everyone who has not entered the aircraft yet */}
            {di === 0 && (
              <QueueStack
                x={PAD} y={top + DECK_TOP} height={heights[di]}
                count={passengers.filter((_, i) =>
                  frame?.[i]?.state === QUEUED).length}
                total={passengers.length}
              />
            )}
          </g>
        );
      })}
    </svg>
  );
}

function QueueStack({ x, y, height, count, total }: {
  x: number; y: number; height: number; count: number; total: number;
}) {
  const maxDots = Math.max(1, Math.floor(height / 9));
  const shown = Math.min(count, maxDots);
  return (
    <g>
      <text x={x} y={y - 6} fontSize="9" fontFamily="IBM Plex Mono"
        fill="#5A6E74" letterSpacing="1">GATE</text>
      {Array.from({ length: shown }).map((_, i) => (
        <circle key={i} cx={x + 8 + (i % 4) * 9}
          cy={y + 6 + Math.floor(i / 4) * 9} r="3" fill="#9FB0AE" />
      ))}
      <text x={x} y={y + height + 12} fontSize="10"
        fontFamily="IBM Plex Mono" fill="#13242E">
        {count}/{total}
      </text>
    </g>
  );
}
