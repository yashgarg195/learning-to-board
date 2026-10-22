import type { Delta, PassengerSpec, SimulateResponse } from "../api/types";
import { BLOCKED, QUEUED, SEATED, STOWING, WALKING } from "../api/types";

/** One passenger's reconstructed state at a given tick. */
export interface PassengerFrame {
  state: number;
  posRow: number;
}

/**
 * Expands the backend's delta-encoded frames into one full state array per
 * tick.
 *
 * The backend only sends passengers whose `(state, posRow)` changed, so the
 * frontend must accumulate. Doing this once up front (rather than during
 * playback) is what lets the scrubber jump to any tick instantly and lets the
 * same trajectory replay at any speed without re-running the simulation.
 */
export function expandFrames(
  frames: Delta[][],
  nPassengers: number,
): PassengerFrame[][] {
  const out: PassengerFrame[][] = [];
  let current: PassengerFrame[] = Array.from(
    { length: nPassengers },
    () => ({ state: QUEUED, posRow: -1 }),
  );
  for (const frame of frames) {
    // Copy-on-write per tick so earlier ticks are never mutated.
    current = current.slice();
    for (const [pid, state, posRow] of frame) {
      current[pid] = { state, posRow };
    }
    out.push(current);
  }
  return out;
}

export interface TickSummary {
  tick: number;
  queued: number;
  walking: number;
  stowing: number;
  blocked: number;
  seated: number;
}

/** Per-tick census for the strip chart, from the backend's own counts. */
export function congestionSeries(sim: SimulateResponse): TickSummary[] {
  const n = sim.passengers.length;
  return sim.congestion.map((c, i) => {
    const [walking, stowing, blocked, seated] = c;
    return {
      tick: i,
      walking,
      stowing,
      blocked,
      seated,
      queued: n - walking - stowing - blocked - seated,
    };
  });
}

export interface SeatOccupancy {
  seated: boolean;
}

/** Index passengers by their seat so the cabin can be drawn seat-first. */
export function seatIndex(
  passengers: PassengerSpec[],
): Map<string, number> {
  const m = new Map<string, number>();
  passengers.forEach((p, i) => m.set(`${p.d}:${p.r}:${p.c}`, i));
  return m;
}

export const STATE_LABEL: Record<number, string> = {
  [QUEUED]: "Waiting at gate",
  [WALKING]: "Walking the aisle",
  [STOWING]: "Stowing / shuffling",
  [BLOCKED]: "Blocked",
  [SEATED]: "Seated",
};

export const STATE_COLOR: Record<number, string> = {
  [QUEUED]: "#9FB0AE",
  [WALKING]: "#2E6DA4",
  [STOWING]: "#E0902F",
  [BLOCKED]: "#B4442E",
  [SEATED]: "#A8966D",
};

/** Ticks are the simulator's unit of time; this is the display convention. */
export function formatTicks(t: number): string {
  return `${Math.round(t)}`;
}
