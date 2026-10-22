import { describe, expect, it } from "vitest";
import { congestionSeries, expandFrames, seatIndex } from "./trajectory";
import type { Delta, SimulateResponse } from "../api/types";
import { QUEUED, SEATED, STOWING, WALKING } from "../api/types";

describe("expandFrames", () => {
  it("starts everyone queued off the aircraft", () => {
    const out = expandFrames([[]], 3);
    expect(out[0]).toEqual([
      { state: QUEUED, posRow: -1 },
      { state: QUEUED, posRow: -1 },
      { state: QUEUED, posRow: -1 },
    ]);
  });

  it("carries unchanged passengers forward between ticks", () => {
    const frames: Delta[][] = [
      [[0, WALKING, 0]],
      [[1, WALKING, 0]],
      [[0, SEATED, 2]],
    ];
    const out = expandFrames(frames, 2);
    // Passenger 0 keeps walking at tick 1 even though it sent no delta.
    expect(out[1][0]).toEqual({ state: WALKING, posRow: 0 });
    expect(out[1][1]).toEqual({ state: WALKING, posRow: 0 });
    expect(out[2][0]).toEqual({ state: SEATED, posRow: 2 });
    expect(out[2][1]).toEqual({ state: WALKING, posRow: 0 });
  });

  it("never mutates an earlier tick when a later one changes", () => {
    const frames: Delta[][] = [
      [[0, WALKING, 0]],
      [[0, STOWING, 1]],
    ];
    const out = expandFrames(frames, 1);
    expect(out[0][0].state).toBe(WALKING);
    expect(out[1][0].state).toBe(STOWING);
  });

  it("produces one full state array per tick", () => {
    const frames: Delta[][] = [[], [[0, WALKING, 0]], [[0, SEATED, 1]]];
    const out = expandFrames(frames, 4);
    expect(out).toHaveLength(3);
    expect(out.every((f) => f.length === 4)).toBe(true);
  });
});

describe("congestionSeries", () => {
  const sim = {
    passengers: new Array(10).fill({}),
    congestion: [
      [0, 0, 0, 0],
      [3, 1, 2, 1],
      [0, 0, 0, 10],
    ],
  } as unknown as SimulateResponse;

  it("derives the queued count as the remainder", () => {
    const s = congestionSeries(sim);
    expect(s[0]).toMatchObject({ tick: 0, queued: 10, seated: 0 });
    expect(s[1]).toMatchObject({ walking: 3, stowing: 1, blocked: 2, queued: 3 });
    expect(s[2]).toMatchObject({ queued: 0, seated: 10 });
  });

  it("keeps every band non-negative and summing to the manifest", () => {
    for (const s of congestionSeries(sim)) {
      const total = s.queued + s.walking + s.stowing + s.blocked + s.seated;
      expect(total).toBe(10);
      expect(s.queued).toBeGreaterThanOrEqual(0);
    }
  });
});

describe("seatIndex", () => {
  it("maps deck:row:col to the passenger index", () => {
    const m = seatIndex([
      { d: 0, r: 1, c: 2 },
      { d: 1, r: 0, c: 0 },
    ] as never);
    expect(m.get("0:1:2")).toBe(0);
    expect(m.get("1:0:0")).toBe(1);
    expect(m.get("0:0:0")).toBeUndefined();
  });
});
