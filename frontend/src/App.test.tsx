import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import App from "./App";
import type { OptionsResponse, SimulateResponse } from "./api/types";

const OPTIONS: OptionsResponse = {
  aircraft: [
    { value: "a320", label: "Airbus A320", decks: 1, rows: 30, seats: 180,
      layout: "3-3", kind: "aircraft" },
    { value: "mid", label: "Mid cabin", decks: 1, rows: 10, seats: 60,
      layout: "3-3", kind: "benchmark" },
  ],
  gates: ["front", "back", "both"],
  arrivals: [
    { value: "random", label: "Random", description: "Any order." },
    { value: "queued", label: "Queued", description: "Perfect compliance." },
    { value: "groups", label: "Travel parties", description: "Together." },
  ],
  strategies: [
    { value: "wilma", label: "WILMA", description: "Window, middle, aisle." },
    { value: "rl", label: "RL agent", description: "Trained policy." },
  ],
  policies: [],
  states: { QUEUED: 0, WALKING: 1, STOWING: 2, BLOCKED: 3, SEATED: 4 },
};

const SIM: SimulateResponse = {
  decks: [{ name: "main", rows: 2, cols: 3, aisleCols: [1], blocks: [1, 1],
    doors: [0] }],
  passengers: [
    { d: 0, r: 0, c: 0, ac: 1, t: "window", bag: 1, id: "1A" },
    { d: 0, r: 1, c: 2, ac: 1, t: "window", bag: 0, id: "2C" },
  ],
  frames: [[], [[0, 1, 0]], [[0, 4, 0], [1, 1, 0]], [[1, 4, 1]]],
  congestion: [[0, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 1], [0, 0, 0, 2]],
  stats: { ticks: 3, blocking_events: 1, blocked_ticks: 2, avg_wait: 1.5,
    n_passengers: 2, seed: 42 },
  strategy: "wilma",
  policy: null,
  config: {
    aircraft: "a320", n_passengers: 120, gate: "front", arrival: "random",
    luggage_fraction: 0.6, stow_ticks: 6, no_bag_ticks: 2, shuffle_ticks: 4,
    walk_ticks_per_row: 1, seed: 42,
  },
};

function mockFetch(handler: (url: string) => [number, unknown]) {
  vi.stubGlobal("fetch", vi.fn(async (url: string) => {
    const [status, body] = handler(String(url));
    return {
      ok: status >= 200 && status < 300,
      status,
      text: async () => JSON.stringify(body),
    } as Response;
  }));
}

afterEach(() => { vi.unstubAllGlobals(); });

describe("App", () => {
  it("shows an actionable error when the API is unreachable", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => { throw new Error("down"); }));
    render(<App />);
    expect(await screen.findByRole("alert")).toHaveTextContent(
      /Cannot reach the simulation API/i,
    );
    expect(screen.getByText(/uvicorn api.main:app/)).toBeInTheDocument();
  });

  it("renders options from the backend and an empty state before any run",
    async () => {
      mockFetch(() => [200, OPTIONS]);
      render(<App />);
      expect(await screen.findByText("No simulation loaded")).toBeInTheDocument();
      // Only backend-advertised aircraft appear.
      const select = screen.getByLabelText("Aircraft") as HTMLSelectElement;
      expect([...select.options].map((o) => o.value))
        .toEqual(["a320", "mid"]);
    });

  it("disables the RL strategy when no policy has been trained", async () => {
    mockFetch(() => [200, OPTIONS]);
    render(<App />);
    await screen.findByLabelText("Boarding strategy");
    const opt = screen.getByRole("option", { name: /RL agent/ });
    expect(opt).toBeDisabled();
    expect(opt).toHaveTextContent(/train one first/i);
  });

  it("runs a simulation and shows metrics from the response", async () => {
    mockFetch((url) =>
      url.includes("/api/simulate") ? [200, SIM] : [200, OPTIONS]);
    render(<App />);
    await userEvent.click(await screen.findByRole("button",
      { name: /Run simulation/i }));

    await waitFor(() =>
      expect(screen.getByText("Cabin — A320")).toBeInTheDocument());
    // Total boarding time comes straight from stats.ticks.
    expect(screen.getByText("3")).toBeInTheDocument();
    expect(screen.getByText("1.5")).toBeInTheDocument();
    expect(screen.getByRole("img",
      { name: /Aircraft cabin plan/i })).toBeInTheDocument();
  });

  it("scrubs the replay without re-calling the API", async () => {
    const fetchMock = vi.fn(async (url: string) => {
      const [status, body] = String(url).includes("/api/simulate")
        ? [200, SIM] : [200, OPTIONS];
      return {
        ok: true, status: status as number,
        text: async () => JSON.stringify(body),
      } as Response;
    });
    vi.stubGlobal("fetch", fetchMock);

    render(<App />);
    await userEvent.click(await screen.findByRole("button",
      { name: /Run simulation/i }));
    await screen.findByLabelText("Scrub to tick");
    const calls = fetchMock.mock.calls.length;

    const slider = screen.getByLabelText("Scrub to tick") as HTMLInputElement;
    expect(slider.max).toBe("3");
    await userEvent.click(screen.getByRole("button",
      { name: /Step forward one tick/i }));
    expect(screen.getByText(/tick 001/)).toBeInTheDocument();
    // Replay is local: no extra network traffic.
    expect(fetchMock.mock.calls.length).toBe(calls);
  });

  it("surfaces a structured backend validation error", async () => {
    mockFetch((url) =>
      url.includes("/api/simulate")
        ? [422, { error: { code: "invalid_request",
            message: "One or more configuration fields are invalid.",
            detail: [{ field: "config.seed", problem: "must be >= 0" }] } }]
        : [200, OPTIONS]);
    render(<App />);
    await userEvent.click(await screen.findByRole("button",
      { name: /Run simulation/i }));
    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent(/configuration fields are invalid/i);
    expect(alert).toHaveTextContent(/config.seed/);
  });
});
