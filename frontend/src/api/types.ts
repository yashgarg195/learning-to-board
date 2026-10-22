/** Types mirroring the FastAPI Pydantic schemas in `api/schemas.py`. */

export type AircraftName =
  | "tiny" | "mid" | "atr72" | "a320" | "b787" | "b777" | "a380";
export type Gate = "front" | "back" | "both";
export type Arrival = "random" | "queued" | "groups";
export type StrategyName =
  | "random" | "back_to_front" | "wilma" | "steffen" | "outside_in" | "rl";

export interface SimConfig {
  aircraft: AircraftName;
  n_passengers: number | null;
  gate: Gate;
  arrival: Arrival;
  luggage_fraction: number;
  stow_ticks: number;
  no_bag_ticks: number;
  shuffle_ticks: number;
  walk_ticks_per_row: number;
  seed: number;
}

export interface Deck {
  name: string;
  rows: number;
  cols: number;
  aisleCols: number[];
  blocks: number[];
  doors: number[];
}

export interface PassengerSpec {
  d: number;
  r: number;
  c: number;
  ac: number;
  t: "window" | "middle" | "aisle";
  bag: number;
  id: string;
}

export interface Stats {
  ticks: number;
  blocking_events: number;
  blocked_ticks: number;
  avg_wait: number;
  n_passengers: number;
  seed: number;
}

/** `[passengerIndex, state, posRow]`, only for passengers that changed. */
export type Delta = [number, number, number];

export interface SimulateResponse {
  decks: Deck[];
  passengers: PassengerSpec[];
  frames: Delta[][];
  congestion: number[][];
  stats: Stats;
  strategy: StrategyName;
  policy: string | null;
  config: SimConfig;
}

export interface CompareRow {
  strategy: StrategyName;
  label: string;
  mean_ticks: number;
  std_ticks: number;
  min_ticks: number;
  max_ticks: number;
  mean_blocked_ticks: number;
  mean_avg_wait: number;
  trials: number;
  per_seed: number[];
}

export interface CompareResponse {
  seeds: number[];
  rows: CompareRow[];
  config: SimConfig;
}

export interface TrainResult {
  saved_as: string | null;
  validation_ticks: number;
  eval_mean_ticks: number;
  baseline_ticks: Record<string, number>;
  states: number;
  per_restart: { restart: number; validation_ticks: number; states: number }[];
  greedy_curve: [number, number][];
  meta: Record<string, unknown>;
}

export interface TrainJob {
  job_id: string;
  status: "queued" | "running" | "done" | "error";
  created_at: number;
  episodes_done: number;
  episodes_total: number;
  message: string | null;
  curve: [number, number][];
  result: TrainResult | null;
}

export interface PolicyInfo {
  name: string;
  meta: Record<string, unknown>;
  /** Display name. `kind: "agent"` policies get a descriptive label. */
  label: string;
  /**
   * `agent`     — trained for use; what the demo should select.
   * `user`      — trained in this app.
   * `reference` — algorithm-comparison artifacts kept to show which algorithms fail
   *               on this problem. Deliberately slower than the heuristics.
   */
  kind: "agent" | "user" | "reference";
  note: string;
}

export interface OptionsResponse {
  aircraft: {
    value: AircraftName; label: string; decks: number;
    rows: number; seats: number; layout: string;
    /** "benchmark" cabins exist to evaluate the agent, not to model a type. */
    kind: "aircraft" | "benchmark";
  }[];
  gates: Gate[];
  arrivals: { value: Arrival; label: string; description: string }[];
  strategies: {
    value: StrategyName; label: string; description: string;
  }[];
  policies: PolicyInfo[];
  states: Record<string, number>;
}

/** Passenger states, mirrored from `boarding/config.py`. */
export const QUEUED = 0;
export const WALKING = 1;
export const STOWING = 2;
export const BLOCKED = 3;
export const SEATED = 4;
