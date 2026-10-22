import type {
  Arrival, Gate, OptionsResponse, SimConfig, StrategyName,
} from "../api/types";
import { Tip } from "./Tip";

interface Props {
  options: OptionsResponse;
  config: SimConfig;
  strategy: StrategyName;
  policy: string | null;
  busy: boolean;
  onConfig: (patch: Partial<SimConfig>) => void;
  onStrategy: (s: StrategyName) => void;
  onPolicy: (p: string | null) => void;
  onRun: () => void;
}

export function ConfigPanel({
  options, config, strategy, policy, busy,
  onConfig, onStrategy, onPolicy, onRun,
}: Props) {
  const craft = options.aircraft.find((a) => a.value === config.aircraft);
  const maxSeats = craft?.seats ?? 1;
  const strat = options.strategies.find((s) => s.value === strategy);
  const noPolicies = options.policies.length === 0;

  return (
    <form
      className="panel p-4 space-y-4"
      onSubmit={(e) => { e.preventDefault(); onRun(); }}
    >
      <h2 className="font-display text-sm font-semibold uppercase tracking-chart">
        Scenario
      </h2>

      <div>
        <label className="field-label" htmlFor="aircraft">Aircraft</label>
        <select
          id="aircraft" className="control" value={config.aircraft}
          onChange={(e) => onConfig({
            aircraft: e.target.value as SimConfig["aircraft"],
            n_passengers: null,
          })}
        >
          {options.aircraft.map((a) => (
            <option key={a.value} value={a.value}>
              {a.label} — {a.seats} seats
            </option>
          ))}
        </select>
        {craft && (
          <p className="mt-1 chart-label normal-case tracking-normal">
            {craft.rows} rows · {craft.layout} ·{" "}
            {craft.decks > 1 ? `${craft.decks} decks` : "single deck"}
          </p>
        )}
      </div>

      <div>
        <label className="field-label" htmlFor="pax">
          Passengers
          <span className="normal-case tracking-normal text-chart-muted">
            {" "}· {config.n_passengers ?? maxSeats} of {maxSeats}
          </span>
        </label>
        <input
          id="pax" type="range" min={1} max={maxSeats} step={1}
          className="w-full accent-magenta"
          value={config.n_passengers ?? maxSeats}
          onChange={(e) => {
            const v = Number(e.target.value);
            onConfig({ n_passengers: v === maxSeats ? null : v });
          }}
        />
      </div>

      <div className="grid grid-cols-2 gap-3">
        <div>
          <label className="field-label" htmlFor="gate">
            Boarding door{" "}
            <Tip text="Which fuselage doors are open. 'Both' opens the front and rear doors and sends each passenger to the nearer one." />
          </label>
          <select id="gate" className="control" value={config.gate}
            onChange={(e) => onConfig({ gate: e.target.value as Gate })}>
            {options.gates.map((g) => (
              <option key={g} value={g}>
                {g === "front" ? "Front" : g === "back" ? "Rear" : "Front + rear"}
              </option>
            ))}
          </select>
        </div>
        <div>
          <label className="field-label" htmlFor="arrival">
            Arrival pattern{" "}
            <Tip text="How well passengers comply with the called order. This is separate from the boarding strategy itself." />
          </label>
          <select id="arrival" className="control" value={config.arrival}
            onChange={(e) => onConfig({ arrival: e.target.value as Arrival })}>
            {options.arrivals.map((a) => (
              <option key={a.value} value={a.value}>{a.label}</option>
            ))}
          </select>
        </div>
      </div>
      <p className="chart-label normal-case tracking-normal -mt-2">
        {options.arrivals.find((a) => a.value === config.arrival)?.description}
      </p>

      <div>
        <label className="field-label" htmlFor="bags">
          Passengers with carry-on ·{" "}
          {Math.round(config.luggage_fraction * 100)}%
        </label>
        <input
          id="bags" type="range" min={0} max={100} step={5}
          className="w-full accent-magenta"
          value={Math.round(config.luggage_fraction * 100)}
          onChange={(e) =>
            onConfig({ luggage_fraction: Number(e.target.value) / 100 })}
        />
      </div>

      <hr className="border-chart-line" />

      <div>
        <label className="field-label" htmlFor="strategy">
          Boarding strategy
        </label>
        <select
          id="strategy" className="control" value={strategy}
          onChange={(e) => onStrategy(e.target.value as StrategyName)}
        >
          {options.strategies.map((s) => (
            <option key={s.value} value={s.value}
              disabled={s.value === "rl" && noPolicies}>
              {s.label}
              {s.value === "rl" && noPolicies ? " — train one first" : ""}
            </option>
          ))}
        </select>
        {strat && (
          <p className="mt-1 chart-label normal-case tracking-normal">
            {strat.description}
          </p>
        )}
      </div>

      {strategy === "rl" && (
        <div>
          <label className="field-label" htmlFor="policy">Saved policy</label>
          <select
            id="policy" className="control" value={policy ?? ""}
            onChange={(e) => onPolicy(e.target.value || null)}
          >
            <option value="">Select a policy…</option>
            {options.policies.map((p) => (
              <option key={p.name} value={p.name}>{p.name}</option>
            ))}
          </select>
        </div>
      )}

      <div>
        <label className="field-label" htmlFor="seed">
          Random seed{" "}
          <Tip text="Fixes the passenger manifest, seat assignment and who carries a bag. The same seed always reproduces the same run." />
        </label>
        <input
          id="seed" type="number" min={0} className="control font-mono"
          value={config.seed}
          onChange={(e) =>
            onConfig({ seed: Math.max(0, Number(e.target.value) || 0) })}
        />
      </div>

      <button type="submit" className="btn-primary w-full" disabled={busy}>
        {busy ? "Running simulation…" : "Run simulation"}
      </button>
    </form>
  );
}
