import { useState } from "react";
import {
  Bar, BarChart, CartesianGrid, Cell, ErrorBar, ResponsiveContainer,
  Tooltip, XAxis, YAxis,
} from "recharts";
import { ApiError, api } from "../api/client";
import type {
  CompareResponse, OptionsResponse, SimConfig, StrategyName,
} from "../api/types";
import { ErrorNote } from "./ErrorNote";
import { Tip } from "./Tip";

const DEFAULT: StrategyName[] = ["random", "back_to_front", "wilma", "steffen"];

export function ComparePanel({ options, config, policy }: {
  options: OptionsResponse; config: SimConfig; policy: string | null;
}) {
  const [picked, setPicked] = useState<StrategyName[]>(DEFAULT);
  const [trials, setTrials] = useState(5);
  const [data, setData] = useState<CompareResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<ApiError | null>(null);

  const toggle = (s: StrategyName) =>
    setPicked((p) =>
      p.includes(s) ? p.filter((x) => x !== s) : [...p, s].slice(0, 6));

  const run = async () => {
    setBusy(true); setErr(null);
    try {
      setData(await api.compare(config, picked, trials, policy));
    } catch (e) {
      setErr(e as ApiError); setData(null);
    } finally {
      setBusy(false);
    }
  };

  const chart = data?.rows.map((r) => ({
    label: r.label,
    mean: Number(r.mean_ticks.toFixed(1)),
    err: Number(r.std_ticks.toFixed(1)),
    strategy: r.strategy,
  })) ?? [];

  return (
    <section className="panel p-4 space-y-4">
      <header className="flex flex-wrap items-baseline justify-between gap-2">
        <h2 className="font-display text-sm font-semibold uppercase tracking-chart">
          Strategy comparison
        </h2>
        <p className="chart-label normal-case tracking-normal">
          Same aircraft, same seed sequence, same luggage draw for every
          strategy.
        </p>
      </header>

      <div className="flex flex-wrap gap-1.5">
        {options.strategies.map((s) => {
          const on = picked.includes(s.value);
          const locked = s.value === "rl" && !policy;
          return (
            <button
              key={s.value}
              type="button"
              aria-pressed={on}
              disabled={locked}
              title={locked ? "Select a saved policy first" : s.description}
              onClick={() => toggle(s.value)}
              className={
                "btn px-2.5 py-1 text-xs " +
                (on
                  ? "bg-chart-ink border-chart-ink text-white"
                  : "bg-white border-chart-line text-chart-muted hover:border-chart-rule")
              }
            >
              {s.label}
            </button>
          );
        })}
      </div>

      <div className="flex flex-wrap items-end gap-3">
        <div className="w-40">
          <label className="field-label" htmlFor="trials">
            Trials per strategy{" "}
            <Tip text="Each trial uses a different seed, so the manifest and bag draw change. The bars show the mean and the error bar one standard deviation." />
          </label>
          <input
            id="trials" type="number" min={1} max={40} value={trials}
            className="control font-mono"
            onChange={(e) =>
              setTrials(Math.min(40, Math.max(1, Number(e.target.value) || 1)))}
          />
        </div>
        <button className="btn-primary" onClick={run}
          disabled={busy || picked.length === 0}>
          {busy ? "Running trials…" : "Compare"}
        </button>
      </div>

      {err && <ErrorNote error={err} />}

      {!data && !busy && !err && (
        <p className="text-sm text-chart-muted border border-dashed
                      border-chart-line p-4">
          Pick two or more strategies and run the comparison. Results are
          measured, not cached — nothing is shown until a run completes.
        </p>
      )}

      {busy && (
        <div className="h-[260px] border border-chart-line bg-white
                        flex items-center justify-center">
          <span className="chart-label animate-pulse">
            Simulating {picked.length * trials} boardings…
          </span>
        </div>
      )}

      {data && !busy && (
        <>
          <div className="h-[260px] bg-white border border-chart-line p-2">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chart}
                margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
                <CartesianGrid stroke="#E4E9E7" vertical={false} />
                <XAxis dataKey="label" tick={{ fontSize: 11, fill: "#5A6E74" }}
                  axisLine={{ stroke: "#CBD4D2" }} tickLine={false} />
                <YAxis tick={{ fontSize: 11, fill: "#5A6E74" }}
                  axisLine={false} tickLine={false}
                  label={{ value: "ticks", angle: -90, position: "insideLeft",
                    fontSize: 10, fill: "#5A6E74" }} />
                <Tooltip
                  cursor={{ fill: "#EDF0EE" }}
                  contentStyle={{
                    border: "1px solid #CBD4D2", borderRadius: 2,
                    fontSize: 12, fontFamily: "IBM Plex Sans",
                  }}
                  formatter={(v: number, k: string) =>
                    [`${v} ticks`, k === "mean" ? "Mean" : k]}
                />
                <Bar dataKey="mean" radius={[2, 2, 0, 0]}>
                  {chart.map((c) => (
                    <Cell key={c.strategy}
                      fill={c.strategy === "rl" ? "#C4347A" : "#2E6DA4"} />
                  ))}
                  <ErrorBar dataKey="err" stroke="#13242E" strokeWidth={1}
                    width={6} direction="y" />
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>

          <table className="w-full text-sm border-collapse">
            <caption className="sr-only">Strategy comparison results</caption>
            <thead>
              <tr className="border-b border-chart-rule">
                {["Strategy", "Mean", "Std", "Best", "Worst", "Avg wait"]
                  .map((h) => (
                    <th key={h} className="chart-label text-left py-1.5 pr-3">
                      {h}
                    </th>
                  ))}
              </tr>
            </thead>
            <tbody className="font-mono tabular-nums">
              {data.rows.map((r, i) => (
                <tr key={r.strategy}
                  className="border-b border-chart-line last:border-0">
                  <td className={"py-1.5 pr-3 font-sans " +
                    (i === 0 ? "font-semibold text-magenta" : "")}>
                    {r.label}
                  </td>
                  <td className="py-1.5 pr-3">{r.mean_ticks.toFixed(1)}</td>
                  <td className="py-1.5 pr-3 text-chart-muted">
                    ±{r.std_ticks.toFixed(1)}
                  </td>
                  <td className="py-1.5 pr-3">{r.min_ticks}</td>
                  <td className="py-1.5 pr-3">{r.max_ticks}</td>
                  <td className="py-1.5 pr-3">{r.mean_avg_wait.toFixed(1)}</td>
                </tr>
              ))}
            </tbody>
          </table>

          <p className="text-xs text-chart-muted leading-relaxed">
            Seeds {data.seeds[0]}–{data.seeds[data.seeds.length - 1]} ·{" "}
            {trials} trial{trials === 1 ? "" : "s"} each. A win over{" "}
            {trials} seeds on one aircraft is evidence, not proof of
            optimality — the spread between seeds is often larger than the gap
            between the top strategies.
          </p>
        </>
      )}
    </section>
  );
}
