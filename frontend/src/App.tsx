import { useCallback, useEffect, useMemo, useState } from "react";
import { ApiError, api } from "./api/client";
import type {
  OptionsResponse, SimConfig, SimulateResponse, StrategyName,
} from "./api/types";
import { CabinLegend, CabinPlan } from "./components/CabinPlan";
import { ComparePanel } from "./components/ComparePanel";
import { CongestionTrace } from "./components/CongestionTrace";
import { ConfigPanel } from "./components/ConfigPanel";
import { ErrorNote } from "./components/ErrorNote";
import { MetricsStrip } from "./components/MetricsStrip";
import { MethodPanel } from "./components/MethodPanel";
import { PlaybackBar } from "./components/PlaybackBar";
import { TrainPanel } from "./components/TrainPanel";
import { usePlayback } from "./hooks/usePlayback";
import { congestionSeries, expandFrames } from "./lib/trajectory";

const INITIAL: SimConfig = {
  aircraft: "a320",
  n_passengers: 120,
  gate: "front",
  arrival: "random",
  luggage_fraction: 0.6,
  stow_ticks: 6,
  no_bag_ticks: 2,
  shuffle_ticks: 4,
  walk_ticks_per_row: 1,
  seed: 42,
};

type Tab = "compare" | "train" | "method";

export default function App() {
  const [options, setOptions] = useState<OptionsResponse | null>(null);
  const [bootError, setBootError] = useState<ApiError | null>(null);
  const [config, setConfig] = useState<SimConfig>(INITIAL);
  const [strategy, setStrategy] = useState<StrategyName>("wilma");
  const [policy, setPolicy] = useState<string | null>(null);
  const [sim, setSim] = useState<SimulateResponse | null>(null);
  const [runError, setRunError] = useState<ApiError | null>(null);
  const [busy, setBusy] = useState(false);
  const [highlight, setHighlight] = useState<number | null>(null);
  const [tab, setTab] = useState<Tab>("compare");

  const loadOptions = useCallback(async () => {
    setBootError(null);
    try {
      const o = await api.options();
      setOptions(o);
      // Default to the strongest trained agent. The list also contains
      // reference policies that are deliberately weaker, and leaving the
      // choice blank invited picking one of those and concluding the RL
      // approach had failed.
      setPolicy((cur) => cur ?? o.policies.find((p) => p.kind === "agent")?.name ?? null);
    } catch (e) {
      setBootError(e as ApiError);
    }
  }, []);

  useEffect(() => { void loadOptions(); }, [loadOptions]);

  const frames = useMemo(
    () => (sim ? expandFrames(sim.frames, sim.passengers.length) : []),
    [sim],
  );
  const series = useMemo(() => (sim ? congestionSeries(sim) : []), [sim]);
  const pb = usePlayback(frames.length);

  const run = useCallback(async () => {
    setBusy(true); setRunError(null);
    try {
      const r = await api.simulate(config, strategy, policy);
      setSim(r);
    } catch (e) {
      setRunError(e as ApiError); setSim(null);
    } finally {
      setBusy(false);
    }
  }, [config, strategy, policy]);

  // Space toggles playback, which is the one shortcut worth having here.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const el = e.target as HTMLElement;
      if (/^(INPUT|SELECT|TEXTAREA|BUTTON)$/.test(el.tagName)) return;
      if (e.code === "Space") { e.preventDefault(); pb.toggle(); }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [pb]);

  if (bootError) {
    return (
      <main className="min-h-screen grid place-items-center p-6">
        <div className="max-w-md w-full space-y-3">
          <h1 className="font-display text-lg font-semibold">
            Learning to Board
          </h1>
          <ErrorNote error={bootError} onRetry={() => void loadOptions()} />
          <p className="text-sm text-chart-muted">
            Start the API with{" "}
            <code className="font-mono bg-white border border-chart-line px-1">
              uvicorn api.main:app --port 8000
            </code>{" "}
            from the project root.
          </p>
        </div>
      </main>
    );
  }

  if (!options) {
    return (
      <main className="min-h-screen grid place-items-center">
        <p className="chart-label animate-pulse">Contacting simulator…</p>
      </main>
    );
  }

  return (
    <div className="min-h-screen flex flex-col">
      <header className="border-b border-chart-rule bg-chart-panel">
        <div className="max-w-[1600px] mx-auto px-5 py-3
                        flex flex-wrap items-baseline gap-x-4 gap-y-1">
          <h1 className="font-display text-base font-bold tracking-tight">
            LEARNING TO BOARD
          </h1>
          <p className="text-sm text-chart-muted">
            Reinforcement learning for aircraft boarding
          </p>
          <span className="ml-auto chart-label">
            All times in simulator ticks
          </span>
        </div>
      </header>

      <main className="flex-1 max-w-[1600px] w-full mx-auto px-5 py-5
                       grid gap-5 lg:grid-cols-[320px_minmax(0,1fr)]">
        <aside className="lg:sticky lg:top-5 lg:self-start space-y-4">
          <ConfigPanel
            options={options} config={config} strategy={strategy}
            policy={policy} busy={busy}
            onConfig={(patch) => setConfig((c) => ({ ...c, ...patch }))}
            onStrategy={setStrategy}
            onPolicy={setPolicy}
            onRun={() => void run()}
          />
          {runError && (
            <ErrorNote error={runError} onRetry={() => void run()} />
          )}
        </aside>

        <div className="space-y-5 min-w-0">
          {!sim && !busy && (
            <section className="panel p-10 text-center">
              <h2 className="font-display text-lg font-semibold">
                No simulation loaded
              </h2>
              <p className="mt-1.5 text-sm text-chart-muted max-w-md mx-auto">
                Configure a scenario and run it. The cabin below replays the
                Python simulator's recorded trajectory tick by tick — every
                passenger position comes from the engine.
              </p>
            </section>
          )}

          {busy && (
            <section className="panel p-10 text-center">
              <p className="chart-label animate-pulse">
                Running the simulator…
              </p>
            </section>
          )}

          {sim && (
            <>
              <MetricsStrip
                stats={sim.stats}
                now={series[pb.tick]}
                tick={pb.tick}
              />

              <section className="panel p-4 space-y-3">
                <header className="flex flex-wrap items-baseline
                                   justify-between gap-2">
                  <h2 className="font-display text-sm font-semibold
                                 uppercase tracking-chart">
                    Cabin — {sim.config.aircraft.toUpperCase()}
                  </h2>
                  <span className="chart-label">
                    {options.strategies.find(
                      (s) => s.value === sim.strategy)?.label}
                    {sim.policy ? ` · ${sim.policy}` : ""} · seed{" "}
                    {sim.stats.seed}
                  </span>
                </header>

                <div className="overflow-x-auto">
                  <div className="min-w-[640px]">
                    <CabinPlan
                      decks={sim.decks}
                      passengers={sim.passengers}
                      frame={frames[pb.tick]}
                      highlight={highlight}
                      onHighlight={setHighlight}
                    />
                  </div>
                </div>

                <CabinLegend />

                <CongestionTrace
                  series={series} tick={pb.tick} onSeek={pb.seek} />

                <PlaybackBar pb={pb} disabled={frames.length === 0} />
              </section>
            </>
          )}

          <nav className="flex gap-1 border-b border-chart-line"
            role="tablist" aria-label="Analysis">
            {(["compare", "train", "method"] as Tab[]).map((t) => (
              <button
                key={t} role="tab" aria-selected={tab === t}
                onClick={() => setTab(t)}
                className={
                  "px-3 py-2 text-sm font-medium border-b-2 -mb-px " +
                  (tab === t
                    ? "border-magenta text-chart-ink"
                    : "border-transparent text-chart-muted hover:text-chart-ink")
                }
              >
                {t === "compare" ? "Compare strategies"
                  : t === "train" ? "Reinforcement learning"
                    : "Method & parameters"}
              </button>
            ))}
          </nav>

          {tab === "compare" ? (
            <ComparePanel options={options} config={config} policy={policy} />
          ) : tab === "train" ? (
            <TrainPanel
              config={config}
              onTrained={(name) => {
                void loadOptions();
                if (name) { setPolicy(name); setStrategy("rl"); }
              }}
            />
          ) : (
            <MethodPanel policies={options.policies} selected={policy} />
          )}
        </div>
      </main>

      <footer className="border-t border-chart-line px-5 py-3">
        <p className="max-w-[1600px] mx-auto chart-label normal-case
                      tracking-normal">
          The Python simulator is the source of truth. The interface replays
          recorded trajectories and never invents simulation events.
        </p>
      </footer>
    </div>
  );
}
