import type { Stats } from "../api/types";
import type { TickSummary } from "../lib/trajectory";
import { Tip } from "./Tip";

interface Props {
  stats: Stats;
  now: TickSummary | undefined;
  tick: number;
}

function Metric({ label, value, unit, tip, accent }: {
  label: string; value: string; unit?: string; tip?: string; accent?: boolean;
}) {
  return (
    <div className="px-3 py-2 border-l border-chart-line first:border-l-0">
      <div className="chart-label flex items-center gap-1">
        {label} {tip && <Tip text={tip} />}
      </div>
      <div className="flex items-baseline gap-1">
        <span className={
          "font-display text-xl font-semibold tabular-nums " +
          (accent ? "text-magenta" : "text-chart-ink")
        }>
          {value}
        </span>
        {unit && <span className="chart-label">{unit}</span>}
      </div>
    </div>
  );
}

export function MetricsStrip({ stats, now, tick }: Props) {
  const n = stats.n_passengers;
  const seated = now?.seated ?? 0;
  const pct = n > 0 ? (seated / n) * 100 : 0;

  return (
    <div className="panel">
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6">
        <Metric
          label="Total boarding" value={String(Math.round(stats.ticks))}
          unit="ticks" accent
          tip="Simulator ticks from the first passenger entering until the last is seated. One tick is one decision step; a passenger walks one row per tick by default."
        />
        <Metric
          label="Elapsed" value={String(tick)} unit="ticks"
          tip="Position of the replay cursor, not a separate measurement."
        />
        <Metric
          label="Seated" value={`${seated}`} unit={`/ ${n}`}
          tip="Passengers who have finished walking, stowing and any seat shuffle."
        />
        <Metric
          label="Remaining" value={String(n - seated)} unit="pax" />
        <Metric
          label="Avg wait" value={stats.avg_wait.toFixed(1)} unit="ticks"
          tip="Mean ticks each passenger spent blocked behind someone else, averaged over all passengers."
        />
        <Metric
          label="Aisle blocks" value={String(Math.round(stats.blocking_events))}
          unit="events"
          tip="Number of times a passenger had to stop because the next aisle cell was occupied. Blocked ticks total is the time cost of those stops."
        />
      </div>
      <div className="h-1 bg-chart-line">
        <div className="h-full bg-magenta transition-[width] duration-150"
          style={{ width: `${pct}%` }}
          role="progressbar" aria-valuenow={Math.round(pct)}
          aria-valuemin={0} aria-valuemax={100}
          aria-label="Boarding progress" />
      </div>
    </div>
  );
}
