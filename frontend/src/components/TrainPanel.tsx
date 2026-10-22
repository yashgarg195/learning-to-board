import { useEffect, useRef, useState } from "react";
import {
  CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import { ApiError, api } from "../api/client";
import type { SimConfig, TrainJob } from "../api/types";
import { ErrorNote } from "./ErrorNote";
import { Tip } from "./Tip";

interface Props {
  config: SimConfig;
  onTrained: (policyName: string | null) => void;
}

export function TrainPanel({ config, onTrained }: Props) {
  const [episodes, setEpisodes] = useState(2000);
  const [restarts, setRestarts] = useState(2);
  const [alpha, setAlpha] = useState(0.25);
  const [batch, setBatch] = useState(16);
  const [saveAs, setSaveAs] = useState("");
  const [job, setJob] = useState<TrainJob | null>(null);
  const [err, setErr] = useState<ApiError | null>(null);
  const poll = useRef<number | null>(null);

  useEffect(() => () => {
    if (poll.current) window.clearInterval(poll.current);
  }, []);

  const start = async () => {
    setErr(null);
    try {
      const j = await api.train({
        config, episodes, restarts, alpha, batch,
        save_as: saveAs.trim() || null,
      });
      setJob(j);
      if (poll.current) window.clearInterval(poll.current);
      poll.current = window.setInterval(async () => {
        try {
          const s = await api.trainStatus(j.job_id);
          setJob(s);
          if (s.status === "done" || s.status === "error") {
            if (poll.current) window.clearInterval(poll.current);
            if (s.status === "done") onTrained(s.result?.saved_as ?? null);
          }
        } catch (e) {
          if (poll.current) window.clearInterval(poll.current);
          setErr(e as ApiError);
        }
      }, 800);
    } catch (e) {
      setErr(e as ApiError);
    }
  };

  const running = job?.status === "queued" || job?.status === "running";
  const pct = job && job.episodes_total > 0
    ? (job.episodes_done / job.episodes_total) * 100 : 0;
  const curve = (job?.curve ?? []).map(([e, v]) => ({ e, v: Number(v.toFixed(1)) }));
  const res = job?.status === "done" ? job.result : null;

  return (
    <section className="panel p-4 space-y-4">
      <header className="flex flex-wrap items-baseline justify-between gap-2">
        <h2 className="font-display text-sm font-semibold uppercase tracking-chart">
          Train an agent
        </h2>
        <p className="chart-label normal-case tracking-normal">
          Tabular REINFORCE · trains on the scenario configured on the left
        </p>
      </header>

      <details className="text-sm">
        <summary className="cursor-pointer chart-label hover:text-chart-ink">
          What the agent actually learns
        </summary>
        <dl className="mt-2 space-y-1.5 text-xs leading-relaxed text-chart-muted">
          <div><dt className="inline font-medium text-chart-ink">State: </dt>
            <dd className="inline">(boarding progress bin, band of the last
              released passenger, seat type of the last released passenger) —
              a few hundred discrete states, independent of cabin size.</dd></div>
          <div><dt className="inline font-medium text-chart-ink">Action: </dt>
            <dd className="inline">release the next passenger from one of
              (distance band × seat type) groups. Unavailable groups are
              masked out rather than encoded in the state.</dd></div>
          <div><dt className="inline font-medium text-chart-ink">Reward: </dt>
            <dd className="inline">−1 per simulator tick, so maximising return
              is exactly minimising boarding time.</dd></div>
          <div><dt className="inline font-medium text-chart-ink">Terminal: </dt>
            <dd className="inline">every passenger seated, or the tick cap.</dd></div>
          <div><dt className="inline font-medium text-chart-ink">Exploration: </dt>
            <dd className="inline">softmax sampling over the masked action
              logits; the reported curve is the <em>greedy</em> policy, which
              is what gets saved.</dd></div>
        </dl>
      </details>

      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <div>
          <label className="field-label" htmlFor="ep">Episodes</label>
          <input id="ep" type="number" min={100} max={40000} step={100}
            className="control font-mono" value={episodes} disabled={running}
            onChange={(e) => setEpisodes(Number(e.target.value) || 100)} />
        </div>
        <div>
          <label className="field-label" htmlFor="rs">
            Restarts <Tip text="Independent training runs. The best one on a held-out validation seed set is kept, which avoids reporting a lucky run." />
          </label>
          <input id="rs" type="number" min={1} max={8}
            className="control font-mono" value={restarts} disabled={running}
            onChange={(e) => setRestarts(Number(e.target.value) || 1)} />
        </div>
        <div>
          <label className="field-label" htmlFor="al">
            Step size α <Tip text="Applied to a normalised advantage, so the same value works across cabin sizes." />
          </label>
          <input id="al" type="number" min={0.01} max={2} step={0.05}
            className="control font-mono" value={alpha} disabled={running}
            onChange={(e) => setAlpha(Number(e.target.value) || 0.25)} />
        </div>
        <div>
          <label className="field-label" htmlFor="bt">
            Batch <Tip text="Episodes accumulated before each gradient step." />
          </label>
          <input id="bt" type="number" min={1} max={128}
            className="control font-mono" value={batch} disabled={running}
            onChange={(e) => setBatch(Number(e.target.value) || 16)} />
        </div>
      </div>

      <div className="flex flex-wrap items-end gap-3">
        <div className="flex-1 min-w-[180px]">
          <label className="field-label" htmlFor="sa">
            Save as (optional)
          </label>
          <input id="sa" className="control font-mono" value={saveAs}
            placeholder="my_policy" disabled={running}
            onChange={(e) => setSaveAs(e.target.value)} />
        </div>
        <button className="btn-primary" onClick={start} disabled={running}>
          {running ? "Training…" : "Start training"}
        </button>
      </div>

      {err && <ErrorNote error={err} onRetry={start} />}

      {job && (
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <span className="chart-label">
              Job <span className="font-mono normal-case">{job.job_id}</span> ·{" "}
              {job.status}
            </span>
            <span className="font-mono text-xs tabular-nums text-chart-muted">
              {job.episodes_done} / {job.episodes_total} episodes
            </span>
          </div>
          <div className="h-1.5 bg-chart-line">
            <div className="h-full bg-magenta transition-[width]"
              style={{ width: `${pct}%` }} role="progressbar"
              aria-valuenow={Math.round(pct)} aria-valuemin={0}
              aria-valuemax={100} aria-label="Training progress" />
          </div>

          {job.status === "error" && (
            <div role="alert" className="border border-alert px-3 py-2 text-sm">
              <p className="font-medium text-alert">Training failed</p>
              <p className="font-mono text-xs mt-1">{job.message}</p>
            </div>
          )}

          {curve.length > 0 && (
            <div className="h-[200px] bg-white border border-chart-line p-2">
              <p className="chart-label px-1">
                Greedy policy evaluated on held-out validation seeds
              </p>
              <ResponsiveContainer width="100%" height="88%">
                <LineChart data={curve}
                  margin={{ top: 8, right: 16, bottom: 4, left: 0 }}>
                  <CartesianGrid stroke="#E4E9E7" vertical={false} />
                  <XAxis dataKey="e" tick={{ fontSize: 10, fill: "#5A6E74" }}
                    axisLine={{ stroke: "#CBD4D2" }} tickLine={false}
                    label={{ value: "episode", position: "insideBottomRight",
                      fontSize: 9, fill: "#5A6E74", dy: 8 }} />
                  <YAxis tick={{ fontSize: 10, fill: "#5A6E74" }}
                    axisLine={false} tickLine={false} domain={["auto", "auto"]} />
                  <Tooltip
                    contentStyle={{ border: "1px solid #CBD4D2",
                      borderRadius: 2, fontSize: 12 }}
                    formatter={(v: number) => [`${v} ticks`, "Greedy"]} />
                  <Line type="monotone" dataKey="v" stroke="#C4347A"
                    strokeWidth={1.8} dot={false} isAnimationActive={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}

          {res && (
            <div className="border border-chart-line bg-white p-3 space-y-2">
              <p className="chart-label">Evaluation (held-out seeds)</p>
              <table className="w-full text-sm font-mono tabular-nums">
                <tbody>
                  <tr className="border-b border-chart-line">
                    <td className="py-1 font-sans text-magenta font-semibold">
                      Trained policy
                    </td>
                    <td className="py-1 text-right">
                      {res.eval_mean_ticks.toFixed(1)} ticks
                    </td>
                  </tr>
                  {Object.entries(res.baseline_ticks).map(([k, v]) => (
                    <tr key={k} className="border-b border-chart-line last:border-0">
                      <td className="py-1 font-sans capitalize">{k}</td>
                      <td className="py-1 text-right">{v.toFixed(1)} ticks</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <p className="text-xs text-chart-muted leading-relaxed">
                {res.states} states visited ·{" "}
                {res.saved_as
                  ? <>saved as <span className="font-mono">{res.saved_as}</span>,
                    now selectable as a boarding strategy.</>
                  : <>not saved — give it a name above to keep it.</>}{" "}
                These are evaluation numbers on seeds the agent never trained
                on; a short run will usually lose to a good heuristic.
              </p>
            </div>
          )}
        </div>
      )}

      {!job && (
        <p className="text-sm text-chart-muted border border-dashed
                      border-chart-line p-4">
          No training run yet. Training happens in the background — you can
          keep simulating while it works. Shipped policies under the strategy
          selector were trained offline with far larger budgets.
        </p>
      )}
    </section>
  );
}
