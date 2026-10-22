import type { PolicyInfo } from "../api/types";

interface Props {
  policies: PolicyInfo[];
  selected: string | null;
}

/**
 * The "what exactly did you train?" reference.
 *
 * Every constant quoted here is mirrored from the Python source, and the
 * per-policy figures are read from the saved metadata rather than typed in,
 * so a retrained agent cannot leave a stale number on screen.
 *
 *   actions / state  -> boarding/envs.py
 *   optimiser        -> boarding/agents/pg.py (train_tabular_pg)
 *   seed pools       -> boarding/agents/pg.py, scripts/train_parity.py
 */

/** Mirrors `boarding/envs.py` and `train_tabular_pg`'s signature. */
const SPEC: { term: string; value: string; detail: string }[] = [
  {
    term: "Algorithm",
    value: "REINFORCE (tabular policy gradient)",
    detail:
      "Monte-Carlo policy gradient over a lookup table. Chosen after "
      + "one-step TD failed structurally on this problem: the reward is "
      + "delayed to the end of the episode.",
  },
  {
    term: "Policy class",
    value: "Softmax table θ[state][action]",
    detail:
      "Invalid actions are set to −1e9 before the softmax, so the agent can "
      + "never pick an empty group.",
  },
  {
    term: "State",
    value: "(progress, last band, last seat type, last parity)",
    detail:
      "12 progress bins × 5 band values × 4 seat types × 3 parities "
      + "(each includes a 'nothing released yet' value). Group availability "
      + "is handled by action masking, not by the state.",
  },
  {
    term: "Action space",
    value: "24 = 4 row bands × 3 seat types × 2 row parities",
    detail:
      "One action releases the next eligible passenger from that group. "
      + "Row parity is what lets the agent alternate odd/even rows — the "
      + "mechanism behind Steffen's advantage.",
  },
  {
    term: "Reward",
    value: "−1 per simulator tick",
    detail:
      "Plus a −2.0 penalty for choosing an empty group. Congestion shaping "
      + "(block_penalty) exists but defaults to 0, so the agent optimises "
      + "total boarding time directly and nothing else.",
  },
  {
    term: "Discount γ",
    value: "1.0 (undiscounted)",
    detail:
      "The objective is total boarding time, so an early tick and a late "
      + "tick cost the same. Discounting would reward pushing congestion "
      + "past the horizon.",
  },
  {
    term: "Termination",
    value: "Every passenger seated, or max_ticks reached",
    detail: "Episodes are finite; truncation is recorded separately.",
  },
  {
    term: "Exploration",
    value: "Sampling from the softmax",
    detail:
      "Not ε-greedy — the stochastic policy is the exploration. This is why "
      + "raw training episodes look slower than the policy actually is.",
  },
  {
    term: "Baseline",
    value: "Paired per-seed running mean (decay 0.3)",
    detail:
      "Essential here: manifest noise between seeds is ±20 ticks, which "
      + "would otherwise swamp the learning signal. The advantage is then "
      + "normalised to standard-deviation units.",
  },
  {
    term: "Step size α",
    value: "0.25 on the normalised advantage",
    detail:
      "Because the advantage is normalised, the same α transfers between a "
      + "60-passenger and a 500-passenger cabin.",
  },
  {
    term: "Batch",
    value: "16 episodes per gradient step",
    detail: "Per-episode updates let one lucky rollout dominate.",
  },
];

const SEEDS: { pool: string; range: string; use: string }[] = [
  { pool: "Training", range: "1 – 48", use: "Cycled during learning." },
  {
    pool: "Validation", range: "300 – 319",
    use: "Picks the best restart. Never reported as a result.",
  },
  {
    pool: "Test", range: "500 – 529",
    use: "Every number quoted in the UI and the README.",
  },
];

function Row({ term, value, detail }:
  { term: string; value: string; detail: string }) {
  return (
    <div className="grid sm:grid-cols-[9rem_1fr] gap-x-4 gap-y-0.5 py-2.5
                    border-b border-chart-line last:border-0">
      <dt className="chart-label pt-0.5">{term}</dt>
      <dd className="m-0">
        <p className="text-sm font-medium text-chart-ink m-0">{value}</p>
        <p className="text-xs text-chart-muted leading-relaxed mt-0.5 m-0">
          {detail}
        </p>
      </dd>
    </div>
  );
}

export function MethodPanel({ policies, selected }: Props) {
  const current = policies.find((p) => p.name === selected) ?? null;
  const meta = (current?.meta ?? {}) as Record<string, unknown>;
  const baselines = meta.baselines as Record<string, number> | undefined;
  const restarts = meta.restart_validation_scores as number[] | undefined;
  const num = (k: string) =>
    typeof meta[k] === "number" ? (meta[k] as number) : null;

  return (
    <section className="panel p-4 space-y-5">
      <header>
        <h2 className="font-display text-sm font-semibold uppercase
                       tracking-chart">
          Method — training &amp; evaluation
        </h2>
        <p className="chart-label normal-case tracking-normal mt-0.5">
          Mirrored from <code>boarding/envs.py</code> and{" "}
          <code>boarding/agents/pg.py</code>. All times in simulator ticks.
        </p>
      </header>

      <div>
        <h3 className="chart-label mb-1">The MDP and the optimiser</h3>
        <dl className="m-0">
          {SPEC.map((s) => <Row key={s.term} {...s} />)}
        </dl>
      </div>

      <div>
        <h3 className="chart-label mb-1.5">
          Evaluation protocol — three disjoint seed pools
        </h3>
        <table className="w-full text-sm border-collapse">
          <thead>
            <tr className="border-b border-chart-rule">
              {["Pool", "Seeds", "Used for"].map((h) => (
                <th key={h} className="chart-label text-left py-1.5 pr-3">
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {SEEDS.map((s) => (
              <tr key={s.pool} className="border-b border-chart-line last:border-0">
                <td className="py-1.5 pr-3 font-medium">{s.pool}</td>
                <td className="py-1.5 pr-3 font-mono tabular-nums">{s.range}</td>
                <td className="py-1.5 pr-3 text-chart-muted">{s.use}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="text-xs text-chart-muted leading-relaxed mt-2">
          Training explores by <em>sampling</em>; evaluation is{" "}
          <strong className="font-semibold">greedy</strong> (argmax over
          unmasked actions), so the reported figure is the policy you would
          actually deploy. Every saved policy is also reloaded from disk and
          re-scored to confirm the round-trip is exact.
        </p>
      </div>

      <div>
        <h3 className="chart-label mb-1.5">Metrics and their definitions</h3>
        <dl className="m-0">
          <Row term="Boarding time" value="Ticks, first entry → last seated"
            detail="The optimisation target. One tick is one decision step; a passenger walks one row per tick by default." />
          <Row term="Avg wait" value="Ticks blocked per passenger"
            detail="Mean ticks a passenger spent unable to advance, averaged over everyone on board." />
          <Row term="Aisle blocks" value="Count of blocking events"
            detail="How many times someone had to stop because the next aisle cell was occupied." />
          <Row term="Std / best / worst" value="Across the test seeds"
            detail="Spread matters as much as the mean: a policy that is fast on average but occasionally terrible is not an operational win." />
        </dl>
      </div>

      <div>
        <h3 className="chart-label mb-1.5">
          Selected policy{current ? ` — ${current.name}` : ""}
        </h3>
        {!current ? (
          <p className="text-sm text-chart-muted">
            No policy selected. Pick one in the configuration panel to see the
            run that produced it.
          </p>
        ) : (
          <>
            <dl className="m-0">
              <Row term="Label" value={current.label}
                detail={current.note} />
              <Row term="Episodes"
                value={num("episodes") !== null
                  ? `${num("episodes")} per restart` : "not recorded"}
                detail={num("restarts") !== null
                  ? `${num("restarts")} independent restarts; the best on the `
                    + "validation pool was kept. Policy gradient is "
                    + "multi-modal here, so restarts matter."
                  : "Restart count not recorded in this policy's metadata."} />
              <Row term="States visited"
                value={num("states") !== null
                  ? `${num("states")} of ~720 reachable` : "not recorded"}
                detail="The table only stores states the agent actually encountered." />
              <Row term="Action space"
                value={typeof meta.action_space === "string"
                  ? String(meta.action_space) : "band × seat_type (12, legacy)"}
                detail={meta.use_parity
                  ? "24 actions, including row parity."
                  : "A pre-parity policy. It is replayed in 12-action mode "
                    + "automatically, inferred from the saved table width."} />
              {num("train_seconds") !== null && (
                <Row term="Training cost"
                  value={`${num("train_seconds")} s total`}
                  detail="CPU only; no GPU or ML framework is involved." />
              )}
            </dl>

            {restarts && restarts.length > 1 && (
              <p className="text-xs text-chart-muted leading-relaxed mt-2">
                <span className="chart-label">Restart spread</span>{" "}
                <span className="font-mono tabular-nums">
                  {restarts.map((r) => r.toFixed(1)).join(" · ")}
                </span>{" "}
                ticks on validation — the gap between the best and worst
                restart is the honest measure of how much the result depends
                on initialisation.
              </p>
            )}

            {baselines && (
              <div className="mt-3">
                <h4 className="chart-label mb-1.5">
                  Result on the test pool (seeds 500–529)
                </h4>
                <table className="w-full text-sm border-collapse">
                  <tbody className="font-mono tabular-nums">
                    {Object.entries(baselines)
                      .sort((a, b) => a[1] - b[1])
                      .map(([k, v]) => {
                        const isAgent = k.startsWith("rl");
                        return (
                          <tr key={k}
                            className="border-b border-chart-line last:border-0">
                            <td className={"py-1.5 pr-3 font-sans "
                              + (isAgent ? "font-semibold text-magenta" : "")}>
                              {k}
                            </td>
                            <td className="py-1.5 pr-3">{v.toFixed(1)}</td>
                            <td className="py-1.5 chart-label">ticks</td>
                          </tr>
                        );
                      })}
                  </tbody>
                </table>
              </div>
            )}
          </>
        )}
      </div>

      <div className="rounded border border-chart-line bg-chart-paper p-3">
        <h3 className="chart-label mb-1.5">Known limits — state these up front</h3>
        <ul className="list-disc pl-4 space-y-1 text-xs text-chart-ink
                       leading-relaxed m-0">
          <li>
            The agent is trained <strong className="font-semibold">per
            cabin</strong>. A policy trained on one cabin does not transfer:
            band indices are tied to that row count.
          </li>
          <li>
            The win is demonstrated on the <strong className="font-semibold">
            mid benchmark cabin</strong>. On the 30-row A320 the same method
            did not beat Steffen within the training budget tried, and that
            policy is deliberately not shipped.
          </li>
          <li>
            Beating a heuristic on 30 seeds of one cabin is{" "}
            <strong className="font-semibold">evidence, not proof of
            optimality</strong>.
          </li>
          <li>
            The simulator abstracts real boarding: no gate-agent behaviour, no
            priority groups, and carry-on stowing is a fixed delay rather than
            a distribution.
          </li>
        </ul>
      </div>
    </section>
  );
}
