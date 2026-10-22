import { useMemo, useRef } from "react";
import type { TickSummary } from "../lib/trajectory";

interface Props {
  series: TickSummary[];
  tick: number;
  onSeek: (t: number) => void;
}

/**
 * Only the *active* passengers are plotted on the main axis. Stacking seated
 * and queued here too would squash walking/stowing/blocked into slivers — on
 * a 180-seat cabin at most a dozen people are moving at once, which is
 * exactly the signal this chart exists to show. Boarding progress is kept as
 * a separate ribbon underneath.
 */
const BANDS = [
  { key: "walking", color: "#2E6DA4", label: "Walking" },
  { key: "stowing", color: "#E0902F", label: "Stowing" },
  { key: "blocked", color: "#B4442E", label: "Blocked" },
] as const;

const H = 96;
const RIBBON = 10;
const PLOT = H - RIBBON - 4;

export function CongestionTrace({ series, tick, onSeek }: Props) {
  const ref = useRef<SVGSVGElement>(null);
  const n = series.length;

  const { paths, peak, total } = useMemo(() => {
    if (n === 0) return { paths: [], peak: 1, total: 1 };
    const tot = series[0].queued + series[0].walking + series[0].stowing
      + series[0].blocked + series[0].seated;
    const pk = Math.max(
      1,
      ...series.map((s) => s.walking + s.stowing + s.blocked),
    );
    const W = 1000;
    const x = (i: number) => (n === 1 ? 0 : (i / (n - 1)) * W);
    const y = (v: number) => PLOT - (v / pk) * PLOT;

    const out: { d: string; color: string; label: string }[] = [];
    const base = new Array(n).fill(0);
    for (const band of BANDS) {
      const top = series.map((s, i) => base[i] + (s[band.key] as number));
      let d = `M ${x(0)} ${y(base[0])}`;
      for (let i = 1; i < n; i++) d += ` L ${x(i)} ${y(base[i])}`;
      for (let i = n - 1; i >= 0; i--) d += ` L ${x(i)} ${y(top[i])}`;
      d += " Z";
      out.push({ d, color: band.color, label: band.label });
      for (let i = 0; i < n; i++) base[i] = top[i];
    }
    return { paths: out, peak: pk, total: tot };
  }, [series, n]);

  const seekFromEvent = (clientX: number) => {
    const rect = ref.current?.getBoundingClientRect();
    if (!rect || n === 0) return;
    const frac = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
    onSeek(Math.round(frac * (n - 1)));
  };

  const cursorX = n > 1 ? (tick / (n - 1)) * 1000 : 0;
  const now = series[tick];
  const active = now ? now.walking + now.stowing + now.blocked : 0;
  const stride = Math.max(1, Math.floor(n / 300));

  return (
    <figure className="m-0">
      <figcaption className="flex flex-wrap items-baseline justify-between
                             gap-x-4 gap-y-1 mb-1.5">
        <span className="chart-label">
          Aisle activity — peak {peak} of {total} passengers moving at once
        </span>
        <span className="flex gap-3 items-center">
          {BANDS.map((b) => (
            <span key={b.key} className="flex items-center gap-1">
              <span className="w-2 h-2 rounded-[1px]"
                style={{ background: b.color }} aria-hidden />
              <span className="chart-label">
                {b.label}
                {now ? ` ${now[b.key]}` : ""}
              </span>
            </span>
          ))}
          <span className="chart-label border-l border-chart-line pl-3">
            Active {active}
          </span>
        </span>
      </figcaption>
      <svg
        ref={ref}
        viewBox={`0 0 1000 ${H}`}
        preserveAspectRatio="none"
        className="w-full cursor-crosshair border border-chart-line bg-white"
        style={{ height: H }}
        role="slider"
        tabIndex={0}
        aria-label="Simulation timeline. Arrow keys step one tick."
        aria-valuemin={0}
        aria-valuemax={Math.max(0, n - 1)}
        aria-valuenow={tick}
        aria-valuetext={
          `Tick ${tick} of ${Math.max(0, n - 1)}, ${active} passengers active`
        }
        onMouseDown={(e) => {
          seekFromEvent(e.clientX);
          const move = (ev: MouseEvent) => seekFromEvent(ev.clientX);
          const up = () => {
            window.removeEventListener("mousemove", move);
            window.removeEventListener("mouseup", up);
          };
          window.addEventListener("mousemove", move);
          window.addEventListener("mouseup", up);
        }}
        onKeyDown={(e) => {
          if (e.key === "ArrowLeft") { e.preventDefault(); onSeek(tick - 1); }
          if (e.key === "ArrowRight") { e.preventDefault(); onSeek(tick + 1); }
          if (e.key === "Home") { e.preventDefault(); onSeek(0); }
          if (e.key === "End") { e.preventDefault(); onSeek(n - 1); }
        }}
      >
        <line x1="0" y1={PLOT / 2} x2="1000" y2={PLOT / 2}
          stroke="#E4E9E7" strokeWidth="1" vectorEffect="non-scaling-stroke" />
        {paths.map((p) => (
          <path key={p.label} d={p.d} fill={p.color} />
        ))}

        <rect x="0" y={H - RIBBON} width="1000" height={RIBBON}
          fill="#EDF0EE" />
        {series.map((s, i) =>
          i % stride === 0 ? (
            <rect
              key={i}
              x={(i / Math.max(1, n - 1)) * 1000}
              y={H - RIBBON}
              width={(stride / Math.max(1, n - 1)) * 1000 + 1}
              height={RIBBON}
              fill="#D8C9A8"
              opacity={s.seated / total}
            />
          ) : null,
        )}

        <line x1={cursorX} y1="0" x2={cursorX} y2={H}
          stroke="#13242E" strokeWidth="2" vectorEffect="non-scaling-stroke" />
      </svg>
      <p className="chart-label mt-1 normal-case tracking-normal">
        Lower strip: share of the cabin seated. Click or drag to scrub.
      </p>
    </figure>
  );
}
