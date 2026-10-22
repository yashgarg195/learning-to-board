import type { Playback } from "../hooks/usePlayback";

const SPEEDS = [0.5, 1, 2, 4, 8];

export function PlaybackBar({ pb, disabled }: {
  pb: Playback; disabled: boolean;
}) {
  const atEnd = pb.total > 0 && pb.tick >= pb.total - 1;
  return (
    <div className="flex flex-wrap items-center gap-2">
      <button
        className="btn-primary min-w-[88px]"
        onClick={pb.toggle} disabled={disabled}
        aria-label={pb.playing ? "Pause replay" : "Play replay"}
      >
        {pb.playing ? "Pause" : atEnd ? "Replay" : "Play"}
      </button>
      <button className="btn-ghost" onClick={() => pb.step(-1)}
        disabled={disabled || pb.tick === 0} aria-label="Step back one tick">
        −1
      </button>
      <button className="btn-ghost" onClick={() => pb.step(1)}
        disabled={disabled || atEnd} aria-label="Step forward one tick">
        +1
      </button>
      <button className="btn-ghost" onClick={pb.reset} disabled={disabled}>
        Reset
      </button>
      <button className="btn-ghost" onClick={pb.replay} disabled={disabled}>
        Replay
      </button>

      <div className="flex items-center gap-1 ml-1" role="group"
        aria-label="Playback speed">
        <span className="chart-label mr-1">Speed</span>
        {SPEEDS.map((s) => (
          <button
            key={s}
            className={
              "btn px-2 py-1 font-mono text-xs " +
              (pb.speed === s
                ? "bg-chart-ink border-chart-ink text-white"
                : "bg-white border-chart-line text-chart-muted hover:border-chart-rule")
            }
            aria-pressed={pb.speed === s}
            onClick={() => pb.setSpeed(s)}
            disabled={disabled}
          >
            {s}×
          </button>
        ))}
      </div>

      <div className="flex-1 min-w-[180px] flex items-center gap-2">
        <input
          type="range" min={0} max={Math.max(0, pb.total - 1)} step={1}
          value={pb.tick} disabled={disabled}
          aria-label="Scrub to tick"
          className="flex-1 accent-magenta"
          onChange={(e) => pb.seek(Number(e.target.value))}
        />
        <span className="font-mono text-xs text-chart-muted tabular-nums
                         whitespace-nowrap">
          t {String(pb.tick).padStart(3, "0")} /{" "}
          {Math.max(0, pb.total - 1)}
        </span>
      </div>
    </div>
  );
}
