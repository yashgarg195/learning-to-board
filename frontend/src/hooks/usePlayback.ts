import { useCallback, useEffect, useRef, useState } from "react";

export interface Playback {
  tick: number;
  playing: boolean;
  speed: number;
  total: number;
  play: () => void;
  pause: () => void;
  toggle: () => void;
  reset: () => void;
  replay: () => void;
  seek: (t: number) => void;
  step: (delta: number) => void;
  setSpeed: (s: number) => void;
}

const BASE_MS = 120; // one simulator tick at 1x

/**
 * Replays a pre-computed trajectory. The backend already produced every
 * frame, so playback only advances an index — changing speed or scrubbing
 * never re-runs the simulation.
 */
export function usePlayback(total: number): Playback {
  const [tick, setTick] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(2);
  const raf = useRef<number | null>(null);
  const acc = useRef(0);
  const last = useRef(0);

  // A new trajectory rewinds to the start rather than stranding the scrubber
  // past the end of a shorter run.
  useEffect(() => {
    setTick(0);
    setPlaying(false);
  }, [total]);

  useEffect(() => {
    if (!playing || total === 0) return;
    last.current = performance.now();
    acc.current = 0;

    const loop = (now: number) => {
      const dt = now - last.current;
      last.current = now;
      acc.current += dt;
      const per = BASE_MS / speed;
      let advance = 0;
      while (acc.current >= per) {
        acc.current -= per;
        advance += 1;
      }
      if (advance > 0) {
        setTick((t) => {
          const next = t + advance;
          if (next >= total - 1) {
            setPlaying(false);
            return total - 1;
          }
          return next;
        });
      }
      raf.current = requestAnimationFrame(loop);
    };
    raf.current = requestAnimationFrame(loop);
    return () => {
      if (raf.current !== null) cancelAnimationFrame(raf.current);
    };
  }, [playing, speed, total]);

  const play = useCallback(() => {
    if (total === 0) return;
    setTick((t) => (t >= total - 1 ? 0 : t));
    setPlaying(true);
  }, [total]);

  const pause = useCallback(() => setPlaying(false), []);

  const toggle = useCallback(() => {
    setPlaying((p) => {
      if (p) return false;
      if (total === 0) return false;
      setTick((t) => (t >= total - 1 ? 0 : t));
      return true;
    });
  }, [total]);

  const reset = useCallback(() => {
    setPlaying(false);
    setTick(0);
  }, []);

  const replay = useCallback(() => {
    setTick(0);
    setPlaying(total > 0);
  }, [total]);

  const seek = useCallback(
    (t: number) => {
      setPlaying(false);
      setTick(Math.max(0, Math.min(total - 1, Math.round(t))));
    },
    [total],
  );

  const step = useCallback(
    (delta: number) => {
      setPlaying(false);
      setTick((t) => Math.max(0, Math.min(total - 1, t + delta)));
    },
    [total],
  );

  return {
    tick, playing, speed, total,
    play, pause, toggle, reset, replay, seek, step, setSpeed,
  };
}
