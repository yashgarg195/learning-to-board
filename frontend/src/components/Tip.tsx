import { useId, useState } from "react";

/** Accessible hover/focus tooltip for metric and RL terminology. */
export function Tip({ text }: { text: string }) {
  const id = useId();
  const [open, setOpen] = useState(false);
  return (
    <span className="relative inline-block align-middle">
      <button
        type="button"
        aria-label="More information"
        aria-describedby={open ? id : undefined}
        className="w-3.5 h-3.5 rounded-full border border-chart-rule
                   text-[9px] leading-none text-chart-muted
                   hover:border-magenta hover:text-magenta transition-colors"
        onMouseEnter={() => setOpen(true)}
        onMouseLeave={() => setOpen(false)}
        onFocus={() => setOpen(true)}
        onBlur={() => setOpen(false)}
        onClick={(e) => { e.preventDefault(); setOpen((v) => !v); }}
      >
        ?
      </button>
      {open && (
        <span
          id={id} role="tooltip"
          className="absolute z-20 left-1/2 -translate-x-1/2 bottom-full mb-1.5
                     w-56 p-2 bg-chart-ink text-white text-xs leading-snug
                     normal-case tracking-normal font-sans rounded-sm shadow-lg"
        >
          {text}
        </span>
      )}
    </span>
  );
}
