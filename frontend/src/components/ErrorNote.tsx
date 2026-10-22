import type { ApiError } from "../api/client";

/** Surfaces the backend's structured error instead of a generic failure. */
export function ErrorNote({ error, onRetry }: {
  error: ApiError; onRetry?: () => void;
}) {
  const detail = Array.isArray(error.detail)
    ? (error.detail as { field: string; problem: string }[])
    : null;
  return (
    <div role="alert"
      className="border border-alert bg-white px-3 py-2.5 text-sm">
      <div className="flex items-start justify-between gap-3">
        <div>
          <p className="font-medium text-alert">{error.message}</p>
          <p className="chart-label mt-0.5">{error.code}</p>
          {detail && (
            <ul className="mt-1.5 space-y-0.5">
              {detail.map((d) => (
                <li key={d.field} className="font-mono text-xs text-chart-muted">
                  {d.field}: {d.problem}
                </li>
              ))}
            </ul>
          )}
        </div>
        {onRetry && (
          <button className="btn-ghost shrink-0" onClick={onRetry}>
            Retry
          </button>
        )}
      </div>
    </div>
  );
}
