import type {
  CompareResponse, OptionsResponse, PolicyInfo, SimConfig,
  SimulateResponse, StrategyName, TrainJob,
} from "./types";

const BASE = import.meta.env.VITE_API_BASE ?? "";

/** An error carrying the backend's structured `{error:{code,message}}` body. */
export class ApiError extends Error {
  code: string;
  detail: unknown;
  status: number;

  constructor(status: number, code: string, message: string, detail?: unknown) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
    this.detail = detail;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(BASE + path, {
      ...init,
      headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
    });
  } catch {
    throw new ApiError(
      0, "network_error",
      "Cannot reach the simulation API. Is the backend running on port 8000?",
    );
  }
  const text = await res.text();
  // A 500 from a proxy or an unhandled server error can be plain text or
  // HTML. Parsing that as JSON throws "Unexpected token 'I'", which hides the
  // real failure, so fall back to reporting the status and the raw body.
  let body: any = null;
  if (text) {
    try {
      body = JSON.parse(text);
    } catch {
      if (!res.ok) {
        throw new ApiError(
          res.status, "server_error",
          `The API returned ${res.status} ${res.statusText || "Error"}.`,
          [{ field: "response", problem: text.slice(0, 300) }],
        );
      }
      throw new ApiError(
        res.status, "bad_response",
        "The API returned a response that was not valid JSON.",
        [{ field: "response", problem: text.slice(0, 300) }],
      );
    }
  }
  if (!res.ok) {
    const e = body?.error;
    throw new ApiError(
      res.status,
      e?.code ?? "http_error",
      e?.message ?? `Request failed with status ${res.status}.`,
      e?.detail,
    );
  }
  return body as T;
}

export const api = {
  health: () => request<{ status: string }>("/api/health"),

  options: () => request<OptionsResponse>("/api/options"),

  simulate: (config: SimConfig, strategy: StrategyName, policy?: string | null) =>
    request<SimulateResponse>("/api/simulate", {
      method: "POST",
      body: JSON.stringify({ config, strategy, policy: policy ?? null }),
    }),

  compare: (
    config: SimConfig, strategies: StrategyName[], trials: number,
    policy?: string | null,
  ) =>
    request<CompareResponse>("/api/compare", {
      method: "POST",
      body: JSON.stringify({ config, strategies, trials, policy: policy ?? null }),
    }),

  train: (body: {
    config: SimConfig; episodes: number; restarts: number;
    alpha: number; batch: number; save_as: string | null;
  }) =>
    request<TrainJob>("/api/train", {
      method: "POST",
      body: JSON.stringify(body),
    }),

  trainStatus: (jobId: string) => request<TrainJob>(`/api/train/${jobId}`),

  policies: () => request<PolicyInfo[]>("/api/policies"),
};
