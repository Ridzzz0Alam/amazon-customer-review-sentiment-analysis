// Same-origin by default: Vite proxies /api in dev and nginx proxies it in Docker.
const BASE = (import.meta.env.VITE_API_URL as string | undefined) ?? "/api";

export type Label = "positive" | "negative";

export interface TermWeight { term: string; weight: number }
export interface Prediction {
  label: Label;
  positive_probability: number;
  confidence: number;
  concerns: string[];
  top_terms: TermWeight[];
}
export interface ConcernMeta { key: string; label: string; type: "product" | "experience" }
export interface ConcernCount extends ConcernMeta { negative_count: number; share_of_negative: number }
export interface BatchItem {
  index: number;
  label: Label | null;
  positive_probability: number | null;
  concerns: string[];
  error: string | null;
}
export interface BatchResponse {
  n_total: number;
  n_scored: number;
  n_negative: number;
  negative_share: number;
  concerns: ConcernCount[];
  items: BatchItem[];
}
export interface ModelInfo {
  model_name: string;
  model_version: string;
  trained_at: string | null;
  params: Record<string, unknown>;
  metrics: Record<string, number | Record<string, number>>;
  data: Record<string, unknown>;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
  });
  if (!res.ok) {
    let detail = `Request failed with status ${res.status}`;
    try {
      const body = await res.json();
      if (typeof body.detail === "string") detail = body.detail;
    } catch { /* keep default */ }
    throw new Error(detail);
  }
  return res.json() as Promise<T>;
}

export const api = {
  predict: (text: string) => request<Prediction>("/v1/predict", { method: "POST", body: JSON.stringify({ text }) }),
  batch: (texts: string[]) => request<BatchResponse>("/v1/predict/batch", { method: "POST", body: JSON.stringify({ texts }) }),
  model: () => request<ModelInfo>("/v1/model"),
  concerns: () => request<ConcernMeta[]>("/v1/concerns"),
};
