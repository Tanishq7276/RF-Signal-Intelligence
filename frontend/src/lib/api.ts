/** Typed API client.  Every call goes to the FastAPI backend with the session header. */
export const getApiBase = (): string => {
  const custom = typeof window !== "undefined" ? localStorage.getItem("sih26147-api-url") : null;
  if (custom && custom.trim()) return custom.trim().replace(/\/+$/, "");
  return (import.meta.env?.VITE_API_URL as string | undefined)?.replace(/\/+$/, "") || "";
};

export const setApiBase = (url: string | null) => {
  if (typeof window === "undefined") return;
  if (url && url.trim()) localStorage.setItem("sih26147-api-url", url.trim().replace(/\/+$/, ""));
  else localStorage.removeItem("sih26147-api-url");
};

export const API_BASE = getApiBase();

export const apiUrl = (path: string): string => {
  if (path.startsWith("http://") || path.startsWith("https://")) return path;
  const base = getApiBase();
  if (!base) return path;
  const cleanPath = path.startsWith("/") ? path : `/${path}`;
  return `${base}${cleanPath}`;
};

export const SESSION_KEY = (() => {
  const k = "sih26147-session";
  let v = localStorage.getItem(k);
  if (!v) {
    v = "s" + Math.random().toString(36).slice(2, 10) + Date.now().toString(36).slice(-4);
    localStorage.setItem(k, v);
  }
  return v;
})();

export class ApiError extends Error {
  status: number;
  detail: any;
  constructor(status: number, detail: any) {
    super(typeof detail === "string" ? detail : detail?.detail || `HTTP ${status}`);
    this.status = status;
    this.detail = detail;
  }
}

export const TOKEN_KEY = "sih26147-token";
export const getToken = () => localStorage.getItem(TOKEN_KEY);
export const setToken = (t: string | null) => { if (t) localStorage.setItem(TOKEN_KEY, t); else localStorage.removeItem(TOKEN_KEY); };

const headers = (): Record<string, string> => {
  const h: Record<string, string> = { "x-session-id": SESSION_KEY };
  const tok = getToken();
  if (tok) h.Authorization = `Bearer ${tok}`;
  return h;
};

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const url = apiUrl(path);
  const res = await fetch(url, {
    ...init,
    headers: { ...headers(), ...(init?.body && !(init.body instanceof FormData) ? { "Content-Type": "application/json" } : {}), ...(init?.headers || {}) },
  });
  const text = await res.text();
  let body: any = null;
  try { body = text ? JSON.parse(text) : null; } catch { body = text; }
  if (!res.ok) throw new ApiError(res.status, body ?? res.statusText);
  return body as T;
}

const post = <T,>(p: string, body?: any) => req<T>(p, { method: "POST", body: body === undefined ? undefined : JSON.stringify(body) });
const del = <T,>(p: string) => req<T>(p, { method: "DELETE" });
const qs = (o: Record<string, any>) => {
  const s = new URLSearchParams();
  Object.entries(o).forEach(([k, v]) => { if (v !== undefined && v !== null && v !== "") s.set(k, String(v)); });
  const str = s.toString();
  return str ? `?${str}` : "";
};

export const Api = {
  sessionKey: SESSION_KEY,
  // auth
  register: (body: any) => post<any>("/api/auth/register", body),
  login: (body: any) => post<any>("/api/auth/login", body),
  logout: () => post<any>("/api/auth/logout"),
  me: () => req<any>("/api/auth/me"),
  authSession: () => req<any>("/api/auth/session"),
  // single-stage DSP endpoints
  preprocess: (body: any) => post<any>("/api/preprocess", body),
  spectrumOf: (body: any) => post<any>("/api/spectrum", body),
  spectrogramOf: (body: any) => post<any>("/api/spectrogram", body),
  detectSignals: (body: any) => post<any>("/api/detect-signals", body),
  extractParameters: (body: any) => post<any>("/api/extract-parameters", body),
  classify: (body: any) => post<any>("/api/modulation/classify", body),
  symbolRate: (body: any) => post<any>("/api/symbol-rate", body),
  constellationOf: (body: any) => post<any>("/api/constellation", body),
  eyeOf: (body: any) => post<any>("/api/eye-diagram", body),
  bitstreamOf: (body: any) => post<any>("/api/bitstream", body),
  verifyGenerated: (body: any) => post<any>("/api/generator/verify", body),
  history: (params: Record<string, any> = {}) => req<any>(`/api/history${qs(params)}`),
  analysisStatus: (id: string) => req<any>(`/api/analyze/${id}/status`),
  // system
  health: () => req<any>("/api/health"),
  capabilities: () => req<any>("/api/capabilities"),
  stats: () => req<any>("/api/stats"),
  settings: () => req<any>("/api/settings"),
  models: () => req<any>("/api/models"),
  clearSession: () => del<any>("/api/session"),
  // files
  listFiles: () => req<any>("/api/files"),
  getFile: (id: string) => req<any>(`/api/files/${id}`),
  deleteFile: (id: string) => del<any>(`/api/files/${id}`),
  preview: (id: string, maxPoints = 4000) => req<any>(`/api/files/${id}/preview${qs({ max_points: maxPoints })}`),
  demos: () => req<any>("/api/demo"),
  loadDemo: (name: string) => post<any>(`/api/demo/${encodeURIComponent(name)}`),
  upload: (file: File, onProgress?: (frac: number) => void) =>
    new Promise<any>((resolve, reject) => {
      const fd = new FormData();
      fd.append("file", file);
      const xhr = new XMLHttpRequest();
      xhr.open("POST", apiUrl("/api/upload"));
      const h = headers();
      Object.entries(h).forEach(([k, v]) => xhr.setRequestHeader(k, v));
      xhr.upload.onprogress = (e) => { if (e.lengthComputable && onProgress) onProgress(e.loaded / e.total); };
      xhr.onload = () => {
        let body: any = xhr.responseText;
        try { body = JSON.parse(xhr.responseText); } catch { /* keep text */ }
        if (xhr.status >= 200 && xhr.status < 300) resolve(body);
        else reject(new ApiError(xhr.status, body));
      };
      xhr.onerror = () => reject(new ApiError(0, "network error during upload"));
      xhr.send(fd);
    }),
  downloadUrl: (id: string) => apiUrl(`/api/files/${id}/download?x-session-id=${SESSION_KEY}`),
  // analysis
  analyze: (body: any) => post<any>("/api/analyze", body),
  blind: (body: any) => post<any>("/api/blind-analysis", body),
  analyses: (params: Record<string, any> = {}) => req<any>(`/api/analyses${qs(params)}`),
  analysis: (id: string) => req<any>(`/api/analysis/${id}`),
  analysisResult: (id: string) => req<any>(`/api/analysis/${id}/result`),
  spectrum: (id: string) => req<any>(`/api/analysis/${id}/spectrum`),
  spectrogram: (id: string, maxFreq = 0, maxTime = 0) => req<any>(`/api/analysis/${id}/spectrogram${qs({ max_freq: maxFreq, max_time: maxTime })}`),
  parameters: (id: string) => req<any>(`/api/analysis/${id}/parameters`),
  evidence: (id: string) => req<any>(`/api/analysis/${id}/evidence`),
  constellation: (id: string) => req<any>(`/api/analysis/${id}/constellation`),
  eye: (id: string) => req<any>(`/api/analysis/${id}/eye`),
  bitstream: (id: string, stream = "auto") => req<any>(`/api/analysis/${id}/bitstream${qs({ stream })}`),
  segments: (id: string) => req<any>(`/api/analysis/${id}/segments`),
  detection: (id: string) => req<any>(`/api/analysis/${id}/detection`),
  region: (id: string, body: any) => post<any>(`/api/analysis/${id}/region`, body),
  selectSignal: (id: string, body: any) => post<any>(`/api/analysis/${id}/select-signal`, body),
  deleteAnalysis: (id: string) => del<any>(`/api/analysis/${id}`),
  // jobs
  job: (id: string) => req<any>(`/api/jobs/${id}`),
  jobs: () => req<any>("/api/jobs"),
  cancelJob: (id: string) => post<any>(`/api/jobs/${id}/cancel`),
  // modules
  demodulate: (body: any) => post<any>("/api/demodulate", body),
  fec: (body: any) => post<any>("/api/fec/analyze", body),
  interleaving: (body: any) => post<any>("/api/interleaving/analyze", body),
  correlate: (body: any) => post<any>("/api/correlate", body),
  compare: (body: any) => post<any>("/api/compare", body),
  comparisons: () => req<any>("/api/comparisons"),
  // generator / benchmark / reports
  generateOptions: () => req<any>("/api/generate/options"),
  generate: (body: any) => post<any>("/api/generate", body),
  generated: () => req<any>("/api/generated"),
  benchmarkConfig: () => req<any>("/api/benchmark/config"),
  benchmark: (body: any) => post<any>("/api/benchmark", body),
  benchmarks: () => req<any>("/api/benchmarks"),
  getBenchmark: (id: string) => req<any>(`/api/benchmarks/${id}`),
  makeReport: (id: string, formats: string[]) => post<any>(`/api/analysis/${id}/report`, { formats }),
  reports: () => req<any>("/api/reports"),
  reportDownloadUrl: (id: string) => apiUrl(`/api/report/${id}/download?x-session-id=${SESSION_KEY}`),
  reportInlineUrl: (id: string, format: string) => apiUrl(`/api/analysis/${id}/report?format=${format}&x-session-id=${SESSION_KEY}`),
};

/** Poll a background job until it finishes (or the caller unmounts). */
export async function pollJob(
  jobId: string,
  onTick: (j: any) => void,
  opts: { intervalMs?: number; signal?: AbortSignal } = {},
): Promise<any> {
  const interval = opts.intervalMs ?? 700;
  for (;;) {
    if (opts.signal?.aborted) throw new ApiError(0, "cancelled by the client");
    const j = await Api.job(jobId);
    onTick(j);
    if (j.status === "done" || j.status === "failed" || j.status === "cancelled") return j;
    await new Promise((r) => setTimeout(r, interval));
  }
}
