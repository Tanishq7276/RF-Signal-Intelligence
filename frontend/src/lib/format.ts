export const num = (v: any, digits = 3, fallback = "—") => {
  if (v === null || v === undefined || v === "" || (typeof v === "number" && !isFinite(v))) return fallback;
  const n = typeof v === "number" ? v : Number(v);
  if (!isFinite(n)) return fallback;
  const a = Math.abs(n);
  if (a !== 0 && (a < 1e-3 || a >= 1e6)) return n.toExponential(2);
  return n.toFixed(digits).replace(/\.?0+$/, "") || "0";
};
export const si = (v: any, unit = "", digits = 3) => {
  if (v === null || v === undefined || !isFinite(Number(v))) return "—";
  const n = Number(v), a = Math.abs(n);
  const scale = a >= 1e9 ? [1e9, "G"] : a >= 1e6 ? [1e6, "M"] : a >= 1e3 ? [1e3, "k"] : a >= 1 ? [1, ""] : a >= 1e-3 ? [1e-3, "m"] : [1e-6, "µ"];
  return `${(n / (scale[0] as number)).toFixed(digits).replace(/\.?0+$/, "")} ${scale[1]}${unit}`.trim();
};
export const bytes = (v: any) => {
  const n = Number(v || 0);
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} kB`;
  if (n < 1024 ** 3) return `${(n / 1024 ** 2).toFixed(2)} MB`;
  return `${(n / 1024 ** 3).toFixed(2)} GB`;
};
export const pct = (v: any, digits = 0) => (v === null || v === undefined || !isFinite(Number(v)) ? "—" : `${(Number(v) * 100).toFixed(digits)} %`);
export const statusClass = (status?: string) => {
  const s = (status || "").toLowerCase();
  if (["ok", "done", "detected", "similar"].includes(s)) return "ok";
  if (["low_confidence", "low", "partial", "partial overlap", "warning", "queued", "running"].includes(s)) return "warn";
  if (["unable", "failed", "error", "cancelled", "none"].includes(s)) return "bad";
  return "info";
};
export const relTime = (iso?: string | null) => {
  if (!iso) return "—";
  const t = new Date(iso).getTime();
  if (!isFinite(t)) return String(iso);
  const dt = (Date.now() - t) / 1000;
  if (dt < 60) return `${Math.max(1, Math.round(dt))} s ago`;
  if (dt < 3600) return `${Math.round(dt / 60)} min ago`;
  if (dt < 86400) return `${Math.round(dt / 3600)} h ago`;
  return new Date(iso).toLocaleString();
};
export const PLOT_BASE = {
  paper_bgcolor: "rgba(0,0,0,0)",
  plot_bgcolor: "#0c1524",
  font: { color: "#c8d6ee", size: 11 },
  margin: { l: 52, r: 16, t: 28, b: 40 },
  xaxis: { gridcolor: "#1c2b45", zerolinecolor: "#243450" },
  yaxis: { gridcolor: "#1c2b45", zerolinecolor: "#243450" },
  legend: { orientation: "h", y: 1.08, font: { size: 10 } },
  hovermode: "closest",
};
export const PLOT_CONFIG = { displaylogo: false, responsive: true, modeBarButtonsToRemove: ["lasso2d", "select2d", "autoScale2d"] as any };
