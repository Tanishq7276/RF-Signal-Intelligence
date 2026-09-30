/** Chart builders: turn the API payloads into Plotly traces (no hard-coded numbers). */
import { num, si } from "../lib/format";

export const spectrumTraces = (spec: any, opts: { color?: string; name?: string } = {}) => {
  const a = spec?.arrays || spec?.spectrum?.arrays || {};
  const f = a.freq_hz || [];
  const p = a.psd_db || a.db || [];
  if (!f.length || !p.length) return [];
  const n = Math.min(f.length, p.length);
  return [{
    type: "scattergl", mode: "lines", x: f.slice(0, n), y: p.slice(0, n),
    line: { color: opts.color || "#22d3ee", width: 1.2 }, name: opts.name || "PSD",
    hovertemplate: "%{x:.0f} Hz<br>%{y:.2f} dB<extra></extra>",
  }];
};

export const spectrumShapes = (spec: any) => {
  const sp = spec?.spectrum || spec || {};
  const shapes: any[] = [], annotations: any[] = [];
  if (sp.center_frequency_hz != null) {
    shapes.push({ type: "line", x0: sp.center_frequency_hz, x1: sp.center_frequency_hz, y0: 0, y1: 1, yref: "paper", line: { color: "#f59e0b", width: 1, dash: "dash" } });
    annotations.push({ x: sp.center_frequency_hz, y: 1, yref: "paper", text: `carrier ${si(sp.center_frequency_hz, "Hz", 2)}`, showarrow: false, font: { size: 10, color: "#f59e0b" }, yanchor: "bottom" });
  }
  if (sp.obw_lo_hz != null && sp.obw_hi_hz != null) {
    shapes.push({ type: "rect", x0: sp.obw_lo_hz, x1: sp.obw_hi_hz, y0: 0, y1: 1, yref: "paper", fillcolor: "rgba(52,211,153,.09)", line: { color: "#34d399", width: 1, dash: "dot" } });
    annotations.push({ x: (sp.obw_lo_hz + sp.obw_hi_hz) / 2, y: 0.06, yref: "paper", text: `99 % OBW ${si(sp.obw_99_hz, "Hz", 2)}`, showarrow: false, font: { size: 10, color: "#34d399" } });
  }
  return { shapes, annotations };
};

export const waterfallTrace = (wf: any) => {
  if (!wf?.db?.length) return [];
  return [{
    type: "heatmap",
    z: wf.db, x: wf.times_s, y: wf.freq_hz,
    colorscale: "Viridis", colorbar: { title: { text: "dB", side: "right" }, thickness: 12, len: 0.9 },
    hovertemplate: "t %{x:.4f} s<br>f %{y:.0f} Hz<br>%{z:.1f} dB<extra></extra>",
  }] as any[];
};

export const waterfallShapes = (detection: any) => {
  const shapes: any[] = [], annotations: any[] = [];
  (detection?.signals || []).slice(0, 8).forEach((s: any, i: number) => {
    shapes.push({ type: "rect", x0: s.t0_s, x1: s.t1_s, y0: s.f_lo_hz, y1: s.f_hi_hz, line: { color: "#f87171", width: 1 }, fillcolor: "rgba(248,113,113,0.05)" });
    annotations.push({ x: s.t0_s, y: s.f_hi_hz, text: s.id || `S${i + 1}`, showarrow: false, font: { size: 10, color: "#fca5a5" } });
  });
  return { shapes, annotations };
};

export const constellationTrace = (con: any) => {
  if (!con?.i?.length) return [];
  const pts = { type: "scattergl", mode: "markers", x: con.i, y: con.q, marker: { size: 3, color: "#22d3ee", opacity: 0.45 }, name: "symbols", hovertemplate: "I %{x:.3f}<br>Q %{y:.3f}<extra></extra>" };
  const out: any[] = [pts];
  if (con.reference_i?.length) {
    out.push({ type: "scatter", mode: "markers", x: con.reference_i, y: con.reference_q, marker: { size: 9, symbol: "x", color: "#f59e0b", line: { width: 1 } }, name: "ideal", hoverinfo: "skip" });
  }
  return out;
};

export const eyeTrace = (eye: any) => {
  const traces: any[] = [];
  const n = eye?.n_traces || 0;
  (eye?.traces_i || []).forEach((tr: number[], i: number) => {
    traces.push({
      type: "scatter", mode: "lines", x: (eye.time_axis_symbols || []).slice(0, tr.length), y: tr,
      line: { color: i === 0 ? "#22d3ee" : "rgba(34,211,238,0.16)", width: i === 0 ? 1.6 : 1 },
      showlegend: false, hoverinfo: "skip", name: `trace ${i}`,
    });
  });
  return { traces: traces.slice(0, n || traces.length), tracesQ: [] };
};

export const byteHistogramTrace = (bs: any) => {
  const h = bs?.byte_histogram || {};
  const counts = h.counts || h.histogram || h.values || [];
  if (!counts.length) return [];
  const bins = h.bins || Array.from({ length: counts.length }, (_, i) => i);
  return [{ type: "bar", x: bins, y: counts, marker: { color: "#3b82f6" }, hovertemplate: "byte %{x}<br>%{y} occurrences<extra></extra>" }];
};

export const autocorrTrace = (bs: any) => {
  const ac = bs?.autocorrelation || [];
  if (!ac.length) return [];
  return [{ type: "scattergl", mode: "lines", x: Array.from({ length: ac.length }, (_, i) => i), y: ac, line: { color: "#a78bfa" }, hovertemplate: "lag %{x} bits<br>correlation %{y:.3f}<extra></extra>" }];
};

export const confusionHeatmap = (classes: string[], matrix: number[][]) => [
  { type: "heatmap", z: matrix, x: classes, y: classes, colorscale: "Blues", colorbar: { title: { text: "count" } }, hovertemplate: "true %{y}<br>predicted %{x}<br>%{z} cases<extra></extra>" },
];

export const hypothesisBars = (hyps: any[], labelKey: (h: any) => string) => [{
  type: "bar", orientation: "h",
  y: hyps.map(labelKey).reverse(),
  x: hyps.map((h) => Number(h.confidence ?? h.probability ?? 0)).reverse(),
  marker: { color: hyps.map((h) => (Number(h.confidence ?? h.probability ?? 0) >= 0.6 ? "#34d399" : Number(h.confidence ?? 0) >= 0.25 ? "#f59e0b" : "#f87171")).reverse() },
  hovertemplate: "%{y}<br>confidence %{x:.2f}<extra></extra>",
}];

export const waterfallLayout = (wf: any) => ({
  xaxis: { title: "time [s]", gridcolor: "#1c2b45" },
  yaxis: { title: "frequency [Hz]", gridcolor: "#1c2b45" },
  title: wf?.nperseg ? `STFT ${wf.nperseg}-point, ${wf.window || "Hann"} window · ${wf.n_frames_full || ""} frames` : undefined,
});
