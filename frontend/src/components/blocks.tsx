import React, { useState } from "react";
import Plot from "./Plot";
import { num, pct, si } from "../lib/format";
import { Alert, Badge, Bar, Card, ConfidenceCell, Empty, Stat } from "./ui";

/* ------------------------------------------------------------------ parameters */
export type Param = {
  name: string; value: any; unit?: string | null; status?: string; confidence?: number | null;
  method?: string | null; evidence?: string[] | null; limitations?: string[] | null; group?: string;
};

export const ParamTable = ({ params, groups }: { params: Param[]; groups?: string[] }) => {
  const [group, setGroup] = useState<string>("all");
  const list = (params || []).filter((p) => group === "all" || (p.group || "signal") === group);
  if (!params?.length) return <Empty>No parameter record was produced for this signal.</Empty>;
  return (
    <div>
      {groups && groups.length > 1 && (
        <div className="tabs">
          <button className={group === "all" ? "active" : ""} onClick={() => setGroup("all")}>all ({params.length})</button>
          {groups.map((g) => (
            <button key={g} className={group === g ? "active" : ""} onClick={() => setGroup(g)}>
              {g} ({params.filter((p) => (p.group || "signal") === g).length})
            </button>
          ))}
        </div>
      )}
      <div className="tblwrap">
        <table>
          <thead><tr>
            <th>Parameter</th><th className="num">Value</th><th>Unit</th><th>Status</th>
            <th>Confidence</th><th>Method</th>
          </tr></thead>
          <tbody>
            {list.map((p, i) => (
              <tr key={i}>
                <td>
                  <div>{p.name}</div>
                  {!!p.evidence?.length && <div className="small muted">{p.evidence.slice(0, 2).join(" · ")}</div>}
                  {!!p.limitations?.length && <div className="small" style={{ color: "#fcd34d" }}>limit: {p.limitations[0]}</div>}
                </td>
                <td className="num">{p.status === "unable" ? <span className="muted">unable</span> : num(p.value, 4)}</td>
                <td className="small">{p.unit || "—"}</td>
                <td><Badge kind={p.status === "ok" ? "ok" : p.status === "unable" ? "bad" : "warn"}>{p.status || "—"}</Badge></td>
                <td><ConfidenceCell v={p.confidence} /></td>
                <td className="small muted">{p.method || "—"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="small muted" style={{ marginTop: 6 }}>
        Every row is a measured or estimated quantity with its own method and confidence. “unable” means the
        quantity could not be estimated reliably from this record — no number is invented for it.
      </div>
    </div>
  );
};

/* ------------------------------------------------------------------ stages */
export const StageList = ({ stages, title }: { stages: any[]; title?: string }) => (
  <div>
    {title && <div className="sub">{title}</div>}
    <div className="tblwrap" style={{ maxHeight: 420 }}>
      <table>
        <thead><tr><th>Stage</th><th>Status</th><th>Detail</th><th>Method / evidence</th></tr></thead>
        <tbody>
          {(stages || []).map((s, i) => (
            <tr key={i}>
              <td>{s.stage || s.name}</td>
              <td><Badge kind={s.status === "ok" ? "ok" : s.status === "failed" ? "bad" : "warn"}>{s.status}</Badge></td>
              <td className="small">{s.detail || s.note || s.error || "—"}
                {s.duration_ms !== undefined && <span className="muted"> ({num(s.duration_ms, 1)} ms)</span>}
              </td>
              <td className="small muted">
                {s.method || ""}
                {!!(s.evidence || []).length && <div>{s.evidence.slice(0, 3).join(" · ")}</div>}
                {!!(s.limitations || []).length && <div style={{ color: "#fcd34d" }}>{s.limitations[0]}</div>}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  </div>
);

/* ------------------------------------------------------------------ hypotheses */
export const HypothesisList = ({ hypotheses, kind }: { hypotheses: any[]; kind: "fec" | "interleave" | "modulation" | "generic" }) => {
  if (!hypotheses?.length) {
    if (kind === "interleave") return <Empty>No interleaver geometry identified</Empty>;
    if (kind === "fec") return <Empty>No FEC identified above evidence threshold</Empty>;
    return <Empty>No hypothesis could be formed for this record.</Empty>;
  }
  return (
    <div className="col">
      {hypotheses.map((h, i) => {
        const conf = h.confidence ?? h.probability ?? 0;
        const fam = (h.family || "").toLowerCase();
        const isLdpc = fam === "ldpc" || (h.hypothesis || "").toLowerCase().includes("ldpc");
        const isUnable = h.status === "unable";

        // Determine clear user-facing status label & badge kind
        let statusLabel = h.status;
        let badgeKind = "info";

        if (h.status === "ok") {
          statusLabel = "Confirmed hypothesis";
          badgeKind = "ok";
        } else if (h.status === "low_confidence" || (conf > 0 && conf < 0.20)) {
          statusLabel = "Low confidence hypothesis";
          badgeKind = "warn";
        } else if (isUnable) {
          if (kind === "fec") {
            if (isLdpc) {
              statusLabel = "LDPC not identifiable blindly";
              badgeKind = "info";
            } else if (fam === "convolutional" || fam === "reed-solomon") {
              statusLabel = "Not enough evidence";
              badgeKind = "info";
            } else {
              statusLabel = "No evidence above threshold";
              badgeKind = "info";
            }
          } else if (kind === "interleave") {
            statusLabel = "No geometry beat control";
            badgeKind = "info";
          } else {
            statusLabel = "Insufficient evidence";
            badgeKind = "info";
          }
        } else if (!h.status) {
          statusLabel = conf >= 0.20 ? "Hypothesis" : "Tested candidate";
          badgeKind = conf >= 0.6 ? "ok" : conf >= 0.20 ? "warn" : "info";
        }

        // Specific explanation for LDPC if not already specified
        let extraExplanation = h.explanation;
        if (isLdpc && (!extraExplanation || !extraExplanation.includes("parity-check matrix"))) {
          extraExplanation = "A parity-check matrix is required for a reliable LDPC identification.";
        }

        return (
          <div key={i} className="card" style={{ margin: 0, padding: 12 }}>
            <div className="row" style={{ gap: 10, flexWrap: "wrap" }}>
              <Badge kind={conf >= 0.6 ? "ok" : conf >= 0.20 ? "warn" : "info"}>
                {kind === "modulation" ? (h.modulation || h.label) : (h.hypothesis || h.kind || "—")}
              </Badge>
              {h.rank && <span className="pill">rank {h.rank}</span>}
              {h.family && <span className="pill">{h.family}</span>}
              {statusLabel && <Badge kind={badgeKind}>{statusLabel}</Badge>}
              {h.rank_excluded && <span className="pill" title="this test measures channel memory, not a permutation, so it is excluded from the ranking">indication only</span>}
              <span style={{ flex: 1 }} />
              <div style={{ minWidth: 150 }}>
                <div className="row" style={{ gap: 6 }}>
                  <span className="small muted">confidence</span>
                  <span className="mono small">{conf.toFixed ? conf.toFixed(2) : num(conf, 2)}</span>
                </div>
                <Bar value={conf} tone={conf >= 0.6 ? "ok" : conf >= 0.20 ? "warn" : undefined} />
              </div>
            </div>
            {(h.evidence || []).length > 0 && (
              <ul className="small" style={{ margin: "8px 0 0 18px", color: "#c8d6ee" }}>
                {h.evidence.map((e: string, j: number) => <li key={j}>{e}</li>)}
              </ul>
            )}
            {!!(h.limitations || []).length && (
              <div className="small" style={{ color: "#fcd34d", marginTop: 6 }}>
                Limitations: {h.limitations.join(" · ")}
              </div>
            )}
            {extraExplanation && <div className="small muted" style={{ marginTop: 6 }}>{extraExplanation}</div>}
            {h.relative_improvement !== undefined && h.relative_improvement !== null && (
              <div className="small mono" style={{ marginTop: 6 }}>
                relative decode improvement {num(h.relative_improvement, 3)}
                {h.align_offset_bits != null && <> · alignment offset {h.align_offset_bits} bits</>}
                {h.beats_control_max != null && <> · beats random-permutation control: {String(h.beats_control_max)}</>}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
};

/* ------------------------------------------------------------------ quality + confidence */
export const QualityPanel = ({ result, summary }: { result: any; summary: any }) => {
  const sig = result?.signal || {};
  const conf = sig.confidence || {};
  const q = result?.quality || {};
  return (
    <div className="grid g4">
      <Stat k="Spectral SNR" v={summary?.snr_db != null ? `${num(summary.snr_db, 1)} dB` : "—"} n="noise-density referenced" />
      <Stat k="Demod EVM" v={summary?.evm_percent != null ? `${num(summary.evm_percent, 2)} %` : "not available"} n={summary?.demodulation_ok ? "demodulated" : "no demodulation"} />
      <Stat k="Symbol rate" v={summary?.symbol_rate_hz ? si(summary.symbol_rate_hz, "sym/s") : "—"} n={sig.symbol_rate_note ? "refined by the demodulator" : "blind estimate"} />
      <Stat k="Overall confidence" v={conf.overall != null ? pct(conf.overall) : "—"} n="weighted over the automatic results" />
      {q.summary && <div className="grid" style={{ gridColumn: "1/-1" }}><Alert kind="info">{q.summary}</Alert></div>}
      <div style={{ gridColumn: "1/-1" }}>
        <div className="tblwrap" style={{ maxHeight: 300 }}>
          <table>
            <thead><tr><th>Automatic result</th><th>Confidence</th><th>Basis</th></tr></thead>
            <tbody>
              {(conf.items || []).map((it: any, i: number) => (
                <tr key={i}><td>{it.name}</td><td><ConfidenceCell v={it.confidence} /></td><td className="small muted">{it.basis}</td></tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

/* ------------------------------------------------------------------ 3-D pipeline / evidence */
export const Pipeline3D = ({ nodes, height = 380 }: { nodes: any[]; height?: number }) => {
  const list = nodes || [];
  if (!list.length) return <Empty>The evidence graph is empty for this analysis.</Empty>;
  const n = list.length;
  const xs: number[] = [], ys: number[] = [], zs: number[] = [], text: string[] = [], colors: string[] = [], labels: string[] = [];
  list.forEach((nd: any, i: number) => {
    const angle = (i / Math.max(1, n - 1)) * Math.PI * 1.6;
    xs.push(Math.cos(angle) * 2.4);
    ys.push(Math.sin(angle) * 2.4);
    zs.push(i);
    const map: Record<string, number> = { ok: 1, partial: 0.6, low_confidence: 0.4, failed: 0.1, none: 0.1, skipped: 0.2 };
    const conf = typeof nd.confidence === "number" ? nd.confidence : (map[nd.status] ?? 0.5);
    colors.push(nd.status === "failed" ? "#f87171" : nd.status === "partial" ? "#f59e0b" : conf > 0.6 ? "#34d399" : "#22d3ee");
    text.push(`<b>${nd.id}</b><br>${nd.label || ""}<br>status: ${nd.status || "—"}<br>${(nd.detail || "").slice(0, 260)}`);
    labels.push(`${i + 1}. ${nd.id}`);
  });
  const lineX: any[] = [], lineY: any[] = [], lineZ: any[] = [];
  for (let i = 0; i < n - 1; i++) {
    lineX.push(xs[i], xs[i + 1], null); lineY.push(ys[i], ys[i + 1], null); lineZ.push(zs[i], zs[i + 1], null);
  }
  return (
    <div>
      <Plot
        data={[
          { type: "scatter3d", mode: "lines", x: lineX, y: lineY, z: lineZ, line: { color: "#24405f", width: 4 }, hoverinfo: "skip", showlegend: false },
          { type: "scatter3d", mode: "markers+text", x: xs, y: ys, z: zs, text: labels, textposition: "top center",
            textfont: { color: "#c8d6ee", size: 9 }, marker: { size: 10, color: colors, line: { color: "#08111f", width: 1 } },
            hovertext: text, hoverinfo: "text", showlegend: false },
        ]}
        layout={{
          scene: {
            bgcolor: "#0b1322",
            xaxis: { title: "", showgrid: true, gridcolor: "#1b2942", zeroline: false, showticklabels: false },
            yaxis: { title: "", showgrid: true, gridcolor: "#1b2942", zeroline: false, showticklabels: false },
            zaxis: { title: "pipeline step", gridcolor: "#1b2942", tickfont: { size: 9 }, autorange: "reversed" },
            camera: { eye: { x: 1.5, y: 1.4, z: 0.9 } },
          },
          margin: { l: 0, r: 0, t: 10, b: 0 }, showlegend: false,
        }}
        height={height}
      />
      <div className="legend" style={{ marginTop: 6 }}>
        <span><i className="dotc" style={{ background: "#34d399" }} />high confidence</span>
        <span><i className="dotc" style={{ background: "#22d3ee" }} />measured</span>
        <span><i className="dotc" style={{ background: "#f59e0b" }} />partial</span>
        <span><i className="dotc" style={{ background: "#f87171" }} />failed / not available</span>
        <span className="muted">rotate with the mouse; each node is one step whose output feeds the next</span>
      </div>
      <div style={{ marginTop: 10 }}>
        {list.map((nd: any, i: number) => (
          <div className="evnode" key={i}>
            <div className="idx">{i + 1}</div>
            <div style={{ flex: 1 }}>
              <div className="row" style={{ gap: 8 }}>
                <b>{nd.id}</b><span className="muted small">{nd.label}</span>
                <Badge kind={nd.status === "ok" ? "ok" : nd.status === "partial" ? "warn" : "bad"}>{nd.status || "—"}</Badge>
              </div>
              <div className="small muted">{nd.detail}</div>
              {!!(nd.inputs || []).length && <div className="small" style={{ color: "#9fb0cc" }}>inputs: {(nd.inputs || []).join(", ")}</div>}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

/* ------------------------------------------------------------------ segments */
export const SegmentTable = ({ segments, onSelect, onRegion }: any) => (
  <div className="tblwrap">
    <table>
      <thead><tr>
        <th>#</th><th className="num">Centre</th><th className="num">Bandwidth</th><th className="num">t₀</th>
        <th className="num">Duration</th><th className="num">Peak</th><th className="num">SNR</th><th>Conf.</th><th>Kind</th><th></th>
      </tr></thead>
      <tbody>
        {(segments || []).map((s: any, i: number) => (
          <tr key={i}>
            <td>{s.label || i + 1}{s.selected && <span className="pill" style={{ marginLeft: 4 }}>analysed</span>}</td>
            <td className="num">{si(s.center_frequency_hz, "Hz")}</td>
            <td className="num">{si(s.bandwidth_hz, "Hz")}</td>
            <td className="num">{num(s.t0_s, 4)} s</td>
            <td className="num">{num(s.duration_s, 4)} s</td>
            <td className="num">{num(s.peak_power_db, 1)} dB</td>
            <td className="num">{num(s.snr_db, 1)} dB</td>
            <td><ConfidenceCell v={s.confidence} /></td>
            <td className="small">{s.kind || "—"}{s.n_bursts ? ` · ${s.n_bursts} bursts` : ""}</td>
            <td className="row" style={{ gap: 4 }}>
              {onSelect && <button className="tiny" onClick={() => onSelect(i)}>analyse</button>}
              {onRegion && <button className="tiny ghost" onClick={() => onRegion(s)}>region</button>}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  </div>
);
