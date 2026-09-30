import { useEffect, useMemo, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { Api } from "../lib/api";
import { num, relTime, si } from "../lib/format";
import { useStore } from "../lib/store";
import { Alert, Badge, Card, Empty, Stat } from "../components/ui";

export default function Dashboard() {
  const { analyses, files, health, setCurrentId, refresh, demoLoad, toast } = useStore();
  const nav = useNavigate();
  const [stats, setStats] = useState<any>(null);
  useEffect(() => { Api.stats().then(setStats).catch(() => undefined); }, [analyses.length]);

  const agg = useMemo(() => {
    const done = analyses.filter((a) => a.status === "done");
    const snrs = done.map((a) => Number(a.summary?.snr_db)).filter((v) => isFinite(v) && v !== null && v!==undefined) as number[];
    const mods: Record<string, number> = {};
    done.forEach((a) => { const m = a.summary?.modulation; if (m) mods[m] = (mods[m] || 0) + 1; });
    return {
      done: done.length,
      failed: analyses.filter((a) => a.status === "failed").length,
      running: analyses.filter((a) => a.status === "running" || a.status === "queued").length,
      avgSnr: snrs.length ? snrs.reduce((s, v) => s + v, 0) / snrs.length : null,
      emissions: done.reduce((s, a) => s + (a.summary?.n_emissions || 0), 0),
      mods,
    };
  }, [analyses]);

  const open = (id: string) => { setCurrentId(id); nav(`/analyze/${id}`); };

  return (
    <div>
      <Card title="Session overview" sub="everything below is read from the database of this workspace"
        right={<div className="row" style={{ gap: 6 }}>
          <Link className="btn tiny" to="/analyze">new analysis</Link>
          <button className="tiny ghost" onClick={refresh}>refresh</button>
        </div>}>
        <div className="grid g4">
          <Stat k="Analyses" v={analyses.length} n={`${agg.done} completed · ${agg.running} running · ${agg.failed} failed`} />
          <Stat k="Files" v={files.length} n={stats ? `${(stats.files_bytes / 1024).toFixed(0)} kB stored` : ""} />
          <Stat k="Average SNR" v={agg.avgSnr !== null ? `${num(agg.avgSnr, 1)} dB` : "—"} n={`over ${agg.done} completed analyses`} />
          <Stat k="Detected emissions" v={agg.emissions} n="summed across all analyses" />
        </div>
        <div className="row" style={{ marginTop: 12, gap: 8 }}>
          <span className="small muted">modulations encountered:</span>
          {Object.entries(agg.mods).length
            ? Object.entries(agg.mods).map(([m, c]) => <span key={m} className="pill">{m} × {c}</span>)
            : <span className="small muted">none yet</span>}
        </div>
        {health && health.status !== "ok" && (
          <Alert kind="warn" title="backend degraded">{[Object.entries(health.checks || {}).filter(([, v]: any) => !v.ok).map(([k]) => k).join(", ")]}</Alert>
        )}
      </Card>

      <div className="grid g3">
        <Card title="Quick start">
          <div className="col" style={{ gap: 8 }}>
            <div className="small muted">Load a shipped demo signal (real recording generated with a known ground truth):</div>
            <div className="row">
              {["sample_qpsk.wav", "sample_bpsk.wav", "sample_16qam.wav", "sample_noisy_qpsk.wav"].map((n) => (
                <button key={n} className="tiny" onClick={async () => { const f = await demoLoad(n); if (f) nav("/analyze"); }}>{n}</button>
              ))}
            </div>
            <div className="hr" />
            <div className="row">
              <Link className="btn tiny" to="/generator">generate a signal</Link>
              <Link className="btn tiny" to="/blind-analysis">blind analysis</Link>
              <Link className="btn tiny" to="/benchmark">self-benchmark</Link>
              <a className="btn tiny ghost" href="/docs" target="_blank" rel="noreferrer">API docs</a>
            </div>
          </div>
        </Card>
        <Card title="Workspace storage" sub="per-session isolation">
          {stats ? (
            <div className="kv">
              {[["files", stats.files], ["analyses", stats.analyses], ["reports", stats.reports],
                ["generated signals", stats.generated_signals], ["disk free", `${stats.disk.free_gb} GB`],
                ["background workers", `${stats.jobs.workers} (${stats.jobs.active} active)`]].map(([k, v], i) => (
                <div key={i} style={{ display: "contents" }}><div className="k">{k as string}</div><div className="v">{String(v)}</div></div>
              ))}
            </div>
          ) : <Empty>statistics unavailable</Empty>}
          <div className="row" style={{ marginTop: 10 }}>
            <button className="tiny danger" onClick={async () => {
              if (!confirm("Delete every file, analysis and report in this workspace?")) return;
              try { const r = await Api.clearSession(); await refresh(); toast(`${r.removed_files} file(s) removed`, "ok"); }
              catch (e: any) { toast(e.message, "err"); }
            }}>clear workspace</button>
          </div>
        </Card>
        <Card title="Capabilities">
          <div className="small muted">what the backend reports it can do (from /api/capabilities)</div>
          <div className="row" style={{ marginTop: 6, gap: 6 }}>
            {["BPSK", "QPSK", "8PSK", "16QAM", "64QAM", "2FSK", "GFSK", "AM", "FM"].map((m) => <span key={m} className="pill">{m}</span>)}
          </div>
          <div className="small muted" style={{ marginTop: 8 }}>Demodulators, FEC families, interleaver geometries, 7 dtypes, WAV/raw/NPY containers and the DSP/MD modules are all exposed through the REST API — see <a href="/docs" target="_blank" rel="noreferrer">/docs</a>.</div>
        </Card>
      </div>

      <Card title="Recent analyses" sub="open any run to inspect its spectra, waterfall, hypotheses and report">
        {analyses.length ? (
          <div className="tblwrap">
            <table>
              <thead><tr><th>file</th><th>when</th><th>type</th><th>modulation</th><th className="num">SNR</th><th className="num">sym rate</th><th>FEC</th><th>status</th><th></th></tr></thead>
              <tbody>
                {analyses.slice(0, 25).map((a) => (
                  <tr key={a.analysis_id}>
                    <td>{a.file?.filename || "—"}</td>
                    <td className="small muted">{relTime(a.created_at)}</td>
                    <td><Badge kind={a.mode === "BLIND" ? "hyp" : "info"}>{a.mode}{a.kind !== "standard" ? ` · ${a.kind}` : ""}</Badge></td>
                    <td>{a.summary?.modulation ? <>{a.summary.modulation} <span className="muted small">{num(a.summary.modulation_confidence, 2)}</span></> : <span className="muted">—</span>}</td>
                    <td className="num">{num(a.summary?.snr_db, 1)}</td>
                    <td className="num">{si(a.summary?.symbol_rate_hz, "Hz", 1)}</td>
                    <td className="small">{a.summary?.fec || <span className="muted">none claimed</span>}</td>
                    <td><Badge kind={a.status === "done" ? "ok" : a.status === "failed" ? "bad" : "warn"}>{a.status}</Badge></td>
                    <td className="row" style={{ gap: 4 }}>
                      <button className="tiny" disabled={a.status !== "done"} onClick={() => open(a.analysis_id)}>open</button>
                      <Link className="btn tiny" to={`/compare?a=${a.analysis_id}`}>compare</Link>
                      <a className="btn tiny ghost" href={Api.reportInlineUrl(a.analysis_id, "pdf")} target="_blank" rel="noreferrer">report</a>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : <Empty>No analyses yet. Upload a capture or load a demo signal from the Analyze page.</Empty>}
      </Card>
    </div>
  );
}
