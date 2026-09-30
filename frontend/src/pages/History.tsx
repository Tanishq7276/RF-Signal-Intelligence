import { useEffect, useMemo, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { Api } from "../lib/api";
import { bytes, num, relTime, si } from "../lib/format";
import { useStore } from "../lib/store";
import { Alert, Badge, Card, Empty } from "../components/ui";

/** /history — every analysis of this workspace with filters, from the database. */
export default function History() {
  const { analyses, files, refresh, setCurrentId, toast } = useStore();
  const nav = useNavigate();
  const [q, setQ] = useState("");
  const [status, setStatus] = useState("");
  const [mode, setMode] = useState("");
  const [openId, setOpenId] = useState<string | null>(null);

  useEffect(() => { refresh(); }, [refresh]);

  const rows = useMemo(() => analyses.filter((a) => {
    const name = (a.file?.filename || "").toLowerCase();
    if (q && !name.includes(q.toLowerCase())) return false;
    if (status && a.status !== status) return false;
    if (mode && a.mode !== mode) return false;
    return true;
  }), [analyses, q, status, mode]);

  return (
    <div>
      <Card title="Analysis history" sub={`${analyses.length} stored runs · ${files.length} files · rows come from the database`}
        right={<div className="row" style={{ gap: 6 }}>
          <button className="tiny ghost" onClick={refresh}>refresh</button>
          <Link className="btn tiny" to="/analyze">new analysis</Link>
        </div>}>
        <div className="row" style={{ gap: 10 }}>
          <div style={{ flex: 1, minWidth: 200 }}><label className="f"><span>search filename</span>
            <input value={q} onChange={(e) => setQ(e.target.value)} placeholder="sample_qpsk" /></label></div>
          <div style={{ width: 150 }}><label className="f"><span>status</span>
            <select value={status} onChange={(e) => setStatus(e.target.value)}>
              <option value="">all</option>{["done", "running", "queued", "failed", "cancelled"].map((s) => <option key={s} value={s}>{s}</option>)}
            </select></label></div>
          <div style={{ width: 150 }}><label className="f"><span>mode</span>
            <select value={mode} onChange={(e) => setMode(e.target.value)}>
              <option value="">all</option>{["AUTO", "BLIND"].map((s) => <option key={s} value={s}>{s}</option>)}
            </select></label></div>
        </div>
        {rows.length ? (
          <div className="tblwrap" style={{ marginTop: 12, maxHeight: 520 }}>
            <table>
              <thead><tr><th>file</th><th>date</th><th>type</th><th>modulation</th><th className="num">SNR</th><th className="num">sym rate</th><th>FEC</th><th>status</th><th></th></tr></thead>
              <tbody>{rows.map((a) => (
                <tr key={a.analysis_id}>
                  <td>{a.file?.filename}</td>
                  <td className="small muted">{relTime(a.created_at)}</td>
                  <td><Badge kind={a.mode === "BLIND" ? "hyp" : "info"}>{a.mode}{a.kind !== "standard" ? `/${a.kind}` : ""}</Badge></td>
                  <td>{a.summary?.modulation || "—"}</td>
                  <td className="num">{num(a.summary?.snr_db, 1)}</td>
                  <td className="num">{si(a.summary?.symbol_rate_hz, "Hz", 1)}</td>
                  <td className="small">{a.summary?.fec || "—"}</td>
                  <td><Badge kind={a.status === "done" ? "ok" : a.status === "failed" ? "bad" : "warn"}>{a.status}</Badge></td>
                  <td className="row" style={{ gap: 4 }}>
                    <button className="tiny" disabled={a.status !== "done"} onClick={() => { setCurrentId(a.analysis_id); nav(`/analyze/${a.analysis_id}`); }}>open</button>
                    <Link className="btn tiny" to={`/compare?a=${a.analysis_id}`}>compare</Link>
                    <a className="btn tiny ghost" href={Api.reportInlineUrl(a.analysis_id, "pdf")} target="_blank" rel="noreferrer">pdf</a>
                    <button className="tiny danger" onClick={async () => {
                      if (!confirm("delete this analysis?")) return;
                      try { await Api.deleteAnalysis(a.analysis_id); await refresh(); toast("analysis deleted", "ok"); }
                      catch (e: any) { toast(e.message, "err"); }
                    }}>×</button>
                  </td>
                </tr>))}
              </tbody>
            </table>
          </div>
        ) : <Empty>no analysis matches the filters</Empty>}
      </Card>
      <Card title="Signal files" sub="stored captures and generated signals of this workspace">
        {files.length ? (
          <div className="tblwrap" style={{ maxHeight: 300 }}>
            <table><thead><tr><th>file</th><th>source</th><th className="num">size</th><th>format</th><th>layout</th><th className="num">rate</th><th></th></tr></thead>
              <tbody>{files.map((f) => (
                <tr key={f.file_id}>
                  <td>{f.filename}</td><td className="small">{f.source}</td>
                  <td className="num">{bytes(f.size_bytes)}</td>
                  <td className="small">{f.format}</td><td className="small">{f.iq_layout}</td>
                  <td className="num">{f.sample_rate ? si(f.sample_rate, "Hz", 0) : <span className="muted">unknown</span>}</td>
                  <td className="row" style={{ gap: 4 }}>
                    <a className="btn tiny" href={Api.downloadUrl(f.file_id)}>download</a>
                    <button className="tiny danger" onClick={async () => {
                      if (!confirm(`delete ${f.filename} and every analysis derived from it?`)) return;
                      try { await Api.deleteFile(f.file_id); await refresh(); toast("file deleted", "ok"); }
                      catch (e: any) { toast(e.message, "err"); }
                    }}>×</button>
                    <button className="tiny" onClick={() => { setOpenId(f.file_id); nav("/explorer"); }}>explore</button>
                  </td>
                </tr>))}
              </tbody></table>
          </div>
        ) : <Empty>no files yet</Empty>}
        <Alert kind="info">
          Deleting a file removes its bytes, its cached analysis results and the analyses that depend on it.
        </Alert>
      </Card>
    </div>
  );
}
