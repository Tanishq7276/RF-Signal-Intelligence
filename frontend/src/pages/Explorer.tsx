import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { Api } from "../lib/api";
import { bytes, num, si } from "../lib/format";
import { useStore } from "../lib/store";
import Plot from "../components/Plot";
import { Alert, Badge, Card, Empty, Json, KV, NumField, Select, Spinner, Stat, Tabs } from "../components/ui";

/** /explorer — the parser view: file metadata, detection evidence, waveform and FFT. */
export default function Explorer() {
  const { files, toast } = useStore();
  const nav = useNavigate();
  const [fileId, setFileId] = useState<string | null>(files[0]?.file_id ?? null);
  const [preview, setPreview] = useState<any>(null);
  const [detail, setDetail] = useState<any>(null);
  const [busy, setBusy] = useState(false);
  const [tab, setTab] = useState("waveform");
  const [fmt, setFmt] = useState<any>({ dtype: "auto", byte_order: "auto", iq_layout: "auto", sample_rate: null });

  useEffect(() => { if (!fileId && files.length) setFileId(files[0].file_id); }, [files, fileId]);

  const load = async () => {
    if (!fileId) return;
    setBusy(true); setPreview(null); setDetail(null);
    try {
      const [pv, de] = await Promise.all([Api.preview(fileId, 4000), Api.getFile(fileId)]);
      setPreview(pv); setDetail(de);
    } catch (e: any) { toast(`could not load the file: ${e.message}`, "err"); }
    finally { setBusy(false); }
  };
  useEffect(() => { load(); /* eslint-disable-next-line */ }, [fileId]);

  const override = Object.fromEntries(Object.entries(fmt).filter(([, v]) => v !== null && v !== "auto"));
  const reanalyze = () => {
    if (!fileId) return;
    runWithOverride(fileId, override);
  };
  const { runAnalysis, setCurrentId } = useStore();
  const runWithOverride = async (id: string, ov: any) => {
    const res = await runAnalysis({ file_id: id, options: Object.keys(ov).length ? { format_override: ov } : {}, wait_s: 0 },
      { label: "Analysis with format override" });
    if (res?.analysis_id) { setCurrentId(res.analysis_id); nav(`/analyze/${res.analysis_id}`); }
  };

  const f = detail?.file;
  const report: any[] = detail?.detection_report || [];

  return (
    <div>
      <Card title="Signal explorer" sub="inspect what the parser found before running a full analysis"
        right={<div className="row" style={{ gap: 6 }}>
          <button className="tiny ghost" onClick={load}>reload</button>
          <button className="tiny" disabled={!fileId} onClick={reanalyze}>analyse with these settings</button>
        </div>}>
        <Select label="file" value={fileId ?? ""} onChange={(v) => setFileId(v || null)}
          options={files.map((x) => ({ value: x.file_id, label: `${x.filename} · ${bytes(x.size_bytes)}` }))} />
      </Card>

      {!files.length && <Empty>No file in this workspace yet — upload one or use the demo buttons on the Analyze page.</Empty>}
      {busy && <Spinner label="reading the file" />}

      {f && (
        <>
          <div className="grid g4">
            <Stat k="Filename" v={<span style={{ fontSize: 13 }}>{f.filename}</span>} n={`${bytes(f.size_bytes)} · sha256 ${String(f.sha256 || "").slice(0, 12)}…`} />
            <Stat k="Format" v={<span style={{ fontSize: 14 }}>{f.format || "unknown"}</span>} n={`confidence ${num(f.confidence, 2)} · ${f.container}`} />
            <Stat k="I/Q layout" v={<span style={{ fontSize: 14 }}>{f.iq_layout || "unknown"}</span>} n={`${f.channels ?? "?"} channel(s) · ${f.dtype || "?"} · ${f.endianness || "?"}`} />
            <Stat k="Sample rate" v={f.sample_rate_known ? si(f.sample_rate, "Hz", 0) : "Unknown / requires estimation"}
              n={f.sample_rate_known ? (f.sample_rate_source || "file metadata") : "no rate metadata in the file — the pipeline estimates it"} />
            <Stat k="Samples" v={num(f.n_samples, 0)} n={f.duration_s ? `${num(f.duration_s, 6)} s at that rate` : "duration follows from the rate"} />
            <Stat k="Centre frequency" v={f.center_frequency_known ? si(f.center_frequency, "Hz", 0) : "Unknown / requires estimation"} n={f.center_frequency_known ? f.center_frequency_source : "not present in the metadata"} />
            <Stat k="Source" v={f.source} n={f.created_at ? new Date(f.created_at).toLocaleString() : ""} />
            <Stat k="Parser confidence" v={num(f.confidence, 3)} n={`${report.length} evidence rows`} />
          </div>

          <Card title="Format detection evidence" sub="how the parser decided, and what a manual override would change">
            {report.length ? (
              <div className="tblwrap" style={{ maxHeight: 280 }}>
                <table><thead><tr><th>item</th><th>value</th><th>evidence</th></tr></thead>
                  <tbody>{report.map((r: any, i: number) => (
                    <tr key={i}><td>{r.name || r.field || r.key}</td><td className="mono small">{String(r.value ?? "")}</td>
                      <td className="small muted">{Array.isArray(r.evidence) ? r.evidence.join(" · ") : String(r.evidence || r.note || "")}</td></tr>
                  ))}</tbody></table>
              </div>
            ) : <Alert kind="warn">No detection report was stored for this file — re-upload to regenerate it.</Alert>}
            <div className="hr" />
            <div className="sub">manual override (use when the automatic detection is wrong or uncertain)</div>
            <div className="grid g4">
              <Select label="dtype" value={fmt.dtype} onChange={(v) => setFmt({ ...fmt, dtype: v })}
                options={["auto", "int8", "uint8", "int16", "uint16", "int32", "float32", "float64"].map((d) => ({ value: d, label: d }))} />
              <Select label="endianness" value={fmt.byte_order} onChange={(v) => setFmt({ ...fmt, byte_order: v })}
                options={["auto", "little", "big"].map((d) => ({ value: d, label: d }))} />
              <Select label="I/Q layout" value={fmt.iq_layout} onChange={(v) => setFmt({ ...fmt, iq_layout: v })}
                options={["auto", "interleaved", "real", "imag_only"].map((d) => ({ value: d, label: d }))} />
              <NumField label="sample rate [Hz]" value={fmt.sample_rate} step={1000} min={1}
                placeholder="leave empty to keep the file/none"
                onChange={(v: number | null) => setFmt({ ...fmt, sample_rate: v })} />
            </div>
            <div className="row" style={{ marginTop: 8 }}>
              <button className="tiny" onClick={reanalyze}>analyse with override</button>
              <span className="small muted">the override is recorded in the analysis options and marked as manual (confidence 1.0 by definition)</span>
            </div>
          </Card>

          <Card title="Signal preview" sub="downsampled for transport; the analysis always uses every sample you uploaded"
            right={<span className="pill">{preview?.downsampled ? `downsampled to ≤${preview.max_points} pts` : "full resolution"}</span>}>
            <Tabs tabs={[{ id: "waveform", label: "I / Q" }, { id: "envelope", label: "envelope & phase" }, { id: "fft", label: "FFT" }, { id: "stats", label: "statistics" }, { id: "raw", label: "raw JSON" }]} value={tab} onChange={setTab} />
            {preview && tab === "waveform" && (
              <Plot height={330} data={[
                { type: "scattergl", mode: "lines", x: preview.t_s, y: preview.i, line: { color: "#22d3ee", width: 1 }, name: "I" },
                ...(preview.is_complex ? [{ type: "scattergl", mode: "lines", x: preview.t_s, y: preview.q, line: { color: "#f472b6", width: 1 }, name: "Q" }] : []),
              ]} layout={{ xaxis: { title: "time [s]" }, yaxis: { title: "amplitude" }, dragmode: "zoom" }} />
            )}
            {preview && tab === "envelope" && (
              <Plot height={330} data={[
                { type: "scattergl", mode: "lines", x: preview.t_s, y: preview.envelope, line: { color: "#34d399" }, name: "|x|" },
                { type: "scattergl", mode: "lines", x: preview.t_s, y: preview.phase_deg, line: { color: "#a78bfa" }, name: "phase [deg] (first 20k samples)" },
              ]} layout={{ xaxis: { title: "time [s]" }, yaxis: { title: "envelope / phase" } }} />
            )}
            {preview && tab === "fft" && (
              <Plot height={330} data={[{ type: "scattergl", mode: "lines", x: preview.fft.freq_hz, y: preview.fft.db, line: { color: "#f59e0b" }, name: "FFT" }]}
                layout={{ xaxis: { title: "frequency [Hz]" }, yaxis: { title: "power [dB]" }, title: `${preview.fft.n_fft}-point ${preview.fft.window} window` }} />
            )}
            {preview && tab === "stats" && (
              <div className="grid g2">
                <KV rows={Object.entries(preview.stats || {}).map(([k, v]) => [k, num(v as number, 5)])} />
                <div>
                  <div className="sub">what the parser reported</div>
                  <KV rows={[["is complex", String(preview.is_complex)], ["fs source", preview.fs_source],
                    ["samples", num(preview.n_samples, 0)], ["notes", (preview.notes || []).join("; ") || "—"]]} />
                </div>
              </div>
            )}
            {preview && tab === "raw" && <Json value={preview} label="preview payload" />}
          </Card>
        </>
      )}
    </div>
  );
}
