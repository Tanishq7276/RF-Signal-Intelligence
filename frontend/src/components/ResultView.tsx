import { useCallback, useEffect, useMemo, useState } from "react";
import { Api } from "../lib/api";
import { bytes, num, pct, si } from "../lib/format";
import { useStore } from "../lib/store";
import { autocorrTrace, byteHistogramTrace, confusionHeatmap, constellationTrace, eyeTrace, hypothesisBars, spectrumShapes, spectrumTraces, waterfallLayout, waterfallShapes, waterfallTrace } from "./charts";
import { HypothesisList, ParamTable, Pipeline3D, QualityPanel, SegmentTable, StageList } from "./blocks";
import Plot from "./Plot";
import { Alert, Badge, Card, ConfidenceCell, Empty, Json, KV, NumField, Select, Spinner, Stat, Tabs } from "./ui";

const TABS = [
  { id: "overview", label: "Overview" },
  { id: "spectrum", label: "Spectrum" },
  { id: "waterfall", label: "Waterfall" },
  { id: "detection", label: "Detection" },
  { id: "constellation", label: "Constellation & eye" },
  { id: "demod", label: "Demodulation" },
  { id: "fec", label: "FEC & interleaving" },
  { id: "bitstream", label: "Bitstream" },
  { id: "evidence", label: "Evidence" },
  { id: "report", label: "Report" },
];

export default function ResultView({ result, analysis, onRegionRequest, onSignalRequest, initialTab }: {
  result: any; analysis: any;
  onRegionRequest?: (r: any) => void; onSignalRequest?: (i: number) => void; initialTab?: string;
}) {
  const { toast } = useStore();
  const [tab, setTab] = useState(initialTab || "overview");
  const [spectrum, setSpectrum] = useState<any>(null);
  const [wf, setWf] = useState<any>(null);
  const [eye, setEye] = useState<any>(null);
  const [bs, setBs] = useState<any>(null);
  const [bsStream, setBsStream] = useState("auto");
  const [region, setRegion] = useState<any>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [fecOut, setFecOut] = useState<any>(null);
  const [intOut, setIntOut] = useState<any>(null);
  const [reports, setReports] = useState<any[]>([]);
  const [corr, setCorr] = useState<any>(null);
  const [pattern, setPattern] = useState("");
  const [patternKind, setPatternKind] = useState("auto");
  const [maxErrors, setMaxErrors] = useState(0);

  const id = analysis?.analysis_id;
  const sig = result?.signal || {};
  const summary = analysis?.summary || {};
  const params: any[] = useMemo(() => {
    const out: any[] = [];
    const groups = sig.params || {};
    Object.entries(groups).forEach(([g, block]: any) => {
      if (block && typeof block === "object") {
        Object.entries(block).forEach(([name, rec]: any) => { if (rec && typeof rec === "object") out.push({ ...rec, group: g, name: rec.name || name }); });
      }
    });
    const recp = (result?.record_spectrum || {}).params || {};
    Object.entries(recp).forEach(([name, rec]: any) => { if (rec && typeof rec === "object") out.push({ ...rec, group: "record", name: rec.name || name }); });
    return out;
  }, [sig, result]);
  const paramGroups = useMemo(() => Array.from(new Set(params.map((p) => p.group))), [params]);

  useEffect(() => {
    if (!id) return;
    if (tab === "spectrum" && !spectrum) Api.spectrum(id).then(setSpectrum).catch((e) => toast(e.message, "err"));
    if (tab === "waterfall" && !wf) Api.spectrogram(id).then(setWf).catch((e) => toast(e.message, "err"));
    if (tab === "constellation" && !eye) Api.eye(id).then(setEye).catch(() => setEye({ eye: null }));
    if (tab === "bitstream") Api.bitstream(id, bsStream).then(setBs).catch((e) => setBs({ ok: false, message: e.message }));
    if (tab === "report") Api.reports().then((r) => setReports((r.reports || []).filter((x: any) => x.analysis_id === id))).catch(() => undefined);
  }, [tab, id, spectrum, wf, eye, bs, bsStream, toast]);

  const runFec = useCallback(async () => {
    setBusy("fec");
    try { const r = await Api.fec({ analysis_id: id }); setFecOut(r); toast(r.best ? `FEC candidate: ${r.best.hypothesis} (${num(r.best.confidence, 2)})` : "no FEC family passed the evidence threshold", "ok"); }
    catch (e: any) { toast(`FEC analysis failed: ${e.message}`, "err"); }
    finally { setBusy(null); }
  }, [id, toast]);

  const runInter = useCallback(async () => {
    setBusy("inter");
    try {
      const r = await Api.interleaving({ analysis_id: id });
      setIntOut(r);
      toast(r.best ? `interleaver candidate: ${r.best.hypothesis} (${num(r.best.confidence, 2)})` : "no interleaver geometry beat the control group", "ok");
    } catch (e: any) { toast(`interleaver analysis failed: ${e.message}`, "err"); }
    finally { setBusy(null); }
  }, [id, toast]);

  if (!id) return <Empty>No analysis selected. Run one from the Analyze page or pick one from the history.</Empty>;

  const specShapes = spectrumShapes(spectrum?.segment_spectrum || sig.spectrum);
  const recShapes = spectrumShapes({ spectrum: result?.record_spectrum });
  const wfDet = wf?.detection || result?.detection;

  return (
    <div>
      <Card
        title={`Analysis ${id} · ${analysis?.mode || "AUTO"} mode`}
        sub={`${analysis?.file?.filename || "—"} · ${analysis?.status} · ${num(analysis?.duration_ms, 0)} ms · ${analysis?.created_at ? new Date(analysis.created_at).toLocaleString() : ""}`}
        right={<div className="row" style={{ gap: 6 }}>
          <Badge kind={analysis?.status === "done" ? "ok" : "bad"}>{analysis?.status}</Badge>
          {summary.modulation && <span className="pill">{summary.modulation} {num(summary.modulation_confidence, 2)}</span>}
          {summary.fec && <span className="pill">FEC: {summary.fec}</span>}
        </div>}
      >
        <Tabs tabs={TABS} value={tab} onChange={setTab} />

        {tab === "overview" && (
          <div className="col" style={{ gap: 14 }}>
            <QualityPanel result={result} summary={summary} />
            <Card title="Extracted parameters" sub="value / confidence / method — grouped by the stage that produced it">
              <ParamTable params={params} groups={paramGroups} />
            </Card>
            {(result?.notes?.length || result?.warnings?.length) > 0 && (
              <Card title="Notes & warnings">
                {(result.notes || []).map((n: string, i: number) => <Alert key={`n${i}`} kind="info">{n}</Alert>)}
                {(result.warnings || []).map((w: string, i: number) => <Alert key={`w${i}`} kind="warn">{w}</Alert>)}
              </Card>
            )}
            <Card title="Hypothesis ranking" sub="each hypothesis is demodulated end-to-end and scored on measured evidence">
              <div className="grid g2">
                <Plot height={300} data={hypothesisBars(sig.hypotheses?.hypotheses || [], (h) => `${h.label || ""} ${h.modulation} @ ${si(h.symbol_rate_hz, "sym/s", 1)}`)}
                  layout={{ xaxis: { title: "score", range: [0, 1] }, margin: { l: 130, r: 12, t: 10, b: 34 } }} />
                <div>
                  <KV rows={[
                    ["best hypothesis", `${sig.hypotheses?.best?.label || "—"} ${sig.hypotheses?.best?.modulation || ""}`],
                    ["score", num(sig.hypotheses?.best?.score, 4)],
                    ["EVM", sig.hypotheses?.best?.evm_percent != null ? `${num(sig.hypotheses.best.evm_percent, 2)} %` : "—"],
                    ["method", sig.hypotheses?.method || "—"],
                  ]} />
                  <div style={{ marginTop: 10 }}>
                    <StageList stages={(result?.stages?.stages || []).map((s: any) => ({ stage: s.name, status: s.status, detail: s.note, duration_ms: s.duration_ms }))} />
                  </div>
                </div>
              </div>
            </Card>
            <Card title="Evidence graph" sub="IQ → spectrum/spectrogram → candidates → modulation → timing → demodulation → FEC → bitstream">
              <Pipeline3D nodes={result?.evidence_graph?.nodes || sig.evidence_graph?.nodes || []} />
            </Card>
          </div>
        )}

        {tab === "spectrum" && (
          <div className="col" style={{ gap: 14 }}>
            <Card title="Record spectrum (before band selection)" sub={result?.record_spectrum?.estimator || ""}>
              <Plot height={300} data={spectrumTraces(result?.record_spectrum, { color: "#64748b", name: "record PSD" })} layout={{ xaxis: { title: "frequency [Hz]" }, yaxis: { title: "power [dB]" }, shapes: recShapes.shapes, annotations: recShapes.annotations }} />
            </Card>
            <Card title="Analysed emission spectrum" sub="the segment that the parameters, modulation and demodulation results describe">
              {spectrum ? (
                <>
                  <Plot height={320} data={spectrumTraces(spectrum.segment_spectrum)} layout={{ xaxis: { title: "frequency [Hz] (baseband of the analysed segment)" }, yaxis: { title: "power [dB]" }, shapes: specShapes.shapes, annotations: specShapes.annotations }} />
                  <div className="grid g4" style={{ marginTop: 10 }}>
                    <Stat k="Noise floor" v={num(spectrum.segment_spectrum?.noise_floor_db, 2)} n={`source: ${spectrum.segment_spectrum?.noise_source || "measured"}`} />
                    <Stat k="SNR" v={`${num(spectrum.segment_spectrum?.snr_db, 2)} dB`} n="spectral, noise-density referenced" />
                    <Stat k="99 % OBW" v={si(spectrum.segment_spectrum?.obw_99_hz, "Hz", 2)} n={`${num(spectrum.segment_spectrum?.occupied_fraction, 3)} of the sampled band`} />
                    <Stat k="Frequency offset" v={si(spectrum.segment_spectrum?.frequency_offset_hz, "Hz", 2)} n="from the capture centre" />
                  </div>
                  <div className="row" style={{ gap: 6, marginTop: 10 }}>
                    <span className="small muted">estimator:</span>
                    <span className="pill">{spectrum.segment_spectrum?.estimator || "—"}</span>
                    <span className="pill">nperseg {spectrum.segment_spectrum?.nperseg || "—"}</span>
                    <span className="pill">dynamic range {num(spectrum.segment_spectrum?.dynamic_range_db, 1)} dB</span>
                    {spectrum.segment_spectrum?.line_like && <span className="pill">line-like</span>}
                    {spectrum.segment_spectrum?.flat_top && <span className="pill">flat-top</span>}
                  </div>
                </>
              ) : <Spinner label="loading spectrum" />}
            </Card>
          </div>
        )}

        {tab === "waterfall" && (
          <div className="col" style={{ gap: 14 }}>
            <Card title="Spectrogram / waterfall" sub="drag a box on the plot to select a time-frequency region and re-analyse it">
              {wf?.ok ? (
                <>
                  <Plot height={430} data={waterfallTrace(wf)} layout={{ ...waterfallLayout(wf.meta || {}), shapes: waterfallShapes(wfDet).shapes, annotations: waterfallShapes(wfDet).annotations, dragmode: "select" } as any}
                    onRelayout={(ev: any) => {
                      if (ev["xaxis.range[0]"] !== undefined || ev["xaxis.range"] || ev["yaxis.range"]) {
                        const xr = ev["xaxis.range"] || [ev["xaxis.range[0]"], ev["xaxis.range[1]"]];
                        const yr = ev["yaxis.range"] || [ev["yaxis.range[0]"], ev["yaxis.range[1]"]];
                        if (xr?.[0] !== undefined && yr?.[0] !== undefined) setRegion({ t0_s: Number(xr[0]), t1_s: Number(xr[1]), f_lo_hz: Number(yr[0]), f_hi_hz: Number(yr[1]) });
                      }
                    }} />
                  <div className="row" style={{ marginTop: 8 }}>
                    <span className="small muted">zoom/pan in the plot; the box below is the current selection</span>
                    <span style={{ flex: 1 }} />
                    <button className="tiny ghost" onClick={() => setRegion(null)}>clear selection</button>
                    <button className="tiny primary" disabled={!region || !onRegionRequest} onClick={() => region && onRegionRequest?.(region)}>re-analyse selected region</button>
                  </div>
                  {region && <div className="small mono" style={{ marginTop: 6 }}>{region.f_lo_hz.toFixed(0)} … {region.f_hi_hz.toFixed(0)} Hz · {region.t0_s.toFixed(5)} … {region.t1_s.toFixed(5)} s</div>}
                  <div className="row" style={{ gap: 8, marginTop: 10 }}>
                    <span className="pill">{wf.shape?.[1]} time bins × {wf.shape?.[0]} frequency bins</span>
                    <span className="pill">{wf.meta?.nperseg}-point STFT · {wf.meta?.window} window</span>
                    <span className="pill">{wf.meta?.df_hz ? `Δf ${num(wf.meta.df_hz, 1)} Hz` : ""} {wf.meta?.dt_s ? `· Δt ${num(wf.meta.dt_s, 6)} s` : ""}</span>
                  </div>
                </>
              ) : wf ? <Alert kind="warn">{wf.message || "no spectrogram available"}</Alert> : <Spinner label="computing waterfall payload" />}
            </Card>
          </div>
        )}

        {tab === "detection" && (
          <div className="col" style={{ gap: 14 }}>
            <Card title="Detected emissions" sub="CFAR detection on the spectrogram; click a row to re-analyse that emission">
              <SegmentTable segments={(result?.detection?.signals || []).map((s: any, i: number) => ({
                label: s.id || `S${i + 1}`, center_frequency_hz: s.center_frequency_hz, bandwidth_hz: s.bandwidth_hz,
                t0_s: s.t0_s, duration_s: s.duration_s, peak_power_db: s.peak_power_db, snr_db: s.snr_db,
                confidence: s.confidence, kind: s.kind, n_bursts: s.n_bursts, selected: i === (result?.selected_signal?.index ?? 0),
              }))} onSelect={onSignalRequest} onRegion={(s: any) => onRegionRequest?.({ f_lo_hz: s.f_lo_hz ?? (s.center_frequency_hz - s.bandwidth_hz / 2), f_hi_hz: s.f_hi_hz ?? (s.center_frequency_hz + s.bandwidth_hz / 2), t0_s: s.t0_s, t1_s: (s.t0_s ?? 0) + (s.duration_s ?? 0) })} />
            </Card>
            <Card title="Detection parameters" sub="thresholds and the CFAR settings that produced the list above">
              <KV rows={[
                ["noise floor", `${num(result?.detection?.noise_floor_db, 2)} dB`],
                ["threshold", `${num(result?.detection?.threshold_db, 2)} dB (${num(result?.detection?.threshold_over_floor_db, 1)} dB over the floor)`],
                ["CFAR", JSON.stringify(result?.detection?.cfar || {})],
                ["rejected candidates", String((result?.detection?.rejected_candidates || []).length)],
                ["spectrogram", `${result?.detection?.spectrogram_shape?.[1]} × ${result?.detection?.spectrogram_shape?.[0]} (time × freq), Δf ${num(result?.detection?.df_hz, 1)} Hz, Δt ${num(result?.detection?.dt_s, 6)} s`],
              ]} />
              {(result?.detection?.notes || []).map((n: string, i: number) => <Alert key={i} kind="info">{n}</Alert>)}
            </Card>
            {result?.segmentation && (
              <Card title="Extracted analysis band" sub="the segment that was shifted to baseband and analysed">
                <KV rows={[
                  ["band", `${si(result.segmentation.f_lo_hz, "Hz", 2)} … ${si(result.segmentation.f_hi_hz, "Hz", 2)}`],
                  ["centre", si(result.segment_center_hz, "Hz", 2)],
                  ["samples", num(result.segmentation.n_samples, 0)],
                  ["decimated from", si(result.segmentation.fs_in, "Hz", 2)],
                  ["analysis rate", si(result.analysis_fs, "Hz", 2)],
                ]} />
              </Card>
            )}
          </div>
        )}

        {tab === "constellation" && (
          <div className="col" style={{ gap: 14 }}>
            <div className="grid g2">
              <Card title="Constellation" sub={sig.constellation?.note || "symbol-sampled I/Q"}>
                {sig.constellation?.i?.length ? (
                  <>
                    <Plot height={360} data={constellationTrace(sig.constellation)}
                      layout={{ xaxis: { title: "in-phase", scaleanchor: "y", scaleratio: 1 }, yaxis: { title: "quadrature" } }} />
                    <div className="row" style={{ gap: 8, marginTop: 8 }}>
                      <span className="pill">EVM {num(sig.constellation.quality?.evm_percent, 2)} %</span>
                      <span className="pill">{num(sig.constellation.quality?.n_symbols, 0)} symbols</span>
                      <span className="pill">cluster coverage {num(sig.constellation.quality?.cluster_coverage, 2)}</span>
                      <span className="pill">ambiguity {num(sig.constellation.quality?.ambiguity_fraction, 2)}</span>
                    </div>
                    <div className="small muted" style={{ marginTop: 6 }}>{sig.constellation.quality?.method}</div>
                  </>
                ) : <Empty>{sig.constellation?.message || "no constellation for this record"}</Empty>}
              </Card>
              <Card title="Eye diagram" sub="timing quality from the matched-filtered signal">
                {eye === null ? <Spinner label="loading" /> : eye?.eye?.ok ? (
                  <>
                    <Plot height={330} data={eyeTrace(eye.eye).traces}
                      layout={{ xaxis: { title: "time [symbols]" }, yaxis: { title: "amplitude" }, showlegend: false }} />
                    <div className="row" style={{ gap: 8, marginTop: 8 }}>
                      <span className="pill">eye opening {num(eye.eye.metrics?.eye_opening_ratio, 3)}</span>
                      <span className="pill">spread {num(eye.eye.metrics?.trace_spread, 3)}</span>
                      <span className="pill">quality {eye.eye.metrics?.eye_quality}</span>
                      <span className="pill">timing line {num(eye.eye.metrics?.timing_line_db, 1)} dB</span>
                      <span className="pill">phase {num(eye.eye.timing_phase_frac, 4)}</span>
                    </div>
                    <div className="small muted" style={{ marginTop: 6 }}>{eye.eye.method}</div>
                    {(eye.eye.limitations || []).map((l: string, i: number) => <Alert key={i} kind="warn">{l}</Alert>)}
                  </>
                ) : <Empty>{eye?.eye?.message || eye?.eye?.error || "no eye diagram for this record"}</Empty>}
              </Card>
            </div>
          </div>
        )}

        {tab === "demod" && (
          <div className="col" style={{ gap: 14 }}>
            <Card title={`Demodulation — ${sig.demodulation?.modulation || "not applied"}`}
              sub={sig.demodulation?.ok ? "modular chain: carrier recovery → symbol-rate refinement → matched filter → timing → decision" : "the demodulator did not complete for this record"}>
              {sig.demodulation?.ok ? (
                <>
                  <div className="grid g4">
                    <Stat k="EVM" v={`${num(sig.demodulation.quality?.evm_percent, 2)} %`} n={`${num(sig.demodulation.quality?.evm_db, 2)} dB`} />
                    <Stat k="Symbols / bits" v={`${num(sig.demodulation.n_symbols, 0)} / ${num(sig.demodulation.n_bits, 0)}`} n="decided from the recovered constellation" />
                    <Stat k="Carrier offset" v={si(sig.demodulation.carrier_offset_hz, "Hz", 2)} n="measured, not assumed" />
                    <Stat k="Symbol rate used" v={si(sig.demodulation.symbol_rate_used_hz, "sym/s", 2)} n={sig.symbol_rate_note ? "refined by timing-drift estimation" : "blind estimate"} />
                  </div>
                  <div style={{ marginTop: 12 }}><StageList stages={sig.demodulation.stages || []} /></div>
                  {sig.demodulation.ber_estimate && (
                    <div style={{ marginTop: 10 }}>
                      <KV rows={Object.entries(sig.demodulation.ber_estimate).map(([k, v]) => [k, typeof v === "number" ? num(v, 6) : String(v)])} />
                    </div>
                  )}
                </>
              ) : <Alert kind="bad">{sig.demodulation?.message || sig.demodulation?.error || "no demodulation result"}</Alert>}
            </Card>
          </div>
        )}

        {tab === "fec" && (
          <div className="col" style={{ gap: 14 }}>
            <Card title="Automatic FEC detection" right={<div className="row" style={{ gap: 6 }}>
              <button className="tiny" disabled={busy === "fec"} onClick={runFec}>{busy === "fec" ? "running…" : "run FEC search"}</button>
            </div>}
              sub={sig.fec?.policy || "a code is only claimed when its evidence beats the random-data null"}>
              {(fecOut?.hypotheses || sig.fec?.hypotheses || []).length ? (
                <>
                  {fecOut?.best === null && !fecOut?.hypotheses && <Alert kind="warn">No FEC family passed the reporting threshold for this record.</Alert>}
                  <HypothesisList kind="fec" hypotheses={(fecOut?.hypotheses || sig.fec?.hypotheses || []).slice(0, 6)} />
                </>
              ) : <Empty>No FEC hypothesis could be formed (a bit stream is required).</Empty>}
              {fecOut?.notes && <div style={{ marginTop: 8 }}>{fecOut.notes.map((n: string, i: number) => <Alert key={i} kind="info">{n}</Alert>)}</div>}
            </Card>
            <Card title="Automatic interleaving detection" right={<button className="tiny" disabled={busy === "inter"} onClick={runInter}>{busy === "inter" ? "running…" : "run interleaver search"}</button>}
              sub="a geometry is only claimed when de-interleaving makes the FEC decoder measurable better than a random-permutation control">
              {(intOut?.hypotheses || sig.interleaving?.hypotheses || []).length
                ? <HypothesisList kind="interleave" hypotheses={(intOut?.hypotheses || sig.interleaving?.hypotheses || []).slice(0, 6)} />
                : <Empty>No interleaving hypothesis was formed.</Empty>}
              {(intOut?.notes || sig.interleaving?.notes || []).map((n: string, i: number) => <Alert key={i} kind="info">{n}</Alert>)}
            </Card>
            <Card title="Before / after comparison" sub="statistics of the bit stream before and after error correction">
              {bs?.comparison ? (
                <div className="grid g2">
                  {(["demod", "decoded"] as const).map((k) => (
                    <div key={k}>
                      <b className="small">{k === "demod" ? "raw demodulator decisions" : "after error correction"}</b>
                      <KV rows={[
                        ["bits", num(bs.comparison[k].n_bits, 0)],
                        ["entropy", `${num(bs.comparison[k].entropy.bits_per_bit, 4)} bit/bit`],
                        ["ones fraction", num(bs.comparison[k].ones_fraction, 4)],
                        ["printable runs", String((bs.comparison[k].printable_ascii || []).length)],
                        ["frame candidates", String((bs.comparison[k].frame_candidates || []).length)],
                      ]} />
                    </div>
                  ))}
                </div>
              ) : <Empty>Open the Bitstream tab (auto stream) to compute the before/after statistics once an FEC hypothesis exists.</Empty>}
            </Card>
          </div>
        )}

        {tab === "bitstream" && (
          <div className="col" style={{ gap: 14 }}>
            <Card title="Bit stream" right={
              <div className="row" style={{ gap: 6 }}>
                <select value={bsStream} onChange={(e) => setBsStream(e.target.value)} style={{ width: 190 }}>
                  <option value="auto">auto (decoded when available)</option>
                  <option value="demod">raw demodulator decisions</option>
                  <option value="decoded">error-corrected information bits</option>
                </select>
              </div>} sub={bs?.stream_note || "bit and byte statistics of the demodulated stream"}>
              {bs?.ok ? (
                <>
                  <div className="grid g4">
                    <Stat k="Bits" v={num(bs.n_bits, 0)} n={`stream: ${bs.stream_used}`} />
                    <Stat k="Entropy" v={`${num(bs.entropy?.bits_per_bit, 4)} bit/bit`} n={`${num(bs.entropy?.bits_per_byte, 3)} bit/byte`} />
                    <Stat k="Ones" v={pct(bs.statistics?.ones_fraction, 1)} n="a fair random stream sits near 50 %" />
                    <Stat k="Periodicity" v={bs.periodicity?.period_bits ? `${bs.periodicity.period_bits} bits` : "none"} n={`score ${num(bs.periodicity?.score, 3)}`} />
                  </div>
                  <div className="hr" />
                  <div className="row" style={{ gap: 10 }}>
                    <span className="small muted">first 512 bits</span>
                    <span className="pill">{(bs.bits_preview || "").length} shown{bs.n_bits_total > bs.n_bits ? ` of ${num(bs.n_bits_total, 0)}` : ""}</span>
                  </div>
                  <pre style={{ whiteSpace: "pre-wrap", wordBreak: "break-all" }}>{bs.bits_preview}</pre>
                  <div className="grid g2">
                    <div>
                      <div className="sub">byte distribution (0–255)</div>
                      <Plot height={250} data={byteHistogramTrace(bs)} layout={{ xaxis: { title: "byte value" }, yaxis: { title: "count" }, margin: { l: 52, r: 12, t: 10, b: 40 } }} />
                    </div>
                    <div>
                      <div className="sub">bit autocorrelation (structure detector)</div>
                      <Plot height={250} data={autocorrTrace(bs)} layout={{ xaxis: { title: "lag [bits]" }, yaxis: { title: "correlation" }, margin: { l: 52, r: 12, t: 10, b: 40 } }} />
                    </div>
                  </div>
                  <div className="grid g2" style={{ marginTop: 10 }}>
                    <div>
                      <div className="sub">printable-ASCII candidate</div>
                      {(bs.printable_ascii || []).length ? (bs.printable_ascii || []).map((p: any, i: number) => (
                        <Alert key={i} kind={p.printable_ratio > 0.9 ? "ok" : "warn"}>
                          <div className="mono small" style={{ wordBreak: "break-all" }}>{p.text_preview}</div>
                          <div className="small muted">bytes {p.start_byte}…{p.start_byte + p.length_bytes} · printable ratio {num(p.printable_ratio, 3)} · {p.label}</div>
                        </Alert>
                      )) : <Empty>no printable-ASCII run found — the stream looks like encoded or scrambled data</Empty>}
                      <Alert kind="info">A printable run is a <b>candidate</b> reading of the bits, not a decoded message.</Alert>
                    </div>
                    <div>
                      <div className="sub">frame / sync candidates (hypotheses)</div>
                      <div className="tblwrap" style={{ maxHeight: 220 }}>
                        <table><thead><tr><th>pattern</th><th>ascii</th><th className="num">hits</th><th className="num">period</th><th className="num">regularity</th><th className="num">score</th></tr></thead>
                          <tbody>{(bs.frame_candidates || []).slice(0, 12).map((f: any, i: number) => (
                            <tr key={i}><td className="mono">{f.pattern_hex}</td><td className="mono">{f.pattern_ascii}</td>
                              <td className="num">{f.occurrences}</td><td className="num">{num(f.period_bytes, 1)} B</td>
                              <td className="num">{num(f.regularity, 2)}</td><td className="num">{num(f.score, 3)}</td></tr>
                          ))}</tbody></table>
                      </div>
                      <div className="sub" style={{ marginTop: 10 }}>hex dump (first 256 bytes)</div>
                      <pre>{(bs.hex_dump || []).join("\n")}</pre>
                    </div>
                  </div>
                </>
              ) : <Alert kind="warn">{bs?.message || "no bit stream available"}{bs?.hint ? ` — ${bs.hint}` : ""}</Alert>}
            </Card>
            <Card title="Correlation search" sub="search the bit stream for a header, byte pattern, hex string or ASCII marker">
              <div className="row" style={{ gap: 10, alignItems: "flex-end" }}>
                <div style={{ minWidth: 260, flex: 1 }}>
                  <label className="f"><span>pattern</span>
                    <input value={pattern} placeholder='e.g. SIH26147 or 534948 or 101010' onChange={(e) => setPattern(e.target.value)} /></label>
                </div>
                <div style={{ width: 150 }}>
                  <Select label="interpretation" value={patternKind} onChange={setPatternKind}
                    options={[{ value: "auto", label: "auto" }, { value: "text", label: "ASCII text" }, { value: "hex", label: "hex bytes" }, { value: "bits", label: "0/1 bits" }]} />
                </div>
                <div style={{ width: 120 }}><NumField label="max errors" value={maxErrors} step={1} min={0} max={64} onChange={(v: number) => setMaxErrors(v ?? 0)} /></div>
                <button className="primary" disabled={!pattern}
                  onClick={async () => {
                    setBusy("corr");
                    try { setCorr(await Api.correlate({ analysis_id: id, pattern, kind: patternKind, max_errors: maxErrors, auto: true })); }
                    catch (e: any) { toast(`correlation failed: ${e.message}`, "err"); }
                    finally { setBusy(null); }
                  }}>{busy === "corr" ? "searching…" : "search"}</button>
              </div>
              {corr && (
                <div style={{ marginTop: 10 }}>
                  <Alert kind={corr.searches?.[0]?.n_hits ? "ok" : "warn"}>{corr.summary}</Alert>
                  {(corr.searches || []).map((s: any, i: number) => (
                    <KV key={i} rows={[
                      ["pattern bits", num(s.pattern_bits, 0)],
                      ["exact hits", num(s.n_hits, 0)],
                      ["best position", s.best_position_bits != null ? `${num(s.best_position_bits, 0)} bits` : "—"],
                      ["matched bits", `${num(s.best_matches, 0)} / ${num(s.pattern_bits, 0)}`],
                    ]} />
                  ))}
                  {!!(corr.auto?.candidates || []).length && (
                    <div style={{ marginTop: 8 }}>
                      <div className="sub">automatically discovered repeated structure</div>
                      <pre>{JSON.stringify(corr.auto.candidates.slice(0, 6), null, 1)}</pre>
                    </div>
                  )}
                </div>
              )}
            </Card>
          </div>
        )}

        {tab === "evidence" && (
          <div className="col" style={{ gap: 14 }}>
            <Card title="Confidence system" sub="every automatic result with the basis of its confidence">
              <QualityPanel result={result} summary={summary} />
            </Card>
            <Card title="Evidence graph" sub="the chain the platform actually followed, with the status of every step">
              <Pipeline3D nodes={result?.evidence_graph?.nodes || sig.evidence_graph?.nodes || []} height={420} />
            </Card>
            <Card title="Raw analysis payload" sub="what the API returned for this analysis (trimmed of the raw bit arrays)">
              <Json value={result} label="result JSON" />
            </Card>
          </div>
        )}

        {tab === "report" && (
          <Card title="Report export" sub="built from the stored result of this analysis id (PDF embeds the real charts)">
            <div className="row">
              <button className="primary" onClick={async () => {
                setBusy("report");
                try { const r = await Api.makeReport(id, ["pdf", "json", "csv", "txt"]); setReports((x) => [...r.reports, ...x]); toast(`${r.reports.length} report file(s) generated`, "ok"); if (r.errors?.length) toast(r.errors.join("; "), "err"); }
                catch (e: any) { toast(`report failed: ${e.message}`, "err"); }
                finally { setBusy(null); }
              }}>{busy === "report" ? "building…" : "generate PDF + JSON + CSV + TXT"}</button>
              {(["pdf", "json", "csv", "txt"] as const).map((f) => (
                <a key={f} className="btn" href={Api.reportInlineUrl(id, f)} target="_blank" rel="noreferrer">download {f.toUpperCase()} (on the fly)</a>
              ))}
            </div>
            <div className="hr" />
            {reports.length ? (
              <div className="tblwrap">
                <table><thead><tr><th>format</th><th>filename</th><th className="num">size</th><th>created</th><th></th></tr></thead>
                  <tbody>{reports.map((r) => (
                    <tr key={r.report_id}><td><Badge kind="info">{r.format}</Badge></td><td className="mono small">{r.filename || r.title}</td>
                      <td className="num">{bytes(r.size_bytes)}</td><td className="small muted">{r.created_at ? new Date(r.created_at).toLocaleString() : ""}</td>
                      <td><a className="btn tiny" href={Api.reportDownloadUrl(r.report_id)} target="_blank" rel="noreferrer">download</a></td></tr>
                  ))}</tbody></table>
              </div>
            ) : <Empty>No stored report for this analysis yet.</Empty>}
          </Card>
        )}
      </Card>
    </div>
  );
}
