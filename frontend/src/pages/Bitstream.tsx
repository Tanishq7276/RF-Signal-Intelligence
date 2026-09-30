import { useEffect, useState } from "react";
import { Api } from "../lib/api";
import { num, pct } from "../lib/format";
import { useStore } from "../lib/store";
import Plot from "../components/Plot";
import { autocorrTrace, byteHistogramTrace } from "../components/charts";
import { Alert, Badge, Card, Empty, Json, KV, NumField, Select, Spinner, Stat } from "../components/ui";

/** /bitstream — bits, bytes, structure and the correlation search. */
export default function Bitstream() {
  const { analyses, toast, current } = useStore();
  const [analysisId, setAnalysisId] = useState<string | null>(current?.analysis?.analysis_id ?? null);
  const [stream, setStream] = useState("auto");
  const [bs, setBs] = useState<any>(null);
  const [busy, setBusy] = useState(false);
  const [pattern, setPattern] = useState("");
  const [kind, setKind] = useState("auto");
  const [maxErrors, setMaxErrors] = useState(0);
  const [corr, setCorr] = useState<any>(null);

  useEffect(() => { if (!analysisId && analyses.length) setAnalysisId(analyses[0].analysis_id); }, [analyses, analysisId]);
  useEffect(() => {
    if (!analysisId) return;
    setBusy(true);
    Api.bitstream(analysisId, stream).then(setBs).catch((e) => { setBs({ ok: false, message: e.message }); })
      .finally(() => setBusy(false));
  }, [analysisId, stream]);

  const search = async () => {
    if (!analysisId || !pattern) return;
    setBusy(true);
    try { setCorr(await Api.correlate({ analysis_id: analysisId, pattern, kind, max_errors: maxErrors, auto: true })); }
    catch (e: any) { toast(e.message, "err"); }
    finally { setBusy(false); }
  };

  return (
    <div>
      <Card title="Bitstream analysis" sub="statistics are computed on the demodulated decisions (or, when an FEC hypothesis passed, on the error-corrected information bits)">
        <div className="row" style={{ gap: 10 }}>
          <div style={{ minWidth: 300, flex: 1 }}>
            <Select label="analysis" value={analysisId ?? ""} onChange={(v) => setAnalysisId(v || null)}
              options={analyses.filter((a) => a.status === "done").map((a) => ({ value: a.analysis_id, label: `${a.file?.filename} · ${a.summary?.modulation || "?"} · ${a.summary?.n_emissions ?? 0} emission(s)` }))} />
          </div>
          <div style={{ width: 210 }}>
            <Select label="stream" value={stream} onChange={setStream}
              options={[{ value: "auto", label: "auto (decoded if available)" }, { value: "demod", label: "raw decisions" }, { value: "decoded", label: "error-corrected" }]} />
          </div>
        </div>
      </Card>

      {busy && <Spinner label="computing statistics" />}
      {bs && !bs.ok && <Alert kind="warn" title="no bit stream">{bs.message}{bs.hint ? ` — ${bs.hint}` : ""}</Alert>}
      {bs?.ok && (
        <>
          <div className="grid g4">
            <Stat k="Bits" v={num(bs.n_bits, 0)} n={`total ${num(bs.n_bits_total, 0)} · ${bs.stream_used}`} />
            <Stat k="Entropy" v={`${num(bs.entropy?.bits_per_bit, 4)} bit/bit`} n={`${num(bs.entropy?.bits_per_byte, 3)} bit/byte`} />
            <Stat k="Ones fraction" v={pct(bs.statistics?.ones_fraction, 2)} n="≈50 % is consistent with a scrambler or compression" />
            <Stat k="Periodicity" v={bs.periodicity?.period_bits ? `${bs.periodicity.period_bits} bits` : "none detected"} n={`score ${num(bs.periodicity?.score, 3)}`} />
          </div>
          <Card title="Stream" sub={bs.stream_note}>
            <div className="row" style={{ gap: 8 }}>
              <Badge kind="info">{bs.stream_used}</Badge>
              {bs.decoded_from && <span className="pill">decoded from {bs.decoded_from}</span>}
              <span className="pill">modulation {bs.modulation || "—"}</span>
              <span className="pill">symbol rate {num(bs.symbol_rate_used_hz, 1)} Hz</span>
            </div>
            <pre style={{ whiteSpace: "pre-wrap", wordBreak: "break-all", marginTop: 8 }}>{bs.bits_preview}</pre>
            <div className="grid g2">
              <div>
                <div className="sub">byte histogram</div>
                <Plot height={230} data={byteHistogramTrace(bs)} layout={{ xaxis: { title: "byte" }, yaxis: { title: "count" }, margin: { l: 50, r: 12, t: 10, b: 36 } }} />
              </div>
              <div>
                <div className="sub">autocorrelation</div>
                <Plot height={230} data={autocorrTrace(bs)} layout={{ xaxis: { title: "lag [bits]" }, yaxis: { title: "correlation" }, margin: { l: 50, r: 12, t: 10, b: 36 } }} />
              </div>
            </div>
            <div className="grid g2" style={{ marginTop: 12 }}>
              <div>
                <div className="sub">printable-ASCII candidate</div>
                {(bs.printable_ascii || []).length ? (bs.printable_ascii || []).map((p: any, i: number) => (
                  <div key={i} className="card" style={{ padding: 10, marginBottom: 8 }}>
                    <div className="mono small" style={{ wordBreak: "break-all" }}>{p.text_preview}</div>
                    <div className="small muted">bytes {p.start_byte}…{p.start_byte + p.length_bytes} · printable ratio {num(p.printable_ratio, 3)}</div>
                  </div>
                )) : <Empty>no printable run found — the stream is consistent with encoded or scrambled data</Empty>}
                <Alert kind="info">A printable run is a <b>candidate</b>, never a decoded message.</Alert>
              </div>
              <div>
                <div className="sub">frame / sync candidates &amp; repeated n-grams</div>
                <div className="tblwrap" style={{ maxHeight: 200 }}>
                  <table><thead><tr><th>hex</th><th>ascii</th><th className="num">hits</th><th className="num">period</th><th className="num">score</th></tr></thead>
                    <tbody>{(bs.frame_candidates || []).slice(0, 10).map((f: any, i: number) => (
                      <tr key={i}><td className="mono">{f.pattern_hex}</td><td className="mono">{f.pattern_ascii}</td>
                        <td className="num">{f.occurrences}</td><td className="num">{num(f.period_bytes, 1)} B</td><td className="num">{num(f.score, 3)}</td></tr>))}
                    </tbody></table>
                </div>
                <Json value={{ ngrams: (bs.ngrams || []).slice(0, 8), run_length: bs.run_length }} label="n-grams / run length" />
              </div>
            </div>
            <div className="sub" style={{ marginTop: 10 }}>hex dump (first 256 bytes)</div>
            <pre>{(bs.hex_dump || []).join("\n")}</pre>
          </Card>

          <Card title="Correlation search" sub="find a header, byte pattern, hex string or ASCII marker in this stream">
            <div className="row" style={{ gap: 10, alignItems: "flex-end" }}>
              <div style={{ flex: 1, minWidth: 240 }}><label className="f"><span>pattern</span>
                <input value={pattern} onChange={(e) => setPattern(e.target.value)} placeholder='SIH26147 · 534948 · 10101010' /></label></div>
              <div style={{ width: 150 }}><Select label="kind" value={kind} onChange={setKind}
                options={[{ value: "auto", label: "auto" }, { value: "text", label: "ASCII" }, { value: "hex", label: "hex" }, { value: "bits", label: "bits" }]} /></div>
              <div style={{ width: 130 }}><NumField label="max errors" value={maxErrors} step={1} min={0} max={32} onChange={(v: number) => setMaxErrors(v ?? 0)} /></div>
              <button className="primary" disabled={!pattern || !analysisId} onClick={search}>search</button>
            </div>
            {corr && (
              <div style={{ marginTop: 10 }}>
                <Alert kind={corr.searches?.[0]?.n_hits ? "ok" : "warn"}>{corr.summary}</Alert>
                <KV rows={[
                  ["stream searched", corr.bit_source?.source || "—"],
                  ["bits", num(corr.n_bits, 0)],
                  ["hits", num(corr.searches?.[0]?.n_hits, 0)],
                  ["best position", corr.searches?.[0]?.best_position_bits != null ? `${num(corr.searches[0].best_position_bits, 0)} bits` : "—"],
                ]} />
                {!!(corr.auto?.candidates || []).length && <><div className="sub" style={{ marginTop: 8 }}>automatically discovered repeated structure</div><pre>{JSON.stringify(corr.auto.candidates.slice(0, 5), null, 1)}</pre></>}
              </div>
            )}
          </Card>
        </>
      )}
    </div>
  );
}
