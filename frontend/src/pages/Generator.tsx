import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { Api } from "../lib/api";
import { bytes, num, si } from "../lib/format";
import { useStore } from "../lib/store";
import Plot from "../components/Plot";
import JobBar from "../components/jobbar";
import { Alert, Card, Empty, KV, NumField, Select, Spinner, Stat } from "../components/ui";

const SPEC_DEFAULT = {
  modulation: "QPSK", fs: 1_000_000, symbol_rate: 100_000, snr_db: 15, n_symbols: 4000,
  carrier_offset_hz: 5000, rolloff: 0.35, fec: "conv_K7_r1_2", interleaver: "none",
  payload: "text", text: "HELLO SIH", seed: 7,
};

/** /generator — the SIH demo: transmit a known message, then recover it and prove it. */
export default function Generator() {
  const { toast, refresh, runAnalysis, setCurrentId } = useStore();
  const nav = useNavigate();
  const [opts, setOpts] = useState<any>(null);
  const [spec, setSpec] = useState<any>(SPEC_DEFAULT);
  const [format, setFormat] = useState("wav");
  const [gen, setGen] = useState<any>(null);
  const [busy, setBusy] = useState(false);
  const [verify, setVerify] = useState<any>(null);
  const [genList, setGenList] = useState<any[]>([]);
  const [preview, setPreview] = useState<any>(null);

  useEffect(() => {
    Api.generateOptions().then(setOpts).catch((e) => toast(e.message, "err"));
    Api.generated().then((r) => setGenList(r.generated || [])).catch(() => undefined);
  }, [toast]);

  const generate = async () => {
    setBusy(true); setVerify(null); setPreview(null);
    try {
      const res = await Api.generate({ spec, format });
      setGen(res);
      await refresh();
      Api.generated().then((r) => setGenList(r.generated || []));
      const pv = await Api.preview(res.file.file_id, 2000);
      setPreview(pv);
      toast(`${res.file.filename} generated (${bytes(res.file.size_bytes)})`, "ok", "Generator");
    } catch (e: any) { toast(`generation failed: ${e.message}`, "err"); }
    finally { setBusy(false); }
  };

  const analyzeAndVerify = async () => {
    if (!gen) return;
    const res = await runAnalysis({ file_id: gen.file.file_id, options: {}, wait_s: 0 }, { label: "Generated signal" });
    if (!res?.analysis_id) return;
    setCurrentId(res.analysis_id);
    try {
      const v = await Api.verifyGenerated({ generated_id: gen.generated_id, analysis_id: res.analysis_id });
      setVerify(v);
      toast(v.ok ? `message recovery measured: ${num(v.recovery_percent, 2)} % of payload bits` : (v.message || "verification unavailable"), v.ok ? "ok" : "info");
    } catch (e: any) { toast(`verification failed: ${e.message}`, "err"); }
    nav(`/analyze/${res.analysis_id}`);
  };

  const gt = gen?.ground_truth || {};

  return (
    <div>
      <JobBar />
      <div className="grid g2">
        <Card title="Synthetic signal lab" sub="the generated file is a real IQ/WAV recording: message → bits → FEC → interleaving → pulse shaping → noise">
          <div className="grid g3">
            <Select label="modulation" value={spec.modulation} onChange={(v) => setSpec({ ...spec, modulation: v })}
              options={(opts?.modulations || ["QPSK"]).map((m: string) => ({ value: m, label: m }))} />
            <NumField label="sample rate [Hz]" value={spec.fs} step={1000} min={1000} onChange={(v: number) => setSpec({ ...spec, fs: v })} />
            <NumField label="symbol rate [sym/s]" value={spec.symbol_rate} step={1000} min={100} onChange={(v: number) => setSpec({ ...spec, symbol_rate: v })} />
            <NumField label="SNR [dB]" value={spec.snr_db} step={1} min={-10} max={60} onChange={(v: number) => setSpec({ ...spec, snr_db: v })} />
            <NumField label="symbols (duration)" value={spec.n_symbols} step={100} min={64} onChange={(v: number) => setSpec({ ...spec, n_symbols: v })} />
            <NumField label="carrier offset [Hz]" value={spec.carrier_offset_hz} step={500} onChange={(v: number) => setSpec({ ...spec, carrier_offset_hz: v })} />
            <Select label="FEC" value={spec.fec} onChange={(v) => setSpec({ ...spec, fec: v })}
              options={(opts?.fec || ["none"]).map((f: string) => ({ value: f, label: f }))} />
            <Select label="interleaving" value={spec.interleaver} onChange={(v) => setSpec({ ...spec, interleaver: v })}
              options={[{ value: "none", label: "none" }, ...(opts?.interleavers || []).filter((i: any) => i.kind !== "none")
                .map((i: any) => ({ value: i.kind + (i.rows ? `:${i.rows}` : "") + (i.depth ? `:${i.depth}` : ""), label: `${i.kind} ${i.rows ? `rows=${i.rows}` : ""} ${i.depth ? `depth=${i.depth}` : ""}` }))]}
              hint="rows/depth are appended to the spec value" />
            <NumField label="roll-off" value={spec.rolloff} step={0.05} min={0.05} max={1} onChange={(v: number) => setSpec({ ...spec, rolloff: v })} />
            <NumField label="seed" value={spec.seed} step={1} onChange={(v: number) => setSpec({ ...spec, seed: v })} />
            <Select label="output format" value={format} onChange={setFormat}
              options={[{ value: "wav", label: ".wav (stereo int16 I/Q)" }, { value: "iq", label: ".iq (complex int16 interleaved)" }]} />
          </div>
          <div className="grid g2" style={{ marginTop: 10 }}>
            <label className="f"><span>message (transmitted payload text)</span>
              <input value={spec.text} onChange={(e) => setSpec({ ...spec, text: e.target.value, payload: "text" })} placeholder="HELLO SIH" />
              <span className="small muted">stored in the ground truth so the recovery can be compared bit-by-bit</span></label>
            <label className="f"><span>payload</span>
              <select value={spec.payload} onChange={(e) => setSpec({ ...spec, payload: e.target.value })}>
                <option value="text">text message (ASCII)</option>
                <option value="random">random bits</option>
              </select></label>
          </div>
          <div className="row" style={{ marginTop: 12 }}>
            <button className="primary" disabled={busy} onClick={generate}>{busy ? "generating…" : "generate signal"}</button>
            {gen && <button disabled={busy} onClick={analyzeAndVerify}>analyse this signal &amp; verify the message</button>}
            {gen && <a className="btn" href={gen.download_url}>download {format.toUpperCase()}</a>}
          </div>
          <Alert kind="info">
            A generated signal is synthetic. The platform never claims to decode unknown real protocol
            traffic; here the message is known because <i>we</i> transmitted it, which makes the recovery
            score an honest measurement of the whole chain.
          </Alert>
        </Card>

        <div className="col" style={{ gap: 14 }}>
          {gen ? (
            <>
              <Card title="Generated signal" sub={gen.file.filename}>
                <KV rows={[
                  ["file id", gen.file.file_id], ["size", bytes(gen.file.size_bytes)],
                  ["format", gen.file.format], ["layout", gen.file.iq_layout],
                  ["sample rate", si(gen.file.sample_rate, "Hz", 0)], ["samples", gen.file.n_samples],
                  ["duration", `${num(gen.file.duration_s, 6)} s`],
                ]} />
                <div className="grid g3" style={{ marginTop: 10 }}>
                  <Stat k="payload bits" v={num(gt.payload_bits, 0)} />
                  <Stat k="coded bits" v={num(gt.n_coded_bits, 0)} />
                  <Stat k="modulation" v={gt.modulation} />
                  <Stat k="symbol rate" v={si(gt.symbol_rate, "sym/s", 1)} />
                  <Stat k="SNR (in-band)" v={`${num(gt.snr_db, 1)} dB`} />
                  <Stat k="FEC" v={gt.fec || "none"} />
                </div>
                {preview && (
                  <Plot height={180} data={[{
                    type: "scattergl", mode: "lines", x: preview.t_s, y: preview.i, line: { color: "#22d3ee", width: 1 }, name: "I",
                  }, { type: "scattergl", mode: "lines", x: preview.t_s, y: preview.q, line: { color: "#f472b6", width: 1 }, name: "Q" }]}
                    layout={{ xaxis: { title: "time [s]" }, yaxis: { title: "amplitude" } }} />
                )}
                <div className="small muted" style={{ marginTop: 6 }}>
                  first 64 payload bits (post-scramble): <span className="mono">{(gt.payload_bitstring || "").slice(0, 64)}</span>
                </div>
                {gt.payload_text && <Alert kind="ok">transmitted message: <b className="mono">{gt.payload_text}</b></Alert>}
              </Card>
              {verify && (
                <Card title="Message recovery check" sub="bit-by-bit comparison against the generator's ground truth">
                  {verify.ok ? (
                    <>
                      <div className="grid g3">
                        <Stat k="recovery" v={`${num(verify.recovery_percent, 2)} %`} n={`${verify.score.matched_bits} / ${verify.score.compared_bits} payload bits`} />
                        <Stat k="BER" v={num(verify.score.bit_error_rate, 5)} n={`alignment shift ${verify.score.shift_bits} bits`} />
                        <Stat k="bit source" v={verify.stream_used} n={verify.stream_source} />
                      </div>
                      <div className="hr" />
                      <KV rows={[
                        ["expected message", verify.expected_text || "—"],
                        ["recovered ASCII", verify.recovered_ascii ? <span className="mono">{verify.recovered_ascii.slice(0, 80)}</span> : "—"],
                        ["text match", verify.text_match === undefined ? "—" : String(verify.text_match)],
                        ["scrambler", verify.scrambler_note],
                      ]} />
                      <Alert kind={verify.text_match ? "ok" : verify.recovery_percent > 99 ? "warn" : "bad"}>
                        {verify.text_match
                          ? `The transmitted message "${verify.expected_text}" was recovered exactly from the analysed signal.`
                          : verify.verdict}
                      </Alert>
                      <Alert kind="info">{verify.policy}</Alert>
                    </>
                  ) : <Alert kind="warn">{verify.message}{verify.note ? ` — ${verify.note}` : ""}</Alert>}
                </Card>
              )}
            </>
          ) : <Card title="Generated signal"><Empty>Nothing generated yet. Set the parameters and press “generate signal”.</Empty></Card>}
          <Card title="Generated in this session">
            {genList.length ? (
              <div className="tblwrap" style={{ maxHeight: 260 }}>
                <table><thead><tr><th>file</th><th>modulation</th><th className="num">SNR</th><th></th></tr></thead>
                  <tbody>{genList.map((g) => (
                    <tr key={g.generated_id}>
                      <td className="small">{g.filename}</td>
                      <td>{g.ground_truth?.modulation} {g.ground_truth?.fec !== "none" && <span className="pill">{g.ground_truth?.fec}</span>}</td>
                      <td className="num">{num(g.ground_truth?.snr_db, 1)}</td>
                      <td><a className="btn tiny" href={g.download_url}>download</a></td>
                    </tr>))}
                  </tbody></table>
              </div>
            ) : <Empty>none yet</Empty>}
          </Card>
        </div>
      </div>
    </div>
  );
}
