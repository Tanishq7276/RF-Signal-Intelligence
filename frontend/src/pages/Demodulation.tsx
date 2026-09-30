import { useEffect, useState } from "react";
import { Api } from "../lib/api";
import { num, si } from "../lib/format";
import { useStore } from "../lib/store";
import Plot from "../components/Plot";
import { StageList } from "../components/blocks";
import { constellationTrace } from "../components/charts";
import { Alert, Card, Empty, KV, NumField, Select, Spinner, Stat, Tabs } from "../components/ui";

/** /demodulation — run one demodulator on a file with explicit parameters. */
export default function Demodulation() {
  const { files, current, toast } = useStore();
  const [fileId, setFileId] = useState<string | null>(files[0]?.file_id ?? null);
  const [mod, setMod] = useState("QPSK");
  const [rs, setRs] = useState<number | null>(null);
  const [rolloff, setRolloff] = useState(0.35);
  const [carrier, setCarrier] = useState<number | null>(null);
  const [diff, setDiff] = useState(false);
  const [out, setOut] = useState<any>(null);
  const [busy, setBusy] = useState(false);
  const [tab, setTab] = useState("stages");

  useEffect(() => { if (!fileId && files.length) setFileId(files[0].file_id); }, [files, fileId]);
  useEffect(() => {
    const s = current?.result?.signal;
    if (s?.symbol_rate_hz && rs === null) setRs(Math.round(s.symbol_rate_hz));
    // eslint-disable-next-line
  }, [current]);

  const run = async () => {
    if (!fileId) return;
    setBusy(true); setOut(null);
    try { setOut(await Api.demodulate({ file_id: fileId, analysis_id: current?.analysis?.analysis_id, modulation: mod, symbol_rate_hz: rs, rolloff, carrier_hint_hz: carrier, differential: diff })); }
    catch (e: any) { toast(`demodulation failed: ${e.message}`, "err"); }
    finally { setBusy(false); }
  };

  const c = out?.constellation;

  return (
    <div>
      <Card title="Demodulation" sub="carrier recovery → symbol-rate refinement → matched filter → timing-drift correction → decisions; each stage reports its own status">
        <div className="grid g4">
          <Select label="file" value={fileId ?? ""} onChange={(v) => setFileId(v || null)}
            options={files.map((f) => ({ value: f.file_id, label: f.filename }))} />
          <Select label="modulation" value={mod} onChange={setMod}
            options={["BPSK", "QPSK", "8PSK", "16QAM", "64QAM", "2FSK", "GFSK", "AM", "FM"].map((m) => ({ value: m, label: m }))} />
          <NumField label="symbol rate [sym/s]" value={rs} step={100} min={1} placeholder="required except AM/FM" onChange={(v: number | null) => setRs(v)} />
          <NumField label="roll-off" value={rolloff} step={0.05} min={0.05} max={1} onChange={(v: number) => setRolloff(v ?? 0.35)} />
          <NumField label="carrier hint [Hz]" value={carrier} step={100} placeholder="optional alias-resolution hint" onChange={(v: number | null) => setCarrier(v)} />
          <label className="chk" style={{ alignSelf: "end" }}><input type="checkbox" checked={diff} onChange={(e) => setDiff(e.target.checked)} />differential decoding</label>
          <div style={{ display: "flex", alignItems: "flex-end" }}>
            <button className="primary" disabled={!fileId || busy} onClick={run}>{busy ? "demodulating…" : "demodulate"}</button>
          </div>
        </div>
        <Alert kind="warn" title="hint vs measurement">
          A carrier hint is used only to resolve the M-th power line's alias ambiguity. The offset itself
          is always measured — trusting a hint blindly cost 44 % EVM in testing.
        </Alert>
      </Card>

      {busy && <Spinner label="running the demodulator" />}
      {out && (
        <Card title={`Result — ${out.modulation}`} sub={out.method}
          right={<div className="row" style={{ gap: 6 }}>
            <button className="tiny" onClick={() => {
              if (!out.bits_preview) return;
              navigator.clipboard?.writeText(out.bits_preview);
              toast("bit preview copied", "ok");
            }}>copy bits</button>
            <button className="tiny" onClick={() => {
              const blob = new Blob([JSON.stringify(out, null, 1)], { type: "application/json" });
              const a = document.createElement("a");
              a.href = URL.createObjectURL(blob); a.download = `demod_${out.modulation}.json`; a.click();
            }}>download JSON</button>
          </div>}>
          {out.ok ? (
            <>
              <div className="grid g4">
                <Stat k="EVM" v={`${num(out.evm_percent, 2)} %`} n={`${out.n_symbols} symbols`} />
                <Stat k="Carrier offset" v={si(out.carrier_offset_hz, "Hz", 2)} n="measured from the M-th power line" />
                <Stat k="Symbol rate used" v={si(out.symbol_rate_used_hz, "sym/s", 2)} n={`requested ${si(out.symbol_rate_requested_hz, "sym/s", 1)}`} />
                <Stat k="Bits" v={num(out.n_bits, 0)} n={`quality ${out.bit_quality} · LLRs ${out.llr_available ? "yes" : "no"}`} />
              </div>
              <div className="hr" />
              <Tabs tabs={[{ id: "stages", label: "stages" }, { id: "constellation", label: "constellation" }, { id: "bits", label: "bits" }, { id: "input", label: "input & quality" }]} value={tab} onChange={setTab} />
              {tab === "stages" && <StageList stages={out.stages || []} />}
              {tab === "constellation" && (c?.i?.length
                ? <Plot height={380} data={constellationTrace(c)} layout={{ xaxis: { title: "I", scaleanchor: "y", scaleratio: 1 }, yaxis: { title: "Q" } }} />
                : <Empty>no symbol samples returned</Empty>)}
              {tab === "bits" && (
                <>
                  <div className="small muted">first {out.bits_preview?.length} bits</div>
                  <pre style={{ whiteSpace: "pre-wrap", wordBreak: "break-all" }}>{out.bits_preview}</pre>
                  <Alert kind="info">Bits are the demodulator's decisions. Error correction, de-interleaving and any protocol interpretation happen separately (FEC &amp; interleaving page).</Alert>
                </>
              )}
              {tab === "input" && (
                <div className="grid g2">
                  <KV rows={Object.entries(out.input || {}).map(([k, v]) => [k, typeof v === "object" ? JSON.stringify(v) : String(v)])} />
                  <KV rows={Object.entries(out.quality || {}).map(([k, v]) => [k, typeof v === "number" ? num(v, 5) : String(v)])} />
                </div>
              )}
            </>
          ) : (
            <Alert kind="bad" title="demodulation did not complete">
              {out.message || "unknown error"}
              <div className="small" style={{ marginTop: 6 }}>
                What you can try: verify the modulation and symbol rate, select a narrower band region
                around the emission, analyse a longer record, or lower the noise by preprocessing.
              </div>
            </Alert>
          )}
        </Card>
      )}
    </div>
  );
}
