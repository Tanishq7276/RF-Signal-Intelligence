import { useEffect, useState } from "react";
import { Api } from "../lib/api";
import { num } from "../lib/format";
import { useStore } from "../lib/store";
import Plot from "../components/Plot";
import { HypothesisList } from "../components/blocks";
import { hypothesisBars } from "../components/charts";
import { Alert, Badge, Card, Empty, KV, NumField, Select, Spinner, Stat } from "../components/ui";

/** /modulation — classification candidates, evidence and the ML vote. */
export default function Modulation() {
  const { files, current, toast } = useStore();
  const [fileId, setFileId] = useState<string | null>(files[0]?.file_id ?? null);
  const [rs, setRs] = useState<number | null>(null);
  const [rolloff, setRolloff] = useState(0.35);
  const [out, setOut] = useState<any>(null);
  const [models, setModels] = useState<any>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => { Api.models().then(setModels).catch(() => undefined); }, []);
  useEffect(() => { if (!fileId && files.length) setFileId(files[0].file_id); }, [files, fileId]);

  const run = async () => {
    if (!fileId) return;
    setBusy(true); setOut(null);
    try { setOut(await Api.classify({ file_id: fileId, rs, rolloff })); }
    catch (e: any) { toast(`classification failed: ${e.message}`, "err"); }
    finally { setBusy(false); }
  };

  const cands = out?.candidates || [];
  const stored = current?.result?.signal?.modulation;

  return (
    <div>
      <Card title="Modulation classification" sub="hybrid: DSP decision rules + symbol statistics + spectral features + an optional ML vote — never a single model">
        <div className="grid g4">
          <Select label="file" value={fileId ?? ""} onChange={(v) => setFileId(v || null)}
            options={files.map((f) => ({ value: f.file_id, label: f.filename }))} />
          <NumField label="symbol rate hint [Hz]" value={rs} step={100} min={1} placeholder="let the platform estimate"
            onChange={(v: number | null) => setRs(v)} />
          <NumField label="roll-off" value={rolloff} step={0.05} min={0.05} max={1} onChange={(v: number) => setRolloff(v ?? 0.35)} />
          <div style={{ display: "flex", alignItems: "flex-end" }}>
            <button className="primary" disabled={!fileId || busy} onClick={run}>{busy ? "classifying…" : "classify"}</button>
          </div>
        </div>
        {busy && <div style={{ marginTop: 10 }}><Spinner label="extracting features and scoring candidates" /></div>}
      </Card>

      {out && (
        <div className="grid g2">
          <Card title="Candidates" sub={`primary: ${out.primary || "none"} — every candidate carries its own confidence and evidence`}>
            {cands.length ? (
              <>
                <Plot height={240} data={hypothesisBars(cands.slice(0, 8), (h: any) => h.modulation)}
                  layout={{ xaxis: { title: "confidence", range: [0, 1] }, margin: { l: 90, r: 12, t: 10, b: 34 } }} />
                <HypothesisList kind="modulation" hypotheses={cands.slice(0, 6)} />
              </>
            ) : <Empty>no candidate was produced</Empty>}
          </Card>
          <div className="col" style={{ gap: 14 }}>
            <Card title="Decision detail">
              <KV rows={[
                ["primary", out.primary || "—"],
                ["classifier probability", num(cands[0]?.probability ?? cands[0]?.confidence, 3)],
                ["features used", num((out.modulation?.features && Object.keys(out.modulation.features).length) || 0, 0)],
                ["ML model", models?.classifier?.available ? `${models.classifier.kind} (offline accuracy ${num(models.classifier.accuracy, 3)})` : "not available"],
                ["method", out.modulation?.method || "—"],
              ]} />
              {out.modulation?.features && (
                <div style={{ marginTop: 8 }}>
                  <div className="sub">measured features</div>
                  <div className="tblwrap" style={{ maxHeight: 220 }}>
                    <table><tbody>{Object.entries(out.modulation.features).slice(0, 30).map(([k, v]) => (
                      <tr key={k}><td className="small">{k}</td><td className="num">{typeof v === "number" ? num(v, 4) : String(v)}</td></tr>
                    ))}</tbody></table>
                  </div>
                </div>
              )}
              {(out.modulation?.limitations || []).map((l: string, i: number) => <Alert key={i} kind="warn">{l}</Alert>)}
            </Card>
            <Card title="From the last full analysis" sub="the classification produced inside the pipeline, for comparison">
              {stored ? (
                <>
                  <Stat k="primary" v={stored.primary || "—"} n={`confidence ${num(stored.confidence, 3)}`} />
                  <div className="row" style={{ marginTop: 8, gap: 6 }}>
                    {(stored.candidates || []).slice(0, 6).map((c: any) => (
                      <span key={c.modulation} className="pill">{c.modulation} {num(c.confidence ?? c.probability, 2)}</span>
                    ))}
                  </div>
                </>
              ) : <Empty>no analysis result loaded yet</Empty>}
            </Card>
            <Card title="Limitations of classification" sub="stated up front">
              <ul className="small" style={{ marginLeft: 16, color: "#c8d6ee" }}>
                <li>Analogue modulations (AM/FM) and FSK are the hardest cases for any blind classifier — low SNR makes them nearly indistinguishable from noise-driven decisions.</li>
                <li>A confident-looking number is still a probability, not proof; the alternatives are always listed.</li>
                <li>Encrypted or scrambled payloads are classified by their waveform only.</li>
              </ul>
            </Card>
          </div>
        </div>
      )}
      {!out && <Card title="Tip"><Alert kind="info">Run a full analysis first (Analyze page) to see the in-pipeline classification, or classify a file directly above.</Alert></Card>}
    </div>
  );
}
