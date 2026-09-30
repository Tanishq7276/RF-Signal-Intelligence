import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { Api } from "../lib/api";
import { num } from "../lib/format";
import { useStore } from "../lib/store";
import { HypothesisList } from "../components/blocks";
import { Alert, Badge, Card, Empty, KV, Select, Spinner, Stat } from "../components/ui";

/** /fec — hypothesis engines for error-correction coding and interleaving. */
export default function FecInterleave() {
  const { analyses, current, toast, setCurrentId } = useStore();
  const [analysisId, setAnalysisId] = useState<string | null>(current?.analysis?.analysis_id ?? null);
  const [fec, setFec] = useState<any>(null);
  const [inter, setInter] = useState<any>(null);
  const [stream, setStream] = useState<any>(null);
  const [busy, setBusy] = useState<string | null>(null);

  useEffect(() => { if (!analysisId && analyses.length) setAnalysisId(analyses[0].analysis_id); }, [analyses, analysisId]);

  const runFec = async () => {
    if (!analysisId) return;
    setBusy("fec");
    try { setFec(await Api.fec({ analysis_id: analysisId })); stream(analysisId); }
    catch (e: any) { toast(`FEC analysis failed: ${e.message}`, "err"); }
    finally { setBusy(null); }
  };
  const runInter = async () => {
    if (!analysisId) return;
    setBusy("inter");
    try { setInter(await Api.interleaving({ analysis_id: analysisId })); }
    catch (e: any) { toast(`interleaver analysis failed: ${e.message}`, "err"); }
    finally { setBusy(null); }
  };
  const stream_ = async (id: string) => {
    try { setStream(await Api.bitstream(id, "auto")); } catch { setStream(null); }
  };
  const streamAlias = stream_;

  return (
    <div>
      <Card title="FEC &amp; interleaving hypothesis engines"
        sub="Both engines test their hypotheses against a random-data null (and, for interleaving, a random-permutation control group). A hypothesis is only reported as the result when it passes the evidence threshold.">
        <div className="row" style={{ gap: 10 }}>
          <div style={{ minWidth: 300, flex: 1 }}>
            <Select label="analysis" value={analysisId ?? ""} onChange={(v) => setAnalysisId(v || null)}
              options={analyses.filter((a) => a.status === "done").map((a) => ({ value: a.analysis_id, label: `${a.file?.filename} · ${a.summary?.modulation || "?"} · ${a.analysis_id.slice(0, 8)}` }))} />
          </div>
          <div style={{ display: "flex", alignItems: "flex-end", gap: 8 }}>
            <button className="primary" disabled={!analysisId || busy !== null} onClick={runFec}>{busy === "fec" ? "testing…" : "test FEC hypotheses"}</button>
            <button disabled={!analysisId || busy !== null} onClick={runInter}>{busy === "inter" ? "testing…" : "test interleaver hypotheses"}</button>
          </div>
        </div>
      </Card>

      <div className="grid g2">
        <Card title="FEC hypotheses" sub={fec?.policy || (current?.result?.signal?.fec?.policy) || "a code is only claimed when its evidence beats the random-data null"}>
          {fec ? (
            <>
              <div className="grid g3">
                <Stat k="leader" v={fec.best?.hypothesis || "none claimed"} n={fec.best ? `confidence ${num(fec.best.confidence, 3)}` : "below the 0.20 reporting threshold"} />
                <Stat k="bits tested" v={num(fec.n_bits_tested, 0)} n={fec.bit_source?.source} />
                <Stat k="families tested" v={new Set((fec.hypotheses || []).map((h: any) => h.family)).size} n="convolutional, Reed-Solomon, concatenated, LDPC" />
              </div>
              <div className="hr" />
              <HypothesisList kind="fec" hypotheses={fec.hypotheses || []} />
              {(fec.notes || []).map((n: string, i: number) => <Alert key={i} kind="info">{n}</Alert>)}
            </>
          ) : <Empty>Run the FEC test to see the ranked hypotheses with their evidence.</Empty>}
        </Card>
        <Card title="Interleaver hypotheses" sub={inter?.policy || (current?.result?.signal?.interleaving?.policy) || "geometry is only claimed when de-interleaving beats a random-permutation control"}>
          {inter ? (
            <>
              <div className="grid g3">
                <Stat k="leader" v={inter.best?.hypothesis || "none claimed"} n={inter.best ? `confidence ${num(inter.best.confidence, 3)}` : "no geometry beat the control group"} />
                <Stat k="bits tested" v={num(inter.n_bits_tested, 0)} n={inter.bit_source?.source} />
                <Stat k="cluster test" v={inter.cluster_test?.ok ? num(inter.cluster_test.z_score, 2) : "n/a"} n={inter.cluster_test?.interpretation || "measures channel memory, not a permutation"} />
              </div>
              <div className="hr" />
              <HypothesisList kind="interleave" hypotheses={inter.hypotheses || []} />
              {(inter.notes || []).map((n: string, i: number) => <Alert key={i} kind="info">{n}</Alert>)}
            </>
          ) : <Empty>Run the interleaver test to see whether any geometry improves the decoder measurably.</Empty>}
        </Card>
      </div>

      <Card title="Before / after error correction" sub="the same record's bit stream before and after the hypothesised decoder">
        {stream?.comparison ? (
          <div className="grid g2">
            {(["demod", "decoded"] as const).map((k) => (
              <div key={k}>
                <b className="small">{k === "demod" ? "raw demodulator decisions" : "after error correction"}</b>
                <KV rows={[
                  ["bits", num(stream.comparison[k].n_bits, 0)],
                  ["entropy", `${num(stream.comparison[k].entropy.bits_per_bit, 4)} bit/bit`],
                  ["printable runs", String((stream.comparison[k].printable_ascii || []).length)],
                  ["frame candidates", String((stream.comparison[k].frame_candidates || []).length)],
                ]} />
              </div>
            ))}
            <div style={{ gridColumn: "1/-1" }}><Alert kind="info">{stream.comparison.note}</Alert></div>
          </div>
        ) : <Empty>Compute it on the Bitstream page (stream = auto) once a hypothesis exists.</Empty>}
        <div className="row" style={{ marginTop: 10 }}>
          <Link className="btn tiny" to="/bitstream">open the bitstream view</Link>
          {analysisId && <button className="tiny ghost" onClick={() => { setCurrentId(analysisId); }}>set as current analysis</button>}
        </div>
      </Card>
      <Alert kind="warn" title="honest limits">
        FEC and interleaver detection cannot prove a protocol: they test a small family of classical codes
        and permutations. “Not detected” means no member of the tested family left evidence above the
        statistical threshold — it does not mean the signal is uncoded. LDPC is reported as
        unsupported rather than guessed.
      </Alert>
    </div>
  );
}
