import { useEffect, useMemo, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { Api, pollJob } from "../lib/api";
import { num, pct } from "../lib/format";
import { useStore } from "../lib/store";
import ResultView from "../components/ResultView";
import { HypothesisList, Pipeline3D } from "../components/blocks";
import { Alert, Badge, Card, Empty, KV, Stat } from "../components/ui";

const STEPS = [
  { id: "problem", label: "1 · Problem" },
  { id: "signal", label: "2 · Signal" },
  { id: "analysis", label: "3 · Automatic analysis" },
  { id: "reasoning", label: "4 · Reasoning" },
  { id: "pipeline", label: "5 · Pipeline" },
  { id: "results", label: "6 · Results" },
  { id: "innovation", label: "7 · Innovation" },
  { id: "benchmark", label: "8 · Benchmark" },
  { id: "report", label: "9 · Report" },
];

const INNOVATION = [
  ["Blind end-to-end analysis", "No modulation, sample rate, symbol rate, carrier offset, FEC or interleaver is supplied in BLIND mode — every one of them is estimated and listed as a hypothesis with its evidence, and the platform says 'Unable to estimate reliably' when it cannot support a claim."],
  ["Multi-hypothesis reasoning", "Alternative pipelines (modulation × symbol rate) are demodulated and scored on margin, EVM, error-correction gain, timing quality and carrier quality; the ranking is shown, not hidden behind one answer."],
  ["Evidence graph & confidence system", "Each automatic result carries status, confidence, method and limitations, and the chain IQ → spectrum → waterfall → candidate → modulation → timing → demodulation → FEC → bits is rendered as a graph you can inspect node by node."],
  ["FEC and interleaver hypothesis testing", "Convolutional (K3/K5/K7, several rates), Reed–Solomon and concatenated hypotheses are each scored against a random-data null; interleaver geometries are only claimed when de-interleaving beats a random-permutation control group. Nothing is claimed without evidence, and LDPC is reported as unsupported rather than guessed."],
  ["Comparison & historical tracking", "Every analysis is stored with its measurements so two recordings can be compared later (similarity verdicts over spectrum, rate, SNR, EVM and byte statistics) and two days of recordings can be diffed."],
  ["Synthetic generator with verification", "Generate any supported modulation with known ground truth, then verify what the platform recovered against that ground truth (bit alignment search, descrambling, BER) — a closed loop that makes the claims testable."],
  ["Self-benchmarking", "The pipeline is scored against its own generator over a grid of modulations and SNRs, producing measured confusion matrices and estimator error tables."],
  ["One browser platform", "Parsing, preprocessing, spectrum, waterfall, detection, parameters, classification, demodulation, FEC, interleaving, bitstream, reports, comparison, benchmarking and live capture in one application with a real API."],
];

const NOT_INNOVATION = "Plotting an FFT, drawing a waterfall, showing a constellation or demodulating a known QPSK stream — these are textbook steps, and this platform treats them as prerequisites rather than achievements.";

export default function Demo() {
  const { demos, demoLoad, runAnalysis, current, currentId, toast, job } = useStore();
  const nav = useNavigate();
  const [step, setStep] = useState(0);
  const [picked, setPicked] = useState<string | null>(null);
  const [bench, setBench] = useState<any>(null);
  const [benchProg, setBenchProg] = useState<any>(null);
  const [benchBusy, setBenchBusy] = useState(false);
  const [reports, setReports] = useState<any[]>([]);
  const [busy, setBusy] = useState(false);

  useEffect(() => { if (picked === null && demos.length) setPicked(demos[0].name); }, [demos, picked]);

  const result = current?.result;
  const sig = result?.signal || {};
  const evidence = sig.evidence_graph || result?.evidence_graph || null;
  const nodes = evidence?.nodes || (Array.isArray(evidence) ? evidence : []);
  const conf = sig.confidence || {};

  const loadSignal = async () => {
    if (!picked) return;
    setBusy(true);
    const f = await demoLoad(picked);
    setBusy(false);
    if (f) { setStep(2); }
  };

  const analyse = async () => {
    const fileId = current?.analysis?.file_id;
    if (!fileId) { toast("load a demo signal first", "err"); return; }
    const res = await runAnalysis({ file_id: fileId, options: {}, wait_s: 0 }, { label: "Guided demo analysis" });
    if (res?.analysis_id) setStep(3);
  };

  const runBench = async () => {
    setBenchBusy(true); setBenchProg({ progress: 0, message: "queued" });
    try {
      const started = await Api.benchmark({ config: { kind: "amc" }, async: true });
      const done = await pollJob(started.job.job_id, setBenchProg);
      if (done.status === "done" && done.result?.benchmark_id) setBench(await Api.getBenchmark(done.result.benchmark_id));
      else toast(done.error || "benchmark failed", "err");
    } catch (e: any) { toast(`benchmark failed: ${e.message}`, "err"); }
    finally { setBenchBusy(false); }
  };

  const buildReports = async () => {
    const id = current?.analysis?.analysis_id;
    if (!id) return;
    setBusy(true);
    try {
      const r = await Api.makeReport(id, ["pdf", "json", "csv", "txt"]);
      setReports(r.reports || []);
      toast(`${(r.reports || []).length} report file(s) written to the database and disk`, "ok");
    } catch (e: any) { toast(`report failed: ${e.message}`, "err"); }
    finally { setBusy(false); }
  };

  const gate = useMemo(() => ({
    analysis: !!currentId && !!result,
    benchmark: !!bench,
    report: reports.length > 0,
  }), [currentId, result, bench, reports]);

  return (
    <div>
      <Card title="SIH demo mode" sub="a guided path through the whole platform: problem → signal → automatic analysis → reasoning → pipeline → results → innovation → benchmark → report"
        right={<Badge kind="info">step {step + 1} / {STEPS.length}</Badge>}>
        <div className="tabs">
          {STEPS.map((s, i) => (
            <button key={s.id} className={i === step ? "active" : ""} onClick={() => setStep(i)}>{s.label}</button>
          ))}
        </div>
        <div className="row" style={{ gap: 8 }}>
          <button className="tiny ghost" disabled={step === 0} onClick={() => setStep((s) => Math.max(0, s - 1))}>← back</button>
          <button className="tiny" disabled={step >= STEPS.length - 1} onClick={() => setStep((s) => Math.min(STEPS.length - 1, s + 1))}>next →</button>
          <span style={{ flex: 1 }} />
          <Link className="btn tiny" to="/documentation">documentation</Link>
        </div>
      </Card>

      {step === 0 && (
        <Card title="The problem statement" sub="SIH26147 — Automated model for analysis of .IQ and .WAV files along with signal parameter extraction">
          <div className="grid g2">
            <div>
              <p className="small" style={{ color: "#c8d6ee" }}>
                An operator is handed a recording — sometimes a well-labelled WAV, sometimes a raw
                interleaved IQ dump with no metadata at all — and needs to know what is inside it:
                what the sample rate and data format are, where the emissions are in time and
                frequency, what modulation each one uses, what the symbol rate is, whether there is
                forward error correction or interleaving in the chain, and what the recovered bit
                stream looks like structurally.
              </p>
              <p className="small" style={{ color: "#c8d6ee" }}>
                The hard parts are not the FFT. They are: working blind, avoiding invented numbers,
                quantifying confidence, testing hypotheses instead of asserting them, and being
                honest about what cannot be determined. That is exactly what this platform is built
                around.
              </p>
              <Alert kind="info" title="what the demo proves">
                Every number you will see in the next eight steps is computed live from a signal this
                application synthesised or you uploaded. Nothing is canned and no chart is decorative.
              </Alert>
            </div>
            <div>
              <KV rows={[
                ["input", ".iq / .IQ / .wav / .WAV / complex binary, plus live capture"],
                ["output", "parameters, ranked modulation hypotheses, demodulated bits, FEC/interleaver hypotheses, reports"],
                ["blind mode", "no modulation, rate, offset, FEC or interleaver hints"],
                ["honesty policy", "'Unknown / requires estimation' and 'Unable to estimate reliably' are first-class results"],
                ["stack", "React + TypeScript · FastAPI · NumPy/SciPy · scikit-learn · SQLite/PostgreSQL · Plotly charts · Docker"],
              ]} />
            </div>
          </div>
        </Card>
      )}

      {step === 1 && (
        <Card title="Pick a signal" sub="demo signals are synthesised on demand with full ground truth (modulation, symbol rate, SNR, FEC, interleaver) so the analysis can be checked">
          <div className="grid g3">
            {demos.map((d: any) => (
              <button key={d.name} className={`card ${picked === d.name ? "" : "ghost"}`} style={{ textAlign: "left", cursor: "pointer" }}
                onClick={() => setPicked(d.name)}>
                <b className="small">{d.name}</b>
                <div className="small muted" style={{ marginTop: 4 }}>
                  {d.ground_truth?.modulation || "—"} · {num(d.ground_truth?.symbol_rate, 0)} sym/s · SNR {num(d.ground_truth?.snr_db, 0)} dB
                  {d.ground_truth?.fec && d.ground_truth.fec !== "none" ? ` · FEC ${d.ground_truth.fec}` : ""}
                </div>
                <div className="small muted">{d.note || d.description || ""}</div>
              </button>
            ))}
          </div>
          <div className="row" style={{ marginTop: 12 }}>
            <button className="primary" disabled={!picked || busy} onClick={loadSignal}>{busy ? "loading…" : `load ${picked || "signal"}`}</button>
            <span className="small muted">or use the upload panel on the Analyze page with your own .iq/.wav file</span>
          </div>
        </Card>
      )}

      {step === 2 && (
        <Card title="Automatic analysis" sub="AUTO mode: the platform estimates every parameter itself and records the evidence for each">
          <div className="row">
            <button className="primary" disabled={!current?.analysis?.file_id || (job && job.status === "running")} onClick={analyse}>
              {job && job.status === "running" ? "analysing…" : "run the full pipeline"}
            </button>
            <span className="pill">{current?.analysis?.file?.filename || "no signal loaded"}</span>
            <Link className="btn tiny" to="/blind-analysis">or run it as BLIND analysis</Link>
          </div>
          {job && (
            <div style={{ marginTop: 10 }}>
              <div className="prog"><i style={{ width: `${(job.progress || 0) * 100}%` }} /></div>
              <div className="small muted" style={{ marginTop: 5 }}>{job.message}</div>
            </div>
          )}
          {result && (
            <div className="grid g4" style={{ marginTop: 12 }}>
              <Stat k="Sample rate" v={result.fs_known === false ? "Unknown / estimated" : `${num(result.fs, 0)} Hz`} n={result.fs_source || ""} />
              <Stat k="Emission bandwidth" v={`${num(sig.spectrum?.obw_99_hz ?? result.spectrum?.obw_99_hz, 0)} Hz`} n="99 % occupied bandwidth" />
              <Stat k="SNR" v={`${num(sig.spectrum?.snr_db ?? result.spectrum?.snr_db, 1)} dB`} n="in-band, from the noise density" />
              <Stat k="Symbol rate" v={sig.symbol_rate_hz ? `${num(sig.symbol_rate_hz, 0)} sym/s` : "Unable to estimate reliably"} n={sig.symbol_rate_method || ""} />
            </div>
          )}
        </Card>
      )}

      {step === 3 && (
        <>
          <Card title="How the platform reasons" sub="each automatic decision is a hypothesis with a status, a confidence, a method and its limitations">
            {result ? (
              <div className="grid g2">
                <div>
                  <KV rows={Object.entries(conf).slice(0, 12).map(([k, v]: any) => [k, typeof v === "object" ? JSON.stringify(v).slice(0, 90) : String(v)])} />
                </div>
                <div>
                  <div className="sub">modulation candidates</div>
                  <HypothesisList kind="modulation" hypotheses={(sig.modulation?.candidates || []).slice(0, 5)} />
                </div>
              </div>
            ) : <Empty>run the analysis in step 3 first</Empty>}
          </Card>
          <Card title="Evidence graph" sub="the chain the platform used: IQ → spectrum → waterfall → candidate → modulation → timing → demodulation → FEC → bitstream">
            <Pipeline3D nodes={nodes} height={400} />
          </Card>
        </>
      )}

      {step === 4 && (
        <Card title="Pipeline view" sub="status of every stage, in order, with the numbers each one produced">
          {result ? (
            <div className="tblwrap">
              <table>
                <thead><tr><th>#</th><th>stage</th><th>status</th><th>detail</th></tr></thead>
                <tbody>{nodes.map((n: any, i: number) => (
                  <tr key={n.id || i}>
                    <td className="mono">{i + 1}</td>
                    <td>{n.label || n.id}</td>
                    <td><Badge kind={n.status === "ok" ? "ok" : n.status === "failed" ? "bad" : "warn"}>{n.status}</Badge></td>
                    <td className="small muted">{n.detail}</td>
                  </tr>
                ))}</tbody>
              </table>
            </div>
          ) : <Empty>no analysis yet</Empty>}
        </Card>
      )}

      {step === 5 && (
        result ? (
          <ResultView result={result} analysis={current?.analysis}
            onRegionRequest={async (r) => { await runAnalysis({ file_id: current?.analysis?.file_id, options: { region: r }, wait_s: 0 }, { label: "region re-analysis" }); }}
            onSignalRequest={(i) => Api.selectSignal(currentId!, { signal_id: i }).then(() => toast("analysis of the selected emission queued", "info")).catch((e) => toast(e.message, "err"))} />
        ) : <Empty>run the analysis first (step 3)</Empty>
      )}

      {step === 6 && (
        <>
          <Card title="What makes this different" sub="the differentiators, in the platform's own terms">
            <div className="col">
              {INNOVATION.map(([t, d]) => (
                <div key={t} className="card" style={{ margin: 0 }}>
                  <b className="small">{t}</b>
                  <div className="small" style={{ color: "#c8d6ee", marginTop: 4 }}>{d}</div>
                </div>
              ))}
            </div>
            <Alert kind="warn" title="what is explicitly not claimed as innovation">{NOT_INNOVATION}</Alert>
            <Alert kind="info" title="and what is explicitly not claimed at all">
              No perfect decoding of arbitrary unknown protocols, no source attribution from similarity
              alone, no FEC identification without statistical evidence, and no invented metadata.
            </Alert>
          </Card>
          <Card title="Try the differentiators directly">
            <div className="row">
              <Link className="btn" to="/blind-analysis">blind analysis</Link>
              <Link className="btn" to="/fec">FEC &amp; interleaving hypotheses</Link>
              <Link className="btn" to="/compare">compare two signals</Link>
              <Link className="btn" to="/generator">synthetic generator + verification</Link>
              <Link className="btn" to="/history">historical tracking</Link>
              <Link className="btn" to="/live">live capture</Link>
            </div>
          </Card>
        </>
      )}

      {step === 7 && (
        <Card title="Self-benchmark" sub="the same pipeline, run over synthetic signals with known ground truth — measured accuracy, not a claim"
          right={bench ? <Badge kind={bench.accuracy > 0.8 ? "ok" : "warn"}>accuracy {pct(bench.accuracy, 1)}</Badge> : null}>
          <div className="row">
            <button className="primary" disabled={benchBusy} onClick={runBench}>{benchBusy ? "running…" : "run the AMC benchmark"}</button>
            <span className="small muted">full grid (all sections) is available on the Benchmark page; this runs the fast classification section</span>
          </div>
          {benchProg && benchBusy && (
            <div style={{ marginTop: 10 }}>
              <div className="prog"><i style={{ width: `${(benchProg.progress || 0) * 100}%` }} /></div>
              <div className="small muted" style={{ marginTop: 5 }}>{benchProg.message}</div>
            </div>
          )}
          {bench && (
            <>
              <div className="grid g4" style={{ marginTop: 12 }}>
                <Stat k="cases" v={num(bench.n_cases, 0)} n={bench.kind} />
                <Stat k="accuracy" v={pct(bench.accuracy, 1)} n="correct modulation decisions" />
                <Stat k="duration" v={`${num((bench.duration_ms || 0) / 1000, 1)} s`} n="wall clock on this machine" />
                <Stat k="classes" v={(bench.classes || []).length} n={(bench.classes || []).join(", ")} />
              </div>
              {bench.per_class && (
                <div className="tblwrap" style={{ marginTop: 12, maxHeight: 300 }}>
                  <table><thead><tr><th>class</th><th className="num">precision</th><th className="num">recall</th><th className="num">support</th></tr></thead>
                    <tbody>{Object.entries(bench.per_class).map(([k, v]: any) => (
                      <tr key={k}><td>{k}</td><td className="num">{num(v.precision, 3)}</td><td className="num">{num(v.recall, 3)}</td><td className="num">{num(v.support, 0)}</td></tr>))}
                    </tbody></table>
                </div>
              )}
            </>
          )}
        </Card>
      )}

      {step === 8 && (
        <Card title="Export the report" sub="PDF with the embedded charts, plus JSON/CSV/TXT of every measured value">
          <div className="row">
            <button className="primary" disabled={!currentId || busy} onClick={buildReports}>{busy ? "building…" : "generate all four formats"}</button>
            {currentId && (["pdf", "json", "csv", "txt"] as const).map((f) => (
              <a key={f} className="btn" href={Api.reportInlineUrl(currentId, f)} target="_blank" rel="noreferrer">on-the-fly {f.toUpperCase()}</a>
            ))}
          </div>
          {!!reports.length && (
            <div className="tblwrap" style={{ marginTop: 12 }}>
              <table><thead><tr><th>format</th><th>file</th><th className="num">size</th><th></th></tr></thead>
                <tbody>{reports.map((r: any) => (
                  <tr key={r.report_id}><td><Badge kind="info">{r.format}</Badge></td><td className="mono small">{r.title}</td>
                    <td className="num">{num((r.size_bytes || 0) / 1024, 1)} kB</td>
                    <td><a className="btn tiny" href={Api.reportDownloadUrl(r.report_id)}>download</a></td></tr>))}
                </tbody></table>
            </div>
          )}
          {!currentId && <Alert kind="warn">Run an analysis first — the report is generated from its stored result.</Alert>}
          <Alert kind="info" title="what a judge should check in the PDF">
            the parameter table with confidence and method per row, the modulation candidates with their
            evidence, the FEC/interleaver hypotheses with the thresholds they had to beat, the bitstream
            statistics, and the limitations section. If a value is not supportable it is absent or marked,
            never invented.
          </Alert>
        </Card>
      )}
    </div>
  );
}
