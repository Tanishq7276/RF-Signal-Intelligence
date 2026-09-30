import { useEffect, useState } from "react";
import { Api, pollJob } from "../lib/api";
import { num, pct } from "../lib/format";
import { useStore } from "../lib/store";
import Plot from "../components/Plot";
import { confusionHeatmap } from "../components/charts";
import { Alert, Badge, Card, Empty, Json, Spinner, Stat } from "../components/ui";

/** /benchmark — the platform scores itself against a known ground truth. */
export default function Benchmark() {
  const { toast, job, cancelJob } = useStore();
  const [cfg, setCfg] = useState<any>(null);
  const [kind, setKind] = useState("amc");
  const [runs, setRuns] = useState<any[]>([]);
  const [res, setRes] = useState<any>(null);
  const [busy, setBusy] = useState(false);
  const [progress, setProgress] = useState<any>(null);

  useEffect(() => {
    Api.benchmarkConfig().then((c) => setCfg(c)).catch(() => undefined);
    Api.benchmarks().then((r) => setRuns(r.benchmarks || [])).catch(() => undefined);
  }, []);

  const start = async () => {
    setBusy(true); setRes(null); setProgress({ progress: 0, message: "queued" });
    try {
      const started = await Api.benchmark({ config: { kind }, async: true });
      const done = await pollJob(started.job.job_id, setProgress);
      if (done.status === "done" && done.result?.benchmark_id) {
        setRes(await Api.getBenchmark(done.result.benchmark_id));
        Api.benchmarks().then((r) => setRuns(r.benchmarks || []));
        toast(`benchmark finished: ${done.result.n_cases} cases in ${num((done.result.duration_ms || 0) / 1000, 1)} s`, "ok");
      } else if (done.status !== "cancelled") toast(done.error || "benchmark failed", "err");
    } catch (e: any) { toast(`benchmark failed: ${e.message}`, "err"); }
    finally { setBusy(false); }
  };

  const classes: string[] = res?.classes || [];
  const matrix: number[][] = res?.confusion_matrix || [];

  return (
    <div>
      <Card title="Benchmark mode" sub="the same pipeline that analyses your uploads is run over a grid of synthetic signals with known ground truth; the confusion matrix and estimator errors are measured, not declared">
        <div className="row" style={{ gap: 12, alignItems: "flex-end" }}>
          <div style={{ width: 220 }}><label className="f"><span>benchmark</span>
            <select value={kind} onChange={(e) => setKind(e.target.value)}>
              {(cfg?.kinds || ["amc", "estimators", "demod", "fec", "interleaving", "full"]).map((k: string) => <option key={k} value={k}>{k}</option>)}
            </select></label></div>
          <button className="primary" disabled={busy} onClick={start}>{busy ? "running…" : "run benchmark"}</button>
          {busy && <button className="danger" onClick={cancelJob}>cancel</button>}
          <span className="small muted">“full” runs every section and takes several minutes on a small container; “amc” is the quick check.</span>
        </div>
        {progress && (
          <div style={{ marginTop: 10 }}>
            <div className="prog"><i style={{ width: `${(progress.progress || 0) * 100}%` }} /></div>
            <div className="small muted" style={{ marginTop: 4 }}>{progress.message}</div>
          </div>
        )}
        {cfg && (
          <div style={{ marginTop: 10 }} className="small muted">
            default grid: {cfg.defaults?.modulations?.length} modulations × SNR {JSON.stringify(cfg.defaults?.snr_db)} × symbol rates {JSON.stringify(cfg.defaults?.symbol_rates)}
          </div>
        )}
      </Card>

      {busy && <Spinner label="generating cases and classifying them" />}
      {res && (
        <>
          <div className="grid g4">
            <Stat k="Cases" v={num(res.n_cases, 0)} n={`kind: ${res.kind}`} />
            <Stat k="Accuracy" v={res.accuracy != null ? pct(res.accuracy, 1) : "n/a"} n="classification accuracy over the grid" />
            <Stat k="Duration" v={`${num((res.duration_ms || 0) / 1000, 1)} s`} n={`${res.status}`} />
            <Stat k="Sections" v={Object.keys(res.metrics || {}).length} n="measured sections in this run" />
          </div>
          {classes.length > 0 && (
            <Card title="Confusion matrix" sub="rows: true modulation, columns: predicted">
              <Plot height={Math.max(320, 42 * classes.length)} data={confusionHeatmap(classes, matrix) as any}
                layout={{ xaxis: { title: "predicted" }, yaxis: { title: "true" }, margin: { l: 70, r: 20, t: 20, b: 60 } }} />
            </Card>
          )}
          {res.per_class && (
            <Card title="Per-class results">
              <div className="tblwrap">
                <table><thead><tr><th>class</th><th className="num">precision</th><th className="num">recall</th><th className="num">f1</th><th className="num">support</th></tr></thead>
                  <tbody>{Object.entries(res.per_class).map(([k, v]: any) => (
                    <tr key={k}><td>{k}</td><td className="num">{num(v.precision, 3)}</td><td className="num">{num(v.recall, 3)}</td>
                      <td className="num">{num(v["f1-score"] ?? v.f1, 3)}</td><td className="num">{num(v.support, 0)}</td></tr>))}
                  </tbody></table>
              </div>
            </Card>
          )}
          <Card title="Raw benchmark payload">
            <Json value={{ metrics: res.metrics, notes: res.notes, config: res.config }} label="benchmark JSON" />
          </Card>
        </>
      )}

      <Card title="Previous runs" sub="stored in the database">
        {runs.length ? (
          <div className="tblwrap" style={{ maxHeight: 260 }}>
            <table><thead><tr><th>kind</th><th>when</th><th className="num">cases</th><th className="num">accuracy</th><th className="num">duration</th><th></th></tr></thead>
              <tbody>{runs.map((r) => (
                <tr key={r.benchmark_id}><td><Badge kind="info">{r.kind}</Badge></td>
                  <td className="small muted">{r.created_at ? new Date(r.created_at).toLocaleString() : ""}</td>
                  <td className="num">{r.n_cases}</td><td className="num">{r.accuracy != null ? num(r.accuracy, 3) : "—"}</td>
                  <td className="num">{num((r.duration_ms || 0) / 1000, 1)} s</td>
                  <td><button className="tiny" onClick={() => Api.getBenchmark(r.benchmark_id).then(setRes)}>open</button></td></tr>))}
              </tbody></table>
          </div>
        ) : <Empty>no benchmark has been run in this workspace yet</Empty>}
      </Card>
    </div>
  );
}
