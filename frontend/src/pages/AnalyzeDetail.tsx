import { useEffect, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { Api } from "../lib/api";
import { num } from "../lib/format";
import { useStore } from "../lib/store";
import JobBar from "../components/jobbar";
import ResultView from "../components/ResultView";
import { Alert, Badge, Card, Empty, Spinner } from "../components/ui";

/** Stored analysis result: /analyze/[id] */
export default function AnalyzeDetail() {
  const { id } = useParams();
  const nav = useNavigate();
  const { analyses, refresh, toast, runAnalysis, setCurrentId } = useStore();
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!id) return;
    setLoading(true);
    Api.analysis(id).then((d) => { setData(d); setCurrentId(id); })
      .catch((e) => toast(`analysis ${id}: ${e.message}`, "err"))
      .finally(() => setLoading(false));
  }, [id, toast, setCurrentId]);

  if (loading) return <Spinner label="loading the stored analysis" />;
  if (!data) return <Empty>analysis not found in this session</Empty>;
  const a = data.analysis;
  const result = data.result;

  const reanalyze = async (body: any, label: string) => {
    const res = await runAnalysis({ file_id: a.file_id, options: body, wait_s: 0 }, { label });
    if (res?.analysis_id) { await refresh(); nav(`/analyze/${res.analysis_id}`); }
  };

  return (
    <div>
      <JobBar />
      {a.status !== "done" && (
        <Alert kind="warn" title={`analysis ${a.status}`}>
          {a.error || a.stage_message || "the analysis has not produced a result yet"}
          {a.status === "running" && <> · progress {num(a.progress, 0)} %</>}
        </Alert>
      )}
      <Card title="Actions">
        <div className="row">
          <button className="tiny" onClick={() => reanalyze({}, "re-analysis")}>re-run analysis</button>
          <button className="tiny" onClick={() => reanalyze({ run_interleaving: false, run_hypotheses: false }, "fast re-analysis")}
            title="skip the two most expensive stages">fast re-run (no interleaver/hypothesis search)</button>
          <button className="tiny" onClick={() => reanalyze({ blind: true }, "blind re-analysis")}>re-run in blind mode</button>
          {a.parent_analysis_id && <span className="pill">derived from {a.parent_analysis_id}</span>}
          <span style={{ flex: 1 }} />
          <Badge kind="info">{data.analysis.kind}</Badge>
        </div>
      </Card>
      {result ? (
        <ResultView
          result={result} analysis={a}
          initialTab={a.kind === "region" ? "spectrum" : "overview"}
          onRegionRequest={(region) => reanalyze({ region }, "region analysis")}
          onSignalRequest={(index) => {
            Api.selectSignal(a.analysis_id, { signal_id: index }).then(async (r) => {
              toast(`analysing emission ${index + 1}: job ${r.job_id}`, "info");
              nav("/analyze");
            }).catch((e) => toast(e.message, "err"));
          }}
        />
      ) : <Empty>no result stored for this analysis</Empty>}
    </div>
  );
}
