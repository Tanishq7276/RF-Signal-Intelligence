import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { useStore } from "../lib/store";
import JobBar from "../components/jobbar";
import RunPanel from "../components/RunPanel";
import { Alert, Card } from "../components/ui";

/** /blind-analysis — no hints at all: the platform must find everything itself. */
export default function Blind() {
  const { runAnalysis, setCurrentId } = useStore();
  const nav = useNavigate();
  const [fileId, setFileId] = useState<string | null>(null);
  const [running, setRunning] = useState(false);

  const start = async () => {
    setRunning(true);
    const res = await runAnalysis({ file_id: fileId, options: { blind: true }, wait_s: 0 }, { blind: true, label: "Blind analysis" });
    setRunning(false);
    if (res?.analysis_id) { setCurrentId(res.analysis_id); nav(`/analyze/${res.analysis_id}`); }
  };

  return (
    <div>
      <JobBar />
      <Card title="Blind analysis" sub="AUTO vs BLIND: in blind mode the platform receives samples only.">
        <Alert kind="info" title="why this matters">
          No modulation, symbol rate, carrier offset, FEC or interleaver information is provided. Every
          one of those is a hypothesis the platform must form and justify — the result page lists the
          ranked candidates with the evidence behind each one.
        </Alert>
        <div className="row" style={{ marginTop: 10 }}>
          <button className="primary" disabled={!fileId || running} onClick={start}>
            {running ? "starting…" : "START BLIND ANALYSIS"}
          </button>
          <span className="small muted">runs the full 16-stage pipeline with all hint fields cleared</span>
        </div>
      </Card>
      <RunPanel blind fileId={fileId} setFileId={setFileId} compact />
    </div>
  );
}
