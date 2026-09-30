import { useStore } from "../lib/store";
import { num } from "../lib/format";

/** Live job progress (real progress from the backend worker) + cancellation. */
export default function JobBar() {
  const { job, cancelJob } = useStore();
  if (!job || (job.status === "done" && job.progress >= 1)) return null;
  const running = job.status === "running" || job.status === "queued" || job.status === "starting";
  return (
    <div className="card" style={{ marginBottom: 12 }}>
      <div className="row" style={{ gap: 10 }}>
        <b style={{ fontSize: 12.5 }}>{job.kind || "analysis"}</b>
        <span className={`badge ${job.status === "failed" ? "bad" : job.status === "done" ? "ok" : "info"}`}>{job.status}</span>
        <span className="mono small">{num((job.progress ?? 0) * 100, 0)} %</span>
        <span style={{ flex: 1 }} />
        {running && <button className="tiny danger" onClick={cancelJob}>cancel</button>}
      </div>
      <div className="prog" style={{ marginTop: 8 }}><i style={{ width: `${(job.progress ?? 0) * 100}%` }} /></div>
      <div className="small muted" style={{ marginTop: 5 }}>{job.message}{job.duration_s ? ` · ${num(job.duration_s, 1)} s` : ""}</div>
      {job.error && <div className="alert bad" style={{ marginTop: 8 }}>{job.error}</div>}
    </div>
  );
}
