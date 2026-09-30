import { useEffect, useState } from "react";
import { Api } from "../lib/api";
import { bytes, relTime } from "../lib/format";
import { useStore } from "../lib/store";
import { Alert, Badge, Card, Empty, Select, Stat } from "../components/ui";

/** /reports — every export of this workspace, plus on-the-fly exports. */
export default function Reports() {
  const { analyses, toast } = useStore();
  const [reports, setReports] = useState<any[]>([]);
  const [analysisId, setAnalysisId] = useState<string>("");
  const [busy, setBusy] = useState(false);

  const load = () => Api.reports().then((r) => setReports(r.reports || [])).catch(() => undefined);
  useEffect(() => { load(); }, []);
  useEffect(() => { if (!analysisId && analyses.length) setAnalysisId(analyses.find((a) => a.status === "done")?.analysis_id || ""); }, [analyses, analysisId]);

  const build = async (formats: string[]) => {
    if (!analysisId) return;
    setBusy(true);
    try {
      const r = await Api.makeReport(analysisId, formats);
      if (r.errors?.length) toast(r.errors.join("; "), "err");
      if (r.reports?.length) toast(`${r.reports.length} file(s) generated`, "ok");
      load();
    } catch (e: any) { toast(`report failed: ${e.message}`, "err"); }
    finally { setBusy(false); }
  };

  const totalBytes = reports.reduce((s, r) => s + (r.size_bytes || 0), 0);

  return (
    <div>
      <Card title="Report export" sub="PDF embeds the real charts of that analysis (spectrum, waterfall, constellation) and the full parameter table with evidence and limitations">
        <div className="grid g4">
          <Stat k="Stored reports" v={reports.length} n={`${bytes(totalBytes)} on disk`} />
          <Stat k="Formats" v="PDF · JSON · CSV · TXT" n="all generated from the cached analysis result" />
          <Stat k="Missing files" v={reports.filter((r) => !r.exists).length} n="rows whose file is no longer on disk" />
        </div>
        <div className="row" style={{ gap: 10, alignItems: "flex-end", marginTop: 12 }}>
          <div style={{ minWidth: 320, flex: 1 }}>
            <Select label="analysis" value={analysisId} onChange={setAnalysisId}
              options={analyses.filter((a) => a.status === "done").map((a) => ({ value: a.analysis_id, label: `${a.file?.filename} · ${a.summary?.modulation || "?"} · ${a.analysis_id.slice(0, 8)}` }))} />
          </div>
          <button className="primary" disabled={!analysisId || busy} onClick={() => build(["pdf", "json", "csv", "txt"])}>
            {busy ? "building…" : "build all four formats"}
          </button>
          {(["pdf", "json", "csv", "txt"] as const).map((f) => (
            <a key={f} className="btn" href={analysisId ? Api.reportInlineUrl(analysisId, f) : "#"} target="_blank" rel="noreferrer"
              onClick={(e) => { if (!analysisId) e.preventDefault(); }}>on-the-fly {f.toUpperCase()}</a>
          ))}
        </div>
        {!analyses.some((a) => a.status === "done") && <Alert kind="warn">No completed analysis yet — run one first.</Alert>}
      </Card>

      <Card title="Stored reports" sub="each row is a real file on disk that can be downloaded again">
        {reports.length ? (
          <div className="tblwrap" style={{ maxHeight: 460 }}>
            <table><thead><tr><th>format</th><th>file</th><th>source signal</th><th className="num">size</th><th>created</th><th></th></tr></thead>
              <tbody>{reports.map((r) => (
                <tr key={r.report_id}>
                  <td><Badge kind="info">{r.format}</Badge></td>
                  <td className="mono small">{r.title}</td>
                  <td className="small">{r.filename || "—"}</td>
                  <td className="num">{bytes(r.size_bytes)}</td>
                  <td className="small muted">{relTime(r.created_at)}</td>
                  <td className="row" style={{ gap: 4 }}>
                    {r.exists
                      ? <a className="btn tiny" href={Api.reportDownloadUrl(r.report_id)}>download</a>
                      : <span className="badge bad">file missing</span>}
                    <a className="btn tiny ghost" href={Api.reportInlineUrl(r.analysis_id, r.format)} target="_blank" rel="noreferrer">rebuild</a>
                  </td>
                </tr>))}
              </tbody></table>
          </div>
        ) : <Empty>No report has been generated in this workspace yet.</Empty>}
      </Card>
      <Alert kind="info" title="what the PDF contains">
        analysis id, input file and its detected format, signal information, spectrum, waterfall, the full
        parameter table (value / confidence / method), modulation candidates with evidence, symbol rate,
        constellation, demodulation stages, FEC and interleaving hypotheses, bitstream statistics,
        confidence summary, the evidence graph and an explicit limitations section.
      </Alert>
    </div>
  );
}
