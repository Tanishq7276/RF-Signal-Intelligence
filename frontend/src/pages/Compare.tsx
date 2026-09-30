import { useEffect, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { Api } from "../lib/api";
import { num, si } from "../lib/format";
import { useStore } from "../lib/store";
import { Alert, Badge, Card, Empty, KV, Spinner } from "../components/ui";

/** /compare — measured similarity between two analyses, with the metric table behind the verdict. */
export default function Compare() {
  const { analyses, toast } = useStore();
  const [params] = useSearchParams();
  const done = analyses.filter((a) => a.status === "done");
  const [a, setA] = useState<string | null>(params.get("a") || done[0]?.analysis_id || null);
  const [b, setB] = useState<string | null>(params.get("b") || done[1]?.analysis_id || null);
  const [out, setOut] = useState<any>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    if (!a && done.length) setA(done[0].analysis_id);
    if (!b && done.length > 1) setB(done[1].analysis_id);
    // eslint-disable-next-line
  }, [analyses]);

  const run = async () => {
    if (!a || !b) return;
    setBusy(true); setOut(null);
    try { setOut(await Api.compare({ analysis_a: a, analysis_b: b })); }
    catch (e: any) { toast(`comparison failed: ${e.message}`, "err"); }
    finally { setBusy(false); }
  };

  return (
    <div>
      <Card title="Signal comparison" sub="compares measured parameters, not filenames; a verdict of 'similar' is never a claim about the source">
        <div className="row" style={{ gap: 12, alignItems: "flex-end" }}>
          <div style={{ flex: 1, minWidth: 240 }}><label className="f"><span>analysis A</span>
            <select value={a ?? ""} onChange={(e) => setA(e.target.value)}>
              <option value="">—</option>{done.map((x) => <option key={x.analysis_id} value={x.analysis_id}>{x.file?.filename} · {x.summary?.modulation || "?"} · {x.analysis_id.slice(0, 8)}</option>)}
            </select></label></div>
          <div style={{ flex: 1, minWidth: 240 }}><label className="f"><span>analysis B</span>
            <select value={b ?? ""} onChange={(e) => setB(e.target.value)}>
              <option value="">—</option>{done.map((x) => <option key={x.analysis_id} value={x.analysis_id}>{x.file?.filename} · {x.summary?.modulation || "?"} · {x.analysis_id.slice(0, 8)}</option>)}
            </select></label></div>
          <button className="primary" disabled={!a || !b || a === b || busy} onClick={run}>{busy ? "comparing…" : "compare"}</button>
        </div>
        {done.length < 2 && <Alert kind="warn">At least two completed analyses are needed. Run another analysis first (Analyze page, or the Generator).</Alert>}
      </Card>

      {busy && <Spinner label="computing similarity" />}
      {out && (
        <>
          <Card title="Verdict" right={<Badge kind={out.verdict === "similar" ? "ok" : out.verdict === "partially similar" ? "warn" : "bad"}>{out.verdict}</Badge>}>
            <div className="grid g4">
              <div className="stat"><div className="k">similarity</div><div className="v">{num(out.similarity, 3)}</div><div className="n">weighted over the metric groups</div></div>
              <div className="stat"><div className="k">analysis A</div><div className="v" style={{ fontSize: 13 }}>{out.a?.file}</div><div className="n">{out.a?.analysis_id?.slice(0, 10)}</div></div>
              <div className="stat"><div className="k">analysis B</div><div className="v" style={{ fontSize: 13 }}>{out.b?.file}</div><div className="n">{out.b?.analysis_id?.slice(0, 10)}</div></div>
              <div className="stat"><div className="k">policy</div><div className="n" style={{ marginTop: 6 }}>{out.policy}</div></div>
            </div>
            {(out.notes || []).map((n: string, i: number) => <Alert key={i} kind="info">{n}</Alert>)}
          </Card>
          <Card title="Measured differences" sub="each row is a real measurement taken from both analyses">
            <div className="tblwrap">
              <table>
                <thead><tr><th>metric</th><th className="num">value</th><th>unit</th><th>detail / basis</th></tr></thead>
                <tbody>
                  {(out.metrics || []).map((m: any, i: number) => (
                    <tr key={i}>
                      <td>{m.metric}</td>
                      <td className="num">{typeof m.value === "number" ? num(m.value, 4) : String(m.value ?? "—")}</td>
                      <td className="small muted">{m.unit || "—"}</td>
                      <td className="small muted">{m.detail || "—"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div style={{ marginTop: 10 }}>
              <KV rows={[
                ["compared", `${out.a?.file} vs ${out.b?.file}`],
                ["analyses", `${out.a?.analysis_id?.slice(0, 10)} vs ${out.b?.analysis_id?.slice(0, 10)}`],
                ["comparison id", out.comparison_id?.slice(0, 10) || "—"],
                ["policy", out.policy],
              ]} />
            </div>
          </Card>
        </>
      )}
      {!out && !busy && <Empty>Choose two analyses and press compare.</Empty>}
    </div>
  );
}
