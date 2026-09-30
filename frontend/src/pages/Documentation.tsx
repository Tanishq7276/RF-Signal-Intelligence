import { useEffect, useState } from "react";
import { Api, apiUrl } from "../lib/api";
import { Markdown, fetchDocs } from "../lib/md";
import { Alert, Card, Empty, KV, Spinner } from "../components/ui";

/** /docs — renders the markdown documents that ship in the repository. */
export default function Documentation() {
  const [docs, setDocs] = useState<any>(null);
  const [md, setMd] = useState<Record<string, string>>({});
  const [current, setCurrent] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [online, setOnline] = useState<any>(null);

  useEffect(() => {
    fetchDocs().then(({ index, markdown }) => {
      setDocs(index); setMd(markdown);
      setCurrent(index?.documents?.[0]?.name ?? null);
      setLoading(false);
    });
    Api.openapi().then(setOnline).catch(() => undefined);
  }, []);

  const list: any[] = docs?.documents || [];

  return (
    <div>
      <Card title="Documentation" sub="the same markdown files that live in the repository (docs/), served by the API so the in-app text and the repository text can never drift apart">
        <div className="row" style={{ gap: 8 }}>
          <a className="btn" href={apiUrl("/docs")} target="_blank" rel="noreferrer">Swagger UI</a>
          <a className="btn" href={apiUrl("/redoc")} target="_blank" rel="noreferrer">ReDoc</a>
          <a className="btn" href={apiUrl("/openapi.json")} target="_blank" rel="noreferrer">OpenAPI JSON</a>
          <span className="pill">{online?.paths ? `${Object.keys(online.paths).length} documented paths` : "API schema not reachable"}</span>
          <span style={{ flex: 1 }} />
          <button className="tiny ghost" onClick={() => Api.stats().then(setOnline).catch(() => undefined)}>refresh</button>
        </div>
      </Card>

      {loading && <Spinner label="loading documents" />}
      {!loading && !list.length && (
        <Alert kind="warn" title="no documents served">
          The backend is not serving <code>/documentation/index.json</code>. In development run the API (uvicorn)
          or the docker-compose stack; the markdown files are in the repository's <code>docs/</code> folder.
        </Alert>
      )}

      {!!list.length && (
        <div className="grid" style={{ gridTemplateColumns: "260px 1fr", gap: 14 }}>
          <Card title="Documents">
            <div className="col" style={{ gap: 4 }}>
              {list.map((d: any) => (
                <button key={d.name} className={`btn tiny ${current === d.name ? "" : "ghost"}`} style={{ textAlign: "left", justifyContent: "flex-start" }}
                  onClick={() => setCurrent(d.name)}>{d.title || d.name}</button>
              ))}
            </div>
            <div className="hr" />
            <KV rows={[["documents", String(list.length)], ["source", "docs/*.md"], ["generated", docs.generated || "—"]]} />
          </Card>
          <Card title={list.find((d: any) => d.name === current)?.title || "Document"}
            sub={list.find((d: any) => d.name === current)?.summary}>
            {current && md[current] ? <Markdown text={md[current]} /> : <Empty>select a document</Empty>}
          </Card>
        </div>
      )}

      <Card title="Keyboard-free quick answers" sub="the questions judges ask most">
        <ul className="small" style={{ marginLeft: 16, color: "#c8d6ee" }}>
          <li><b>Where is the AI?</b> Not in one model. Classification is a hybrid vote of DSP decision rules, symbol/spectral statistics and an optional scikit-learn model; every automatic result carries a confidence, its evidence and its limitations. See Moderation → the classification card, and Analysis → Evidence.</li>
          <li><b>What happens with an unknown sample rate?</b> Nothing is invented. If the file has no rate metadata the pipeline works in normalised frequency, reports <code>Unknown / requires estimation</code> for absolute values and marks every frequency-derived parameter accordingly.</li>
          <li><b>How do you avoid fake numbers?</b> Every value is produced by an estimator in <code>dsp/</code> from the uploaded samples; when an estimator cannot support a claim the field reads <i>Unable to estimate reliably</i> with the reason.</li>
          <li><b>How is FEC detection not a guess?</b> Each code hypothesis must beat a random-data null; only hypotheses above the 0.20 confidence threshold are called the result, and LDPC is explicitly reported as unsupported.</li>
          <li><b>Can it decode an unknown protocol perfectly?</b> No — and the platform says so. It recovers bits and hypotheses, exposes uncertainty, and never claims a perfect decode of an arbitrary protocol.</li>
        </ul>
      </Card>
    </div>
  );
}
