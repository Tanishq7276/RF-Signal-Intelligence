import { useEffect, useState } from "react";
import { Api, getApiBase, setApiBase } from "../lib/api";
import { num } from "../lib/format";
import { useStore } from "../lib/store";
import { Alert, Card, Empty, Json, KV, Spinner } from "../components/ui";

export default function Settings() {
  const { health, refreshHealth, refresh, toast } = useStore();
  const [cfg, setCfg] = useState<any>(null);
  const [models, setModels] = useState<any>(null);
  const [caps, setCaps] = useState<any>(null);
  const [busy, setBusy] = useState(false);
  const [apiUrlInput, setApiUrlInput] = useState(getApiBase());

  useEffect(() => {
    Api.settings().then(setCfg).catch((e) => toast(e.message, "err"));
    Api.models().then(setModels).catch(() => undefined);
    Api.capabilities().then(setCaps).catch(() => undefined);
  }, [toast]);

  const handleSaveApiUrl = async () => {
    setApiBase(apiUrlInput);
    toast("Backend API URL saved. Re-connecting...", "info");
    await refreshHealth();
    await refresh();
  };

  return (
    <div>
      <Card title="Backend API Server Connection" sub="Configure which backend endpoint this Vercel web application connects to for DSP analysis and processing.">
        <div className="row" style={{ gap: 10, alignItems: "center" }}>
          <input
            type="text"
            placeholder="e.g. https://rf-signal-intelligence-api.onrender.com or http://localhost:8000"
            value={apiUrlInput}
            onChange={(e) => setApiUrlInput(e.target.value)}
            style={{ flex: 1, padding: "7px 12px", background: "#0b1626", border: "1px solid #1e385c", color: "#eef", borderRadius: 4 }}
          />
          <button className="btn" onClick={handleSaveApiUrl}>Save &amp; Connect</button>
          {apiUrlInput && (
            <button className="tiny ghost" onClick={() => { setApiUrlInput(""); setApiBase(null); refreshHealth(); refresh(); }}>Reset to Default</button>
          )}
        </div>
        <div className="small muted" style={{ marginTop: 8 }}>
          Current API Base: <code>{getApiBase() || "(same-origin / relative)"}</code>
        </div>
      </Card>

      <Card title="Environment" sub="deployment configuration (environment variables) — shown so a run can be reproduced, not silently rewritten by the UI"
        right={<button className="tiny ghost" onClick={refreshHealth}>re-check health</button>}>
        {health ? (
          <div className="grid g4">
            <div className="stat"><div className="k">backend</div><div className="v">{health.status}</div><div className="n">uptime {num(health.uptime_s, 0)} s · v{health.version}</div></div>
            {Object.entries(health.checks || {}).map(([k, v]: any) => (
              <div className="stat" key={k}><div className="k">{k}</div><div className="v">{v.ok ? "ok" : "failed"}</div>
                <div className="n">{Object.entries(v).filter(([kk]) => kk !== "ok").map(([kk, vv]) => `${kk}: ${String(vv)}`).join(" · ")}</div></div>
            ))}
          </div>
        ) : <Spinner label="checking" />}
      </Card>

      {cfg && (
        <div className="grid g2">
          <Card title="Limits &amp; paths">
            <KV rows={[
              ...Object.entries(cfg.limits || {}).map(([k, v]) => [k, typeof v === "number" && k.includes("bytes") ? `${(v as number) / 1e6} MB` : String(v)] as [string, string]),
              ...Object.entries(cfg.paths || {}).map(([k, v]) => [`path: ${k}`, String(v)] as [string, string]),
            ]} />
          </Card>
          <Card title="Security">
            <KV rows={[
              ["session header", cfg.security?.session_header],
              ["uploads executed", String(cfg.security?.uploads_are_never_executed)],
              ["allowed extensions", (cfg.security?.allowed_extensions || []).join(" ")],
              ["refused extensions", (cfg.security?.refused_extensions || []).slice(0, 12).join(" ") + " …"],
              ["password hashing", "scrypt (n=2^14, r=8, p=1) with per-user salt"],
              ["tokens", "HS256 JWT, 24 h — rotate SIH_JWT_SECRET to invalidate"],
              ["rate limits", "auth 10/min · upload 60/min · analyze 30/min per client"],
            ]} />
          </Card>
          <Card title="Machine learning models">
            {models ? (
              <>
                <KV rows={[
                  ["classifier available", String(models.classifier?.available)],
                  ["kind", models.classifier?.kind || "—"],
                  ["measured accuracy", num(models.classifier?.accuracy, 3)],
                  ["notes", String(models.classifier?.notes || "—")],
                ]} />
                <Alert kind="info">{models.usage}</Alert>
              </>
            ) : <Empty>no model information</Empty>}
          </Card>
          <Card title="Capabilities">
            {caps ? <Json value={caps} label="capabilities JSON" /> : <Empty>unavailable</Empty>}
          </Card>
        </div>
      )}

      <Card title="Data handling" sub="what this platform stores and where">
        <ul className="small" style={{ marginLeft: 16, color: "#c8d6ee" }}>
          <li>Uploads are written to the server's data directory with a sanitised name and a UUID prefix; nothing is ever executed from an upload (extensions are allow-listed, scripts/binaries refused).</li>
          <li>Analyses are stored twice: queryable rows (segments, parameters, hypotheses, bitstream summary) in the relational schema, and the full result — including the downsampled chart arrays — as a gzipped cache file. Raw IQ arrays are never stored in the relational tables.</li>
          <li>Every request carries an identity: a bearer token for signed-in accounts, otherwise an isolated <code>x-session-id</code>. Files, analyses, reports and jobs are scoped to it.</li>
          <li>Temporary upload spools are removed after the file is stored; deleting a file removes its cache directory and the analyses derived from it.</li>
        </ul>
        <div className="row" style={{ marginTop: 10 }}>
          <button className="danger" disabled={busy} onClick={async () => {
            if (!confirm("Delete every file, analysis and report of this workspace?")) return;
            setBusy(true);
            try { const r = await Api.clearSession(); toast(`${r.removed_files} file(s) removed`, "ok"); }
            catch (e: any) { toast(e.message, "err"); }
            finally { setBusy(false); }
          }}>wipe this workspace</button>
        </div>
      </Card>

      <Card title="Authorised use" sub="intended scope">
        <Alert kind="warn">
          This platform is built for authorised recordings, synthetic signals, laboratory testing and
          research datasets. It performs signal analysis only: it does not intercept traffic, extract
          credentials, break encryption or recover keys. If a payload is encrypted, the platform reports
          that the recovered data remains encrypted.
        </Alert>
      </Card>
    </div>
  );
}
