import { useEffect, useState } from "react";
import { Api, getApiBase, getEnvApiUrl, normalizeApiUrl, setApiBase } from "../lib/api";
import { num, relTime } from "../lib/format";
import { useStore } from "../lib/store";
import { Alert, Badge, Card, Empty, Json, KV, Spinner, Stat } from "../components/ui";

export default function Settings() {
  const { health, refreshHealth, refresh, toast } = useStore();
  const [cfg, setCfg] = useState<any>(null);
  const [models, setModels] = useState<any>(null);
  const [caps, setCaps] = useState<any>(null);
  const [busy, setBusy] = useState(false);
  const [testing, setTesting] = useState(false);
  const [apiUrlInput, setApiUrlInput] = useState(getApiBase());
  const [connectResult, setConnectResult] = useState<{ status: "idle" | "connected" | "disconnected"; message?: string } | null>(null);

  useEffect(() => {
    Api.settings().then(setCfg).catch((e) => toast(e.message, "err"));
    Api.models().then(setModels).catch(() => undefined);
    Api.capabilities().then(setCaps).catch(() => undefined);
  }, [toast]);

  const handleSaveApiUrl = async () => {
    const normalized = normalizeApiUrl(apiUrlInput);
    setApiUrlInput(normalized);
    setTesting(true);
    setConnectResult(null);

    // Perform explicit health check before committing
    const check = await Api.checkHealthAt(normalized);
    setTesting(false);

    if (check.ok) {
      setApiBase(normalized);
      setConnectResult({ status: "connected", message: `Connected to ${normalized || "same-origin / relative"} (version ${check.data?.version || "1.0.0"})` });
      toast("Backend connected successfully", "ok", "API Connection");
      await refreshHealth();
      await refresh();
    } else {
      setConnectResult({ status: "disconnected", message: check.error || "Connection failed" });
      toast(`Connection failed: ${check.error}`, "err", "API Connection");
    }
  };

  const handleResetToDefault = async () => {
    setApiBase(null);
    const envDefault = getEnvApiUrl();
    setApiUrlInput(envDefault);
    setConnectResult(null);
    toast("Reset to environment default API URL", "info", "API Connection");
    await refreshHealth();
    await refresh();
  };

  const activeApiUrl = getApiBase();
  const envDefaultUrl = getEnvApiUrl();
  const isCustom = typeof window !== "undefined" && !!localStorage.getItem("sih26147-api-url");

  return (
    <div>
      <Card
        title="Backend API Server Connection"
        sub="Configure which backend endpoint this web application connects to for DSP analysis and processing."
        right={
          health?.status === "ok" ? (
            <Badge kind="ok">● Connected</Badge>
          ) : (
            <Badge kind="bad">○ Disconnected</Badge>
          )
        }
      >
        <div className="row" style={{ gap: 10, alignItems: "center" }}>
          <input
            type="text"
            placeholder="e.g. https://api.yourdomain.com or https://rf-signal-intelligence-api.onrender.com"
            value={apiUrlInput}
            onChange={(e) => setApiUrlInput(e.target.value)}
            disabled={testing}
            style={{ flex: 1, padding: "7px 12px", background: "#0b1626", border: "1px solid #1e385c", color: "#eef", borderRadius: 4 }}
          />
          <button className="primary btn" disabled={testing} onClick={handleSaveApiUrl}>
            {testing ? "Testing…" : "Save & Connect"}
          </button>
          {isCustom && (
            <button className="tiny ghost" disabled={testing} onClick={handleResetToDefault}>
              Reset to Default
            </button>
          )}
        </div>

        {connectResult && (
          <div style={{ marginTop: 10 }}>
            {connectResult.status === "connected" ? (
              <Alert kind="ok" title="Connection successful">
                {connectResult.message}
              </Alert>
            ) : (
              <Alert kind="bad" title="Connection failed">
                {connectResult.message}
              </Alert>
            )}
          </div>
        )}

        <div className="grid g3" style={{ marginTop: 14 }}>
          <Stat
            k="Active API Base URL"
            v={activeApiUrl || "(same-origin / relative)"}
            n={isCustom ? "custom override stored in browser session" : "configured via environment variable"}
          />
          <Stat
            k="Environment Default"
            v={envDefaultUrl || "(none configured)"}
            n="NEXT_PUBLIC_API_BASE_URL or VITE_API_URL"
          />
          <Stat
            k="Health Status"
            v={health?.status === "ok" ? "Healthy (200 OK)" : (health?.status || "Unreachable")}
            n={health?.lastChecked ? `checked ${relTime(health.lastChecked)}` : "not checked"}
          />
        </div>

        {health?.error && (
          <div style={{ marginTop: 10 }}>
            <Alert kind="bad" title="Connection error">
              {health.error}
            </Alert>
          </div>
        )}

        {!activeApiUrl && (
          <div style={{ marginTop: 10 }}>
            <Alert kind="warn" title="Production backend URL is not configured">
              No backend API URL is configured in the environment (<code>NEXT_PUBLIC_API_BASE_URL</code> or <code>VITE_API_URL</code>) or saved locally. The application will not silently fall back to an obsolete tunnel or localhost. Please configure your permanent backend URL above or in your hosting environment.
            </Alert>
          </div>
        )}

        <div className="small muted" style={{ marginTop: 12 }}>
          💡 <b>Permanent production architecture:</b> In production, the backend runs on a permanent cloud host (e.g. Render or persistent Cloudflare Named Tunnel) at a stable URL. Do not use ephemeral quick tunnels.
        </div>
      </Card>

      <Card title="Environment &amp; Diagnostics" sub="deployment configuration (environment variables) — shown so a run can be reproduced, not silently rewritten by the UI"
        right={<button className="tiny ghost" onClick={refreshHealth}>re-check health</button>}>
        {health ? (
          <div className="grid g4">
            <div className="stat"><div className="k">backend</div><div className="v">{health.status}</div><div className="n">uptime {num(health.uptime_s, 0)} s · v{health.version || "1.0.0"}</div></div>
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
