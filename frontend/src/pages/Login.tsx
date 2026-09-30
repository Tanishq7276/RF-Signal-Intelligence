import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { login } from "../lib/auth";
import { useStore } from "../lib/store";
import { Alert, Card } from "../components/ui";

export default function Login({ registerMode = false }: { registerMode?: boolean }) {
  const nav = useNavigate();
  const { toast, refresh } = useStore();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [fullName, setFullName] = useState("");
  const [organisation, setOrganisation] = useState("");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setBusy(true); setErr(null);
    try {
      const { register } = await import("../lib/auth");
      const res = registerMode
        ? await register({ email, password, full_name: fullName || undefined, organisation: organisation || undefined })
        : await login(email, password);
      toast(`${registerMode ? "account created" : "signed in"} as ${res.user.email}`, "ok", "Authentication");
      await refresh();
      nav("/dashboard");
    } catch (e: any) {
      setErr(e.message || "authentication failed");
    } finally { setBusy(false); }
  };

  return (
    <div style={{ maxWidth: 520, margin: "20px auto" }}>
      <Card title={registerMode ? "Create an account" : "Sign in"} sub={
        registerMode
          ? "Accounts own an isolated workspace: your files, analyses and reports are never mixed with another user's."
          : "Sign in to keep your own workspace, or use the platform anonymously (your session is then isolated by a browser key)."}>
        <form className="col" style={{ gap: 12 }} onSubmit={submit}>
          {registerMode && (
            <>
              <label className="f"><span>full name</span><input value={fullName} onChange={(e) => setFullName(e.target.value)} placeholder="optional" /></label>
              <label className="f"><span>organisation</span><input value={organisation} onChange={(e) => setOrganisation(e.target.value)} placeholder="optional" /></label>
            </>
          )}
          <label className="f"><span>email</span><input type="email" required value={email} onChange={(e) => setEmail(e.target.value)} placeholder="engineer@example.org" /></label>
          <label className="f"><span>password {registerMode && <span className="muted">(min 8 characters)</span>}</span>
            <input type="password" required minLength={registerMode ? 8 : 1} value={password} onChange={(e) => setPassword(e.target.value)} /></label>
          {err && <Alert kind="bad">{err}</Alert>}
          <div className="row">
            <button className="primary" disabled={busy} type="submit">{busy ? "working…" : registerMode ? "register" : "sign in"}</button>
            <Link className="btn" to={registerMode ? "/login" : "/register"}>{registerMode ? "I already have an account" : "create an account"}</Link>
            <a className="btn ghost" href="/analyze">continue anonymously</a>
          </div>
        </form>
        <div className="hr" />
        <div className="small muted">
          Passwords are stored as scrypt digests; sessions use signed HS256 JWTs (24 h). Anonymous use
          is fully functional for a demo — signing in only changes which workspace your data belongs to.
        </div>
      </Card>
    </div>
  );
}
