import { useEffect, useState } from "react";
import { Link, NavLink, Route, Routes, useLocation, useNavigate } from "react-router-dom";
import { Api } from "./lib/api";
import { logout, whoami } from "./lib/auth";
import { num } from "./lib/format";
import { useStore } from "./lib/store";
import Analyze from "./pages/Analyze";
import AnalyzeDetail from "./pages/AnalyzeDetail";
import Benchmark from "./pages/Benchmark";
import Bitstream from "./pages/Bitstream";
import Blind from "./pages/Blind";
import Compare from "./pages/Compare";
import Dashboard from "./pages/Dashboard";
import Demo from "./pages/Demo";
import Demodulation from "./pages/Demodulation";
import Documentation from "./pages/Documentation";
import Explorer from "./pages/Explorer";
import FecInterleave from "./pages/FecInterleave";
import Generator from "./pages/Generator";
import History from "./pages/History";
import Landing from "./pages/Landing";
import Live from "./pages/Live";
import Login from "./pages/Login";
import Modulation from "./pages/Modulation";
import Reports from "./pages/Reports";
import Settings from "./pages/Settings";

const NAV: { group: string; items: [string, string, string][] }[] = [
  { group: "overview", items: [["/dashboard", "Dashboard", "▤"], ["/demo", "SIH demo mode", "★"], ["/", "About", "◈"]] },
  { group: "capture & analyse", items: [["/analyze", "Analyze signal", "◎"], ["/blind-analysis", "Blind analysis", "◐"], ["/live", "Live analysis", "◉"], ["/explorer", "Signal explorer", "⌗"]] },
  { group: "analysis modules", items: [["/modulation", "Modulation", "∿"], ["/demodulation", "Demodulation", "⊞"], ["/fec", "FEC & interleaving", "⛨"], ["/bitstream", "Bitstream", "⏻"]] },
  { group: "compare & evaluate", items: [["/compare", "Compare signals", "⇄"], ["/generator", "Synthetic generator", "⚗"], ["/benchmark", "Benchmark", "▣"]] },
  { group: "manage", items: [["/history", "History & sessions", "⧗"], ["/reports", "Reports", "⎙"], ["/settings", "Settings", "⚙"], ["/documentation", "Documentation", "✎"]] },
];

const TITLES: Record<string, string> = {
  "/": "about this platform",
  "/dashboard": "dashboard",
  "/demo": "SIH demo mode",
  "/analyze": "analyze signal",
  "/live": "live analysis",
  "/explorer": "signal explorer",
  "/blind-analysis": "blind analysis",
  "/modulation": "modulation classification",
  "/demodulation": "demodulation",
  "/fec": "FEC & interleaving hypotheses",
  "/bitstream": "bitstream analysis",
  "/compare": "signal comparison",
  "/generator": "synthetic signal generator",
  "/benchmark": "benchmark mode",
  "/history": "history & sessions",
  "/reports": "reports",
  "/settings": "settings",
  "/documentation": "documentation",
  "/login": "account",
};

export default function App() {
  const { health, job, currentId, analyses, files } = useStore();
  const loc = useLocation();
  const nav = useNavigate();
  const [user, setUser] = useState<any>(null);
  const [menu, setMenu] = useState(false);

  useEffect(() => {
    whoami().then((s) => setUser(s?.authenticated ? s.user : null)).catch(() => setUser(null));
  }, [loc.pathname]);

  const title = TITLES[loc.pathname] || (loc.pathname.startsWith("/analyze/") ? `analysis ${loc.pathname.split("/")[2]?.slice(0, 8)}` : "workspace");
  const running = job && (job.status === "running" || job.status === "queued");
  const done = analyses.filter((a: any) => a.status === "done").length;

  return (
    <div className="app">
      <aside className="side">
        <div className="brand">
          <span className="dot" />
          <div>
            <b>RF Signal Intelligence</b>
            <span>SIH26147 · IQ / WAV analysis</span>
          </div>
        </div>
        <nav className="nav">
          {NAV.map((g) => (
            <div key={g.group}>
              <div className="navgroup">{g.group}</div>
              {g.items.map(([to, label, icon]) => (
                <NavLink key={to} to={to} end={to === "/"} onClick={() => setMenu(false)}
                  className={({ isActive }) => (isActive ? "active" : "")}>
                  <span className="ic">{icon}</span>{label}
                </NavLink>
              ))}
            </div>
          ))}
        </nav>
      </aside>

      <main className="main">
        <header className="top">
          <button className="tiny ghost" style={{ display: "none" }} />
          <h1>{title}</h1>
          <span className="spacer" />
          <Link to="/settings" style={{ textDecoration: "none" }}>
            <span className={`badge ${health?.status === "ok" ? "ok" : "bad"}`}
              title={health?.status === "ok" ? "Backend connected (click to view settings)" : `Backend unreachable (${health?.error || "check backend status"}) - click to configure`}>
              {health?.status === "ok" ? "api connected" : "api unreachable"}
            </span>
          </Link>
          <span className="pill" title="files in this workspace">{files.length} file{files.length === 1 ? "" : "s"} · {done} analysis</span>
          {running && <span className="pill" title={job.message}>{num((job.progress || 0) * 100, 0)} % · {job.kind || "job"}</span>}
          {currentId && <Link className="btn tiny" to={`/analyze/${currentId}`}>current analysis</Link>}
          {user
            ? <button className="tiny ghost" title={user.email} onClick={async () => { await logout(); setUser(null); nav("/"); }}>sign out {user.email?.split("@")[0]}</button>
            : <Link className="btn tiny" to="/login">sign in</Link>}
          <button className="tiny ghost" onClick={() => setMenu((m) => !m)} title="workspace menu">⋯</button>
        </header>

        {menu && (
          <div className="card" style={{ margin: "10px 18px 0" }}>
            <div className="row" style={{ gap: 10 }}>
              <span className="small muted">workspace</span>
              <span className="pill">{Api.sessionKey}</span>
              <span className="small muted">{files.length} files · {analyses.length} analyses · {done} completed</span>
              <span style={{ flex: 1 }} />
              <Link className="btn tiny" to="/history">history</Link>
              <Link className="btn tiny" to="/reports">reports</Link>
              <Link className="btn tiny" to="/settings">wipe workspace</Link>
            </div>
          </div>
        )}

        <div className="content">
          <Routes>
            <Route path="/" element={<Landing />} />
            <Route path="/login" element={<Login />} />
            <Route path="/dashboard" element={<Dashboard />} />
            <Route path="/demo" element={<Demo />} />
            <Route path="/analyze" element={<Analyze />} />
            <Route path="/analyze/:id" element={<AnalyzeDetail />} />
            <Route path="/live" element={<Live />} />
            <Route path="/explorer" element={<Explorer />} />
            <Route path="/blind-analysis" element={<Blind />} />
            <Route path="/modulation" element={<Modulation />} />
            <Route path="/demodulation" element={<Demodulation />} />
            <Route path="/fec" element={<FecInterleave />} />
            <Route path="/bitstream" element={<Bitstream />} />
            <Route path="/compare" element={<Compare />} />
            <Route path="/generator" element={<Generator />} />
            <Route path="/benchmark" element={<Benchmark />} />
            <Route path="/history" element={<History />} />
            <Route path="/reports" element={<Reports />} />
            <Route path="/settings" element={<Settings />} />
            <Route path="/documentation" element={<Documentation />} />
            <Route path="*" element={<div className="card"><b>page not found</b><div className="small muted" style={{ marginTop: 6 }}>this route does not exist in the platform — use the navigation on the left.</div></div>} />
          </Routes>
        </div>

        <footer className="foot">
          SIH26147 · Automated model for analysis of .IQ and .WAV files along with signal parameter
          extraction · every figure in this interface is computed from the signal you loaded ·
          unsupported values are reported as “Unknown / requires estimation” or “Unable to estimate
          reliably” rather than invented · authorised analysis of your own or synthetic recordings only.
        </footer>
      </main>
    </div>
  );
}
