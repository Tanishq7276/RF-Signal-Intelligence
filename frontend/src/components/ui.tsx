import React from "react";
import { num, pct, statusClass } from "../lib/format";

export const Card = ({ title, sub, right, children, className }: any) => (
  <div className={`card ${className || ""}`}>
    {(title || right) && <h3>{title}{right && <span style={{ marginLeft: "auto" }}>{right}</span>}</h3>}
    {sub && <div className="sub">{sub}</div>}
    {children}
  </div>
);

export const Stat = ({ k, v, n, tone }: { k: string; v: React.ReactNode; n?: React.ReactNode; tone?: string }) => (
  <div className="stat" style={tone ? { borderColor: tone } : undefined}>
    <div className="k">{k}</div>
    <div className="v">{v}</div>
    {n !== undefined && <div className="n">{n}</div>}
  </div>
);

export const Badge = ({ children, kind }: { children: React.ReactNode; kind?: string }) => (
  <span className={`badge ${kind || statusClass(String(children))}`}>{children}</span>
);

export const Bar = ({ value, tone }: { value: number | null | undefined; tone?: "ok" | "warn" | "bad" }) => {
  const v = value === null || value === undefined || !isFinite(Number(value)) ? 0 : Math.max(0, Math.min(1, Number(value)));
  const cls = tone || (v >= 0.7 ? "ok" : v >= 0.35 ? "" : "bad");
  return <div className={`bar ${cls}`} title={value === null || value === undefined ? "not available" : pct(value, 1)}><i style={{ width: `${v * 100}%` }} /></div>;
};

export const KV = ({ rows }: { rows: [React.ReactNode, React.ReactNode][] }) => (
  <div className="kv">{rows.map(([k, v], i) => (<React.Fragment key={i}><div className="k">{k}</div><div className="v">{v}</div></React.Fragment>))}</div>
);

export const Empty = ({ children }: any) => <div className="empty">{children}</div>;

export const Alert = ({ kind = "info", children, title }: any) => (
  <div className={`alert ${kind}`}>{title && <b>{title}: </b>}{children}</div>
);

export const Tabs = ({ tabs, value, onChange }: { tabs: { id: string; label: string }[]; value: string; onChange: (id: string) => void }) => (
  <div className="tabs">{tabs.map((t) => (
    <button key={t.id} className={t.id === value ? "active" : ""} onClick={() => onChange(t.id)}>{t.label}</button>
  ))}</div>
);

export const Spinner = ({ label }: { label?: string }) => (
  <span className="row" style={{ gap: 8 }}><span className="spin" />{label && <span className="muted small">{label}</span>}</span>
);

export const ConfidenceCell = ({ v }: { v: any }) => (
  <div className="row" style={{ gap: 6, minWidth: 90 }}>
    <span className="mono small">{v === null || v === undefined ? "—" : v.toFixed ? v.toFixed(2) : num(v, 2)}</span>
    <Bar value={v} />
  </div>
);

export const Json = ({ value, label }: { value: any; label?: string }) => {
  const [open, setOpen] = React.useState(false);
  return (
    <div>
      <button className="tiny ghost" onClick={() => setOpen((o) => !o)}>{open ? "hide" : "show"} {label || "raw JSON"}</button>
      {open && <pre>{JSON.stringify(value, null, 1)}</pre>}
    </div>
  );
};

export const Select = ({ label, value, onChange, options, hint }: {
  label: string; value: any; onChange: (v: any) => void;
  options: { value: any; label: string }[]; hint?: string;
}) => (
  <label className="f">
    <span>{label}</span>
    <select value={value ?? ""} onChange={(e) => onChange(e.target.value)}>
      {options.map((o) => <option key={String(o.value)} value={o.value}>{o.label}</option>)}
    </select>
    {hint && <span className="small muted">{hint}</span>}
  </label>
);

export const NumField = ({ label, value, onChange, step = 1, min, max, hint, placeholder }: any) => (
  <label className="f">
    <span>{label}</span>
    <input type="number" value={value ?? ""} step={step} min={min} max={max} placeholder={placeholder}
      onChange={(e) => onChange(e.target.value === "" ? null : Number(e.target.value))} />
    {hint && <span className="small muted">{hint}</span>}
  </label>
);
