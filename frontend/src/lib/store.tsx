import React, { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from "react";
import { Api, ApiError, pollJob } from "./api";

type Toast = { id: number; msg: string; kind: "ok" | "err" | "info"; title?: string };
type Ctx = {
  health: any; files: any[]; analyses: any[]; demos: any[];
  currentId: string | null; current: any; currentLoading: boolean;
  job: any; setCurrentId: (id: string | null) => void;
  refresh: () => Promise<void>; refreshHealth: () => Promise<void>; refreshCurrent: () => Promise<void>;
  toast: (msg: string, kind?: Toast["kind"], title?: string) => void;
  runAnalysis: (body: any, opts?: { blind?: boolean; label?: string }) => Promise<any>;
  cancelJob: () => Promise<void>;
  demoLoad: (name: string) => Promise<any>;
};
const Store = createContext<Ctx>(null as any);
export const useStore = () => useContext(Store);

export function StoreProvider({ children }: { children: React.ReactNode }) {
  const [health, setHealth] = useState<any>(null);
  const [files, setFiles] = useState<any[]>([]);
  const [analyses, setAnalyses] = useState<any[]>([]);
  const [demos, setDemos] = useState<any[]>([]);
  const [currentId, setCurrentId] = useState<string | null>(null);
  const [current, setCurrent] = useState<any>(null);
  const [currentLoading, setCurrentLoading] = useState(false);
  const [job, setJob] = useState<any>(null);
  const [toasts, setToasts] = useState<Toast[]>([]);
  const abort = useRef<AbortController | null>(null);

  const toast = useCallback((msg: string, kind: Toast["kind"] = "info", title?: string) => {
    const id = Date.now() + Math.random();
    setToasts((t) => [...t.slice(-4), { id, msg, kind, title }]);
    setTimeout(() => setToasts((t) => t.filter((x) => x.id !== id)), kind === "err" ? 12000 : 7000);
  }, []);

  const refreshHealth = useCallback(async () => {
    try {
      const h = await Api.health(8000);
      setHealth({
        ...h,
        lastChecked: new Date().toISOString(),
        error: null,
      });
    } catch (e: any) {
      setHealth({
        status: "unreachable",
        lastChecked: new Date().toISOString(),
        error: String(e?.message || e),
      });
    }
  }, []);

  const refresh = useCallback(async () => {
    try {
      const [f, a, d] = await Promise.all([Api.listFiles(), Api.analyses({ limit: 200 }), Api.demos()]);
      setFiles(f.files || []); setAnalyses(a.analyses || []); setDemos(d.demo_signals || []);
    } catch (e: any) {
      // Avoid spamming error toast if already logged or backend offline
    }
  }, []);

  useEffect(() => { refreshHealth(); refresh(); }, [refresh, refreshHealth]);

  const refreshCurrent = useCallback(async () => {
    if (!currentId) { setCurrent(null); return; }
    setCurrentLoading(true);
    try { setCurrent(await Api.analysis(currentId)); }
    catch (e: any) { setCurrent(null); toast(`analysis ${currentId} could not be loaded: ${e.message}`, "err"); }
    finally { setCurrentLoading(false); }
  }, [currentId, toast]);

  useEffect(() => { refreshCurrent(); }, [refreshCurrent]);

  const demoLoad = useCallback(async (name: string) => {
    try {
      const res = await Api.loadDemo(name);
      await refresh();
      toast(`demo signal "${name}" is ready (${res.info?.source || "sample_data"})`, "ok", "Demo signal");
      return res.file;
    } catch (e: any) { toast(`demo signal failed: ${e.message}`, "err"); return null; }
  }, [refresh, toast]);

  const runAnalysis = useCallback(async (body: any, opts: { blind?: boolean; label?: string } = {}) => {
    abort.current = new AbortController();
    setJob({ status: "starting", progress: 0, message: "submitting" });
    try {
      const started = opts.blind ? await Api.blind(body) : await Api.analyze(body);
      setJob(started.job || started);
      const finished = await pollJob(started.job_id, (j) => setJob(j), { signal: abort.current.signal });
      setJob(finished);
      if (finished.status === "done") {
        const aid = finished.result?.analysis_id;
        await refresh();
        if (aid) { setCurrentId(aid); toast(`analysis finished in ${(finished.duration_s || 0).toFixed(1)} s`, "ok", opts.label || "Analysis"); }
        return finished.result;
      }
      if (finished.status === "cancelled") toast("analysis cancelled", "info");
      else toast(`analysis failed: ${finished.error || "unknown error"}`, "err");
      return null;
    } catch (e: any) {
      setJob(null);
      if (e instanceof ApiError && e.message === "cancelled by the client") return null;
      toast(`could not start the analysis: ${e.message}`, "err");
      return null;
    }
  }, [refresh, toast]);

  const cancelJob = useCallback(async () => {
    const jid = job?.job_id;
    abort.current?.abort();
    if (jid) { try { const r = await Api.cancelJob(jid); setJob(r.job); toast("cancellation requested - the pipeline stops at the next stage boundary", "info"); } catch (e: any) { toast(`cancel failed: ${e.message}`, "err"); } }
  }, [job, toast]);

  const value = useMemo<Ctx>(() => ({
    health, files, analyses, demos, currentId, current, currentLoading, job,
    setCurrentId, refresh, refreshHealth, refreshCurrent, toast, runAnalysis, cancelJob, demoLoad,
  }), [health, files, analyses, demos, currentId, current, currentLoading, job, refresh, refreshHealth, refreshCurrent, toast, runAnalysis, cancelJob, demoLoad]);

  return (
    <Store.Provider value={value}>
      {children}
      <div className="toasts">
        {toasts.map((t) => (
          <div key={t.id} className={`toast ${t.kind === "err" ? "err" : t.kind === "ok" ? "ok" : ""}`}>
            {t.title && <b>{t.title}</b>}<span>{t.msg}</span>
          </div>
        ))}
      </div>
    </Store.Provider>
  );
}
