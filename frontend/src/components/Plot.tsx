import { useEffect, useRef, useState } from "react";
import Plotly from "plotly.js-dist-min";
import { PLOT_BASE, PLOT_CONFIG } from "../lib/format";

type Props = {
  data: any[]; layout?: any; height?: number; onRelayout?: (e: any) => void; className?: string;
};
/** Thin Plotly wrapper: real charts from the arrays the API returned. */
export default function Plot({ data, layout, height = 300, onRelayout, className }: Props) {
  const ref = useRef<HTMLDivElement>(null);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => {
    if (!ref.current) return;
    const el = ref.current;
    try {
      Plotly.react(el, data as any, { ...PLOT_BASE, ...(layout || {}), height, autosize: true } as any, PLOT_CONFIG as any);
      setErr(null);
    } catch (e: any) { setErr(String(e?.message || e)); }
    const handler = (ev: any) => onRelayout && onRelayout(ev);
    (el as any).on?.("plotly_relayout", handler);
    const ro = new ResizeObserver(() => { try { Plotly.Plots.resize(el); } catch { /* ignore */ } });
    ro.observe(el);
    return () => { ro.disconnect(); try { (el as any).removeAllListeners?.("plotly_relayout"); } catch { /* ignore */ } };
  }, [data, layout, height, onRelayout]);
  if (err) return <div className="alert bad">the chart could not be drawn: {err}</div>;
  return <div ref={ref} className={className || "plotbox"} style={{ height }} />;
}
