import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { Api, getApiBase, getEnvApiUrl } from "../lib/api";
import { bytes, si } from "../lib/format";
import { useStore } from "../lib/store";
import { Alert, Badge, Card, Empty, KV, NumField, Select } from "./ui";

export type RunOptions = {
  blind: boolean;
  preprocess: boolean;
  spectrogram: boolean;
  run_fec: boolean;
  run_interleaving: boolean;
  run_correlation: boolean;
  run_hypotheses: boolean;
  run_eye: boolean;
  modulation_hint?: string | null;
  symbol_rate_hint?: number | null;
  rolloff: number;
  max_fec_bits: number;
  signal_id?: number | null;
};

const DEFAULTS: RunOptions = {
  blind: false, preprocess: true, spectrogram: true, run_fec: true, run_interleaving: true,
  run_correlation: true, run_hypotheses: true, run_eye: true, modulation_hint: null,
  symbol_rate_hint: null, rolloff: 0.35, max_fec_bits: 6000,
};

/** File selection (upload / demo / already-uploaded) plus the real analysis options. */
export default function RunPanel({ blind = false, options, setOptions, fileId, setFileId, compact }: {
  blind?: boolean; options?: RunOptions; setOptions?: (o: RunOptions) => void;
  fileId?: string | null; setFileId?: (id: string | null) => void; compact?: boolean;
}) {
  const { files, demos, refresh, toast, runAnalysis, demoLoad, analyses, health } = useStore();
  const [internalFile, setInternalFile] = useState<string | null>(fileId ?? null);
  const [internalOpts, setInternalOpts] = useState<RunOptions>({ ...DEFAULTS, blind });
  const [uploadPct, setUploadPct] = useState<number | null>(null);
  const [over, setOver] = useState(false);
  const opts = options ?? internalOpts;
  const setOpts = setOptions ?? setInternalOpts;
  const sel = fileId !== undefined ? fileId : internalFile;
  const setSel = setFileId ?? setInternalFile;

  useEffect(() => { if (blind) setOpts({ ...(options ?? internalOpts), blind: true }); /* eslint-disable-next-line */ }, [blind]);

  const analyseCount = useMemo(() => {
    const m: Record<string, number> = {};
    analyses.forEach((a: any) => { if (a.file_id) m[a.file_id] = (m[a.file_id] || 0) + 1; });
    return m;
  }, [analyses]);

  const doUpload = async (f: File) => {
    setUploadPct(0);
    try {
      const res = await Api.upload(f, setUploadPct);
      await refresh();
      setSel(res.file.file_id);
      toast(`${f.name}: detected ${res.file.format || res.file.dtype} · ${res.file.iq_layout || "?"}` +
        (res.file.sample_rate ? ` · ${si(res.file.sample_rate, "Hz", 0)} sample rate from file metadata` :
          " · sample rate unknown (will be estimated)"), "ok", "Upload");
    } catch (e: any) {
      const isNetwork = e.status === 0 || e.message?.includes("network error") || e.message?.includes("Failed to fetch");
      const msg = isNetwork
        ? `Upload failed: Backend API is not reachable. Please connect your backend in the Settings page or via VITE_API_URL.`
        : `Upload refused: ${e.message || "Invalid file format or server error"}`;
      toast(msg, "err", "Upload");
    } finally { setUploadPct(null); }
  };

  const selected = files.find((f) => f.file_id === sel);
  const canRun = !!sel && !opts.blind ? true : !!sel;
  const isBackendOffline = health && health.status !== "ok";

  return (
    <Card
      title={blind ? "Blind analysis input" : "Input signal"}
      sub={blind
        ? "No modulation, symbol rate, carrier offset, FEC or interleaver input is used: the platform searches for all of them."
        : "Upload an .iq / .wav / complex-IQ capture, load one of the shipped demo signals, or pick a file already in this session."}
      right={<button className="tiny ghost" onClick={refresh}>refresh files</button>}
    >
      {isBackendOffline && (
        <Alert kind="warn" title={getEnvApiUrl() || getApiBase() ? "Backend API not reachable" : "Production backend URL is not configured"}>
          {getEnvApiUrl() || getApiBase()
            ? `The DSP analysis engine at ${getApiBase()} is currently unreachable. Check your backend status or configure the backend URL in `
            : "Production backend URL is not configured. Please set NEXT_PUBLIC_API_BASE_URL or VITE_API_URL in your deployment, or connect in "}
          <Link to="/settings" style={{ color: "#7cb3ff", fontWeight: "bold" }}>Settings → Backend API Server Connection</Link>.
        </Alert>
      )}

      <div
        className={`drop ${over ? "hot" : ""}`}
        onDragOver={(e) => { e.preventDefault(); setOver(true); }}
        onDragLeave={() => setOver(false)}
        onDrop={(e) => { e.preventDefault(); setOver(false); const f = e.dataTransfer.files?.[0]; if (f) doUpload(f); }}
        onClick={() => document.getElementById("file-input")?.click()}
      >
        <div style={{ fontSize: 13 }}>drop a file here or click to choose</div>
        <div className="small muted" style={{ marginTop: 4 }}>
          .iq · .wav · .npy · .bin · .cf32/.cs16/.cs8 · .dat — up to 512 MB; executables and scripts are refused
        </div>
        {uploadPct !== null && <div className="prog" style={{ marginTop: 10 }}><i style={{ width: `${uploadPct * 100}%` }} /></div>}
        <input id="file-input" type="file" style={{ display: "none" }}
          accept=".iq,.IQ,.wav,.WAV,.npy,.bin,.dat,.cfile,.cf32,.cs16,.cs8,.cu8,.float,.f32,.sc16,.s16,.raw,.complex"
          onChange={(e) => { const f = e.target.files?.[0]; if (f) doUpload(f); e.target.value = ""; }} />
      </div>

      {!!demos.length && (
        <div style={{ marginTop: 12 }}>
          <div className="small muted" style={{ marginBottom: 6 }}>shipped demo signals (generated with a known ground truth)</div>
          <div className="row">
            {demos.map((d: any) => (
              <button key={d.name} className="tiny" title={d.description}
                onClick={async () => { const f = await demoLoad(d.name); if (f) setSel(f.file_id); }}>
                {d.name}{d.registered ? " ✓" : ""}
              </button>
            ))}
          </div>
        </div>
      )}

      <div style={{ marginTop: 12 }}>
        <Select label="selected file" value={sel ?? ""} onChange={setSel}
          options={[{ value: "", label: files.length ? "— choose a file —" : "no files yet" },
            ...files.map((f) => ({
              value: f.file_id,
              label: `${f.filename} · ${bytes(f.size_bytes)} · ${f.format || f.dtype || "?"}` +
                (f.sample_rate ? ` · ${si(f.sample_rate, "Hz", 0)}` : " · rate unknown") +
                (analyseCount[f.file_id] ? ` · ${analyseCount[f.file_id]} analysis` : ""),
            }))]} />
      </div>

      {selected && (
        <div style={{ marginTop: 8 }}>
          <KV rows={[
            ["filename", selected.filename],
            ["size", bytes(selected.size_bytes)],
            ["detected format", selected.format || "—"],
            ["layout / channels", `${selected.iq_layout || "?"} / ${selected.channels ?? "?"}`],
            ["sample rate", selected.sample_rate ? `${si(selected.sample_rate, "Hz", 3)} (${selected.sample_rate_source || "metadata"})` : "Unknown / requires estimation"],
            ["centre frequency", selected.center_frequency ? si(selected.center_frequency, "Hz") : "Unknown / requires estimation"],
            ["samples · duration", `${selected.n_samples ?? "?"} · ${selected.duration_s ? selected.duration_s.toFixed(4) + " s" : "—"}`],
          ]} />
        </div>
      )}

      {!compact && (
        <details style={{ marginTop: 12 }}>
          <summary style={{ cursor: "pointer", fontSize: 12.5 }}>analysis options (all optional)</summary>
          <div className="grid g3" style={{ marginTop: 10 }}>
            <NumField label="roll-off (matched filter)" value={opts.rolloff} step={0.05} min={0.05} max={1}
              onChange={(v: number) => setOpts({ ...opts, rolloff: v ?? 0.35 })} />
            <NumField label="max bits for FEC search" value={opts.max_fec_bits} step={1000} min={1000} max={100000}
              onChange={(v: number) => setOpts({ ...opts, max_fec_bits: v ?? 6000 })} />
            <NumField label="emission index to analyse" value={opts.signal_id ?? null} step={1} min={0}
              placeholder="strongest"
              onChange={(v: number | null) => setOpts({ ...opts, signal_id: v })} />
            <Select label="modulation hint (optional)" value={opts.modulation_hint ?? ""}
              onChange={(v) => setOpts({ ...opts, modulation_hint: v || null })}
              options={[{ value: "", label: "let the platform decide" }, ...["BPSK", "QPSK", "8PSK", "16QAM", "64QAM", "2FSK", "GFSK", "AM", "FM"].map((m) => ({ value: m, label: m }))]}
              hint="a hint only steers the demodulator; the classifier still reports its own ranking" />
            <NumField label="symbol-rate hint [Hz]" value={opts.symbol_rate_hint ?? null} step={100} min={1}
              onChange={(v: number | null) => setOpts({ ...opts, symbol_rate_hint: v })} />
          </div>
          <div className="row" style={{ marginTop: 8 }}>
            {([["preprocess", "preprocessing"], ["spectrogram", "spectrogram"], ["run_fec", "FEC search"],
              ["run_interleaving", "interleaver search"], ["run_correlation", "correlation search"],
              ["run_hypotheses", "multi-hypothesis"], ["run_eye", "eye diagram"]] as [keyof RunOptions, string][]).map(([k, label]) => (
              <label className="chk" key={String(k)}>
                <input type="checkbox" checked={!!opts[k]} onChange={(e) => setOpts({ ...opts, [k]: e.target.checked })} />{label}
              </label>
            ))}
          </div>
          <Alert kind="warn">
            Disabling FEC/interleaver/hypothesis search makes the run much faster but removes those results
            from the evidence chain — the report then states that they were not searched.
          </Alert>
        </details>
      )}

      <div className="row" style={{ marginTop: 12 }}>
        <button className="primary" disabled={!canRun}
          onClick={() => {
            const options: any = {
              preprocess: opts.preprocess, spectrogram: opts.spectrogram, run_fec: opts.run_fec,
              run_interleaving: opts.run_interleaving, run_correlation: opts.run_correlation,
              run_hypotheses: opts.run_hypotheses, run_eye: opts.run_eye, rolloff: opts.rolloff,
              max_fec_bits: opts.max_fec_bits,
            };
            if (opts.modulation_hint) options.modulation_hint = opts.modulation_hint;
            if (opts.symbol_rate_hint) options.symbol_rate_hint = opts.symbol_rate_hint;
            if (opts.signal_id !== null && opts.signal_id !== undefined) options.signal_id = opts.signal_id;
            if (blind) options.blind = true;
            runAnalysis({ file_id: sel, options, wait_s: 0 }, { blind, label: blind ? "Blind analysis" : "Analysis" });
          }}>
          {blind ? "Run blind analysis" : "Run full analysis"}
        </button>
        <span className="small muted">
          runs: format check → preprocessing → PSD → spectrogram → detection → parameters → classification →
          {blind ? " blind demodulation →" : " demodulation →"} FEC → interleaving → bitstream → hypotheses → evidence
        </span>
      </div>
      {!files.length && !compact && <div style={{ marginTop: 8 }}><Empty>No file yet. Load a demo signal or upload a capture to begin.</Empty></div>}
    </Card>
  );
}
