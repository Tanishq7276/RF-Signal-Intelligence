import { useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { Api } from "../lib/api";
import { num } from "../lib/format";
import { useStore } from "../lib/store";
import Plot from "../components/Plot";
import { Alert, Badge, Card, Empty, KV, Stat } from "../components/ui";

/* ------------------------------------------------------------------ tiny FFT (radix-2, real input) */
function fftMagDb(buf: Float32Array, nfft: number, hann: Float32Array) {
  const re = new Float32Array(nfft), im = new Float32Array(nfft);
  for (let i = 0; i < nfft; i++) re[i] = (buf[i] || 0) * hann[i];
  // bit-reversal permutation
  for (let i = 1, j = 0; i < nfft; i++) {
    let bit = nfft >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) { const t = re[i]; re[i] = re[j]; re[j] = t; }
  }
  for (let len = 2; len <= nfft; len <<= 1) {
    const ang = (-2 * Math.PI) / len, wr = Math.cos(ang), wi = Math.sin(ang);
    for (let i = 0; i < nfft; i += len) {
      let cwr = 1, cwi = 0;
      for (let k = 0; k < len / 2; k++) {
        const ur = re[i + k], ui = im[i + k];
        const vr = re[i + k + len / 2] * cwr - im[i + k + len / 2] * cwi;
        const vi = re[i + k + len / 2] * cwi + im[i + k + len / 2] * cwr;
        re[i + k] = ur + vr; im[i + k] = ui + vi;
        re[i + k + len / 2] = ur - vr; im[i + k + len / 2] = ui - vi;
        const nwr = cwr * wr - cwi * wi;
        cwi = cwr * wi + cwi * wr; cwr = nwr;
      }
    }
  }
  const half = nfft / 2, out = new Float32Array(half);
  for (let i = 0; i < half; i++) {
    const p = (re[i] * re[i] + im[i] * im[i]) / nfft;
    out[i] = 10 * Math.log10(Math.max(p, 1e-20));
  }
  return out;
}

const NFFT = 1024;
const HANN = (() => { const h = new Float32Array(NFFT); for (let i = 0; i < NFFT; i++) h[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (NFFT - 1)); return h; })();
const MAX_FRAMES = 160;

/* ------------------------------------------------------------------ WAV writer (16-bit PCM) */
function wavBytes(samples: Float32Array, fs: number) {
  const n = samples.length, buf = new ArrayBuffer(44 + n * 2), v = new DataView(buf);
  const str = (o: number, s: string) => { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)); };
  str(0, "RIFF"); v.setUint32(4, 36 + n * 2, true); str(8, "WAVE"); str(12, "fmt ");
  v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
  v.setUint32(24, fs, true); v.setUint32(28, fs * 2, true); v.setUint16(32, 2, true); v.setUint16(34, 16, true);
  str(36, "data"); v.setUint32(40, n * 2, true);
  for (let i = 0; i < n; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    v.setInt16(44 + i * 2, s < 0 ? s * 32768 : s * 32767, true);
  }
  return new Blob([buf], { type: "audio/wav" });
}

/** /live — capture from the microphone / sound card and analyse it on the fly. */
export default function Live() {
  const { toast, refresh, setCurrentId } = useStore();
  const nav = useNavigate();
  const [on, setOn] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [fs, setFs] = useState<number | null>(null);
  const [stats, setStats] = useState<any>(null);
  const [frames, setFrames] = useState<{ z: number[][]; peakHz: number } | null>(null);
  const [wave, setWave] = useState<any>(null);
  const [recording, setRecording] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [recorded, setRecorded] = useState(0);

  const ctxRef = useRef<AudioContext | null>(null);
  const nodeRef = useRef<any>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const rowsRef = useRef<number[][]>([]);
  const recRef = useRef<Float32Array[]>([]);
  const recLenRef = useRef(0);
  const frameRef = useRef(0);
  const recordingRef = useRef(false);

  const stop = () => {
    nodeRef.current?.disconnect?.();
    streamRef.current?.getTracks?.().forEach((t: any) => t.stop());
    ctxRef.current?.close?.();
    nodeRef.current = null; streamRef.current = null; ctxRef.current = null;
    setOn(false);
  };
  useEffect(() => () => stop(), []);

  const start = async () => {
    setErr(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1, echoCancellation: false, noiseSuppression: false, autoGainControl: false },
      });
      const Ctx = (window as any).AudioContext || (window as any).webkitAudioContext;
      if (!Ctx) throw new Error("this browser has no Web Audio implementation");
      const ctx = new Ctx();
      await ctx.resume();
      const src = ctx.createMediaStreamSource(stream);
      const node = ctx.createScriptProcessor(2048, 1, 1);
      nodeRef.current = node; streamRef.current = stream; ctxRef.current = ctx;
      setFs(ctx.sampleRate);
      node.onaudioprocess = (e: any) => {
        const buf: Float32Array = new Float32Array(e.inputBuffer.getChannelData(0));
        if (recordingRef.current) {
          recRef.current.push(buf.slice());
          recLenRef.current += buf.length;
          if (frameRef.current % 8 === 0) setRecorded(recLenRef.current / ctx.sampleRate);
        }
        frameRef.current++;
        if (frameRef.current % 6 !== 0) return;                   // ≈8 fps of spectral frames
        const db = fftMagDb(buf, NFFT, HANN);
        const sorted = Float32Array.from(db).sort();
        const floor = sorted[Math.floor(sorted.length * 0.3)];
        let pk = 0, pki = 0;
        for (let i = 1; i < db.length - 1; i++) if (db[i] > pk) { pk = db[i]; pki = i; }
        const peakHz = (pki * ctx.sampleRate) / NFFT;
        const rms = Math.sqrt(buf.reduce((s, x) => s + x * x, 0) / buf.length);
        const peak = buf.reduce((s, x) => Math.max(s, Math.abs(x)), 0);
        const dc = buf.reduce((s, x) => s + x, 0) / buf.length;
        setStats({
          rms, peak, dc, crest: peak > 0 ? 20 * Math.log10(peak / Math.max(rms, 1e-12)) : null,
          floor_db: floor, peak_db: pk, snr_db: pk - floor, peakHz, n: buf.length,
          detected: pk - floor > 10,
        });
        rowsRef.current.push(Array.from(db));
        if (rowsRef.current.length > MAX_FRAMES) rowsRef.current.shift();
        setFrames({ z: rowsRef.current.slice(), peakHz });
        setWave({ t: Array.from({ length: 256 }, (_, i) => (i * buf.length) / 256 / ctx.sampleRate), y: Array.from({ length: 256 }, (_, i) => buf[Math.floor((i * buf.length) / 256)]) });
      };
      src.connect(node);
      node.connect(ctx.destination);
      setOn(true);
      recRef.current = []; recLenRef.current = 0;
    } catch (e: any) {
      setErr(e?.message || String(e));
      toast(`microphone capture refused: ${e?.message || e}`, "err");
    }
  };

  // keep the recording flag readable inside the audio callback
  useEffect(() => { recordingRef.current = recording; }, [recording]);

  const saveAndAnalyse = async () => {
    const ctx = ctxRef.current;
    if (!recRef.current.length || !ctx) return;
    setUploading(true);
    try {
      const total = recRef.current.reduce((s, b) => s + b.length, 0);
      const flat = new Float32Array(total);
      let o = 0;
      recRef.current.forEach((b) => { flat.set(b, o); o += b.length; });
      const blob = wavBytes(flat, Math.round(ctx.sampleRate));
      const f = new File([blob], `microphone_${new Date().toISOString().replace(/[:.]/g, "-")}.wav`, { type: "audio/wav" });
      const up = await Api.upload(f);
      toast(`saved ${(total / ctx.sampleRate).toFixed(1)} s of live audio as ${up.file.filename}`, "ok");
      await refresh();
      const res = await Api.analyze({ file_id: up.file.file_id, options: {}, wait_s: 0 });
      if (res?.job?.job_id) {
        const { pollJob } = await import("../lib/api");
        const done = await pollJob(res.job.job_id, () => undefined);
        if (done.status === "done" && done.result?.analysis_id) {
          setCurrentId(done.result.analysis_id);
          nav(`/analyze/${done.result.analysis_id}`);
          return;
        }
        toast(done.error || "analysis of the live capture failed", "err");
      }
      recRef.current = []; recLenRef.current = 0; setRecorded(0);
    } catch (e: any) { toast(`could not save the live capture: ${e.message}`, "err"); }
    finally { setUploading(false); }
  };

  return (
    <div>
      <Card title="Live analysis" sub="capture the sound card / microphone with the Web Audio API and analyse it in the browser with the same spectral maths the backend uses. Works on any real capture device — no simulated data."
        right={<Badge kind={on ? "ok" : "warn"}>{on ? "capturing" : "idle"}</Badge>}>
        <div className="row" style={{ gap: 8 }}>
          <button className="primary" disabled={on} onClick={start}>start capture</button>
          <button className="danger" disabled={!on} onClick={stop}>stop</button>
          <button disabled={!on} onClick={() => { recRef.current = []; recLenRef.current = 0; setRecording(true); setRecorded(0); }}>● record</button>
          <button disabled={!recording || !recRef.current.length || uploading} onClick={() => { setRecording(false); saveAndAnalyse(); }}>
            {uploading ? "uploading…" : `stop & analyse (${recorded.toFixed(1)} s)`}
          </button>
          <button className="ghost" disabled={!on} onClick={() => { rowsRef.current = []; setFrames({ z: [], peakHz: 0 }); }}>clear waterfall</button>
        </div>
        <Alert kind="info" title="what this mode can and cannot tell you">
          A sound card captures baseband audio: the sample rate is real (the AudioContext rate shown below)
          but there is <b>no RF centre frequency</b>, so frequencies are shown relative to 0 Hz of that
          capture. Recorded audio is written as a real 16-bit WAV with the true rate in its header, so the
          whole pipeline (detection, classification, demodulation) can then run on it.
        </Alert>
        {err && <Alert kind="warn" title="capture failed">
          {err}
          <div className="small" style={{ marginTop: 6 }}>
            The browser only exposes a microphone after an explicit user gesture and in a secure context.
            If this machine has no capture device, use the upload or demo path instead — the analysis
            pipeline is identical.
          </div>
        </Alert>}
      </Card>

      {stats && (
        <div className="grid g4">
          <Stat k="Sample rate (AudioContext)" v={fs ? `${fs} Hz` : "—"} n={`buffer ${stats.n} samples · FFT ${NFFT}`} />
          <Stat k="RMS / peak" v={`${num(stats.rms, 4)} / ${num(stats.peak, 4)}`} n={`DC offset ${num(stats.dc, 5)} · crest factor ${num(stats.crest, 2)} dB`} />
          <Stat k="Noise floor" v={`${num(stats.floor_db, 1)} dB`} n="30th percentile of the current frame" />
          <Stat k="Peak vs floor" v={`${num(stats.snr_db, 1)} dB`} n={`peak bin ${num(stats.peakHz, 0)} Hz (relative to capture centre)`} />
        </div>
      )}

      {on && stats?.detected && <Alert kind="ok">A narrow-band peak stands {num(stats.snr_db, 1)} dB above the noise floor at {num(stats.peakHz, 0)} Hz (relative). That is a detection, not an identification — classify it by recording and running the analyzer.</Alert>}
      {on && stats && !stats.detected && <Alert kind="warn">No peak stands more than 10 dB above the local noise floor in the current frame; either there is no emission in this band, or it is below the capture chain's noise.</Alert>}

      <div className="grid g2">
        <Card title="Live spectrum" sub="Hann window, 1024-point FFT, dB, updated ≈8 times per second">
          {frames?.z.length ? (
            <Plot height={300} data={[{
              type: "scattergl", mode: "lines",
              x: Array.from({ length: NFFT / 2 }, (_, i) => (i * (fs || 48000)) / NFFT),
              y: frames.z[frames.z.length - 1], line: { color: "#22d3ee" },
            }]} layout={{ xaxis: { title: "frequency relative to capture centre [Hz]" }, yaxis: { title: "power [dB]" }, margin: { l: 52, r: 12, t: 10, b: 40 } }} />
          ) : <Empty>start the capture to see the live spectrum</Empty>}
        </Card>
        <Card title="Live waterfall" sub={`last ${frames?.z.length ?? 0} frames, oldest at the top`}>
          {frames?.z.length ? (
            <Plot height={300} data={[{
              type: "heatmap", z: frames.z,
              x: Array.from({ length: NFFT / 2 }, (_, i) => (i * (fs || 48000)) / NFFT),
              y: Array.from({ length: frames.z.length }, (_, i) => i - frames.z.length),
              colorscale: "Jet", hovertemplate: "%{x:.0f} Hz<br>frame %{y}<br>%{z:.1f} dB<extra></extra>",
              colorbar: { title: { text: "dB" } },
            }]} layout={{ xaxis: { title: "frequency [Hz]" }, yaxis: { title: "age [frames]" }, margin: { l: 52, r: 12, t: 10, b: 40 } }} />
          ) : <Empty>waterfall appears once frames are captured</Empty>}
        </Card>
      </div>

      <div className="grid g2">
        <Card title="Live waveform" sub="one captured buffer (2048 samples), decimated for display">
          {wave ? <Plot height={220} data={[{ type: "scattergl", mode: "lines", x: wave.t, y: wave.y, line: { color: "#34d399" } }]}
            layout={{ xaxis: { title: "time [s]" }, yaxis: { title: "amplitude" }, margin: { l: 52, r: 12, t: 10, b: 40 } }} /> : <Empty>no buffer yet</Empty>}
        </Card>
        <Card title="Live measurements" sub="measured in the browser on the captured samples">
          {stats ? <KV rows={[
            ["sample rate", `${fs} Hz (from AudioContext)`],
            ["buffer size", `${stats.n} samples`],
            ["rms", num(stats.rms, 5)],
            ["peak", num(stats.peak, 5)],
            ["dc offset", num(stats.dc, 6)],
            ["crest factor", `${num(stats.crest, 2)} dB`],
            ["noise floor (frame)", `${num(stats.floor_db, 1)} dB`],
            ["peak bin", `${num(stats.peakHz, 0)} Hz relative`],
            ["peak − floor", `${num(stats.snr_db, 1)} dB`],
            ["centre frequency", "not present in an audio capture — Unknown / requires estimation"],
          ]} /> : <Empty>start the capture to measure</Empty>}
        </Card>
      </div>
      <Alert kind="warn" title="absolute RF frequency">
        A sound-card or microphone capture contains no RF metadata. The platform will never invent a
        centre frequency for it: it reports <code>Unknown / requires estimation</code> and analyses the
        record as baseband. For RF captures use a recorded .iq/.wav file that carries its own metadata.
      </Alert>
    </div>
  );
}
