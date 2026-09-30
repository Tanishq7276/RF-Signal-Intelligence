import { Link } from "react-router-dom";
import { useStore } from "../lib/store";
import { Card, Stat } from "../components/ui";

const CHAIN = [
  ["IQ / WAV", "format detection: dtype, endianness, I/Q layout, sample rate"],
  ["DSP", "DC removal, filtering, resampling, denoising"],
  ["Spectrum", "Welch PSD, noise floor, SNR, 99 % occupied bandwidth"],
  ["Waterfall", "STFT spectrogram and CFAR emission detection"],
  ["Parameters", "carrier, offset, symbol rate, amplitude & phase statistics"],
  ["Modulation", "hybrid DSP + statistics + ML classification with evidence"],
  ["Demodulation", "carrier recovery, timing, matched filter, decisions"],
  ["FEC", "convolutional / RS / concatenated hypotheses vs random null"],
  ["Bitstream", "bits, bytes, entropy, structure and message recovery"],
  ["Report", "PDF / JSON / CSV / TXT with confidence and limitations"],
];

export default function Landing() {
  const { health, analyses, files } = useStore();
  const done = analyses.filter((a: any) => a.status === "done");
  return (
    <div>
      <div className="card" style={{ padding: 26, background: "linear-gradient(135deg,#0e1b2f,#101a2b 45%,#140f22)" }}>
        <div className="row" style={{ gap: 12, alignItems: "center" }}>
          <span className="badge info">SIH26147</span>
          <span className="badge">Automated model for analysis of .IQ and .WAV files</span>
          {health && <span className={`badge ${health.status === "ok" ? "ok" : "warn"}`}>backend {health.status}</span>}
        </div>
        <h1 style={{ fontSize: 30, margin: "14px 0 8px", lineHeight: 1.15 }}>RF Signal Intelligence Platform</h1>
        <p style={{ fontSize: 15, maxWidth: 880, color: "#c8d6ee" }}>
          AI-assisted IQ and WAV signal analysis, automated parameter extraction, modulation
          classification and signal characterization — blind, evidence-backed and fully runnable in
          the browser. Every number shown anywhere in this application is computed from the signal
          you load; nothing is canned.
        </p>
        <div className="row" style={{ marginTop: 16 }}>
          <Link className="btn primary" to="/analyze">Start analysis</Link>
          <Link className="btn" to="/generator">Synthetic signal lab</Link>
          <Link className="btn" to="/blind-analysis">Blind analysis</Link>
          <Link className="btn" to="/documentation">Documentation</Link>
        </div>
        <div className="grid g4" style={{ marginTop: 20 }}>
          <Stat k="Signal files in session" v={files.length} />
          <Stat k="Analyses run" v={analyses.length} n={`${done.length} completed`} />
          <Stat k="Detected emissions" v={done.reduce((s: number, a: any) => s + (a.summary?.n_emissions || 0), 0)} />
          <Stat k="DSP stages per analysis" v="16" n="format → report" />
        </div>
      </div>

      <div className="grid g2">
        <Card title="Processing chain" sub="the same chain runs for uploads, blind analyses and generated signals">
          <div className="col" style={{ gap: 6 }}>
            {CHAIN.map(([name, why], i) => (
              <div key={name} className="row" style={{ gap: 10, alignItems: "flex-start" }}>
                <span className="pill mono">{String(i + 1).padStart(2, "0")}</span>
                <div>
                  <b style={{ fontSize: 12.5 }}>{name}</b>
                  <div className="small muted">{why}</div>
                </div>
              </div>
            ))}
          </div>
        </Card>
        <div className="col" style={{ gap: 14 }}>
          <Card title="What makes this different" sub="not another FFT viewer">
            <ul className="small" style={{ marginLeft: 16, color: "#c8d6ee" }}>
              <li><b>Blind end-to-end analysis</b> — no modulation, symbol rate, carrier offset, FEC or interleaver input required.</li>
              <li><b>Multi-hypothesis reasoning</b> — every candidate pipeline is demodulated and scored; you inspect the ranking.</li>
              <li><b>Evidence &amp; confidence for every automatic result</b>, with the method and its limitations.</li>
              <li><b>FEC and interleaver hypothesis testing</b> against a random-data control group — never claimed without evidence.</li>
              <li><b>Synthetic generator with message recovery</b> — transmit a known message, analyse it, compare bit-by-bit.</li>
              <li><b>Self-benchmarking</b> — the platform measures its own accuracy and publishes the confusion matrix.</li>
            </ul>
          </Card>
          <Card title="Scientific honesty in the UI" sub="how uncertainty is reported">
            <div className="row" style={{ gap: 6 }}>
              <span className="badge ok">MEASURED</span><span className="badge info">ESTIMATED</span>
              <span className="badge info">CLASSIFIED</span><span className="badge hyp">HYPOTHESIZED</span>
              <span className="badge warn">LOW CONFIDENCE</span><span className="badge bad">UNABLE / UNKNOWN</span>
            </div>
            <ul className="small" style={{ marginLeft: 16, marginTop: 8, color: "#c8d6ee" }}>
              <li>Absent metadata is shown as <i>Unknown / requires estimation</i> — never invented.</li>
              <li>A parameter that cannot be estimated keeps <code>value = null</code> and states why.</li>
              <li>Recovered bits are only called text when the bytes actually decode as text.</li>
              <li>Encrypted or scrambled payloads are reported as such, not decoded.</li>
            </ul>
          </Card>
        </div>
      </div>
    </div>
  );
}
