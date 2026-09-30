import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { Api } from "../lib/api";
import { useStore } from "../lib/store";
import JobBar from "../components/jobbar";
import RunPanel from "../components/RunPanel";
import { Card, Empty } from "../components/ui";

/** Main workspace: upload/choose a file, run the pipeline, jump into the result. */
export default function Analyze() {
  const { currentId, setCurrentId, analyses, runAnalysis, toast } = useStore();
  const nav = useNavigate();
  const [fileId, setFileId] = useState<string | null>(null);

  useEffect(() => {
    if (currentId) { const a = analyses.find((x) => x.analysis_id === currentId); if (a?.file_id) setFileId(a.file_id); }
  }, [currentId, analyses]);

  const run = async () => {
    const res = await runAnalysis({ file_id: fileId, options: {}, wait_s: 0 }, { label: "Analysis" });
    if (res?.analysis_id) { setCurrentId(res.analysis_id); nav(`/analyze/${res.analysis_id}`); }
  };

  return (
    <div>
      <JobBar />
      <div className="grid g2">
        <RunPanel fileId={fileId} setFileId={setFileId} />
        <Card title="What happens when you press run" sub="16 stages, all reported with status, duration and notes">
          <ol className="small" style={{ marginLeft: 16, color: "#c8d6ee" }}>
            <li>format detection (dtype, endianness, I/Q layout, sample rate) with confidence</li>
            <li>preprocessing: DC removal, optional filtering/denoising, before/after metrics</li>
            <li>Welch PSD: noise floor, SNR, peak, 99 % occupied bandwidth, frequency offset</li>
            <li>STFT spectrogram → CFAR detection of every emission in the record</li>
            <li>band extraction of the selected emission to complex baseband</li>
            <li>parameter estimation (carrier, symbol rate, amplitude/phase statistics)</li>
            <li>hybrid modulation classification: DSP rules + statistics + ML vote, ranked candidates</li>
            <li>demodulation: carrier recovery → timing → matched filter → symbols → bits</li>
            <li>constellation and eye diagram of the recovered symbols</li>
            <li>bitstream statistics, FEC hypotheses, interleaver hypotheses, correlation search</li>
            <li>multi-hypothesis ranking and the evidence graph</li>
            <li>report export (PDF/JSON/CSV/TXT) built from the stored result</li>
          </ol>
          <div className="row" style={{ marginTop: 10 }}>
            <button className="primary" disabled={!fileId} onClick={run}>run full analysis</button>
            <span className="small muted">or open a previous run from the list below</span>
          </div>
        </Card>
      </div>
      <Card title="Previous runs" sub="click to open the stored result (charts and reports come from the cache, nothing is recomputed)">
        {analyses.length ? (
          <div className="tblwrap" style={{ maxHeight: 320 }}>
            <table>
              <thead><tr><th>file</th><th>mode</th><th>modulation</th><th className="num">SNR</th><th>status</th><th></th></tr></thead>
              <tbody>{analyses.slice(0, 40).map((a) => (
                <tr key={a.analysis_id}>
                  <td>{a.file?.filename}</td><td className="small">{a.mode}</td>
                  <td>{a.summary?.modulation || "—"}</td><td className="num">{a.summary?.snr_db ?? "—"}</td>
                  <td>{a.status}</td>
                  <td><button className="tiny" disabled={a.status !== "done"} onClick={() => { setCurrentId(a.analysis_id); nav(`/analyze/${a.analysis_id}`); }}>open</button></td>
                </tr>))}
              </tbody>
            </table>
          </div>
        ) : <Empty>nothing analysed yet</Empty>}
      </Card>
    </div>
  );
}
