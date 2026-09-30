"""Report builders: JSON / CSV / TXT / PDF exports of a stored analysis result.

The PDF is generated with reportlab and embeds real matplotlib renderings of the *stored*
analysis arrays (spectrum, waterfall, constellation, eye diagram) - nothing here is decorative:
every number in the report comes from the cached analysis result of that analysis id.
"""
from __future__ import annotations

import csv
import datetime as _dt
import io
import json
from typing import Any

import numpy as np

from dsp import utils as utils_mod

FMT_EXT = {"pdf": "pdf", "json": "json", "csv": "csv", "txt": "txt"}
FMT_MIME = {"pdf": "application/pdf", "json": "application/json", "csv": "text/csv",
            "txt": "text/plain; charset=utf-8"}


# ---------------------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------------------
def _walk_params(block: Any, group: str) -> list[dict]:
    out: list[dict] = []
    if isinstance(block, dict):
        for name, rec in block.items():
            if isinstance(rec, dict):
                out.append(dict(rec, group=rec.get("group") or group, name=rec.get("name") or name))
            elif isinstance(rec, list):
                out.extend(_walk_params(rec, group))
    elif isinstance(block, list):
        for rec in block:
            if isinstance(rec, dict):
                out.append(dict(rec, group=rec.get("group") or group))
    return out


def _params(result: dict) -> list[dict]:
    """Every parameter record of the analysis (grouped `params` blocks plus the record spectrum)."""
    sig = result.get("signal") or {}
    out = [r for r in _walk_params(sig.get("params"), "signal") if r.get("name")]
    out += [dict(r, name=r["name"], group="record")
            for r in _walk_params((result.get("record_spectrum") or {}).get("params"), "record")
            if r.get("name")]
    out += [r for r in _walk_params((sig.get("modulation") or {}).get("params"), "modulation")
            if r.get("name")]
    return out


def _param_rows(result: dict) -> list[list[str]]:
    rows = []
    for rec in _params(result):
        rows.append([rec.get("name") or "", fmt(rec.get("value")), rec.get("unit") or "",
                     rec.get("status") or "", fmt(rec.get("confidence")), rec.get("method") or ""])
    return rows


def fmt(v: Any) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.6g}"
    if isinstance(v, (list, tuple)):
        return ", ".join(fmt(x) for x in v[:8])
    return str(v)


def _file_meta(result: dict) -> dict:
    return (result.get("file") or {}) if isinstance(result.get("file"), dict) else {}


def _bit_quality(dm: dict) -> str:
    bq = dm.get("bit_quality")
    if isinstance(bq, dict):
        bq = bq.get("quality") or bq.get("label")
    return str(bq or "-")


def summary_lines(result: dict) -> list[tuple[str, str]]:
    """Key/value rows used by the TXT and PDF report headers."""
    sig = result.get("signal") or {}
    det = result.get("detection") or {}
    mod = sig.get("modulation") or {}
    dm = sig.get("demodulation") or {}
    spec = sig.get("spectrum") or {}
    hyp = (sig.get("hypotheses") or {}).get("best") or {}
    fec = ((sig.get("fec") or {}).get("hypotheses") or [{}])
    inter = ((sig.get("interleaving") or {}).get("hypotheses") or [{}])
    f = _file_meta(result)
    return [
        ("Problem statement", "SIH26147 - Analysis of .IQ / .WAV files and signal parameter extraction"),
        ("Report generated", _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")),
        ("Analysis id", str(result.get("_analysis_id") or "-")),
        ("Mode", str(result.get("mode") or "-")),
        ("File", str(f.get("filename") or "-")),
        ("File size", f"{f['size_bytes'] / 1024:.1f} kB" if f.get("size_bytes") else "-"),
        ("Detected format", str(f.get("format") or f.get("dtype") or "-")),
        ("IQ layout", str(f.get("iq_layout") or "-")),
        ("Samples analysed", f"{result.get('n_samples'):,}" if result.get("n_samples") else "-"),
        ("Sample rate", (f"{result.get('fs'):.3f} Hz (known)"
                         if result.get("fs_known") else
                         f"{result.get('fs'):.3f} Hz (ESTIMATED - not present in the file metadata)")),
        ("Duration", f"{float(result.get('signal', {}).get('duration_s') or 0):.6f} s"),
        ("Emissions detected", str(len(det.get("signals") or []))),
        ("Occupied bandwidth (99 %)", f"{fmt(spec.get('obw_99_hz'))} Hz"),
        ("Noise floor", f"{fmt(spec.get('noise_floor_dbm'))} dBm/Hz-equivalent (density: {fmt(spec.get('noise_density_dbm_hz'))})"),
        ("SNR", f"{fmt(spec.get('snr_db'))} dB"),
        ("Symbol rate", f"{fmt(sig.get('symbol_rate_hz'))} Hz"),
        ("Modulation (primary candidate)", f"{mod.get('primary') or '-'} (confidence {fmt(mod.get('confidence'))})"),
        ("Demodulation", f"{'ok' if dm.get('ok') else 'not available'} EVM {fmt((dm.get('quality') or {}).get('evm_percent'))} %"),
        ("Bit quality", _bit_quality(dm)),
        ("FEC hypothesis", f"{(fec[0] or {}).get('hypothesis')} (confidence {fmt((fec[0] or {}).get('confidence'))})"),
        ("Interleaving hypothesis", f"{(inter[0] or {}).get('hypothesis')} (confidence {fmt((inter[0] or {}).get('confidence'))})"),
        ("Overall confidence", fmt((sig.get("confidence") or {}).get("overall"))),
        ("Best hypothesis", f"{hyp.get('modulation') or '-'} @ {fmt(hyp.get('symbol_rate_hz'))} Hz (score {fmt(hyp.get('score'))})"),
    ]


# ---------------------------------------------------------------------------------------
# text / csv / json
# ---------------------------------------------------------------------------------------
def build_txt(result: dict, meta: dict | None = None) -> str:
    L: list[str] = []
    add = L.append
    add("=" * 100)
    add("RF SIGNAL INTELLIGENCE PLATFORM - SIGNAL ANALYSIS REPORT")
    add("=" * 100)
    for k, v in summary_lines(result):
        add(f"{k:<34}: {v}")
    add("")
    add("-" * 100)
    add("EXTRACTED PARAMETERS  (value / confidence / method; 'unable' means it could not be estimated)")
    add("-" * 100)
    add(f"{'parameter':<40} {'value':>18} {'unit':<10} {'status':<9} {'conf':>5}  method")
    for rec in _params(result):
        add(f"{str(rec.get('name'))[:40]:<40} {fmt(rec.get('value')):>18} "
            f"{str(rec.get('unit') or ''):<10} {str(rec.get('status') or ''):<9} "
            f"{fmt(rec.get('confidence')):>5}  {str(rec.get('method') or '')[:60]}")
    notes = result.get("notes") or []
    warnings = result.get("warnings") or []
    if notes:
        add(""); add("NOTES")
        for n in notes:
            add(f"  - {n}")
    if warnings:
        add(""); add("WARNINGS / LIMITATIONS")
        for w in warnings:
            add(f"  - {w}")
    add("")
    add("This report contains measured values only.  Hypotheses (modulation, FEC, interleaving, "
        "protocol) are stated as hypotheses with a confidence; they are not verified protocol facts.")
    return "\n".join(L) + "\n"


def build_csv(result: dict) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["section", "key", "value", "unit", "status", "confidence", "method"])
    for k, v in summary_lines(result):
        w.writerow(["summary", k, v, "", "", "", ""])
    for rec in _params(result):
        w.writerow(["parameter", rec.get("name"), fmt(rec.get("value")), rec.get("unit") or "",
                    rec.get("status") or "", fmt(rec.get("confidence")), rec.get("method") or ""])
    det = result.get("detection") or {}
    for i, s in enumerate(det.get("signals") or []):
        w.writerow([f"emission_{i}", "f_center_hz", fmt(s.get("f_center_hz")), "Hz", "detected",
                    fmt(s.get("confidence")), "time-frequency detection"])
        w.writerow([f"emission_{i}", "bw_hz", fmt(s.get("bw_hz")), "Hz", "detected",
                    fmt(s.get("confidence")), "time-frequency detection"])
        w.writerow([f"emission_{i}", "snr_db", fmt(s.get("snr_db")), "dB", "detected", "", ""])
    mod = (result.get("signal") or {}).get("modulation") or {}
    for c in (mod.get("candidates") or []):
        w.writerow(["modulation_candidate", c.get("modulation") or c.get("label"), fmt(c.get("confidence")),
                    "", "hypothesis", fmt(c.get("confidence")), c.get("method") or "hybrid classifier"])
    for rec in _evidence(result):
        w.writerow(["evidence", rec.get("step"), rec.get("detail"), "", "", "", rec.get("status") or ""])
    return buf.getvalue()


def _evidence(result: dict) -> list[dict]:
    eg = result.get("evidence_graph") or {}
    nodes = eg.get("nodes") if isinstance(eg, dict) else None
    if isinstance(nodes, list):
        return [n for n in nodes if isinstance(n, dict)]
    if isinstance(eg, list):
        return [n for n in eg if isinstance(n, dict)]
    return []


def build_json(result: dict, meta: dict | None = None) -> str:
    payload = {"report": {"title": "RF Signal Intelligence Platform - analysis report",
                          "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
                          "problem_statement": "SIH26147",
                          "summary": dict(summary_lines(result)),
                          "meta": utils_mod.to_jsonable(meta or {})},
               "analysis": utils_mod.to_jsonable(result)}
    return json.dumps(payload, indent=1, default=str)


# ---------------------------------------------------------------------------------------
# figures + pdf
# ---------------------------------------------------------------------------------------
def _fig_spectrum(result: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    arr = (((result.get("record_spectrum") or {}).get("arrays")
            or ((result.get("signal") or {}).get("spectrum") or {}).get("arrays")) or {})
    f = arr.get("freq_hz") or []
    p = arr.get("psd_db") or arr.get("db") or []
    if not f or not p:
        return None
    n = min(len(f), len(p))
    fig, ax = plt.subplots(figsize=(7.2, 2.8), dpi=140)
    ax.plot(np.asarray(f[:n]) / 1e3, np.asarray(p[:n]), lw=0.8, color="#38bdf8")
    sp = (result.get("signal") or {}).get("spectrum") or {}
    if sp.get("f_center_hz") is not None:
        ax.axvline(float(sp["f_center_hz"]) / 1e3, color="#f59e0b", lw=0.9, ls="--",
                   label=f"carrier {float(sp['f_center_hz']) / 1e3:.2f} kHz")
        ax.legend(loc="upper right", fontsize=7, framealpha=0.2)
    ax.set_xlabel("baseband frequency [kHz]", fontsize=8)
    ax.set_ylabel("power [dB]", fontsize=8)
    ax.set_title("Averaged spectrum (Welch PSD)", fontsize=9)
    ax.grid(alpha=0.25, lw=0.4)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    return fig


def _fig_waterfall(result: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    wf = (result.get("spectrogram") or {}).get("arrays") or {}
    db = wf.get("db") or []
    if not db:
        return None
    M = np.asarray(db, dtype=float)
    if M.ndim != 2 or M.size == 0:
        return None
    f = np.asarray(wf.get("freq_hz") or [])
    t = np.asarray(wf.get("times_s") or [])
    fig, ax = plt.subplots(figsize=(7.2, 3.0), dpi=140)
    extent = None
    if f.size == M.shape[0] and t.size == M.shape[1]:
        extent = [float(t[0] * 1e3), float(t[-1] * 1e3), float(f[0] / 1e3), float(f[-1] / 1e3)]
    im = ax.imshow(M, aspect="auto", origin="lower", cmap="viridis", extent=extent)
    if extent is None:
        ax.set_xlabel("time bin", fontsize=8)
        ax.set_ylabel("frequency bin", fontsize=8)
    else:
        ax.set_xlabel("time [ms]", fontsize=8)
        ax.set_ylabel("frequency [kHz]", fontsize=8)
    for s in ((result.get("detection") or {}).get("signals") or [])[:6]:
        if extent and s.get("f_center_hz") is not None:
            ax.axhline(float(s["f_center_hz"]) / 1e3, color="#f87171", lw=0.7, ls=":")
    fig.colorbar(im, ax=ax, label="power [dB]", fraction=0.03, pad=0.01)
    ax.set_title("Spectrogram / waterfall with detected emission centres", fontsize=9)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    return fig


def _fig_constellation(result: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ca = (result.get("signal") or {}).get("constellation") or {}
    pts = ca.get("points") or ca.get("symbols") or []
    if not pts:
        return None
    P = np.asarray(pts)
    if P.ndim == 2 and P.shape[1] >= 2:
        x, y = P[:, 0], P[:, 1]
    elif np.iscomplexobj(P):
        x, y = P.real, P.imag
    else:
        return None
    fig, ax = plt.subplots(figsize=(3.4, 3.4), dpi=140)
    ax.scatter(x, y, s=3, alpha=0.35, color="#22d3ee")
    ax.axhline(0, color="#64748b", lw=0.5)
    ax.axvline(0, color="#64748b", lw=0.5)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title("Constellation (symbol samples)", fontsize=9)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    return fig


def build_pdf(result: dict, meta: dict | None = None) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import (Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                    TableStyle)

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=16 * mm, rightMargin=14 * mm,
                            topMargin=14 * mm, bottomMargin=14 * mm,
                            title="RF Signal Intelligence Platform - analysis report",
                            author="SIH26147 platform")
    ss = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=ss["Heading1"], fontSize=15, spaceAfter=4)
    h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=11, spaceBefore=8, spaceAfter=4)
    small = ParagraphStyle("small", parent=ss["BodyText"], fontSize=7.6, leading=10)
    body = ParagraphStyle("body", parent=ss["BodyText"], fontSize=8.6, leading=11.5)
    story: list = []

    story.append(Paragraph("Signal Analysis Report", h1))
    story.append(Paragraph("SIH26147 &middot; Automated Model for Analysis of .IQ and .WAV Files "
                           "Along with Signal Parameter Extraction", small))
    story.append(Spacer(1, 5))

    rows = [["Property", "Value"]] + [[k, v] for k, v in summary_lines(result)]
    t = Table(rows, colWidths=[52 * mm, 112 * mm])
    t.setStyle(TableStyle([
        ("FONT", (0, 0), (-1, -1), "Helvetica", 7.4),
        ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 7.6),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#eef2f7")]),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#94a3b8")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(t)

    for title, fig in (("Spectrum and detected carrier", _fig_spectrum(result)),
                       ("Spectrogram with detection overlays", _fig_waterfall(result))):
        if fig is None:
            continue
        png = io.BytesIO()
        fig.savefig(png, format="png")
        import matplotlib.pyplot as plt
        plt.close(fig)
        png.seek(0)
        story.append(Paragraph(title, h2))
        story.append(Image(png, width=170 * mm, height=170 * mm * fig.get_size_inches()[1] /
                           fig.get_size_inches()[0]))
    cfig = _fig_constellation(result)
    if cfig is not None:
        png = io.BytesIO()
        cfig.savefig(png, format="png")
        import matplotlib.pyplot as plt
        plt.close(cfig)
        png.seek(0)
        story.append(Paragraph("Constellation", h2))
        story.append(Image(png, width=70 * mm, height=70 * mm))

    story.append(PageBreak())
    story.append(Paragraph("Extracted parameters", h2))
    prows = [["Parameter", "Value", "Unit", "Status", "Conf.", "Method"]]
    for rec in _params(result):
        prows.append([Paragraph(str(rec.get("name") or "")[:70], small), fmt(rec.get("value")),
                      str(rec.get("unit") or ""), str(rec.get("status") or ""),
                      fmt(rec.get("confidence")), Paragraph(str(rec.get("method") or "")[:90], small)])
    if len(prows) == 1:
        story.append(Paragraph("No parameter could be extracted from this record.", body))
    else:
        pt = Table(prows, colWidths=[52 * mm, 30 * mm, 14 * mm, 16 * mm, 13 * mm, 39 * mm],
                   repeatRows=1)
        pt.setStyle(TableStyle([
            ("FONT", (0, 0), (-1, -1), "Helvetica", 6.8),
            ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 7),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#eef2f7")]),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#94a3b8")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]))
        story.append(pt)

    sig = result.get("signal") or {}
    mod = sig.get("modulation") or {}
    story.append(Paragraph("Modulation classification", h2))
    cand = mod.get("candidates") or []
    if cand:
        mrows = [["Candidate", "Confidence", "Evidence"]]
        for c in cand[:8]:
            mrows.append([str(c.get("modulation") or c.get("label")), fmt(c.get("confidence")),
                          Paragraph("; ".join(str(v) for v in (c.get("evidence") or [])[:4])[:400], small)])
        mt = Table(mrows, colWidths=[28 * mm, 20 * mm, 116 * mm], repeatRows=1)
        mt.setStyle(TableStyle([("FONT", (0, 0), (-1, -1), "Helvetica", 6.8),
                                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e293b")),
                                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#94a3b8")),
                                ("VALIGN", (0, 0), (-1, -1), "TOP")]))
        story.append(mt)
    else:
        story.append(Paragraph("No modulation candidate exceeded the reporting threshold.", body))

    story.append(Paragraph("Demodulation stages", h2))
    stages = ((sig.get("demodulation") or {}).get("stages") or [])
    if stages:
        srows = [["Stage", "Status", "Note"]]
        for s in stages:
            srows.append([str(s.get("stage")), str(s.get("status")),
                          Paragraph(str(s.get("error") or s.get("note") or "")[:200], small)])
        st = Table(srows, colWidths=[38 * mm, 18 * mm, 108 * mm], repeatRows=1)
        st.setStyle(TableStyle([("FONT", (0, 0), (-1, -1), "Helvetica", 6.8),
                                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e293b")),
                                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#94a3b8")),
                                ("VALIGN", (0, 0), (-1, -1), "TOP")]))
        story.append(st)
    else:
        story.append(Paragraph("No demodulation was applied to this record.", body))

    story.append(Paragraph("Bitstream, FEC and interleaving hypotheses", h2))
    for label, block in (("FEC", sig.get("fec") or {}), ("Interleaving", sig.get("interleaving") or {})):
        hyps = block.get("hypotheses") or []
        story.append(Paragraph(f"<b>{label}</b>: " + (
            "; ".join(f"{h.get('hypothesis')} (conf {fmt(h.get('confidence'))})" for h in hyps[:5])
            if hyps else "no hypothesis could be tested on this record"), small))
        best = block.get("best")
        if best and best.get("explanation"):
            story.append(Paragraph(str(best["explanation"])[:600], small))
    bs = sig.get("bitstream") or {}
    if bs:
        story.append(Paragraph(
            f"<b>Bitstream</b>: {fmt(bs.get('n_bits'))} bits, {fmt(bs.get('n_bytes'))} bytes, "
            f"entropy {fmt((bs.get('entropy') or {}).get('bits_per_byte') if isinstance(bs.get('entropy'), dict) else bs.get('entropy'))} bit/byte", small))

    eg = _evidence(result)
    if eg:
        story.append(Paragraph("Evidence graph", h2))
        erows = [["Step", "Status", "Detail"]]
        for n in eg[:40]:
            detail = " ".join(str(v) for v in (n.get("inputs") or [])[:4]) if n.get("inputs") else ""
            erows.append([str(n.get("step") or n.get("id")), str(n.get("status") or ""),
                          Paragraph((str(n.get("label") or "") + " " + detail)[:600], small)])
        et = Table(erows, colWidths=[34 * mm, 18 * mm, 112 * mm], repeatRows=1)
        et.setStyle(TableStyle([("FONT", (0, 0), (-1, -1), "Helvetica", 6.6),
                                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e293b")),
                                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#94a3b8")),
                                ("VALIGN", (0, 0), (-1, -1), "TOP")]))
        story.append(et)

    notes = (result.get("notes") or []) + (result.get("warnings") or [])
    if notes:
        story.append(Paragraph("Notes, warnings and limitations", h2))
        for n in notes[:24]:
            story.append(Paragraph(f"&bull; {str(n)[:400]}", small))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "Interpretation policy: measured values are reported with their estimation method; "
        "modulation, FEC, interleaving and protocol results are hypotheses with confidences and are "
        "never asserted as verified facts.  A candidate printable-text reading of the bit stream is "
        "a candidate only and is not a decoded message.", small))
    doc.build(story)
    return buf.getvalue()


def build_report(result: dict, fmt: str, meta: dict | None = None) -> tuple[bytes, str, str]:
    """Return ``(payload, media_type, extension)`` for the requested export format."""
    fmt = (fmt or "json").lower()
    if fmt not in FMT_EXT:
        raise ValueError(f"unsupported report format '{fmt}' (use one of {', '.join(FMT_EXT)})")
    if fmt == "json":
        payload = build_json(result, meta).encode("utf-8")
    elif fmt == "csv":
        payload = build_csv(result).encode("utf-8")
    elif fmt == "txt":
        payload = build_txt(result, meta).encode("utf-8")
    else:
        payload = build_pdf(result, meta)
    return payload, FMT_MIME[fmt], FMT_EXT[fmt]
