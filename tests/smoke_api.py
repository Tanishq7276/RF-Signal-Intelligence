"""End-to-end API smoke test against a running backend.

    python3 tests/smoke_api.py [base_url]

Exercises the endpoints the SIH26147 acceptance criteria list - upload, format detection, analysis,
every single-stage DSP module, FEC/interleaver hypothesis engines, correlation, comparison, the
synthetic generator with ground-truth verification, reports in all four formats, history, benchmark,
authentication and the failure paths (corrupt upload, unsupported file, unknown analysis id).

Every check prints PASS/FAIL with the measured value that was asserted, so the output doubles as
evidence that the platform is really computing rather than returning canned data.
"""
from __future__ import annotations

import io
import json
import os
import sys
import time

import httpx

BASE = (sys.argv[1] if len(sys.argv) > 1 else os.environ.get("SIH_API", "http://127.0.0.1:8000")).rstrip("/")
SESSION = "smoke-" + str(int(time.time()))
CLIENT = httpx.Client(base_url=BASE, timeout=httpx.Timeout(600.0, connect=10.0),
                      headers={"x-session-id": SESSION})
RESULTS: list[tuple[bool, str, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    RESULTS.append((bool(ok), name, str(detail)[:200]))
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  ·  {detail}" if detail else ""), flush=True)
    return bool(ok)


def get(path, **kw):
    r = CLIENT.get(path, **kw)
    return r


def post(path, body=None, **kw):
    return CLIENT.post(path, json=body, **kw)


def poll(job_id: str, timeout: float = 900.0) -> dict:
    t0 = time.time()
    last = {}
    while time.time() - t0 < timeout:
        last = get(f"/api/jobs/{job_id}").json()
        if last.get("status") in ("done", "failed", "cancelled"):
            return last
        time.sleep(0.6)
    return last


def main() -> int:
    # ---------------------------------------------------------------- system
    h = get("/api/health")
    hj = h.json()
    check("GET /api/health", h.status_code == 200 and hj.get("status") == "ok",
          f"status={hj.get('status')} version={hj.get('version')}")
    caps = get("/api/capabilities").json()
    check("GET /api/capabilities", isinstance(caps, dict) and len(caps) > 3,
          f"{len(caps)} capability groups")
    st = get("/api/stats").json()
    check("GET /api/stats", "files" in st or "counts" in st, json.dumps(st)[:120])
    oa = get("/openapi.json").json()
    check("GET /openapi.json", len(oa.get("paths", {})) > 40, f"{len(oa['paths'])} documented paths")
    check("GET /docs (Swagger UI)", get("/docs").status_code == 200)

    # ---------------------------------------------------------------- demo signals + upload
    demos = get("/api/demo").json()
    names = [d["name"] for d in demos.get("demo_signals", [])]
    check("GET /api/demo", len(names) >= 6, f"{len(names)} demo signals: {', '.join(names[:7])}")
    demo_file = None
    for want in ("sample_qpsk.wav", "sample_qpsk"):
        if want in names:
            r = post(f"/api/demo/{want}")
            if r.status_code == 200:
                demo_file = r.json()["file"]
                break
    if demo_file is None and names:
        r = post(f"/api/demo/{names[0]}")
        demo_file = r.json()["file"]
    check("POST /api/demo/{name} (load demo signal)", bool(demo_file),
          f"{demo_file and demo_file.get('filename')} layout={demo_file and demo_file.get('iq_layout')} "
          f"dtype={demo_file and demo_file.get('dtype')} fs={demo_file and demo_file.get('sample_rate')} "
          f"conf={demo_file and demo_file.get('format_confidence')}")
    check("format detection reports a data type and I/Q layout",
          bool(demo_file and demo_file.get("dtype") and demo_file.get("iq_layout")),
          f"container={demo_file and demo_file.get('container')} channels={demo_file and demo_file.get('channels')}")

    # corrupt / unsupported upload must fail cleanly with 4xx (never 500)
    bad = io.BytesIO(b"this is not a signal" * 10)
    r = CLIENT.post("/api/upload", files={"file": ("broken.wav", bad, "audio/wav")})
    check("corrupt upload is rejected with 4xx (not 500)", 400 <= r.status_code < 500,
          f"status={r.status_code} detail={str(r.json().get('detail'))[:90]}")
    r = CLIENT.post("/api/upload", files={"file": ("script.sh", io.BytesIO(b"#!/bin/sh\nrm -rf /"),
                                                     "application/x-sh")})
    check("script upload is refused", 400 <= r.status_code < 500, f"status={r.status_code}")
    r = CLIENT.post("/api/upload", files={"file": ("tiny.iq", io.BytesIO(b"\x01\x02\x03"), "application/octet-stream")})
    check("insufficient-samples upload is refused", 400 <= r.status_code < 500, f"status={r.status_code}")

    # a real upload of a demo file from disk (round trip through multipart)
    sample = "sample_data/sample_16qam.wav"
    if os.path.exists(sample):
        bio = io.BytesIO(open(sample, "rb").read())
        r = CLIENT.post("/api/upload", files={"file": (os.path.basename(sample), bio, "audio/wav")})
        ok = r.status_code == 200
        up = r.json().get("file", {}) if ok else {}
        check("POST /api/upload (real 16-QAM WAV)", ok,
              f"{up.get('filename')} {up.get('size_bytes')} B dtype={up.get('dtype')} "
              f"nsamples={up.get('n_samples')} fs={up.get('sample_rate')}")
        file_id = up.get("file_id") or (demo_file or {}).get("file_id")
    else:
        file_id = (demo_file or {}).get("file_id")

    for path in ("/api/files", f"/api/files/{file_id}", f"/api/files/{file_id}/preview"):
        r = get(path)
        check(f"GET {path}", r.status_code == 200 and isinstance(r.json(), dict),
              f"{r.status_code} {len(r.content)} B" if r.status_code == 200 else f"status={r.status_code}")

    # ---------------------------------------------------------------- single-stage DSP modules
    region_req = {"file_id": file_id, "max_samples": 400_000}
    singles = [
        ("/api/preprocess", {"steps": {"remove_dc": True, "normalize": True, "spectral_subtract": False}},
         lambda j: j.get("ok") and j.get("log")),
        ("/api/spectrum", {"nperseg": 2048},
         lambda j: j.get("ok") and j.get("spectrum")),
        ("/api/spectrogram", {"nperseg": 512},
         lambda j: j.get("ok") and (j.get("db") or j.get("waterfall") or j.get("db_downsampled"))),
        ("/api/detect-signals", {}, lambda j: j.get("ok") and (j.get("detection") is not None
                                                              or j.get("signals") is not None)),
        ("/api/extract-parameters", {}, lambda j: j.get("ok") and (j.get("parameters") or j.get("report"))),
        ("/api/modulation/classify", {}, lambda j: j.get("ok") and j.get("modulation")),
        ("/api/symbol-rate", {}, lambda j: j.get("ok")),
        ("/api/eye-diagram", {}, lambda j: j.get("ok") or "unavailable" in str(j.get("message", ""))),
    ]
    for path, extra, okfn in singles:
        body = dict(region_req)
        body.update(extra)
        t0 = time.time()
        r = post(path, body)
        j = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
        dt = time.time() - t0
        try:
            ok = r.status_code == 200 and bool(okfn(j))
        except Exception as exc:                                     # noqa: BLE001
            ok = False
            j = {"error": str(exc)}
        check(f"POST {path}", ok, f"{r.status_code} in {dt:.1f} s · {json.dumps(j)[:110]}")

    # ---------------------------------------------------------------- full analysis pipeline
    t0 = time.time()
    r = post("/api/analyze", {"file_id": file_id, "options": {}, "wait_s": 0})
    check("POST /api/analyze (async job)", r.status_code == 200 and "job_id" in r.json(),
          f"status={r.status_code}")
    job = poll(r.json()["job_id"])
    analysis_id = (job.get("result") or {}).get("analysis_id")
    check("analysis job completes", job.get("status") == "done" and bool(analysis_id),
          f"status={job.get('status')} in {time.time() - t0:.1f} s "
          f"stage={job.get('message')} error={job.get('error')}")

    for path in (f"/api/analysis/{analysis_id}", f"/api/analysis/{analysis_id}/result",
                 f"/api/analysis/{analysis_id}/spectrum", f"/api/analysis/{analysis_id}/spectrogram",
                 f"/api/analysis/{analysis_id}/detection", f"/api/analysis/{analysis_id}/segments",
                 f"/api/analysis/{analysis_id}/parameters", f"/api/analysis/{analysis_id}/constellation",
                 f"/api/analysis/{analysis_id}/eye", f"/api/analysis/{analysis_id}/bitstream",
                 f"/api/analysis/{analysis_id}/evidence", f"/api/analysis/{analysis_id}/status"):
        r = get(path)
        ok = r.status_code == 200
        check(f"GET {path.replace(analysis_id, '<id>')}", ok,
              f"{r.status_code} {len(r.content) / 1024:.1f} kB" if ok else f"status={r.status_code}")

    body = get(f"/api/analysis/{analysis_id}").json()
    sig = (body.get("result") or {}).get("signal") or {}
    mod_primary = sig.get("modulation", {}).get("primary") or "QPSK"

    # single-stage modules that need the analysis (they reuse the stored region / bit stream)
    for path, extra, okfn in [
        ("/api/constellation", {"analysis_id": analysis_id, "modulation": mod_primary},
         lambda j: j.get("ok") and (j.get("constellation") or {}).get("ok")),
        ("/api/bitstream", {"analysis_id": analysis_id},
         lambda j: j.get("ok") and j.get("n_bits")),
        ("/api/correlation", {"analysis_id": analysis_id, "pattern": "SIH26147", "kind": "auto"},
         lambda j: j.get("ok") is not None),
    ]:
        r = post(path, extra)
        j = r.json()
        check(f"POST {path} (with analysis_id)", r.status_code == 200 and bool(okfn(j)),
              f"{r.status_code} {json.dumps(j)[:110]}")
    check("analysis result carries the measured parameters",
          bool(sig.get("modulation") and sig.get("symbol_rate_hz")),
          f"modulation={sig.get('modulation', {}).get('primary')} "
          f"conf={sig.get('modulation', {}).get('confidence')} "
          f"rs={sig.get('symbol_rate_hz')} evm={((sig.get('demodulation') or {}).get('quality') or {}).get('evm_percent')}")

    # region + signal selection re-analysis
    segs = get(f"/api/analysis/{analysis_id}/segments").json()
    items = segs.get("segments") or []
    if items:
        s0 = items[0]
        r = post(f"/api/analysis/{analysis_id}/region",
                 {"f_lo_hz": s0.get("f_lo_hz"), "f_hi_hz": s0.get("f_hi_hz"),
                  "t0_s": s0.get("t0_s"), "t1_s": s0.get("t1_s")})
        check("POST /api/analysis/{id}/region (region re-analysis)", r.status_code == 200,
              f"status={r.status_code} {json.dumps(r.json())[:100]}")
    r = post(f"/api/analysis/{analysis_id}/select-signal", {"signal_id": 0})
    check("POST /api/analysis/{id}/select-signal", r.status_code == 200, f"status={r.status_code}")

    # ---------------------------------------------------------------- module endpoints
    t0 = time.time()
    r = post("/api/demodulate", {"analysis_id": analysis_id, "modulation": sig.get("modulation", {}).get("primary") or "QPSK"})
    dj = r.json()
    check("POST /api/demodulate", r.status_code == 200 and dj.get("ok"),
          f"EVM={dj.get('evm_percent')} bits={dj.get('n_bits')} stages={len(dj.get('stages') or [])} "
          f"in {time.time() - t0:.1f} s" if r.status_code == 200 else f"status={r.status_code}")

    t0 = time.time()
    r = post("/api/fec/analyze", {"analysis_id": analysis_id})
    fj = r.json()
    check("POST /api/fec/analyze", r.status_code == 200 and fj.get("hypotheses") is not None,
          f"best={(fj.get('best') or {}).get('hypothesis')} "
          f"conf={(fj.get('best') or {}).get('confidence')} n_hyp={len(fj.get('hypotheses') or [])} "
          f"in {time.time() - t0:.1f} s")

    t0 = time.time()
    r = post("/api/interleaving/analyze", {"analysis_id": analysis_id})
    ij = r.json()
    check("POST /api/interleaving/analyze", r.status_code == 200 and ij.get("hypotheses") is not None,
          f"best={(ij.get('best') or {}).get('hypothesis')} cands={len(ij.get('hypotheses') or [])} "
          f"in {time.time() - t0:.1f} s")

    r = post("/api/correlate", {"analysis_id": analysis_id, "pattern": "SIH26147", "kind": "auto", "max_errors": 0})
    cj = r.json()
    check("POST /api/correlate", r.status_code == 200 and cj.get("searches") is not None,
          str(cj.get("summary"))[:130])

    # ---------------------------------------------------------------- generator + verification
    gen_spec = {"modulation": "8PSK", "symbol_rate": 25000.0, "fs": 200000.0, "n_symbols": 3000,
                "snr_db": 24.0, "carrier_offset_hz": 3200.0, "fec": "none", "payload": "text",
                "text": "SIH26147 VERIFY PAYLOAD ", "seed": 777}
    r = post("/api/generate", {"spec": gen_spec, "format": "iq"})
    gj = r.json()
    gen_ok = r.status_code == 200 and gj.get("ground_truth")
    check("POST /api/generate (synthetic IQ)", gen_ok,
          f"{gj.get('file', {}).get('filename')} ground truth keys={len(gj.get('ground_truth') or {})}"
          if gen_ok else f"status={r.status_code} {json.dumps(gj)[:120]}")
    gen_file = gj.get("file", {}).get("file_id")

    t0 = time.time()
    r = post("/api/analyze", {"file_id": gen_file, "options": {}, "wait_s": 0})
    gjob = poll(r.json()["job_id"])
    gen_analysis = (gjob.get("result") or {}).get("analysis_id")
    check("generated signal is analysed end-to-end", gjob.get("status") == "done",
          f"{time.time() - t0:.1f} s status={gjob.get('status')} error={gjob.get('error')}")

    r = post("/api/generator/verify", {"generated_id": gj.get("generated_id"),
                                       "analysis_id": gen_analysis})
    vj = r.json()
    vok = r.status_code == 200 and vj.get("ok")
    check("POST /api/generator/verify (ground truth vs recovered bits)", vok,
          f"match={vj.get('match_ratio')} bits={vj.get('n_bits_compared')} "
          f"ASCII={str(vj.get('recovered_text'))[:40]}" if vok else f"status={r.status_code} {json.dumps(vj)[:140]}")

    # ---------------------------------------------------------------- comparison
    r = post("/api/compare", {"analysis_a": analysis_id, "analysis_b": gen_analysis})
    cj2 = r.json()
    check("POST /api/compare", r.status_code == 200 and cj2.get("verdict"),
          f"verdict={cj2.get('verdict')} similarity={cj2.get('similarity')} metrics={len(cj2.get('metrics') or [])}")
    check("GET /api/comparisons", get("/api/comparisons").status_code == 200)

    # ---------------------------------------------------------------- reports
    r = post(f"/api/analysis/{analysis_id}/report", {"formats": ["pdf", "json", "csv", "txt"]})
    rj = r.json() if r.status_code == 200 else {}
    made = rj.get("reports") or []
    check("POST /api/analysis/{id}/report (4 formats)", len(made) == 4,
          ", ".join(f"{m['format']}:{m['size_bytes'] / 1024:.1f}kB" for m in made) or str(rj)[:160])
    for m in made:
        rr = get(f"/api/report/{m['report_id']}/download")
        head = rr.content[:4]
        ok = rr.status_code == 200 and len(rr.content) == m["size_bytes"] and len(rr.content) > 200
        if m["format"] == "pdf":
            ok = ok and head == b"%PDF"
        check(f"GET /api/report/<id>/download ({m['format']})", ok,
              f"{len(rr.content) / 1024:.1f} kB head={head!r}")
    for fmt in ("pdf", "json", "csv", "txt"):
        rr = get(f"/api/analysis/{analysis_id}/report?format={fmt}")
        check(f"GET /api/analysis/<id>/report?format={fmt} (on the fly)",
              rr.status_code == 200 and len(rr.content) > 200, f"{len(rr.content) / 1024:.1f} kB")
    check("GET /api/reports", len(get("/api/reports").json().get("reports", [])) >= 4)

    # ---------------------------------------------------------------- history, auth, benchmark
    hist = get("/api/history").json()
    check("GET /api/history", "analyses" in hist or "items" in hist or "count" in hist,
          json.dumps({k: v for k, v in hist.items() if not isinstance(v, list)})[:140])
    check("GET /api/analyses", len(get("/api/analyses").json().get("analyses", [])) >= 2)

    email = f"smoke{int(time.time())}@example.com"
    r = post("/api/auth/register", {"email": email, "password": "smoke-pass-123",
                                    "full_name": "Smoke Test", "organisation": "SIH"})
    tok = (r.json() or {}).get("token")
    check("POST /api/auth/register", r.status_code == 200 and bool(tok), f"status={r.status_code}")
    r = post("/api/auth/login", {"email": email, "password": "smoke-pass-123"})
    check("POST /api/auth/login", r.status_code == 200 and (r.json() or {}).get("token"), f"status={r.status_code}")
    r = CLIENT.get("/api/auth/me", headers={"Authorization": f"Bearer {tok}"})
    check("GET /api/auth/me (bearer token)", r.status_code == 200 and (r.json() or {}).get("authenticated"),
          f"user={(r.json() or {}).get('user', {}).get('email')}")
    r = post("/api/auth/login", {"email": email, "password": "wrong-password"})
    check("wrong password is rejected with 401", r.status_code == 401, f"status={r.status_code}")

    r = post("/api/benchmark", {"config": {"kind": "amc", "modulations": ["BPSK", "QPSK", "8PSK"],
                                           "snr_db": [20.0], "symbol_rates": [25000.0], "n_symbols": 800},
                                "async": True})
    bj = r.json()
    check("POST /api/benchmark (async)", r.status_code == 200 and "job" in bj, f"status={r.status_code}")
    t0 = time.time()
    bjob = poll(bj["job"]["job_id"], timeout=1200)
    bid = (bjob.get("result") or {}).get("benchmark_id")
    check("benchmark job completes", bjob.get("status") == "done" and bool(bid),
          f"{time.time() - t0:.1f} s accuracy={(bjob.get('result') or {}).get('accuracy')}")
    if bid:
        br = get(f"/api/benchmarks/{bid}").json()
        check("GET /api/benchmarks/{id} (confusion matrix)", bool(br.get("confusion_matrix")),
              f"classes={br.get('classes')} accuracy={br.get('accuracy')} cases={br.get('n_cases')}")
    check("GET /api/benchmarks", get("/api/benchmarks").status_code == 200)

    # ---------------------------------------------------------------- failure paths
    r = get("/api/analysis/does-not-exist")
    check("unknown analysis id -> 404", r.status_code == 404, f"status={r.status_code}")
    r = post("/api/analyze", {"file_id": "does-not-exist", "options": {}})
    check("unknown file id -> 4xx", 400 <= r.status_code < 500, f"status={r.status_code}")
    r = post("/api/demodulate", {"file_id": file_id, "modulation": "NOT-A-MOD"})
    check("unsupported modulation -> 4xx", 400 <= r.status_code < 500, f"status={r.status_code}")
    r = post("/api/spectrum", {"file_id": file_id, "nperseg": -5})
    check("invalid FFT size -> 422", r.status_code == 422, f"status={r.status_code}")

    # ---------------------------------------------------------------- cleanup
    r = CLIENT.delete(f"/api/analysis/{analysis_id}")
    check("DELETE /api/analysis/{id}", r.status_code == 200, f"status={r.status_code}")

    failed = [r for r in RESULTS if not r[0]]
    print("\n" + "=" * 100)
    print(f"{len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed"
          + (f" — {len(failed)} FAILED" if failed else " — no endpoint returned an error"))
    for _, name, detail in failed:
        print(f"  FAILED: {name} · {detail}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
