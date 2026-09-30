<!-- summary: Component diagram, request flow, module map, storage model and performance strategy -->
# Architecture

## Component view

```
                      ┌──────────────────────────────────────────────────────────────┐
                      │  Browser (React + TypeScript + Plotly, dark RF-lab theme)     │
                      │  Dashboard · Analyze · Live · Explorer · Blind · Modulation   │
                      │  Demod · FEC/Interleaving · Bitstream · Compare · Generator   │
                      │  Benchmark · History · Reports · Settings · Documentation      │
                      └───────────────┬──────────────────────────────────────────────┘
                                      │  fetch/JSON + x-session-id  (same origin in production)
┌─────────────────────────────────────▼───────────────────────────────────────────────┐
│ FastAPI application  (backend/app)                                                   │
│                                                                                      │
│  routes/    system · auth · files · analysis · modules · dspops · generate ·          │
│             benchmark · reports · docs      (thin: validate → service → serialise)    │
│  services/  orchestration, persistence, report building, comparison                   │
│  jobs/      thread pool, progress, per-stage messages, cancellation, history          │
│  storage/   streamed uploads, format detection cache, loaders, result cache (.gz)     │
│  security   session isolation, safe filenames, extension allow/deny lists, limits     │
│  reports/   PDF (reportlab) · JSON · CSV · TXT                                        │
└───────────┬─────────────────────────┬──────────────────────────┬─────────────────────┘
            │                         │                          │
     ┌──────▼──────┐          ┌───────▼────────┐         ┌───────▼────────┐
     │  dsp/       │          │  ml/           │         │  storage       │
     │  engine     │          │  classifier    │         │  SQLite/PG     │
     │  (NumPy/    │          │  (scikit-learn │         │  data/app.db   │
     │   SciPy)    │          │   joblib)      │         │  uploads/ cache/│
     └─────────────┘          └────────────────┘         │  reports/      │
                                                          └────────────────┘
```

## Request flow (a full analysis)

1. **Upload** — `POST /api/upload` streams the body to disk with a size cap and a sanitised filename,
   detects container/sample format (`dsp/iqformats.detect_format`) and stores the detection report
   with its evidence in the file row's `meta`.
2. **Analyse** — `POST /api/analyze` (or `/api/blind-analysis`) validates the request, creates an
   `AnalysisSession` row and enqueues a job. The worker loads the samples memory-bounded, runs
   `dsp/pipeline.analyse_record`, writes progress after every stage, and finally persists the
   queryable metadata (parameters, hypotheses, bitstream statistics) and a compressed cache file with
   the full result.
3. **Read** — the UI polls `GET /api/jobs/{id}` for progress and then reads
   `GET /api/analysis/{id}` (summary), `/spectrum`, `/spectrogram`, `/detection`,
   `/parameters`, `/constellation`, `/eye`, `/bitstream`, `/evidence`. Regional work
   (`POST /api/analysis/{id}/region`, `/select-signal`) re-enters the pipeline on the selected segment
   only.
4. **Modules** — `POST /api/demodulate`, `/fec/analyze`, `/interleaving/analyze`, `/correlate`,
   `/compare` operate on a stored analysis (or on an uploaded file) and return the same
   value/confidence/evidence record structure as the pipeline.
5. **Report** — `POST /api/analysis/{id}/report` renders PDF/JSON/CSV/TXT into `data/reports` and
   links it to the session.

## DSP engine map

| Module | Responsibility | Key entry points |
| --- | --- | --- |
| `iqformats.py` | container + dtype + I/Q structure identification, loader | `detect_format`, `file_report`, `load_iq` |
| `preprocess.py` | DC removal, detrend, normalise, filters, resample, denoise, windows, before/after metrics | `run_pipeline`, `apply_step` |
| `spectrum.py` | Welch PSD, peak/carrier, occupied bandwidth, noise floor, SNR, spectral shape | `analyse_spectrum` |
| `waterfall.py` | STFT spectrogram, downsampled payload, band extraction, region measurement | `waterfall_payload`, `extract_band`, `analyse_region` |
| `detect.py` | cell-averaging/smallest-of CFAR detection, region labelling and measurement | `detect_signals` |
| `params.py` | symbol rate (three estimators reconciled), excess bandwidth, FSK deviation, amplitude/phase stats | `estimate_symbol_rate`, `full_parameter_report` |
| `modulation.py` | hybrid classification, constellation view, eye diagram | `classify`, `constellation_view`, `eye_diagram` |
| `demod.py` | modular demodulators with per-stage reporting, EVM, symbol-rate refinement | `demodulate`, `refine_symbol_rate` |
| `fec.py` | convolutional (Viterbi) / Reed–Solomon / concatenated hypothesis testing vs a null | `analyse_fec` |
| `interleave.py` | geometry search, dispersion clustering test, de-interleave decode test with a time budget | `analyse_interleaving` |
| `bitstream.py` | entropy, byte histogram, runs, periodicity, frame candidates, printable-ASCII candidate | `summarise_bits` |
| `correlate.py` | FFT-assisted pattern search (text/hex/bits, tolerant) + repeated-structure discovery | `analyse_correlation` |
| `hypothesis.py` | multi-hypothesis pipeline ranking | `analyse_hypotheses` |
| `compare.py` | spectral/modulation/rate/quality comparison and verdict | `compare_results` |
| `synth.py` | signal generator with ground truth, scrambler, FEC, interleavers, impairments | `generate_signal`, `write_iq` |
| `benchmark.py` | self-benchmark and confusion matrices | `run_benchmark` |
| `scenarios.py` | deterministic test scenarios (also used by the demo data and pytest) | `build` |
| `pipeline.py` | orchestrator that runs the stages, records per-stage success/failure and builds the evidence graph | `analyse_record` |
| `utils.py` | parameter records (`value`, `confidence`, `method`, `evidence`, `status`), plotting decimation, JSON helpers | `param`, `unable`, `decimate_for_plot` |

## Front-end structure

* `src/lib/api.ts` — typed client; adds the `x-session-id` header (per browser session), upload with
  progress, job polling with cancellation.
* `src/lib/store.tsx` — one context holding health, files, analyses, the current analysis and its
  result, toasts and the job bar; `runAnalysis` starts a job and polls it.
* `src/components/Plot.tsx` — a thin Plotly wrapper (zoom/pan/reset/PNG export are native).
* `src/components/ResultView.tsx` — the whole result surface (overview, spectrum, waterfall,
  detection, constellation & eye, demodulation, FEC & interleaving, bitstream, evidence, report).
* `src/pages/*` — the 20 routes; every button calls a real endpoint.

## Storage model

| Data | Where | Why |
| --- | --- | --- |
| raw samples | `data/uploads/<uuid>.<ext>` | streamed, size-limited, never held in the database |
| generated signals | `data/generated/<uuid>.<ext>` + ground truth | downloadable, used by `POST /api/generator/verify` |
| analysis result | `data/cache/<analysis_id>.json.gz` | charts, symbols, bit previews — reload without recomputation |
| queryable metadata | SQLite/PostgreSQL (15 tables) | history, comparison and reporting are SQL queries |
| reports | `data/reports/<uuid>.{pdf,json,csv,txt}` | download, never regenerated implicitly |

**Rule: no raw IQ arrays in relational tables.** The database stores scalars, JSON blobs of *summaries*
and links to files.

## Performance and robustness

* **Jobs, not blocking requests** — analyses and benchmarks run on a small worker pool with progress,
  stage messages, cancellation and a history log; the UI shows a live job bar.
* **Chunked/streamed I/O** — uploads and generated files are streamed to disk; the loader is
  sample-bounded (`SIH_MAX_SAMPLES`) and analysis is bounded (`SIH_MAX_SAMPLES_ANALYSE`) with an
  explicit truncation note instead of a crash.
* **Caching** — the full result is cached per analysis; single-stage endpoints (`/spectrum`,
  `/spectrogram`, `/bitstream`) reuse the analysis cache when the parameters match.
* **Bounded visuals** — spectrum/waterfall payloads are decimated for plotting while the numeric
  analysis uses full resolution.
* **Bounded search** — interleaver search has a two-stage probe with `interleave_time_budget_s`;
  the FEC scan and hypothesis chain have explicit caps surfaced as `truncated: true`.
* **Never crash on bad input** — corrupt WAV, truncated IQ, wrong dtype, DC-only or noise-only
  captures, unknown sample rate, failed synchronisation and memory pressure all resolve into a
  structured *unable* result with a reason (see [Limitations](08_limitations.md)).
* **Security** — extension allow/deny lists, safe filename generation, size limits, session-scoped
  rows on every query (no cross-session read is possible), temporary file cleanup, and uploads are
  only ever parsed as data — never imported or executed.
