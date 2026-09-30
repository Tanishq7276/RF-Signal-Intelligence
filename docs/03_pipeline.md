<!-- summary: Stage-by-stage explanation of the blind analysis pipeline, the evidence graph and the confidence system -->
# Analysis pipeline

`dsp/pipeline.analyse_record` is the orchestrator used by both **Analyze Signal** (guided) and
**Blind Analysis** (nothing but the samples). Every stage is executed through a stage recorder: a
stage that fails does not abort the analysis — it is marked `ok: false` with the exception type, and
the downstream stages that depend on it are skipped, while the rest still run.

## Stages

| # | Stage | Input → output | Typical failure and how it is reported |
| --- | --- | --- | --- |
| 1 | format / container | bytes → dtype, structure, endianness, metadata, evidence | unsupported or ambiguous → low-confidence detection + manual override hint |
| 2 | preprocessing | samples → cleaned samples + before/after metrics | never fatal; steps that cannot run are listed as skipped with the reason |
| 3 | spectrum / PSD | samples → power spectrum, peak, occupied bandwidth, noise floor, SNR, carrier | noise-only or all-DC input → `status="unable"` for bandwidth/SNR with the reason |
| 4 | spectrogram / waterfall | samples → time–frequency matrix + detected regions | too few samples for an STFT frame → stage marked unavailable, charts show the reason |
| 5 | multi-signal detection | spectrogram → emissions (start/end, centre, bandwidth, peak power, SNR, confidence) | nothing above the CFAR floor → `signal_count: 0`, no emission is invented |
| 6 | parameter estimation | cleaned samples → symbol rate, deviation, amplitude/phase statistics, timing quality | no cyclostationary line → `Unable to estimate reliably` plus the tests that were run |
| 7 | modulation classification | samples + parameters → ranked candidates, confidences, rules fired, measured EVM | all candidates weak → top candidate returned with its low confidence and the ambiguity note |
| 8 | constellation / eye | samples + (modulation, symbol rate) → symbols, clusters, EVM, eye opening | failed synchronisation → stage `ok: false`, no constellation claimed |
| 9 | demodulation | samples → bits through carrier → timing → matched filter → decision, per-stage status | carrier/timing lock failure → the failing stage is named, bits are not reported as if valid |
| 10 | FEC hypothesis testing | bits → ranked code hypotheses with z-scores and control comparisons | none above threshold → `best: null`, hypotheses still listed with their evidence |
| 11 | interleaving hypothesis testing | soft decisions / bits → geometry hypotheses, dispersion z-score, decode-test improvement | budget exhausted → `truncated: true`, `geometries_tested/geometries_total` |
| 12 | bitstream analysis | bits → entropy, byte histogram, runs, periodicity, frame candidates, printable-ASCII candidate | empty bitstream → `ok: false` with the reason (no fabricated statistics) |
| 13 | correlation search | bits → verified pattern hits and repeated-structure candidates | no pattern supplied and no repetition found → zero hits, no candidate claimed |
| 14 | multi-hypothesis analysis | samples → ranked complete pipelines | always runs; hypotheses can be empty for noise-only input |
| 15 | evidence graph + confidence | all of the above → nodes, edges, confidence summary | always runs |

## Modes

* **AUTO (guided)** — the user may pin the sample rate, dtype, band, modulation or symbol rate. Pinned
  values are used as *hints*; every stage still reports what it measured, and a pin that contradicts
  the measurement is shown next to it.
* **BLIND** — no hints at all. Format detection, band selection, symbol rate, carrier offset,
  modulation, FEC and interleaving all come from the estimators, each with confidence and evidence.
  This is the mode the innovation claim rests on.
* **Region / selected signal** — a rectangle on the waterfall (`POST /api/analysis/{id}/region`) or a
  detected emission (`/select-signal`) re-enters the pipeline with the band shifted to baseband and
  decimated, so downstream stages work on the emission instead of the whole record.

## The evidence graph

Every result carries `evidence_graph`:

```
IQ samples ─► spectrum ─► spectrogram ─► detection ─► candidate signal
                                                   │
                        parameters ◄───────────────┘
                             │
                    modulation hypotheses ─► timing recovery ─► demodulation ─► bits
                                                                   │
                                                     FEC ─► interleaving ─► bitstream
```

Nodes carry `ok`, `confidence`, `status` and a one-line summary of the measurement that produced
them; edges record which stage consumed which. The UI renders it as an interactive 3-D-style staged
graph (Analyze → Evidence, and step 4 of the demo), so a judge can see *why* a conclusion was reached —
including which stages failed.

## The Analysis Confidence System

Three different things are deliberately kept apart, in the API and in the UI:

| Kind | Meaning | Where it appears |
| --- | --- | --- |
| **measured** | a value computed from the samples by a named estimator; the method and evidence are attached | every parameter record |
| **hypothesis** | a ranked candidate whose confidence comes from evidence tests (votes, z-scores, EVM, decode improvement) | modulation, FEC, interleaving, protocol structure |
| **unable** | the estimator cannot support a value for this input; the record keeps `value: null` and the reason | any parameter or module |

Rules enforced throughout the code base:

* a confidence is never invented — it is derived from the same measurements that are returned;
* a hypothesis is never presented as a verified fact: the top candidate is labelled as the top
  candidate, with its confidence and the runner-up;
* FEC and interleaving are only *claimed* when the test beats an explicit null/control comparison by a
  documented margin; otherwise they are reported as indication-only or excluded;
* everything numeric is JSON-serialisable and re-readable from the cached result, so the report shows
  exactly what the analysis showed.

## Multi-hypothesis ranking

`dsp/hypothesis.py` scores complete pipelines (a modulation plus a symbol rate) on measurements only:

| Term | Weight | Measurement |
| --- | --- | --- |
| demodulator quality | 0.30 | EVM / decision quality of an actual demodulation attempt |
| classification support | 0.25 | the classifier's confidence for that modulation |
| symbol-rate consistency | 0.15 | agreement between the three rate estimators and the timing loop |
| spectral fit | 0.10 | occupied bandwidth and spectral shape against the candidate's expected shape |
| bit-quality proxy | 0.10 | constellation cluster separation / soft-decision margin |
| evidence agreement | 0.10 | how many independent evidence sources agree |

Each hypothesis is returned with its component scores, the measurements behind them and an
`overall_confidence`. A hypothesis is inspectable and can be re-run on its own
(`POST /api/demodulate` with that modulation/rate), so "the platform thinks it is QPSK at
25 kSym/s" can be checked immediately.
