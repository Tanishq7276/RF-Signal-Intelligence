<!-- summary: The exact click-path and expected output for a live judge demonstration -->
# The judge script (SIH DEMO MODE)

Open `http://localhost:8000/demo`. The page is a guided, eight-step walk-through that performs real
work at every step — nothing here is a slideshow. Expected outputs are stated so a judge can verify
that the platform is telling the truth (numbers below are from the bundled demo signals on a 2-vCPU
container; small variations are normal).

## Step 1 — Problem (5 s)

The banner shows the problem statement, the platform's honest scope and the non-claims
(no perfect decoding of unknown protocols, no invented metadata, no single-model AI).

## Step 2 — Upload (10 s)

Pick one of the bundled demo signals (`sample_qpsk.wav` is the best first demonstration) or press
**Upload** and drop your own `.iq`/`.wav`/`.cbin` file.

* The file card shows filename, size, container, dtype, I/Q structure, sample rate, centre frequency
  (or `Unknown / requires estimation`) and the detection evidence.
* The six bundled signals were produced by the repository's own generator, so `sample_data/GROUND_TRUTH.json`
  is the answer key for everything that follows.

## Step 3 — Automatic analysis (30–60 s)

Press **Run automatic analysis**. The job bar shows the stage messages live
(`spectrum → spectrogram → detection → parameters → classification → demodulation → FEC → …`) and can be
cancelled. Expected for `sample_qpsk.wav`: one detected emission around the nominal centre, QPSK as
the top modulation candidate, an occupied bandwidth of roughly the symbol rate × (1 + roll-off), an
SNR in the 18–22 dB region, and a bitstream of ~50 000 bits.

## Step 4 — AI/DSP reasoning (20 s)

The reasoning panel lists, in order: the detection evidence, the classifier's **rules that fired with
their weights**, the measured EVM per candidate modulation, the symbol-rate evidence from the three
independent estimators, and the FEC/interleaver z-scores against their nulls. The optional ML vote is
shown with its measured accuracy — and the panel states that the decision does not depend on it.

## Step 5 — 3-D pipeline visualisation (10 s)

The evidence graph (IQ → spectrum → spectrogram → detection → modulation hypotheses → timing →
demodulation → FEC → interleaving → bitstream) is drawn as staged columns; every node shows whether
that stage succeeded, its confidence and the measurement behind it. Failed stages are visible, not
hidden.

## Step 6 — Results (30 s)

The full result view opens on the same analysis: spectrum with peak/bandwidth/noise-floor/SNR markers,
waterfall with the detected region and click-to-re-analyse, constellation (raw/filtered/symbol-sampled)
with EVM, eye diagram with timing quality, demodulation stages, FEC and interleaving hypotheses,
bitstream statistics, correlation search, and the report export buttons (PDF/JSON/CSV/TXT).

Useful things to demonstrate here:

* **Waterfall region → re-analysis**: drag a box over an emission, press *analyse region*; the pipeline
  re-runs on that segment only (the timing shows a shorter record).
* **Constellation**: switch raw → filtered → symbol-sampled to show the four QPSK clusters forming, and
  read the measured EVM.
* **FEC panel**: the hypotheses are ranked with z-scores; press *verify with decode* to see the
  before/after bit-error comparison when a code is claimed.
* **Bitstream panel**: entropy per bit, byte histogram, run lengths, repeated-structure candidates and
  the correlation search box (try a header from your own frame format).

## Step 7 — Innovation (20 s)

The innovation panel re-uses the *current* result to show the differentiators with live numbers:
blind end-to-end analysis, multi-hypothesis ranking, evidence + confidence on every statement,
FEC/interleaver hypothesis testing, comparison and historical tracking, the synthetic generator with
ground truth, self-benchmarking, and the fact that all of it runs in one browser platform.

## Step 8 — Benchmark and report (60 s)

* **Benchmark**: run the AMC benchmark (nine modulations × SNRs) and show the confusion matrix; the
  accuracy is computed live from the generator's ground truth.
* **Report**: press *build report*, then download the PDF (a complete analysis document), the JSON
  (machine-readable), the CSV (parameter table) or the TXT (human-readable summary).

## Things to try live (the "it really works" checks)

| Try this | What should happen |
| --- | --- |
| Upload a text file renamed to `.wav` | the API rejects it with a 4xx explaining that the container is invalid; no analysis is started |
| Upload a 3-byte `.iq` file | rejected with a 4xx ("too little data to identify a sample format") |
| Upload a pure-noise capture | analysis completes, detection reports zero emissions, parameters are `Unable to estimate reliably`, and no modulation is claimed |
| Run **Blind Analysis** on `sample_16qam.wav` with no hints | the platform estimates the rate, classifies 16-QAM, demodulates, and reports its confidence honestly |
| Delete the ML model file (`ml/models/modclass.joblib`) and re-run | classification still works (DSP + constellation evidence) and `GET /api/models` says the model is unavailable |
| Generate a signal with FEC + interleaver in **Synthetic Generator**, then use *Recover* | the platform recovers the payload and scores it bit-by-bit against the generator's ground truth |
| Compare two analyses in **Compare Signals** | a verdict of similar / partially similar / significantly different with the measured metrics behind it |
| Restart the API and refresh the browser | CSS, charts and history survive (results are cached on disk and served by the API) |
