<!-- summary: How each estimator works, which tests back every claim, and where the thresholds come from -->
# Methods

This page is the methodology summary: what each estimator computes, which test supports a claim, and
where the thresholds come from. Every threshold in the code carries a comment naming the measurement
that calibrated it.

## 1. Container and sample-format identification (`dsp/iqformats.py`)

* **WAV** — RIFF/RF64 parsing: sample rate, channels, bits per sample and data chunk are read from the
  header; a 2-channel 16/32-bit file with near-equal I/Q power is treated as complex baseband, a
  1-channel file as real, and the decision is reported with the evidence (channel correlation, I/Q
  power balance, header fields).
* **Raw / interleaved binary** — every candidate `(dtype, interleaving, endianness)` is *scored*
  against measurable properties of the byte stream: value-distribution plausibility, float-validity
  fraction (NaN/Inf/huge exponents rule float out), quantisation stride (integer formats produce
  lattice-valued samples), DC structure, I/Q balance and, for real-valued decoding of complex data,
  the image/aliasing signature. The winner is returned with a confidence and the full evidence list,
  and the UI offers a manual override that is recorded in the result.
* **NumPy `.npy`** and **SigMF sidecars** — header/metadata are parsed; absent fields stay
  `Unknown / requires estimation`.

The sample rate is **never invented**: WAV carries it, SigMF carries it, raw IQ usually does not — in
that case every frequency-derived number is reported in normalised units or flagged as requiring an
assumption, and the platform states it works in normalised frequency.

## 2. Preprocessing (`dsp/preprocess.py`)

DC removal (mean subtraction), detrending (linear trend), normalisation (unit RMS), IIR/FIR filtering
(LPF/BPF/HPF with explicit order and cut-offs), resampling (polyphase), spectral-subtraction denoising
with a configurable strength, I/Q imbalance correction, and windowing. Each step returns before/after
measurements (RMS, peak, DC, occupied bandwidth) so the UI comparison is checkable rather than
decorative.

## 3. Spectrum, bandwidth, noise floor, SNR (`dsp/spectrum.py`)

* **PSD** — Welch averaged periodogram with a Hann window, `scaling="density"`; two-sided for complex
  input, one-sided (power-doubled) for real input.
* **Peak / carrier** — maximum of the smoothed PSD, refined by parabolic interpolation on the three
  bins around the peak.
* **Occupied bandwidth** — the 99 % (configurable 90/95/99) power-containment bandwidth from the
  cumulative PSD.
* **Noise floor** — median of the PSD in the *quiet* part of the band, estimated with a robust
  (MAD-based) statistic and reported with the window used; the estimator refuses (status `unable`)
  when the band is fully occupied, because a floor cannot be separated from the signal.
* **SNR** — signal power inside the occupied bandwidth minus the estimated noise density integrated
  over the same band; the noise density is measured *outside* the signal band in the same record.
* **Centre frequency / offset** — centre of the occupied band; the offset is reported relative to the
  requested or nominal centre when one is known.

## 4. Waterfall and detection (`dsp/waterfall.py`, `dsp/detect.py`)

* **STFT** — Welch-style overlapping frames; the numeric analysis uses the full-resolution matrix, the
  browser payload is aggregated to a bounded number of cells.
* **CFAR detection** — a *smallest-of* (grey erosion) noise-floor map over a small time–frequency
  window, so a continuous carrier cannot raise its own detection floor; a cell is occupied when it
  exceeds the local floor by `snr_threshold_db` or by the chi-squared CFAR level derived from the
  number of averaged frames (bounded false-alarm rate).
* **Region measurement** — connected-component labelling, then per-region: start/end time, centre
  frequency, bandwidth, peak and mean power, SNR, occupancy fraction and a confidence that combines
  the SNR margin with the region's time-frequency support.
* **Band extraction** — a selected region is shifted to baseband, low-pass filtered and decimated so
  the modulation stage works on the emission, not on the whole record.

## 5. Parameter estimation (`dsp/params.py`)

Symbol rate uses **three independent estimators** and reconciles them, because no single blind
estimator works for every modulation:

1. **squared-envelope spectral line** — for PSK/QAM/ASK with excess bandwidth, |x|² is cyclostationary
   at the symbol rate (in the record's own units);
2. **instantaneous-frequency derivative line** — for (G)FSK the |dφ/dt| spectrum peaks at the symbol
   rate;
3. **autocorrelation of the symbol-rate candidates** — the lag-domain equivalent, used as a
   cross-check and to break ties.

Agreement between the estimators raises the confidence; disagreement is reported as an ambiguity with
the competing values. Excess bandwidth (roll-off) is estimated from the PSD edges, FSK deviation from
the instantaneous-frequency histogram, and amplitude/phase statistics (mean, standard deviation,
skewness of |x| and ∠x, envelope constancy) are measured on the baseband samples.

## 6. Modulation classification (`dsp/modulation.py`)

Hybrid, and every vote is inspectable:

* **DSP feature rules** — envelope constancy (piecewise-constant instantaneous frequency ⇒ FSK; measured
  linear/FSK separator 1.80, calibrated on the nine reference modulations), phase clustering after
  de-rotation, envelope-spectrum tone at the symbol rate (AM detection; the AM rule requires >45 dB at
  the symbol rate and *not* at the symbol rate, calibrated against 2FSK/GFSK/FM), carrier-line
  presence, spectral flatness (OFDM/noise-like ⇒ explicitly outside the supported set), amplitude
  histograms.
* **Constellation evidence** — each candidate modulation is actually demodulated (blind carrier and
  timing recovery) and scored by the **measured EVM** of the resulting symbol cloud; this is the
  strongest discriminator between PSK orders and QAM orders.
* **Statistical/spectral features** — cumulants, kurtosis, instantaneous-frequency statistics, PSD
  shape.
* **Optional ML vote** — a scikit-learn RandomForest trained on the generator's synthetic data
  (`ml/train_classifier.py`). If the model file is present its measured accuracy is exposed through
  `GET /api/models`; if it is absent the classifier runs on DSP + constellation evidence only. The ML
  vote never overrides the measurement-based evidence on its own.

Output: ranked candidates, confidences, the rules that fired with their weights, the measured EVM per
candidate, and ambiguity notes (for example: a single-tone FM test signal is waveform-identical to
two-level FSK, so FM is reported as an alternative with a zero-weight evidence note).

## 7. Demodulation (`dsp/demod.py`)

Stages are reported individually so a failure is localised:

```
detected signal → carrier recovery → timing recovery → matched filter → symbol decision → bits
```

* **Carrier recovery** — blind M-th-power coarse estimate plus decision-directed fine tracking for
  PSK/QAM; non-coherent discriminator (with a coherent correlator-bank cross-check) for (G)FSK;
  envelope detection for AM; phase-difference discrimination for FM.
* **Timing recovery** — a two-pass symbol-rate refinement (coarse ±2 % scan, then a fine ±0.15 % scan
  around the best candidate), using the measured EVM as the objective, followed by a Gardner-style
  timing loop for the sample phase.
* **Matched filter** — root-raised-cosine with the estimated roll-off; the constellation view offers
  raw, filtered and symbol-sampled clouds.
* **Decisions and quality** — hard decisions with measured EVM, SER/BER proxies and per-stage status.

## 8. FEC hypothesis testing (`dsp/fec.py`)

Each candidate code is *tested*, not guessed:

* **Convolutional** — a hard-decision Viterbi decoder runs for every standard code
  (K = 3/5/7/9, rates 1/2 and 1/3). The path metric per received bit is the Hamming distance to the
  nearest codeword; the same statistic is measured on randomised data (the null distribution), so the
  evidence is a z-score. Random uncoded data sits at the null mean; a genuinely coded stream sits far
  below it.
* **Reed–Solomon** — the bitstream is grouped into symbol blocks for the RS(255,k) family; a block is
  scored by syndrome weight and by how many symbol errors the decoder corrects, again against a
  randomised control.
* **Concatenated** — RS(outer) + convolutional(inner) chains are tested as ordered combinations.
* **LDPC** — not implemented; the platform says so instead of pretending.
* Only hypotheses that beat their control by the documented margin are promoted; the rest stay in the
  ranked list as indication-only with `rank_excluded` and their measured z-scores.

## 9. Interleaving hypothesis testing (`dsp/interleave.py`)

* **Candidate geometries** — block (8/16/32 rows), convolutional (rows × depth), diagonal (8/16/32),
  pseudo-random (seeded permutations), plus the identity (no interleaver).
* **Dispersion / clustering test** — low-|LLR| (unreliable) symbols cluster in time when the channel has
  memory and no interleaver is present, and are spread when an interleaver disperses them. A dispersion
  z-score against the memoryless null measures this; it is *indication only*.
* **De-interleave decode test** — each geometry is applied and the downstream decoder's improvement per
  bit is measured against randomised controls (the same geometry applied to shuffled data). A claim
  requires ≥ 0.10 relative improvement *and* beating the control maximum; the reporting threshold is
  0.20.
* **Time budget** — a two-stage probe (cheap geometries first, then promotion of the best candidates)
  with `time_budget_s`, so the search always terminates; `truncated` and
  `geometries_tested/geometries_total` tell the user whether the search was complete.

## 10. Bitstream and correlation (`dsp/bitstream.py`, `dsp/correlate.py`)

* Bit/byte packing, entropy per bit, ones fraction, byte histogram, run-length distribution,
  autocorrelation and periodicity, candidate frame boundaries from folding the stream at plausible
  lengths and looking for a stable header pattern.
* **Printable-ASCII candidate** — the recovered bytes are *searched* for printable runs and reported
  with a match score. The platform never calls this "the decoded message".
* **Correlation search** — the user supplies a header as text, hex or a bit string, optionally with a
  Hamming tolerance. Coarse candidates come from an FFT cross-correlation on ±1 sequences, and every
  candidate is then verified bit by bit, so a reported hit is a verified hit (position, occurrence
  count, matches, score).
* **Repeated structure** — automatic discovery of a periodically repeating pattern, reported as a
  header *candidate* with its length, support and score.

## 11. Comparison and benchmarking (`dsp/compare.py`, `dsp/benchmark.py`)

* **Comparison** — spectral shape similarity (normalised cross-correlation of the PSDs after aligning
  on the detected centre frequency), modulation agreement with both confidences, symbol-rate ratio,
  occupied-bandwidth ratio, SNR difference and constellation agreement. The verdict is one of
  *similar*, *partially similar*, *significantly different*. The platform explicitly does **not** claim
  a common source: that would need protocol evidence the platform does not have.
* **Benchmark** — the generator is the ground truth. Every configuration (modulation × SNR × symbol
  rate) is synthesised, analysed by the same estimators the API uses, and compared; the output is a
  confusion matrix, per-estimator error tables and FEC/interleaver detection results. Because both
  sides come from this repository, the numbers describe this implementation's behaviour on its own
  signal model — which is exactly what "self-benchmark" means, and the report says so.
