<!-- summary: Known limitations, honest uncertainty policy and future research directions -->
# Limitations and future research

The platform is deliberately explicit about what it cannot do. This page is the same list the UI
shows where relevant ("Limitations" on the results, `limitations` in every hypothesis record).

## Where the platform is weak, and why

| Limitation | What happens in the app | Cause |
| --- | --- | --- |
| **Single-tone FM is indistinguishable from 2FSK** | FM signals carry 2FSK as the top candidate with FM as an alternative, plus a zero-weight evidence note saying the two waveforms are identical for a pure tone | a constant-envelope, two-level instantaneous-frequency signal *is* the same waveform; separating them needs the message's statistics (speech/music), which a test signal does not have |
| **Blind classification without a symbol-rate estimate** | confidence drops and the ranked list is shown with lower scores; the platform reports the ambiguity instead of a single answer | the discriminator between PSK/QAM orders is the symbol cloud, which requires timing; without it only statistical features remain |
| **FEC detection needs enough coded bits** | the hypothesis list is returned with z-scores and `n_bits_tested`; when the stream is too short the platform says so and ranks nothing | Viterbi/RS statistics need a population; a few hundred bits cannot separate a code from random data |
| **Interleaving detection is not always decidable** | block and pseudo-random interleavers are frequently missed; the result lists `geometries_tested/geometries_total` and `truncated` | interleaving is a permutation of the bits: with no errors to disperse, or with a decoder that cannot see the gain, the measurement is genuinely ambiguous. The search is time-bounded on purpose |
| **LDPC / turbo / polar codes** | reported as *unsupported* in `capabilities` and in the FEC result | implementing a blind LDPC detector is a research project; claiming it without a decoder would be dishonest |
| **Absolute frequencies need a sample rate** | parameters are flagged `requires_estimated_fs`; time/frequency axes are shown in normalised units | a raw IQ capture has no rate metadata. The platform estimates relative structure but will not invent an absolute rate |
| **Symbol rate for very low SNR** | `Unable to estimate reliably` with the tests that were run | the cyclostationary lines disappear into the noise; the estimator refuses rather than reporting a spurious value |
| **Blind carrier recovery for high-order QAM** | EVM degrades and the constellation view shows the smear; the demodulator reports the failing stage | decision-directed recovery needs a usable initial point; 64-QAM at low SNR has no blind anchor |
| **Own-signal benchmark** | accuracy numbers are labelled "self-benchmark: synthetic signals from this repository's generator" | ground truth is only available for synthesised data; the platform does not claim performance on unknown field captures |
| **PostgreSQL and multi-worker deployment** | supported through `SIH_DATABASE_URL`; the bundled compose file runs one API container with a thread-pool job runner | the demo needs determinism and a small footprint; horizontal scaling would need the job runner moved to a broker |
| **Browser capture** | Live Analysis records from an audio input (`.wav` at the sound-card rate), not from an SDR | a web page cannot open an SDR device; the page states this and offers file upload instead |

## The uncertainty policy (enforced in code)

1. A value that cannot be supported is returned as `value: null`, `status: "unable"` plus a reason —
   never as an approximate fiction.
2. Every parameter carries `method` and `evidence`; every hypothesis carries its competing candidates.
3. Confidences come from the measurements that are returned alongside them; there is no random or
   hard-coded confidence anywhere in the code base.
4. AI/ML output is labelled as a *vote* with its measured accuracy; it never silently overrides DSP
   evidence.
5. Bits are never interpreted as text: the strongest claim is "printable-ASCII candidate" with a match
   score.
6. Two signals are never claimed to share a source; the comparison verdict is limited to
   similar / partially similar / significantly different.
7. FEC and interleaving are never claimed without beating an explicit null/control comparison.

## Future research

* **Soft-decision and iterative decoding** — feed the demodulator's LLRs into the FEC search (currently
  hard decisions) and iterate with the interleaver hypothesis; a turbo-style loop between the
  equaliser, the de-interleaver and the decoder is the natural next step.
* **LDPC/turbo blind detection** — parity-check consistency tests on candidate mother codes, with a
  belief-propagation decoder and a proper null model.
* **Protocol inference** — frame-boundary detection currently folds the stream at plausible lengths;
  a grammar/sequence-model approach (and cross-record session tracking) would let the platform infer
  frame structure with evidence instead of candidates.
* **Field-data validation** — the benchmark is self-referential by construction. Recording real
  captures (SDR front-ends, known transmitters) and publishing the measured error tables would turn the
  self-benchmark into an external one.
* **Deeper ML, still as a vote** — a small CNN on the raw I/Q with calibrated probabilities
  (`sklearn.calibration` or temperature scaling) could add a third evidence source; the architecture
  already isolates the ML vote so it can be swapped or removed without touching the DSP path.
* **Streaming/large-capture mode** — the current analysis is record-oriented with sample caps; a
  streaming CFAR + incremental symbol-rate estimator would allow multi-GB captures.
* **Multi-emission separation** — co-channel signals are currently detected as overlapping regions;
  blind source separation (or per-region iterative cancellation) would let each be demodulated
  independently.
