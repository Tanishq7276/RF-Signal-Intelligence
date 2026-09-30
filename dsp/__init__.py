"""SIH26147 RF Signal Analysis Platform - DSP/ML core library.

Public modules:
    iqformats   - file/format detection and IQ/WAV loading
    synth       - synthetic signal generator (ground-truth data)
    preprocess  - DC removal, filtering, resampling, denoising
    spectrum    - Welch PSD, occupied bandwidth, SNR, noise floor
    waterfall   - STFT spectrogram / waterfall
    detect      - time-frequency signal detection (candidate regions)
    params      - signal parameter extraction (symbol rate, offsets, stats)
    modulation  - hybrid automatic modulation classification
    demod       - modular demodulators (PSK/QAM/FSK/AM/FM)
    fec         - FEC hypothesis engine (convolutional / RS / concatenated)
    interleave  - interleaving hypothesis engine (block/conv/diagonal/pseudo)
    bitstream   - bit/byte utilities, entropy, frame candidate detection
    correlate   - bit and byte level pattern search + header discovery
    hypothesis  - multi-hypothesis ranking engine
    pipeline    - end-to-end orchestrator (guided + blind modes)
"""
__version__ = "1.0.0"
