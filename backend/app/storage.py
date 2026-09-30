"""File storage, format detection, sample loading and the analysis cache.

Security notes
--------------
* uploads are stored under ``data/uploads/<session>/`` with a generated file id and a sanitised
  name; the original name is kept only as a display string in the database,
* only allow-listed extensions are accepted and blocklisted (executable/script) extensions are
  rejected outright,
* nothing from an upload is ever executed or ``eval``-ed; binary payloads are only ever read as
  arrays,
* every file is owned by a session key and lookups always filter on it, so one browser session
  cannot see another session's data.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import shutil
import uuid
from typing import Any, BinaryIO

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from dsp import bitstream as bits_mod
from dsp import correlate as corr_mod
from dsp import fec as fec_mod
from dsp import hypothesis as hyp_mod
from dsp import interleave as inter_mod
from dsp import iqformats as iqf
from dsp import modulation as mod_mod
from dsp import params as params_mod
from dsp import pipeline as pipe
from dsp import preprocess as pre_mod
from dsp import scenarios as scen
from dsp import spectrum as spec_mod
from dsp import synth as synth_mod
from dsp import utils as utils_mod
from dsp import waterfall as wf

from .config import DANGEROUS_EXTENSIONS, settings
from .models import (AnalysisSession, BenchmarkResult, Bitstream, Comparison,
                     DemodulationResult, FECHypothesis, GeneratedSignal, InterleavingHypothesis,
                     ModulationResult, Report, SignalParameters, SignalSegment, UploadedFile)

_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def sanitise_filename(name: str | None) -> str:
    base = os.path.basename((name or "upload").replace("\\", "/")).strip()
    base = _SAFE.sub("_", base)[:120] or "upload"
    if base.startswith("."):
        base = "_" + base[1:]
    return base


def validate_upload(name: str, size: int) -> dict:
    ext = os.path.splitext(name.lower())[1]
    if ext in DANGEROUS_EXTENSIONS:
        return {"ok": False, "reason": f"the extension '{ext}' is not an accepted signal-file "
                                       "format (executables and scripts are refused)"}
    if ext not in settings.allowed_extensions:
        return {"ok": False, "reason": f"unsupported extension '{ext or '(none)'}'. Accepted: "
                                       + ", ".join(settings.allowed_extensions)}
    if size <= 0:
        return {"ok": False, "reason": "the file is empty"}
    if size > settings.max_upload_bytes:
        return {"ok": False, "reason": f"the file is {size / 1e6:.1f} MB, larger than the "
                                       f"{settings.max_upload_bytes / 1e6:.0f} MB limit"}
    return {"ok": True, "extension": ext}


# A file can pass the name/size checks and still not be a recording at all: a text file renamed to
# .wav, a 3-byte stub, a truncated container.  Those are rejected here with a reason the UI shows,
# *before* anything is stored as an analysis input - an upload that cannot be parsed must never end
# up as an analysis that silently reports nonsense.
MIN_BYTES = 16


def content_reject_reason(spec: dict, size: int, filename: str) -> str | None:
    ext = os.path.splitext((filename or "").lower())[1]
    container = str(spec.get("container") or "").lower()
    # containers report their frame count under different keys: raw IQ under "n_samples", a WAV
    # under "n_frames" (one frame = one complex sample for an I/Q WAV)
    n = int(spec.get("n_samples") or spec.get("n_frames") or 0)
    # The most specific reason wins, so the user is told what is actually wrong with the file.
    if container == "text":
        return ("the file contains printable text, not signal samples: a text file renamed to "
                f"'{ext or 'no extension'}' is not an IQ/WAV recording")
    if size < MIN_BYTES:
        return (f"only {size} bytes: too little data to identify a sample format, let alone a "
                f"signal (a single complex float32 sample already needs 8 bytes)")
    if container == "invalid":
        return ("the file could not be parsed as a signal container"
                + (f" ({spec['detection_error']})" if spec.get("detection_error") else "")
                + ": it is corrupt or truncated, so it is rejected instead of being guessed at")
    if ext in (".wav", ".wave") and container != "wav":
        return ("the '.wav' extension is present but no usable RIFF/WAVE header was found: the "
                "container is invalid or truncated, so the file is rejected instead of being "
                "guessed at")
    if ext == ".npy" and container != "npy":
        return ("the '.npy' extension is present but the NumPy magic header is missing: the file is "
                "not a NumPy array")
    if n < 4:
        return (f"the container could only be parsed into {n} sample(s): no analysis is possible "
                f"from that (upload at least a few hundred samples)")
    if container in ("", "unknown"):
        return (f"the container could not be identified (detection confidence "
                f"{float(spec.get('confidence') or 0):.2f}): the file is not a supported IQ/WAV "
                f"recording")
    return None


def path_for(session_key: str, file_id: str, filename: str, subdir: str | None = None) -> str:
    folder = os.path.join(settings.upload_dir, session_key) if subdir is None else subdir
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, f"{file_id}__{filename}")


def save_stream(session_key: str, filename: str, stream: BinaryIO, source: str = "upload",
                subdir: str | None = None) -> dict:
    """Validate + persist an uploaded/generated file, then detect its format."""
    safe = sanitise_filename(filename)
    file_id = uuid.uuid4().hex
    tmp = os.path.join(settings.data_dir, f".tmp_{file_id}")
    size = 0
    h = hashlib.sha256()
    with open(tmp, "wb") as fh:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            if size > settings.max_upload_bytes:
                fh.close()
                os.remove(tmp)
                raise ValueError(f"upload exceeds the {settings.max_upload_bytes / 1e6:.0f} MB limit")
            h.update(chunk)
            fh.write(chunk)
    check = validate_upload(safe, size)
    if not check.get("ok"):
        os.remove(tmp)
        raise ValueError(check["reason"])
    dest = path_for(session_key, file_id, safe, subdir=subdir)
    shutil.move(tmp, dest)
    spec = iqf.detect_format(dest, safe)
    reason = content_reject_reason(spec, size, safe)
    if reason:
        try:
            os.remove(dest)
        except OSError:                                                  # pragma: no cover
            pass
        raise ValueError(f"{reason} (nothing was stored, no analysis was started)")
    spec["sha256"] = h.hexdigest()
    spec["file_id"] = file_id
    spec["stored_path"] = dest
    spec["source"] = source
    return spec


def file_model_from_spec(session_key: str, spec: dict) -> UploadedFile:
    return UploadedFile(
        id=spec["file_id"], session_key=session_key, filename=spec.get("filename"),
        stored_path=spec["stored_path"], source=spec.get("source", "upload"),
        size_bytes=int(spec.get("file_size_bytes") or 0), sha256=spec.get("sha256"),
        container=spec.get("container"), detected_format=spec.get("format"),
        dtype=spec.get("dtype"), endianness=spec.get("endian_name"),
        iq_layout=spec.get("iq_layout"), channels=spec.get("channels"),
        n_samples=spec.get("n_samples"), duration_s=spec.get("duration_s"),
        sample_rate=spec.get("sample_rate"), sample_rate_source=spec.get("sample_rate_source"),
        center_frequency=spec.get("center_frequency"),
        center_frequency_source=spec.get("center_frequency_source"),
        confidence=spec.get("confidence"),
        metadata_source="detected" if spec.get("sample_rate") else "unknown",
        meta={"detection": {k: v for k, v in spec.items() if k not in ("stored_path",)}},
    )


# ---------------------------------------------------------------------------------------
# sample loading
# ---------------------------------------------------------------------------------------
def load_samples(file_row: UploadedFile, override: dict | None = None,
                 max_samples: int | None = None) -> dict:
    """Load the samples of a stored file, applying an optional manual format override."""
    spec = dict((file_row.meta or {}).get("detection") or {})
    spec.setdefault("filename", file_row.filename)
    spec.setdefault("container", file_row.container)
    spec.setdefault("dtype", file_row.dtype)
    spec.setdefault("endian_name", file_row.endianness)
    spec.setdefault("iq_layout", file_row.iq_layout)
    spec.setdefault("channels", file_row.channels)
    spec.setdefault("is_complex", (file_row.iq_layout or "").startswith(("interleaved", "complex")))
    notes: list[str] = []
    if override:
        mapping = {"dtype": "dtype", "byte_order": "endian_name", "endianness": "endian_name",
                   "iq_layout": "iq_layout", "format": "format", "channels": "channels",
                   "sample_rate": "sample_rate", "center_frequency": "center_frequency",
                   "offset_bytes": "offset_bytes", "n_samples": "n_samples"}
        for k, v in override.items():
            if v in (None, "", "auto"):
                continue
            key = mapping.get(k)
            if key:
                spec[key] = v
                notes.append(f"manual override: {k} = {v}")
        spec["confidence"] = 1.0
        spec["manual_override"] = True
    n_lim = int(max_samples or settings.max_samples_analyse)
    out = iqf.load_iq(file_row.stored_path, spec=spec, max_samples=n_lim)
    out["load_notes"] = notes + list(out.get("load_notes") or [])
    out["spec"] = spec
    return out


def effective_fs(file_row: UploadedFile, loaded: dict, override: dict | None = None) -> tuple[float, bool, str]:
    """Return (fs, fs_known, source).  An unknown sample rate is *never* invented."""
    ov = (override or {}).get("sample_rate")
    if ov:
        return float(ov), True, "user-specified sample rate"
    if file_row.sample_rate:
        return float(file_row.sample_rate), True, file_row.sample_rate_source or "file metadata"
    container_fs = (loaded.get("spec") or {}).get("sample_rate")
    if container_fs:
        return float(container_fs), True, "file metadata"
    return 1.0, False, "unknown - no sample-rate metadata in the file"


# ---------------------------------------------------------------------------------------
# analysis cache
# ---------------------------------------------------------------------------------------
def _options_hash(options: dict | None) -> str:
    payload = json.dumps(options or {}, sort_keys=True, default=str)
    return hashlib.sha1(payload.encode()).hexdigest()[:16]


def cache_write(analysis_id: str, options: dict | None, result: dict) -> str:
    folder = os.path.join(settings.cache_dir, analysis_id)
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{_options_hash(options)}.json.gz")
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        json.dump(result, fh, default=str)
    return path


def cache_read(path: str | None) -> dict | None:
    if not path or not os.path.exists(path):
        return None
    try:
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:                                                     # pragma: no cover
        return None


def delete_file(row: UploadedFile) -> None:
    for path in (row.stored_path,):
        with contextlib_suppress():
            os.remove(path)
    with contextlib_suppress():
        shutil.rmtree(os.path.join(settings.cache_dir, row.id))


class contextlib_suppress:                                                    # noqa: N801
    """Tiny local replacement for contextlib.suppress so the import list stays short."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return exc_type is not None
