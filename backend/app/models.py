"""SQLAlchemy ORM models.

Design rule from the problem statement: **no raw IQ arrays are stored in relational tables.**
Samples live on disk (original upload) or in compact cache files; the database stores metadata,
measured parameters and the pointers needed to find them.
"""
from __future__ import annotations

import datetime as _dt
import json
import uuid
from typing import Any

from sqlalchemy import (JSON, Boolean, Column, DateTime, Float, ForeignKey, Integer, String, Text,
                        func)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


def _uuid() -> str:
    return uuid.uuid4().hex


def _now() -> _dt.datetime:
    return _dt.datetime.now(_dt.timezone.utc)


class JsonMixin:
    """Small helper so routes can attach plain dicts to JSON columns without escaping games."""

    @staticmethod
    def dumps(value: Any) -> Any:
        return json.loads(json.dumps(value, default=str)) if value is not None else None


class User(Base):
    """Platform account.  Passwords are stored as scrypt digests, never in clear text."""

    __tablename__ = "users"

    id = Column(String(32), primary_key=True, default=_uuid)
    email = Column(String(190), unique=True, index=True, nullable=False)
    full_name = Column(String(120))
    organisation = Column(String(160))
    role = Column(String(24), default="engineer")
    password_hash = Column(Text, nullable=False)
    is_active = Column(Boolean, default=True)
    is_admin = Column(Boolean, default=False)
    last_login_at = Column(DateTime(timezone=True))
    login_count = Column(Integer, default=0)
    created_at = Column(DateTime(timezone=True), default=_now)

    def public(self) -> dict:
        return {"user_id": self.id, "email": self.email, "full_name": self.full_name,
                "organisation": self.organisation, "role": self.role, "is_active": bool(self.is_active),
                "login_count": self.login_count,
                "created_at": self.created_at.isoformat() if self.created_at else None,
                "last_login_at": self.last_login_at.isoformat() if self.last_login_at else None}


class AnalysisSession(Base):
    """One analysis run (standard, blind or region re-analysis)."""

    __tablename__ = "analysis_sessions"

    id = Column(String(32), primary_key=True, default=_uuid)
    session_key = Column(String(64), index=True, nullable=False)
    file_id = Column(String(32), ForeignKey("uploaded_files.id", ondelete="CASCADE"), index=True)
    kind = Column(String(24), default="standard")          # standard | blind | region | manual
    mode = Column(String(24), default="AUTO")             # AUTO | BLIND
    status = Column(String(16), default="queued")          # queued | running | done | failed | cancelled
    progress = Column(Float, default=0.0)
    stage_message = Column(String(200))
    options = Column(JSON)
    summary = Column(JSON)                                # compact result summary for list views
    notes = Column(JSON)
    cache_path = Column(Text)
    error = Column(Text)
    duration_ms = Column(Float)
    created_at = Column(DateTime(timezone=True), default=_now, index=True)
    finished_at = Column(DateTime(timezone=True))

    file = relationship("UploadedFile", back_populates="analyses")
    segments = relationship("SignalSegment", back_populates="analysis", cascade="all, delete-orphan")
    parameters = relationship("SignalParameters", back_populates="analysis", cascade="all, delete-orphan")
    modulation = relationship("ModulationResult", back_populates="analysis", cascade="all, delete-orphan")
    demodulation = relationship("DemodulationResult", back_populates="analysis", cascade="all, delete-orphan")
    fec = relationship("FECHypothesis", back_populates="analysis", cascade="all, delete-orphan")
    interleaving = relationship("InterleavingHypothesis", back_populates="analysis", cascade="all, delete-orphan")
    bitstream = relationship("Bitstream", back_populates="analysis", cascade="all, delete-orphan")
    correlations = relationship("CorrelationResult", back_populates="analysis", cascade="all, delete-orphan")
    reports = relationship("Report", back_populates="analysis", cascade="all, delete-orphan")


class UploadedFile(Base):
    __tablename__ = "uploaded_files"

    id = Column(String(32), primary_key=True, default=_uuid)
    session_key = Column(String(64), index=True, nullable=False)
    filename = Column(String(255), nullable=False)           # original name (display only)
    stored_path = Column(Text, nullable=False)
    source = Column(String(16), default="upload")            # upload | generated | demo
    size_bytes = Column(Integer)
    sha256 = Column(String(64), index=True)
    container = Column(String(24))                           # wav | raw | npy
    detected_format = Column(String(48))
    dtype = Column(String(16))
    endianness = Column(String(8))
    iq_layout = Column(String(24))
    channels = Column(Integer)
    n_samples = Column(Integer)
    duration_s = Column(Float)
    sample_rate = Column(Float)
    sample_rate_source = Column(String(40))
    center_frequency = Column(Float)
    center_frequency_source = Column(String(40))
    confidence = Column(Float)
    metadata_source = Column(String(24), default="detected")   # detected | manual_override | unknown
    meta = Column(JSON)
    created_at = Column(DateTime(timezone=True), default=_now, index=True)

    analyses = relationship("AnalysisSession", back_populates="file", cascade="all, delete-orphan")


class SignalSegment(Base):
    """One detected emission (frequency band and/or time gate) inside a record."""

    __tablename__ = "signal_segments"

    id = Column(String(32), primary_key=True, default=_uuid)
    analysis_id = Column(String(32), ForeignKey("analysis_sessions.id", ondelete="CASCADE"), index=True)
    file_id = Column(String(32), ForeignKey("uploaded_files.id", ondelete="CASCADE"), index=True)
    segment_index = Column(Integer, default=0)
    label = Column(String(32))
    kind = Column(String(24))                                 # modulated | tone | burst | mixed
    f_lo_hz = Column(Float)
    f_hi_hz = Column(Float)
    center_frequency_hz = Column(Float)
    bandwidth_hz = Column(Float)
    peak_frequency_hz = Column(Float)
    t0_s = Column(Float)
    t1_s = Column(Float)
    duration_s = Column(Float)
    duty_cycle = Column(Float)
    peak_power_db = Column(Float)
    mean_power_db = Column(Float)
    snr_db = Column(Float)
    confidence = Column(Float)
    n_bursts = Column(Integer)
    selected = Column(Boolean, default=False)
    meta = Column(JSON)

    analysis = relationship("AnalysisSession", back_populates="segments")


class SignalParameters(Base):
    """Value / confidence / method record for every extracted parameter."""

    __tablename__ = "signal_parameters"

    id = Column(String(32), primary_key=True, default=_uuid)
    analysis_id = Column(String(32), ForeignKey("analysis_sessions.id", ondelete="CASCADE"), index=True)
    segment_id = Column(String(32), ForeignKey("signal_segments.id", ondelete="SET NULL"))
    group = Column(String(32))                                # record | spectrum | modulation | ...
    name = Column(String(96), nullable=False)
    value_num = Column(Float)
    value_text = Column(Text)
    unit = Column(String(24))
    status = Column(String(24))                               # ok | low_confidence | unable
    confidence = Column(Float)
    method = Column(Text)
    evidence = Column(JSON)
    limitations = Column(JSON)

    analysis = relationship("AnalysisSession", back_populates="parameters")


class ModulationResult(Base):
    __tablename__ = "modulation_results"

    id = Column(String(32), primary_key=True, default=_uuid)
    analysis_id = Column(String(32), ForeignKey("analysis_sessions.id", ondelete="CASCADE"), index=True)
    segment_id = Column(String(32), ForeignKey("signal_segments.id", ondelete="SET NULL"))
    modulation = Column(String(24), nullable=False)
    rank = Column(Integer, default=1)
    probability = Column(Float)
    confidence = Column(Float)
    probability_ml = Column(Float)
    probability_dsp = Column(Float)
    evm_percent = Column(Float)
    method = Column(Text)
    evidence = Column(JSON)
    rules_fired = Column(JSON)
    features = Column(JSON)
    ml_model = Column(String(64))
    ml_model_accuracy = Column(Float)
    created_at = Column(DateTime(timezone=True), default=_now)

    analysis = relationship("AnalysisSession", back_populates="modulation")


class DemodulationResult(Base):
    __tablename__ = "demodulation_results"

    id = Column(String(32), primary_key=True, default=_uuid)
    analysis_id = Column(String(32), ForeignKey("analysis_sessions.id", ondelete="CASCADE"), index=True)
    segment_id = Column(String(32), ForeignKey("signal_segments.id", ondelete="SET NULL"))
    modulation = Column(String(24))
    symbol_rate_hz = Column(Float)
    rolloff = Column(Float)
    ok = Column(Boolean, default=False)
    message = Column(Text)
    n_symbols = Column(Integer)
    n_bits = Column(Integer)
    evm_percent = Column(Float)
    evm_db = Column(Float)
    ber_estimate = Column(Float)
    esn0_db = Column(Float)
    bit_quality = Column(String(24))
    carrier_offset_hz = Column(Float)
    timing_phase_frac = Column(Float)
    stages = Column(JSON)
    symbols_preview = Column(JSON)
    bits_preview = Column(Text)
    created_at = Column(DateTime(timezone=True), default=_now)

    analysis = relationship("AnalysisSession", back_populates="demodulation")


class FECHypothesis(Base):
    __tablename__ = "fec_hypotheses"

    id = Column(String(32), primary_key=True, default=_uuid)
    analysis_id = Column(String(32), ForeignKey("analysis_sessions.id", ondelete="CASCADE"), index=True)
    family = Column(String(32))
    hypothesis = Column(String(96), nullable=False)
    rank = Column(Integer, default=1)
    confidence = Column(Float)
    support = Column(Float)
    z_score = Column(Float)
    null_level = Column(Float)
    n_blocks = Column(Integer)
    n_valid = Column(Integer)
    n_corrected = Column(Integer)
    n_clean = Column(Integer)
    parameters = Column(JSON)
    evidence = Column(JSON)
    limitations = Column(JSON)
    created_at = Column(DateTime(timezone=True), default=_now)

    analysis = relationship("AnalysisSession", back_populates="fec")


class InterleavingHypothesis(Base):
    __tablename__ = "interleaving_hypotheses"

    id = Column(String(32), primary_key=True, default=_uuid)
    analysis_id = Column(String(32), ForeignKey("analysis_sessions.id", ondelete="CASCADE"), index=True)
    kind = Column(String(32))
    hypothesis = Column(String(96), nullable=False)
    rank = Column(Integer, default=1)
    confidence = Column(Float)
    dispersion = Column(Float)
    z_score = Column(Float)
    improvement_per_bit = Column(Float)
    relative_improvement = Column(Float)
    code = Column(String(64))
    parameters = Column(JSON)
    evidence = Column(JSON)
    limitations = Column(JSON)
    created_at = Column(DateTime(timezone=True), default=_now)

    analysis = relationship("AnalysisSession", back_populates="interleaving")


class Bitstream(Base):
    __tablename__ = "bitstreams"

    id = Column(String(32), primary_key=True, default=_uuid)
    analysis_id = Column(String(32), ForeignKey("analysis_sessions.id", ondelete="CASCADE"), index=True)
    segment_id = Column(String(32), ForeignKey("signal_segments.id", ondelete="SET NULL"))
    n_bits = Column(Integer)
    n_bits_analysed = Column(Integer)
    entropy_per_bit = Column(Float)
    ones_fraction = Column(Float)
    bit_preview = Column(Text)
    hex_preview = Column(Text)
    byte_histogram = Column(JSON)
    run_length = Column(JSON)
    periodic = Column(JSON)
    frame_candidates = Column(JSON)
    printable_ascii = Column(JSON)
    autocorrelation = Column(JSON)
    method = Column(Text)
    created_at = Column(DateTime(timezone=True), default=_now)

    analysis = relationship("AnalysisSession", back_populates="bitstream")


class CorrelationResult(Base):
    __tablename__ = "correlation_results"

    id = Column(String(32), primary_key=True, default=_uuid)
    analysis_id = Column(String(32), ForeignKey("analysis_sessions.id", ondelete="CASCADE"), index=True)
    pattern_text = Column(Text)
    pattern_kind = Column(String(16))
    pattern_bits = Column(Integer)
    max_errors = Column(Integer)
    n_hits = Column(Integer)
    n_occurrences = Column(Integer)
    score = Column(Float)
    best_position_bits = Column(Integer)
    best_matches = Column(Integer)
    positions = Column(JSON)
    auto_candidates = Column(JSON)
    notes = Column(JSON)
    created_at = Column(DateTime(timezone=True), default=_now)

    analysis = relationship("AnalysisSession", back_populates="correlations")


class BenchmarkResult(Base):
    __tablename__ = "benchmark_results"

    id = Column(String(32), primary_key=True, default=_uuid)
    session_key = Column(String(64), index=True)
    kind = Column(String(32), default="full")                # amc | estimators | fec | full
    status = Column(String(16), default="done")
    config = Column(JSON)
    accuracy = Column(Float)
    confusion_matrix = Column(JSON)
    classes = Column(JSON)
    per_class = Column(JSON)
    metrics = Column(JSON)
    n_cases = Column(Integer)
    duration_ms = Column(Float)
    notes = Column(JSON)
    created_at = Column(DateTime(timezone=True), default=_now, index=True)


class GeneratedSignal(Base):
    __tablename__ = "generated_signals"

    id = Column(String(32), primary_key=True, default=_uuid)
    session_key = Column(String(64), index=True)
    file_id = Column(String(32), ForeignKey("uploaded_files.id", ondelete="SET NULL"))
    spec = Column(JSON)
    ground_truth = Column(JSON)
    seed = Column(Integer)
    n_samples = Column(Integer)
    duration_s = Column(Float)
    created_at = Column(DateTime(timezone=True), default=_now)


class Report(Base):
    __tablename__ = "reports"

    id = Column(String(32), primary_key=True, default=_uuid)
    analysis_id = Column(String(32), ForeignKey("analysis_sessions.id", ondelete="CASCADE"), index=True)
    session_key = Column(String(64), index=True)
    title = Column(String(160))
    format = Column(String(8))                                # pdf | json | csv | txt
    path = Column(Text)
    size_bytes = Column(Integer)
    created_at = Column(DateTime(timezone=True), default=_now, index=True)

    analysis = relationship("AnalysisSession", back_populates="reports")


class Comparison(Base):
    """Signal comparison result (how similar two analysed records are, with the evidence)."""

    __tablename__ = "comparisons"

    id = Column(String(32), primary_key=True, default=_uuid)
    session_key = Column(String(64), index=True)
    file_a_id = Column(String(32), ForeignKey("uploaded_files.id", ondelete="CASCADE"))
    file_b_id = Column(String(32), ForeignKey("uploaded_files.id", ondelete="CASCADE"))
    analysis_a_id = Column(String(32))
    analysis_b_id = Column(String(32))
    verdict = Column(String(32))
    similarity = Column(Float)
    metrics = Column(JSON)
    notes = Column(JSON)
    created_at = Column(DateTime(timezone=True), default=_now, index=True)
