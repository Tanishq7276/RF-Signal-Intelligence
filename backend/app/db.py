"""Database engine, session handling and a few query helpers."""
from __future__ import annotations

import contextlib
from typing import Iterator

from sqlalchemy import create_engine, event, func, select
from sqlalchemy.orm import Session, sessionmaker

from .config import settings
from .models import (AnalysisSession, Base, BenchmarkResult, Bitstream, Comparison,
                     DemodulationResult, FECHypothesis, GeneratedSignal,
                     InterleavingHypothesis, ModulationResult, Report, SignalParameters,
                     SignalSegment, UploadedFile)

_connect_args = {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}
engine = create_engine(settings.database_url, future=True, pool_pre_ping=True,
                       connect_args=_connect_args)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False,
                            future=True)


@event.listens_for(engine, "connect")
def _sqlite_pragmas(dbapi_connection, _record):          # pragma: no cover - sqlite only
    if not settings.database_url.startswith("sqlite"):
        return
    cur = dbapi_connection.cursor()
    try:
        cur.execute("PRAGMA foreign_keys=ON")
        cur.execute("PRAGMA journal_mode=WAL")
        cur.execute("PRAGMA synchronous=NORMAL")
    finally:
        cur.close()


def init_db() -> None:
    Base.metadata.create_all(bind=engine)


@contextlib.contextmanager
def session_scope() -> Iterator[Session]:
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def get_db() -> Iterator[Session]:
    """FastAPI dependency."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def counts(db: Session) -> dict:
    out = {}
    for model in (UploadedFile, AnalysisSession, SignalSegment, SignalParameters, ModulationResult,
                  DemodulationResult, FECHypothesis, InterleavingHypothesis, Bitstream,
                  Report, BenchmarkResult, GeneratedSignal, Comparison):
        out[model.__tablename__] = int(db.scalar(select(func.count()).select_from(model)) or 0)
    return out
