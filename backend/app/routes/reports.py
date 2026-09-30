"""Report generation and download (PDF / JSON / CSV / TXT)."""
from __future__ import annotations

import datetime as _dt
import os
import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..config import settings
from ..db import get_db
from ..models import AnalysisSession, Report, UploadedFile
from ..reports import FMT_MIME, build_report
from ..storage import cache_read
from .common import require_analysis, session_key

router = APIRouter(tags=["reports"])


def _meta(row: AnalysisSession, db: Session) -> dict:
    f = db.get(UploadedFile, row.file_id) if row.file_id else None
    return {"title": f"Analysis report - {f.filename if f else row.id}",
            "analysis_id": row.id, "file_id": row.file_id,
            "filename": f.filename if f else None,
            "created_at": row.created_at.isoformat() if row.created_at else None,
            "duration_ms": row.duration_ms, "mode": row.mode, "kind": row.kind}


def _safe_stem(row: AnalysisSession, f: UploadedFile | None) -> str:
    name = (f.filename if f else "signal") or "signal"
    stem = os.path.splitext(os.path.basename(name))[0]
    stem = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in stem)[:60]
    return f"{stem}_{row.id[:8]}"


class ReportRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    formats: list[str] = Field(default_factory=lambda: ["pdf", "json", "csv", "txt"])


@router.post("/analysis/{analysis_id}/report", summary="Build and store report files")
def make_report(analysis_id: str, req: ReportRequest | None = None, key: str = Depends(session_key),
                db: Session = Depends(get_db)) -> dict:
    row = require_analysis(db, analysis_id, key)
    if row.status != "done":
        raise HTTPException(status_code=409, detail=f"the analysis is '{row.status}'; there is "
                                                    "nothing to report yet")
    result = cache_read(row.cache_path)
    if result is None:
        raise HTTPException(status_code=410, detail="the cached analysis result is no longer available")
    result["_analysis_id"] = row.id
    f = db.get(UploadedFile, row.file_id) if row.file_id else None
    meta = _meta(row, db)
    formats = (req.formats if req else ["pdf", "json", "csv", "txt"]) or ["pdf"]
    folder = os.path.join(settings.report_dir, key)
    os.makedirs(folder, exist_ok=True)
    out = []
    errors = []
    for fmt in formats:
        try:
            payload, media, ext = build_report(result, fmt, meta)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        except Exception as exc:                                              # pragma: no cover
            errors.append(f"{fmt}: {type(exc).__name__}: {exc}")
            continue
        name = f"{_safe_stem(row, f)}.{ext}"
        path = os.path.join(folder, f"{uuid.uuid4().hex[:8]}__{name}")
        with open(path, "wb") as fh:
            fh.write(payload)
        rep = Report(analysis_id=row.id, session_key=key, title=name, format=fmt, path=path,
                     size_bytes=len(payload))
        db.add(rep)
        db.commit()
        db.refresh(rep)
        out.append({"report_id": rep.id, "format": fmt, "filename": name,
                    "size_bytes": len(payload), "media_type": media,
                    "download_url": f"/api/report/{rep.id}/download"})
    return {"analysis_id": row.id, "reports": out, "errors": errors,
            "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            "note": "reports are built from the stored analysis result of this analysis id"}


@router.get("/analysis/{analysis_id}/report", summary="Download a report built on the fly")
def report_inline(analysis_id: str, format: str = Query("pdf", pattern="^(pdf|json|csv|txt)$"),
                  key: str = Depends(session_key), db: Session = Depends(get_db)) -> Response:
    row = require_analysis(db, analysis_id, key)
    result = cache_read(row.cache_path)
    if row.status != "done" or result is None:
        raise HTTPException(status_code=409, detail="no completed, cached analysis result for this id")
    result["_analysis_id"] = row.id
    f = db.get(UploadedFile, row.file_id) if row.file_id else None
    try:
        payload, media, ext = build_report(result, format, _meta(row, db))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"report generation failed: {exc}") from exc
    name = f"{_safe_stem(row, f)}.{ext}"
    return Response(content=payload, media_type=media,
                    headers={"Content-Disposition": f'attachment; filename="{name}"'})


@router.post("/reports/{fmt}", summary="Alias: build one report format for an analysis and download it")
def report_format_alias(fmt: str, analysis_id: str = Query(...),
                        key: str = Depends(session_key), db: Session = Depends(get_db)) -> Response:
    if fmt not in FMT_MIME:
        raise HTTPException(status_code=422, detail=f"unsupported format '{fmt}' (pdf, json, csv, txt)")
    return report_inline(analysis_id, format=fmt, key=key, db=db)


@router.get("/reports", summary="Stored reports for this session")
def list_reports(analysis_id: str | None = None, limit: int = Query(50, ge=1, le=200),
                 key: str = Depends(session_key), db: Session = Depends(get_db)) -> dict:
    stmt = select(Report).where(Report.session_key == key)
    if analysis_id:
        stmt = stmt.where(Report.analysis_id == analysis_id)
    rows = db.scalars(stmt.order_by(Report.created_at.desc()).limit(limit)).all()
    out = []
    for r in rows:
        a = db.get(AnalysisSession, r.analysis_id) if r.analysis_id else None
        f = db.get(UploadedFile, a.file_id) if a and a.file_id else None
        out.append({"report_id": r.id, "analysis_id": r.analysis_id, "title": r.title,
                    "format": r.format, "size_bytes": r.size_bytes,
                    "exists": bool(r.path and os.path.exists(r.path)),
                    "filename": f.filename if f else None,
                    "created_at": r.created_at.isoformat() if r.created_at else None,
                    "download_url": f"/api/report/{r.id}/download"})
    return {"reports": out, "count": len(out)}


@router.get("/report/{report_id}/download", summary="Download a stored report")
def download(report_id: str, key: str = Depends(session_key), db: Session = Depends(get_db)):
    row = db.get(Report, report_id)
    if row is None or row.session_key != key:
        raise HTTPException(status_code=404, detail="report not found in this session")
    if not row.path or not os.path.exists(row.path):
        raise HTTPException(status_code=410, detail="the report file has been removed from disk")
    return FileResponse(row.path, filename=row.title or f"report.{row.format}",
                        media_type=FMT_MIME.get(row.format, "application/octet-stream"))
