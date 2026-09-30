"""Documentation API + static index for the repository's ``docs/`` folder.

Why two layers?

* ``/documentation`` (static mount in :mod:`backend.app.main`) serves the markdown files exactly as
  they are stored in the repository, so what a judge reads in the app is byte-identical to what is
  in the repository;
* ``/api/docs`` returns the same content as JSON for programmatic use (and a document list with
  titles and one-line summaries for the UI navigation);
* :func:`sync_index` writes ``docs/index.json`` at start-up from the markdown files themselves.
  The index is therefore *derived* state: it can never drift away from the documents, and a missing
  index is repaired the next time the API starts.

Only files that sit directly inside the documentation root and carry a markdown-ish suffix are ever
served: no path separators, no ``..``, no symlink escapes.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import re
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query

from ..config import settings
from ..views import api_view

router = APIRouter()

DOC_SUFFIXES = (".md", ".markdown", ".txt")
INDEX_NAME = "index.json"
SUMMARY_RE = re.compile(r"<!--\s*summary:\s*(.*?)\s*-->", re.IGNORECASE)
MAX_BYTES = 512 * 1024


def docs_root() -> Path:
    """Documentation root: ``SIH_DOCS_DIR`` when set, otherwise ``<repo>/docs``."""
    configured = os.environ.get("SIH_DOCS_DIR") or getattr(settings, "docs_dir", "")
    if configured:
        return Path(configured).resolve()
    return (Path(settings.data_dir).resolve().parent / "docs").resolve()


def _title_and_summary(path: Path) -> tuple[str, str]:
    """Title = first markdown heading; summary = ``<!-- summary: ... -->`` or first prose line."""
    title = path.stem.replace("_", " ").replace("-", " ").strip()
    summary = ""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:                                                       # pragma: no cover
        return title, summary
    default_title = path.stem.replace("_", " ").replace("-", " ").strip()
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if not summary:
            m = SUMMARY_RE.search(stripped)          # <!-- summary: one line --> wins if present
            if m:
                summary = m.group(1)
        if title == default_title and stripped.startswith("#"):
            title = stripped.lstrip("#").strip() or title
        if (not summary and not stripped.startswith(("#", ">", "|", "```", "-", "*", "!"))
                and ":" in stripped):
            summary = re.sub(r"[*`_]", "", stripped)[:220]
        if summary and title != default_title:
            break
    return title, summary


def build_index() -> dict:
    """Describe every servable document (name, id, title, summary, size, mtime)."""
    root = docs_root()
    documents: list[dict] = []
    if root.is_dir():
        for entry in sorted(root.iterdir(), key=lambda p: p.name.lower()):
            if not entry.is_file() or entry.suffix.lower() not in DOC_SUFFIXES:
                continue
            try:
                stat = entry.stat()
            except OSError:                                               # pragma: no cover
                continue
            title, summary = _title_and_summary(entry)
            documents.append({"name": entry.stem, "file": entry.name, "id": entry.stem,
                              "title": title, "summary": summary, "size_bytes": stat.st_size,
                              "modified": _dt.datetime.fromtimestamp(stat.st_mtime).isoformat(
                                  timespec="seconds")})
    return {"root": str(root), "generated": _dt.datetime.now().isoformat(timespec="seconds"),
            "count": len(documents), "documents": documents}


def sync_index() -> Path | None:
    """Write ``docs/index.json`` from the markdown files.  Never fatal, never load-bearing."""
    root = docs_root()
    if not root.is_dir():
        return None
    path = root / INDEX_NAME
    payload = build_index()
    payload["note"] = ("generated from the markdown files in this folder at API start-up; the "
                       "documents are the source of truth, this index only lists them")
    payload["read_by"] = {"ui": "/documentation/index.json", "api": "/api/docs"}
    try:
        if path.exists():
            try:
                if json.loads(path.read_text(encoding="utf-8")).get("count") == payload["count"] and \
                        payload["count"] == 0:
                    return path
            except Exception:
                pass
        path.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    except OSError:                                                       # pragma: no cover
        return None
    return path


def _resolve(doc_id: str) -> Path:
    if not doc_id or "/" in doc_id or "\\" in doc_id or ".." in doc_id:
        raise HTTPException(status_code=400, detail="invalid document id")
    root = docs_root()
    for doc in build_index()["documents"]:
        if doc_id in (doc["id"], doc["name"], doc["file"]):
            path = (root / doc["file"]).resolve()
            if root not in path.parents and path.parent != root:
                raise HTTPException(status_code=400, detail="document outside the docs root")
            return path
    raise HTTPException(status_code=404, detail=f"no document named {doc_id!r} is served by this API")


@router.get("/docs", summary="List the project documentation")
@api_view
def list_docs() -> dict:
    idx = build_index()
    return {"ok": True, "count": idx["count"], "documents": idx["documents"], "root": idx["root"],
            "static_mount": "/documentation", "message": None if idx["count"] else
            ("no documentation files were found: the docs/ folder is empty or was not mounted into "
             "the container (set SIH_DOCS_DIR to point at it)")}


@router.get("/docs/{doc_id:int}", include_in_schema=False)
@router.get("/docs/{doc_id}", summary="One documentation file as markdown")
@api_view
def get_doc(doc_id: str, max_chars: int = Query(200000, ge=1000, le=1000000)) -> dict:
    path = _resolve(str(doc_id))
    try:
        if path.stat().st_size > MAX_BYTES:
            raise HTTPException(status_code=413, detail="the document is too large to serve")
        text = path.read_text(encoding="utf-8", errors="replace")
    except HTTPException:
        raise
    except OSError as exc:                                                # pragma: no cover
        raise HTTPException(status_code=500, detail=f"the document could not be read: {exc}") from exc
    title, summary = _title_and_summary(path)
    return {"ok": True, "id": path.stem, "name": path.name, "title": title, "summary": summary,
            "markdown": text[:max_chars], "truncated": len(text) > max_chars, "n_chars": len(text)}
