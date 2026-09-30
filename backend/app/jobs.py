"""Background job runner: progress, cancellation and result hand-off.

The platform runs the heavy DSP work in worker threads so the HTTP API stays responsive.  Each job
has a cancellable flag that the analysis pipeline polls, and a progress fraction/text pair that the
UI polls to show a real progress bar (never a fake one).
"""
from __future__ import annotations

import threading
import time
import traceback
import uuid
from collections import OrderedDict
from typing import Any, Callable

from .config import settings


class Job:
    def __init__(self, kind: str, params: dict | None = None) -> None:
        self.id = uuid.uuid4().hex[:16]
        self.kind = kind
        self.params = params or {}
        self.status = "queued"          # queued | running | done | failed | cancelled
        self.progress = 0.0
        self.message = "queued"
        self.result: dict | None = None
        self.error: str | None = None
        self.traceback: str | None = None
        self.created_at = time.time()
        self.started_at: float | None = None
        self.finished_at: float | None = None
        self._cancel = threading.Event()
        self._lock = threading.Lock()

    # -- control -----------------------------------------------------------------------
    def cancel(self) -> None:
        self._cancel.set()

    def cancelled(self) -> bool:
        return self._cancel.is_set()

    def set_progress(self, progress: float, message: str | None = None) -> None:
        with self._lock:
            self.progress = float(max(0.0, min(1.0, progress)))
            if message:
                self.message = str(message)[:200]

    def as_dict(self, include_result: bool = False) -> dict:
        out = {
            "job_id": self.id, "kind": self.kind, "status": self.status,
            "progress": round(self.progress, 4), "message": self.message,
            "error": self.error, "created_at": self.created_at,
            "started_at": self.started_at, "finished_at": self.finished_at,
            "duration_s": (round((self.finished_at or time.time()) -
                                 (self.started_at or self.created_at), 3)),
            "params": self.params,
        }
        if include_result and self.result is not None:
            out["result"] = self.result
        return out


class JobManager:
    def __init__(self, workers: int | None = None, history: int | None = None) -> None:
        self.workers = int(workers or settings.job_workers)
        self.history = int(history or settings.job_history)
        self._jobs: "OrderedDict[str, Job]" = OrderedDict()
        self._lock = threading.Lock()
        self._queue: list[Job] = []
        self._cv = threading.Condition(self._lock)
        self._threads: list[threading.Thread] = []
        self._stopped = False

    # -- lifecycle ---------------------------------------------------------------------
    def start(self) -> None:
        with self._lock:
            if self._threads:
                return
            for i in range(max(1, self.workers)):
                t = threading.Thread(target=self._worker, name=f"sih-worker-{i}", daemon=True)
                t.start()
                self._threads.append(t)

    def stop(self) -> None:
        with self._cv:
            self._stopped = True
            self._cv.notify_all()

    # -- api ---------------------------------------------------------------------------
    def submit(self, kind: str, fn: Callable[[Job], Any], params: dict | None = None) -> Job:
        job = Job(kind, params)
        job._fn = fn                                                       # type: ignore[attr-defined]
        with self._cv:
            self._jobs[job.id] = job
            while len(self._jobs) > self.history:
                self._jobs.popitem(last=False)
            self._queue.append(job)
            self._cv.notify()
        self.start()
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def cancel(self, job_id: str) -> Job | None:
        job = self.get(job_id)
        if job:
            job.cancel()
            if job.status == "queued":
                job.status = "cancelled"
                job.message = "cancelled before start"
                job.finished_at = time.time()
        return job

    def list(self, limit: int = 25) -> list[dict]:
        with self._lock:
            jobs = list(self._jobs.values())[-limit:]
        return [j.as_dict() for j in reversed(jobs)]

    def active(self) -> int:
        with self._lock:
            return sum(1 for j in self._jobs.values() if j.status in ("queued", "running"))

    # -- worker ------------------------------------------------------------------------
    def _worker(self) -> None:
        while True:
            with self._cv:
                while not self._queue and not self._stopped:
                    self._cv.wait(timeout=1.0)
                if self._stopped:
                    return
                job = self._queue.pop(0)
            if job.status == "cancelled":
                continue
            job.status = "running"
            job.started_at = time.time()
            job.set_progress(0.01, "starting")
            try:
                job.result = job._fn(job)                                  # type: ignore[attr-defined]
                job.status = "cancelled" if job.cancelled() else "done"
                if job.status == "cancelled":
                    job.message = "cancelled"
                else:
                    job.set_progress(1.0, "done")
            except Exception as exc:                                       # pragma: no cover
                job.status = "failed"
                job.error = f"{type(exc).__name__}: {exc}"
                job.traceback = traceback.format_exc(limit=6)
                job.message = job.error
            finally:
                job.finished_at = time.time()


job_manager = JobManager()
