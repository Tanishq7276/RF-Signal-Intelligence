"""Application configuration.

Everything is environment-overridable so the same image runs in development, in a container and
behind a reverse proxy.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.environ.get("SIH_DATA_DIR", os.path.join(BASE_DIR, "data"))


def _flag(name: str, default: str = "1") -> bool:
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class Settings:
    app_name: str = "RF Signal Intelligence Platform"
    problem_statement: str = "SIH26147 - Automated Model for Analysis of .IQ and .WAV Files Along with Signal Parameter Extraction"
    version: str = "1.0.0"
    api_prefix: str = "/api"

    data_dir: str = DATA_DIR
    upload_dir: str = field(default_factory=lambda: os.path.join(DATA_DIR, "uploads"))
    generated_dir: str = field(default_factory=lambda: os.path.join(DATA_DIR, "generated"))
    report_dir: str = field(default_factory=lambda: os.path.join(DATA_DIR, "reports"))
    cache_dir: str = field(default_factory=lambda: os.path.join(DATA_DIR, "cache"))
    sample_dir: str = field(default_factory=lambda: os.path.join(BASE_DIR, "sample_data"))
    frontend_dist: str = field(default_factory=lambda: os.environ.get(
        "SIH_FRONTEND_DIST", os.path.join(BASE_DIR, "frontend", "dist")))
    docs_dir: str = field(default_factory=lambda: os.environ.get(
        "SIH_DOCS_DIR", os.path.join(BASE_DIR, "docs")))
    docs_url_path: str = os.environ.get("SIH_DOCS_URL", "/documentation")

    database_url: str = field(default_factory=lambda: os.environ.get(
        "SIH_DATABASE_URL", f"sqlite:///{os.path.join(DATA_DIR, 'app.db')}"))

    # upload limits
    max_upload_bytes: int = int(os.environ.get("SIH_MAX_UPLOAD_BYTES", str(512 * 1024 * 1024)))
    allowed_extensions: tuple = (".iq", ".wav", ".npy", ".bin", ".dat", ".cfile", ".cf32", ".cs16",
                                ".cs8", ".cu8", ".float", ".f32", ".sc16", ".s16", ".raw", ".complex")
    max_samples_analyse: int = int(os.environ.get("SIH_MAX_SAMPLES_ANALYSE", str(12_000_000)))
    max_samples_display: int = int(os.environ.get("SIH_MAX_SAMPLES", str(4_000_000)))

    # job runner
    job_workers: int = int(os.environ.get("SIH_JOB_WORKERS", "2"))
    job_history: int = 200

    # security
    session_header: str = "x-session-id"
    cors_origins: tuple = tuple(o for o in os.environ.get(
        "SIH_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173,http://localhost:8000").split(",") if o)
    trust_proxy: bool = _flag("SIH_TRUST_PROXY", "1")

    def ensure_dirs(self) -> None:
        for d in (self.data_dir, self.upload_dir, self.generated_dir, self.report_dir,
                  self.cache_dir):
            os.makedirs(d, exist_ok=True)


settings = Settings()
settings.ensure_dirs()

#: extensions that are never accepted (executables / scripts) - defence in depth on top of the
#: allow-list, because the platform must never execute anything that was uploaded
DANGEROUS_EXTENSIONS = (
    ".exe", ".dll", ".so", ".bat", ".cmd", ".com", ".sh", ".ps1", ".py", ".pyc", ".js", ".jar",
    ".msi", ".scr", ".php", ".pl", ".rb", ".html", ".htm", ".svg", ".vbs", ".apk", ".dmg",
)
