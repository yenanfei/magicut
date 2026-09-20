"""Runtime configuration for the MagiCut API server."""

from __future__ import annotations

import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("MAGICUT_DATA_DIR", ROOT_DIR / "server_data"))
UPLOAD_DIR = DATA_DIR / "uploads"
RESULT_DIR = DATA_DIR / "results"
JOBS_DIR = DATA_DIR / "jobs"
CONFIG_PATH = Path(os.environ.get("MAGICUT_CONFIG", ROOT_DIR / "configs" / "config.yaml"))

# auto | mock | real
PIPELINE_MODE = os.environ.get("MAGICUT_PIPELINE_MODE", "auto").lower()
API_HOST = os.environ.get("MAGICUT_API_HOST", "0.0.0.0")
API_PORT = int(os.environ.get("MAGICUT_API_PORT", "8080"))
API_TOKEN = os.environ.get("MAGICUT_API_TOKEN", "")  # empty = open for local prototype
MAX_UPLOAD_MB = int(os.environ.get("MAGICUT_MAX_UPLOAD_MB", "200"))
DEFAULT_MAX_FRAMES = int(os.environ.get("MAGICUT_DEFAULT_MAX_FRAMES", "60"))
CORS_ORIGINS = [
    o.strip()
    for o in os.environ.get("MAGICUT_CORS_ORIGINS", "*").split(",")
    if o.strip()
]


def ensure_dirs() -> None:
    for path in (DATA_DIR, UPLOAD_DIR, RESULT_DIR, JOBS_DIR):
        path.mkdir(parents=True, exist_ok=True)
