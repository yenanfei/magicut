#!/usr/bin/env bash
# Launch MagiCut cloud API (mobile / App Store backend prototype)
set -euo pipefail
cd "$(dirname "$0")/.."
export MAGICUT_PIPELINE_MODE="${MAGICUT_PIPELINE_MODE:-auto}"
export MAGICUT_API_HOST="${MAGICUT_API_HOST:-0.0.0.0}"
export MAGICUT_API_PORT="${MAGICUT_API_PORT:-8080}"
exec python3 -m uvicorn server.main:app --host "$MAGICUT_API_HOST" --port "$MAGICUT_API_PORT" --reload
