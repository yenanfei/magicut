#!/usr/bin/env bash
# Bootstrap MagiCut API on an Ubuntu SSH GPU host.
# Usage (on the GPU machine, from repo root):
#   bash scripts/remote_gpu_bootstrap.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "==> MagiCut GPU host bootstrap"
if [[ ! -f .env ]]; then
  cp .env.example .env
  # Prefer real mode on GPU hosts
  sed -i 's/^MAGICUT_PIPELINE_MODE=.*/MAGICUT_PIPELINE_MODE=auto/' .env
  TOKEN="$(openssl rand -hex 16 2>/dev/null || python3 -c 'import secrets;print(secrets.token_hex(16))')"
  sed -i "s/^MAGICUT_API_TOKEN=.*/MAGICUT_API_TOKEN=${TOKEN}/" .env
  echo "Wrote .env (API token generated). Edit MAGICUT_PUBLIC_HOST / CORS as needed."
fi

# shellcheck disable=SC1091
set -a
source .env
set +a

if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "==> NVIDIA GPU detected; running setup_ubuntu.sh weight/deps helpers if needed"
  bash setup_ubuntu.sh || true
else
  echo "==> No nvidia-smi; keeping MAGICUT_PIPELINE_MODE=${MAGICUT_PIPELINE_MODE:-auto} (may fall back to mock)"
fi

sudo mkdir -p /opt/magicut
sudo rsync -a --delete \
  --exclude .git --exclude server_data/uploads --exclude server_data/results --exclude server_data/jobs \
  --exclude .venv --exclude outputs --exclude weights/*.pt \
  "$ROOT/" /opt/magicut/
sudo mkdir -p /opt/magicut/server_data /opt/magicut/weights
if [[ -d "$ROOT/weights" ]]; then
  sudo rsync -a "$ROOT/weights/" /opt/magicut/weights/ || true
fi
sudo cp "$ROOT/.env" /opt/magicut/.env
sudo cp "$ROOT/deploy/magicut-api.service" /etc/systemd/system/magicut-api.service

# Recreate venv under /opt for the service user
if [[ ! -d /opt/magicut/.venv ]]; then
  sudo python3 -m venv /opt/magicut/.venv
fi
sudo /opt/magicut/.venv/bin/pip install -U pip
sudo /opt/magicut/.venv/bin/pip install -r /opt/magicut/requirements.txt

sudo systemctl daemon-reload
sudo systemctl enable magicut-api
sudo systemctl restart magicut-api
sudo systemctl --no-pager --full status magicut-api || true

echo ""
echo "API should answer on :8080. Put Caddy in front:"
echo "  sudo apt install -y caddy"
echo "  sudo cp deploy/Caddyfile /etc/caddy/Caddyfile   # set your domain"
echo "  sudo systemctl reload caddy"
echo "Health: curl -s http://127.0.0.1:8080/health"
echo "Token is in /opt/magicut/.env (MAGICUT_API_TOKEN)."
