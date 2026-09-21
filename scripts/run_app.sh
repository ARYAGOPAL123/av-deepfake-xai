#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON="$ROOT/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  echo "Creating .venv..."
  python3 -m venv .venv
  "$PYTHON" -m pip install -r requirements.txt
  "$PYTHON" -m pip install -e .
fi
export PYTHONPATH=src
URL="http://127.0.0.1:8000/"
echo "AVDF forensic app: $URL"
if command -v xdg-open >/dev/null 2>&1; then xdg-open "$URL" >/dev/null 2>&1 & fi
exec "$PYTHON" -m uvicorn api.main:app --host 127.0.0.1 --port 8000
