#!/usr/bin/env bash
# Launch Inkasso automatically when its GitHub Codespace starts.
set -euo pipefail
cd "$(dirname "$0")/.."

PID_FILE="${TMPDIR:-/tmp}/inkasso-streamlit.pid"
LOG_FILE="${TMPDIR:-/tmp}/inkasso-streamlit.log"

if [[ -f "$PID_FILE" ]]; then
  current_pid="$(cat "$PID_FILE")"
  if [[ "$current_pid" =~ ^[0-9]+$ ]] && kill -0 "$current_pid" 2>/dev/null; then
    echo "Inkasso dashboard already running (PID $current_pid)."
    exit 0
  fi
fi

# If a Streamlit server is already healthy on our port, avoid a collision.
if python - <<'PY' >/dev/null 2>&1
from urllib.request import urlopen
with urlopen("http://127.0.0.1:8501/_stcore/health", timeout=2) as r:
    assert r.status == 200
PY
then
  echo "Inkasso dashboard already available on port 8501."
  exit 0
fi

nohup python -m streamlit run app.py \
  --server.address 0.0.0.0 \
  --server.port 8501 \
  --server.headless true >"$LOG_FILE" 2>&1 < /dev/null &
echo "$!" >"$PID_FILE"
echo "Starting Inkasso dashboard on port 8501 (PID $(cat "$PID_FILE"))."
echo "Dashboard log: $LOG_FILE"
