#!/bin/bash
set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"

# Start Vite dev server in background (accessible on LAN via 0.0.0.0:5173)
echo "[start] Vite → http://$(ipconfig getifaddr en0 2>/dev/null || hostname -I | awk '{print $1}'):5173"
(cd "$ROOT/art/web" && npm run dev --) &
VITE_PID=$!

# Kill Vite when this script exits
trap "kill $VITE_PID 2>/dev/null; wait $VITE_PID 2>/dev/null" EXIT INT TERM

# Start TUI with WebSocket bridge in foreground
cd "$ROOT"
uv run python scripts/tui.py --resume --web "$@"
