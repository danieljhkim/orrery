#!/usr/bin/env bash
# Serve the repo root so lab/sims can import lab/lib and lab/vendor relatively.
# Usage: lab/tools/serve.sh [port]   (default 8000; gallery at /lab/gallery/)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
exec python3 -m http.server "${1:-8000}"
