#!/usr/bin/env bash
# Serve the repo root so sims can import lib/ and vendor/ with relative paths.
# Usage: tools/serve.sh [port]   (default 8000; gallery at /gallery/)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
exec python3 -m http.server "${1:-8000}"
