#!/usr/bin/env bash
#
# new-sim.sh — scaffold a new sim from templates/.
#
# Usage:
#   lab/tools/new-sim.sh <slug> [--kind web|py] [--title "Human Title"]
#
# Creates lab/sims/<slug>/ with a runnable starter + sim.json, then rebuilds the
# gallery. Slug is kebab-case; kind defaults to web.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SLUG="${1:?usage: new-sim.sh <slug> [--kind web|py] [--title \"...\"]}"
shift
KIND="web"
TITLE="$SLUG"

while [ $# -gt 0 ]; do
  case "$1" in
    --kind)  KIND="$2"; shift 2 ;;
    --title) TITLE="$2"; shift 2 ;;
    *) echo "unknown option: $1" >&2; exit 1 ;;
  esac
done

case "$KIND" in web|py) ;; *) echo "kind must be web or py" >&2; exit 1 ;; esac
[[ "$SLUG" =~ ^[a-z0-9][a-z0-9-]*$ ]] || { echo "slug must be kebab-case" >&2; exit 1; }

DEST="$ROOT/sims/$SLUG"
[ -e "$DEST" ] && { echo "lab/sims/$SLUG already exists" >&2; exit 1; }

cp -R "$ROOT/templates/$KIND" "$DEST"
DATE="$(date +%Y-%m-%d)"
for f in "$DEST"/*; do
  [ -f "$f" ] || continue
  sed -i.bak -e "s/__SLUG__/$SLUG/g" -e "s/__TITLE__/$TITLE/g" -e "s/__DATE__/$DATE/g" "$f"
  rm -f "$f.bak"
done

python3 "$ROOT/tools/build-gallery.py"
echo "created lab/sims/$SLUG (kind=$KIND)"
[ "$KIND" = web ] && echo "view: lab/tools/serve.sh then http://localhost:8000/lab/sims/$SLUG/"
[ "$KIND" = py ] && echo "run:  uv run lab/sims/$SLUG/main.py"
