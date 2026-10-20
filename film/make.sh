#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p build/song out

SONG_SRC="${SONG_SRC:-/home/user/uploaded_files/Official Opening Credits Game of Thrones (HBO) (1).mp3}"
test -f "$SONG_SRC"

python3 -c "import geo; geo.ensure_cache(force=True)"
bash render_chunks.sh
SONG_SRC="$SONG_SRC" bash mux.sh
