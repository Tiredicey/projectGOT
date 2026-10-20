#!/usr/bin/env bash
# Rebuild a working sandbox after a reset: system libs, Python deps, map cache, song, user clips.
set -euo pipefail
cd "$(dirname "$0")"
UP="${UPLOADS:-/home/user/uploaded_files}"
echo "[1/5] system libraries"
sudo apt-get install -y -q libegl1 libegl-mesa0 libgl1-mesa-dri >/dev/null 2>&1 || echo "apt step failed (continuing)"
sudo systemctl stop apt-daily.timer apt-daily-upgrade.timer 2>/dev/null || true
echo "[2/5] python packages"
pip install -q -r requirements.txt 2>&1 | grep -v notice || true
python3 -c "import moderngl, cv2, librosa, shapely, networkx; print('python deps ok')"
echo "[3/5] map + terrain cache"
python3 -c "import geo; geo.ensure_cache(force=True)" >/dev/null
echo "[4/5] supplied song"
mkdir -p build/song build/refs build/kanal
SONG="$UP/Official Opening Credits Game of Thrones (HBO) (1).mp3"
if [[ -f "$SONG" ]]; then cp "$SONG" build/song/theme.mp3; echo "song ok"; else echo "song MISSING: upload it again"; fi
echo "[5/5] reference videos and kanal clips"
for f in "$UP"/kanal_0*.mp4; do [[ -f "$f" ]] && ln -sf "$f" build/kanal/; done
[[ -f "$UP/Game-of-Thrones-SnapYT.App.mp4" ]] && ln -sf "$UP/Game-of-Thrones-SnapYT.App.mp4" build/refs/refA_GoT.mp4
[[ -f "$UP/Opening-Credits-SnapYT.App.mp4" ]] && ln -sf "$UP/Opening-Credits-SnapYT.App.mp4" build/refs/refB_Opening.mp4
[[ -f "$UP/Official-Opening-Credits-SnapYT.App.mp4" ]] && ln -sf "$UP/Official-Opening-Credits-SnapYT.App.mp4" build/refs/refC_Official.mp4
ls build/kanal build/refs
echo "ready. finished chunks listed in CHUNKS.md can be re-downloaded instead of re-rendered."
