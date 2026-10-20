#!/usr/bin/env bash
# Mux the supplied song onto build/video4_only.mp4 (bit-exact, no re-render).
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p build/song out
SONG_SRC="${SONG_SRC:-/home/user/uploaded_files/Official Opening Credits Game of Thrones (HBO) (1).mp3}"
cp "$SONG_SRC" build/song/theme.mp3
ffmpeg -loglevel error -y -i build/video4_only.mp4 -i build/song/theme.mp3 \
  -map 0:v -map 1:a -c:v copy -c:a copy -movflags +faststart \
  out/lipa_v4_orbital_1080p24.mp4

SRC_MD5=$(ffmpeg -v error -i build/song/theme.mp3 -map 0:a -c copy -f md5 -)
OUT_MD5=$(ffmpeg -v error -i out/lipa_v4_orbital_1080p24.mp4 -map 0:a -c copy -f md5 -)
echo "song packets  $SRC_MD5"
echo "muxed packets $OUT_MD5"
if [[ "$SRC_MD5" != "$OUT_MD5" ]]; then
  echo "audio is not bit-exact" >&2
  exit 1
fi

python3 measure.py out/lipa_v4_orbital_1080p24.mp4 --out out/measure4.json
ffmpeg -loglevel error -y -ss 94.5 -i out/lipa_v4_orbital_1080p24.mp4 -frames:v 1 out/poster4.jpg
ffmpeg -loglevel error -y -i out/lipa_v4_orbital_1080p24.mp4 \
  -vf "fps=1/3.5,scale=384:-1,tile=6x5:padding=4" -frames:v 1 out/contact4.jpg
