#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
TOTAL=$(python3 -c "from timeline import NFRAMES; print(NFRAMES)")
CHUNK=${CHUNK:-120}
OUT=build/chunks
mkdir -p "$OUT"
: > "$OUT/list.txt"
for ((s = 0; s < TOTAL; s += CHUNK)); do
  e=$((s + CHUNK < TOTAL ? s + CHUNK : TOTAL))
  f=$(printf "%s/c_%05d.mp4" "$OUT" "$s")
  if [[ ! -s "$f.done" ]]; then
    python3 render.py --start "$s" --end "$e" --raw "$OUT/raw.rgb"
    ffmpeg -loglevel error -y -f rawvideo -pix_fmt rgb24 -s 1920x1080 -r 24 -i "$OUT/raw.rgb" \
      -c:v libx264 -preset medium -crf 16 -pix_fmt yuv420p -color_primaries bt709 -color_trc bt709 \
      -colorspace bt709 -threads 2 -movflags +faststart "$f.tmp.mp4"
    rm -f "$OUT/raw.rgb"
    n=$(ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=nb_read_frames -of csv=p=0 "$f.tmp.mp4")
    if [[ "$n" != "$((e - s))" ]]; then
      echo "chunk $s: expected $((e - s)) frames, got $n" >&2
      exit 1
    fi
    mv "$f.tmp.mp4" "$f"
    echo "$n" > "$f.done"
  fi
  echo "file '$(basename "$f")'" >> "$OUT/list.txt"
done
ffmpeg -loglevel error -y -f concat -safe 0 -i "$OUT/list.txt" -c copy -movflags +faststart build/video_only.mp4
echo "joined $(ffprobe -v error -count_packets -select_streams v:0 -show_entries stream=nb_read_packets -of csv=p=0 build/video_only.mp4) frames"
