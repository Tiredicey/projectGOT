#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
TOTAL=$(python3 -c "from timeline4 import NFRAMES; print(NFRAMES)")
CHUNK=${CHUNK:-120}
OUT=build/chunks4
LEDGER=records/chunks4.tsv
mkdir -p "$OUT" records
touch "$LEDGER"
restore() {
  local name=$1 url
  url=$(awk -v n="$name" '$1==n {print $2}' "$LEDGER" | tail -1)
  [[ -z "$url" ]] && return 1
  local dl
  dl=$(gsk download "$url" 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('data',{}).get('download_url') or d.get('download_url') or '')" 2>/dev/null || true)
  [[ -z "$dl" ]] && return 1
  curl -sfL "$dl" -o "$OUT/$name.tmp" || return 1
  mv "$OUT/$name.tmp" "$OUT/$name"
}
: > "$OUT/list.txt"
for ((s = 0; s < TOTAL; s += CHUNK)); do
  e=$((s + CHUNK < TOTAL ? s + CHUNK : TOTAL))
  name=$(printf "c_%05d.mp4" "$s")
  f="$OUT/$name"
  if [[ ! -s "$f.done" ]] && restore "$name"; then
    n=$(ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=nb_read_frames -of csv=p=0 "$f" || echo 0)
    if [[ "$n" == "$((e - s))" ]]; then echo "$n" > "$f.done"; echo "restored $name"; else rm -f "$f"; fi
  fi
  if [[ ! -s "$f.done" ]]; then
    python3 render4.py --start "$s" --end "$e" --raw "$OUT/raw.rgb"
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
    url=$(gsk upload "$f" 2>/dev/null | python3 -c "import sys,json; print(json.load(sys.stdin)['data']['file_wrapper_url'])" || true)
    if [[ -n "$url" ]]; then
      printf "%s\t%s\t%s\n" "$name" "$url" "$(git rev-parse --short HEAD)" >> "$LEDGER"
      (cd .. && git add "film/$LEDGER" && git commit -qm "v4 chunk $name uploaded" && git push -q origin HEAD:main) || echo "git push of ledger failed (kept locally)"
    fi
  fi
  echo "file '$name'" >> "$OUT/list.txt"
done
ffmpeg -loglevel error -y -f concat -safe 0 -i "$OUT/list.txt" -c copy -movflags +faststart build/video4_only.mp4
echo "joined $(ffprobe -v error -count_packets -select_streams v:0 -show_entries stream=nb_read_packets -of csv=p=0 build/video4_only.mp4) frames"
