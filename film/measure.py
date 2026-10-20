import argparse
import hashlib
import json
import logging
import re
import subprocess
import sys
from pathlib import Path

import cv2
import librosa
import numpy as np

log = logging.getLogger("measure")


def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def probe(path):
    r = run(["ffprobe", "-v", "error", "-show_entries",
             "stream=codec_type,codec_name,width,height,r_frame_rate,pix_fmt,sample_rate,channels,nb_frames:format=duration,bit_rate",
             "-of", "json", path])
    return json.loads(r.stdout)


def scene_cuts(path, thr=0.3):
    r = run(["ffmpeg", "-hide_banner", "-i", path, "-vf", f"select='gt(scene,{thr})',showinfo", "-an", "-f", "null", "-"])
    return [float(m) for m in re.findall(r"pts_time:([0-9.]+)", r.stderr)]


def frame_stats(path, step=1):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24
    prev, diffs, lum = None, [], []
    i = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if i % step == 0:
            g = cv2.cvtColor(cv2.resize(fr, (320, 180)), cv2.COLOR_BGR2GRAY).astype(np.float32)
            lin = (g / 255.0) ** 2.2
            lum.append(float(lin.mean()))
            if prev is not None:
                diffs.append(float(np.abs(g - prev).mean()))
            prev = g
        i += 1
    cap.release()
    return fps, np.array(diffs), np.array(lum)


def flash_count(lum, fps):
    d = np.diff(lum)
    big = np.abs(d) >= 0.1
    worst = 0
    w = int(round(fps))
    for s in range(0, max(1, len(d) - w)):
        seg = d[s:s + w][big[s:s + w]]
        if len(seg) < 2:
            continue
        flips = int(np.sum(np.sign(seg[1:]) != np.sign(seg[:-1])))
        worst = max(worst, (flips + 1) // 2)
    return worst


def loudness(path):
    r = run(["ffmpeg", "-hide_banner", "-nostats", "-i", path, "-filter_complex", "ebur128=peak=true", "-f", "null", "-"])
    tail = r.stderr[r.stderr.rfind("Summary:"):]
    get = lambda k: float(re.search(k + r":\s+(-?[0-9.]+)", tail).group(1))
    return {"integrated_lufs": get("I"), "lra_lu": get("LRA"), "true_peak_dbtp": get("Peak")}


def audio_md5(path):
    r = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-map", "0:a", "-f", "md5", "-"], capture_output=True, text=True)
    return r.stdout.strip()


def tempo_onsets(path):
    y, sr = librosa.load(path, sr=22050, mono=True)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    on = librosa.onset.onset_detect(y=y, sr=sr, units="time")
    env = librosa.onset.onset_strength(y=y, sr=sr)
    return float(np.atleast_1d(tempo)[0]), on, env, librosa.times_like(env, sr=sr)


def main():
    ap = argparse.ArgumentParser(description="M1 teardown measurements for a video file.")
    ap.add_argument("video")
    ap.add_argument("--cues", help="cue_sheet.json to check hit sync against onsets")
    ap.add_argument("--out", default="build/measure.json")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    res = {"file": args.video, "sha256": hashlib.sha256(Path(args.video).read_bytes()).hexdigest()}
    res["probe"] = probe(args.video)
    dur = float(res["probe"]["format"]["duration"])
    cuts = scene_cuts(args.video)
    res["scene_cuts_gt0.3"] = cuts
    res["cuts_per_minute"] = round(len(cuts) / dur * 60, 2)
    fps, diffs, lum = frame_stats(args.video)
    res["near_still_share"] = round(float(np.mean(diffs < 0.5)), 3)
    res["mean_frame_diff"] = round(float(diffs.mean()), 3)
    res["general_flashes_max_per_second"] = flash_count(lum, fps)
    res["loudness"] = loudness(args.video)
    res["audio_md5"] = audio_md5(args.video)
    tempo, on, env, et = tempo_onsets(args.video)
    res["tempo_bpm_librosa"] = round(tempo, 2)
    res["onset_count"] = int(len(on))
    if args.cues:
        cues = json.load(open(args.cues))["cues"]
        errs = []
        for c in cues:
            if c["type"] == "taiko":
                continue
            m = (et > c["t"] - 0.25) & (et < c["t"] + 0.25)
            peak_t = float(et[m][np.argmax(env[m])])
            errs.append({"cue": c, "onset_peak_offset_ms": round((peak_t - c["t"]) * 1000, 1),
                         "peak_vs_median": round(float(env[m].max() / np.median(env)), 2)})
        res["hit_sync"] = errs
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k not in ("scene_cuts_gt0.3", "probe")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
