#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""retry_failed_h264.py — for videos that failed precompute (not H264/HEVC encoded), transcode to H.264 first, then recompute visidx.

Background: cv_reader only supports H264/HEVC; in benchmarks like mvbench some mp4s are
MPEG-4 etc., raising "Only support H264/HEVC video, aborting". This script transcodes these
videos missing visidx to H.264 using the system ffmpeg (content unchanged, re-encoded only to
produce readable motion vectors/residuals), then saves visidx under the original key.

The t dimension of visidx is the index of the "t-th sampled frame" (0..num_frames-1), independent
of the absolute frame numbers of the original/transcoded video; as long as both uniformly sample
num_frames frames they align, so the tiny total-frame drift from transcoding does not affect the evaluator.

Usage (grid parameters must exactly match the main preprocessing and evaluation):
  python scripts/precompute_codec_visidx/retry_failed_h264.py \
    --jsonl   codec_visidx_cache/mvbench_videos.jsonl \
    --out_root codec_visidx_cache/mvbench \
    --num_video_frames 64 --segment_t_size 16 --image_size 224 --patch_size 14 --keep_ratio 0.5
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from precompute_video_visidx import compute_visidx_for_video, _HAS_CV_READER  # noqa: E402

FFMPEG = os.environ.get("FFMPEG_BIN", "ffmpeg")


def _transcode_h264(src: str, dst: str) -> bool:
    """Re-encode src to H.264 (preserving frame rate/size); returns True on success."""
    cmd = [
        FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
        "-i", src,
        # libx264(yuv420p) requires even width and height; some videos have odd sizes (e.g. 500x375)
        # and raise "height not divisible by 2"; use scale to round down to even (≤1px, later resized to a square)
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-pix_fmt", "yuv420p", "-an",
        dst,
    ]
    try:
        r = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        if r.returncode != 0:
            sys.stderr.write(f"[ffmpeg-fail] {Path(src).name}: {r.stderr.decode(errors='ignore')[:160]}\n")
            return False
        return os.path.exists(dst) and os.path.getsize(dst) > 0
    except Exception as e:
        sys.stderr.write(f"[ffmpeg-exc] {Path(src).name}: {e}\n")
        return False


def main():
    ap = argparse.ArgumentParser("Retry failed videos via H.264 transcode")
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--num_video_frames", type=int, default=64)
    ap.add_argument("--segment_t_size", type=int, default=16)
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--patch_size", type=int, default=14)
    ap.add_argument("--keep_ratio", type=float, default=0.5)
    ap.add_argument("--keep_first_per_segment", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--tmp_dir", type=str, default=None, help="temporary directory for transcoding (default: system tmp)")
    args = ap.parse_args()

    if not _HAS_CV_READER:
        print("[FATAL] cv_reader not available", file=sys.stderr); sys.exit(2)

    jobs = [json.loads(l) for l in open(args.jsonl) if l.strip()]
    out_root = Path(args.out_root)
    missing = [j for j in jobs if not (out_root / j["key"] / "visidx.npy").exists()]
    print(f"[INFO] total={len(jobs)} missing={len(missing)}")
    if not missing:
        print("[DONE] no missing visidx, nothing to recompute"); return

    tmp_root = args.tmp_dir or tempfile.mkdtemp(prefix="codec_retry_")
    os.makedirs(tmp_root, exist_ok=True)

    t0 = time.time(); ok = fail = 0
    for idx, j in enumerate(missing):
        key, src = str(j["key"]), str(j["video"])
        tmp_mp4 = os.path.join(tmp_root, f"{key}.h264.mp4")
        try:
            if not _transcode_h264(src, tmp_mp4):
                fail += 1; continue
            visidx, frame_ids, meta = compute_visidx_for_video(
                tmp_mp4,
                num_frames=args.num_video_frames, segment_t_size=args.segment_t_size,
                image_size=args.image_size, patch_size=args.patch_size,
                keep_ratio=args.keep_ratio, keep_first_per_segment=args.keep_first_per_segment,
            )
            d = out_root / key
            d.mkdir(parents=True, exist_ok=True)
            meta["key"] = key
            meta["video"] = os.path.abspath(src)   # record the original video, not the temporary transcoded file
            meta["transcoded_h264"] = True
            np.save(d / "visidx.npy", visidx)
            np.save(d / "frame_ids.npy", frame_ids)
            with open(d / "meta.json", "w") as f:
                json.dump(meta, f, indent=2)
            ok += 1
        except Exception as e:
            fail += 1
            sys.stderr.write(f"[FAIL] {key}: {type(e).__name__}: {str(e)[:140]}\n")
        finally:
            if os.path.exists(tmp_mp4):
                os.remove(tmp_mp4)
        if (idx + 1) % 50 == 0:
            print(f"  ... {idx+1}/{len(missing)}  ok={ok} fail={fail}")

    print(f"[DONE] retry ok={ok} fail={fail} elapsed={time.time()-t0:.1f}s -> {out_root}")
    if not args.tmp_dir:
        try:
            os.rmdir(tmp_root)
        except OSError:
            pass


if __name__ == "__main__":
    main()
