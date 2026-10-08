#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""precompute_video_visidx.py — precompute codec-filtered patch indices (visidx) for video benchmarks.

Follows the codec token filtering idea of OneVision-Encoder/llava_next
(Compressed_Video_Reader/tool/offline_precompute_llava_codec_assets.py),
but outputs align with the visidx interface of the CoVisco_sft model's ViT,
rather than mosaic + patch_positions.

Key differences between the two:
  - OneVision-Encoder's llava_ov_encoder packs the selected patches into a mosaic image and
    feeds it into the ViT with (t,h,w) patch_positions (576/16 grid).
  - This repo's llava_covisco ViT natively supports visidx (models/covisco_vit.py
    ::forward(..., visidx=...)): it keeps full-resolution video frames, computes only the
    selected patches by visible_indices, and derives 4D-RoPE from them. So here we produce
    visidx directly, no mosaic needed.

visidx convention (aligned with models/_covisco_encoder_src.py::get_segments):
  - Grid: num_frames frames, (image_size/patch_size)^2 patches per frame.
    Default 64 frames × (224/14)^2 = 64 × 256 = 16384 patches.
  - Global index idx = t * (h*w) + row * w + col, t∈[0,num_frames), row/col∈[0, image_size/patch).
  - The number of selected tokens per segment (segment_t_size frames) must be equal, and the
    whole thing is sorted ascending (get_segments reshapes to (num_seg, L//num_seg), see its
    assert L % num_seg == 0).

Output (one directory per video <out_root>/<key>/):
  - visidx.npy   int64, shape (num_seg * K_seg,)  ascending global patch indices
  - frame_ids.npy int32, shape (num_frames,)      sampled original frame numbers
  - meta.json    records grid parameters, for the evaluator to verify consistency

Dependency: cv_reader (compiled from Compressed_Video_Reader, reads motion vectors + residuals of H.264/H.265).
"""

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import cv2

# ---- Codec scoring/reading tools, vendored from OneVision-Encoder (MIT) ----
# The Compressed_Video_Reader source lives in this directory (./Compressed_Video_Reader),
# so this script is self-contained: no external repo needed. Set OV_ENCODER_TOOL_DIR
# to override (e.g. point at a separately installed copy). The cv_reader C++ extension
# itself still needs a one-time build: bash Compressed_Video_Reader/install.sh
_OV_ENCODER_TOOL_DIR = os.environ.get(
    "OV_ENCODER_TOOL_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "Compressed_Video_Reader", "tool"),
)
if _OV_ENCODER_TOOL_DIR not in sys.path:
    sys.path.insert(0, _OV_ENCODER_TOOL_DIR)

try:
    from offline_precompute_llava_codec_assets import (  # type: ignore
        _cv_reader_fetch_mvres_by_frame_ids,
        _residual_energy_norm,
        _mv_energy_norm,
        _HAS_CV_READER,
    )
except Exception as e:  # pragma: no cover - give clear guidance when import fails
    raise RuntimeError(
        f"Failed to import codec tool functions from {_OV_ENCODER_TOOL_DIR}: {e}\n"
        f"Please confirm the OneVision-Encoder repo path is correct, or override with OV_ENCODER_TOOL_DIR."
    )


def _get_total_frames_cv2(video_path: str) -> int:
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return 0
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return max(0, n)
    except Exception:
        return 0


def _sample_frame_ids(total_frames: int, num_frames: int) -> List[int]:
    """Uniform sampling consistent with the evaluator's _extract_video_frames: np.linspace(0, total-1, num_frames)."""
    if total_frames <= 0:
        return [0] * int(num_frames)
    ids = np.linspace(0, total_frames - 1, int(num_frames)).astype(int)
    return [int(x) for x in ids.tolist()]


def _resize_shorter_centercrop_map(arr2d: np.ndarray, out_size: int) -> np.ndarray:
    """Map a (H,W) energy map to (out_size,out_size) via "Resize shorter side to out_size + center crop".

    Spatially consistent with the evaluator's build_image_transform Resize(out_size)+CenterCrop(out_size),
    ensuring the scoring grid aligns with the patch grid the ViT actually sees.
    """
    H, W = arr2d.shape[:2]
    out_size = int(out_size)
    if H <= 0 or W <= 0:
        return np.zeros((out_size, out_size), dtype=np.float32)
    scale = float(out_size) / float(min(H, W))
    Hn = max(out_size, int(round(H * scale)))
    Wn = max(out_size, int(round(W * scale)))
    resized = cv2.resize(arr2d.astype(np.float32), (Wn, Hn), interpolation=cv2.INTER_LINEAR)
    top = (Hn - out_size) // 2
    left = (Wn - out_size) // 2
    return resized[top:top + out_size, left:left + out_size].astype(np.float32)


def _frame_scores(
    fused_hw: np.ndarray,
    image_size: int,
    patch_size: int,
) -> np.ndarray:
    """Pool a frame's fused energy map (H,W) into (h*w,) patch scores, row-major (row*w+col)."""
    sq = _resize_shorter_centercrop_map(fused_hw, out_size=int(image_size))  # (S,S)
    p = int(patch_size)
    hb = int(image_size) // p
    wb = int(image_size) // p
    # (hb, p, wb, p) -> sum over pixels within a patch
    s = sq[: hb * p, : wb * p].reshape(hb, p, wb, p).sum(axis=(1, 3))  # (hb, wb)
    return s.reshape(-1).astype(np.float32)  # (hb*wb,)


def compute_visidx_for_video(
    video_path: str,
    num_frames: int,
    segment_t_size: int,
    image_size: int,
    patch_size: int,
    keep_ratio: float,
    mv_unit_div: float = 4.0,
    mv_pct: float = 95.0,
    res_pct: float = 95.0,
    w_mv: float = 1.0,
    w_res: float = 1.0,
    mv_compensate: str = "median",
    res_use_grad: bool = False,
    keep_first_per_segment: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Compute visidx for a single video. Returns (visidx int64[L], frame_ids int32[T], meta)."""
    if not _HAS_CV_READER:
        raise RuntimeError("cv_reader not available: please compile and install Compressed_Video_Reader first.")

    assert image_size % patch_size == 0, (
        f"image_size({image_size}) must be divisible by patch_size({patch_size}), otherwise it won't align with the ViT patch grid"
    )
    per_frame = (image_size // patch_size) ** 2
    assert num_frames % segment_t_size == 0, (
        f"num_frames({num_frames}) must be divisible by segment_t_size({segment_t_size})"
    )
    num_seg = num_frames // segment_t_size

    total_frames = _get_total_frames_cv2(video_path)
    frame_ids = _sample_frame_ids(total_frames, num_frames)

    items = _cv_reader_fetch_mvres_by_frame_ids(
        video_path, frame_ids, with_residual=True, seek_to_frame=None, decode_len=None,
    )
    if not isinstance(items, (list, tuple)) or len(items) == 0:
        raise RuntimeError(f"cv_reader returned empty: {video_path}")

    # Infer original resolution (mv scoring needs H/W)
    H0, W0 = 0, 0
    try:
        cap = cv2.VideoCapture(str(video_path))
        if cap.isOpened():
            W0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            H0 = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
    except Exception:
        pass
    if (H0 <= 0 or W0 <= 0) and items[0] is not None and "residual_y" in items[0]:
        ry0 = np.asarray(items[0]["residual_y"])
        if ry0.ndim >= 2:
            H0, W0 = int(ry0.shape[0]), int(ry0.shape[1])

    scores = np.zeros((int(num_frames), int(per_frame)), dtype=np.float32)
    for t in range(int(num_frames)):
        it = items[int(t)]
        mv = np.asarray(it["motion_vector"])
        res_y = np.asarray(it["residual_y"])
        res_norm = _residual_energy_norm(res_y, pct=float(res_pct), use_grad=bool(res_use_grad))
        mv_norm = _mv_energy_norm(
            mv, H=int(H0), W=int(W0), mv_unit_div=float(mv_unit_div),
            pct=float(mv_pct), compensate=str(mv_compensate),
        )
        denom = float(w_mv + w_res) if (w_mv + w_res) != 0 else 1.0
        fused = np.clip((float(w_mv) * mv_norm + float(w_res) * res_norm) / denom, 0.0, 1.0)
        scores[t] = _frame_scores(fused, image_size=int(image_size), patch_size=int(patch_size))

    # Equal top-K per segment (get_segments requires the same token count per segment), concatenated and sorted ascending.
    # When keep_first_per_segment=True, each segment first forcibly keeps all per_frame patches of the first frame,
    # then fills the rest from the segment's other frames by top-(k_seg-per_frame) scores — each segment's total stays k_seg, compatible with both.
    k_seg = max(1, int(round(float(keep_ratio) * segment_t_size * per_frame)))
    k_seg = min(k_seg, segment_t_size * per_frame)
    # Guarantee "always keep each segment's first frame": the budget must at least fit a full frame (if keep_ratio is too small, force it up to per_frame)
    k_floored_to_first = False
    if keep_first_per_segment and k_seg < per_frame:
        k_seg = per_frame
        k_floored_to_first = True
    selected: List[np.ndarray] = []
    for s in range(int(num_seg)):
        f0 = s * segment_t_size
        seg = scores[f0:f0 + segment_t_size].reshape(-1)  # (seg_t*per_frame,)  idx = local_t*per_frame + hw

        if keep_first_per_segment:
            # Forcibly keep all patches of the first frame (local_t=0); here k_seg>=per_frame always holds
            first = np.arange(per_frame, dtype=np.int64)
            rest_budget = k_seg - per_frame
            if rest_budget > 0:
                seg_rest = seg.copy()
                seg_rest[:per_frame] = -np.inf  # exclude the first frame to avoid double counting
                top_rest = np.argpartition(seg_rest, -rest_budget)[-rest_budget:]
                local = np.concatenate([first, top_rest.astype(np.int64)])
            else:
                local = first
        else:
            # Pure per-segment top-K (no forced first frame)
            k = min(k_seg, seg.size)
            local = np.argpartition(seg, -k)[-k:].astype(np.int64)

        local_t = local // per_frame
        hw = local % per_frame
        gidx = (f0 + local_t) * per_frame + hw
        selected.append(gidx.astype(np.int64))

    visidx = np.sort(np.concatenate(selected)).astype(np.int64)

    meta = {
        "num_frames": int(num_frames),
        "segment_t_size": int(segment_t_size),
        "num_seg": int(num_seg),
        "image_size": int(image_size),
        "patch_size": int(patch_size),
        "per_frame_patches": int(per_frame),
        "total_patches": int(num_frames * per_frame),
        "keep_ratio": float(keep_ratio),
        "k_per_segment": int(k_seg),
        "keep_first_per_segment": bool(keep_first_per_segment),
        "k_floored_to_first": bool(k_floored_to_first),
        "visidx_len": int(visidx.size),
        "total_frames": int(total_frames),
        "orig_hw": [int(H0), int(W0)],
        "vi_min": int(visidx.min()) if visidx.size else -1,
        "vi_max": int(visidx.max()) if visidx.size else -1,
        "video": os.path.abspath(str(video_path)),
        "w_mv": float(w_mv), "w_res": float(w_res),
        "mv_compensate": str(mv_compensate), "res_use_grad": bool(res_use_grad),
    }
    return visidx, np.asarray(frame_ids, dtype=np.int32), meta


def _process_one(job: Dict, args: argparse.Namespace) -> Tuple[str, bool, str]:
    key = str(job["key"])
    video = str(job["video"])
    out_dir = Path(args.out_root) / key
    visidx_path = out_dir / "visidx.npy"
    if visidx_path.exists() and not args.overwrite:
        return key, True, "skip(exists)"
    try:
        visidx, frame_ids, meta = compute_visidx_for_video(
            video,
            num_frames=args.num_video_frames,
            segment_t_size=args.segment_t_size,
            image_size=args.image_size,
            patch_size=args.patch_size,
            keep_ratio=args.keep_ratio,
            mv_unit_div=args.mv_unit_div,
            mv_pct=args.mv_pct,
            res_pct=args.res_pct,
            w_mv=args.w_mv,
            w_res=args.w_res,
            mv_compensate=args.mv_compensate,
            res_use_grad=args.res_use_grad,
            keep_first_per_segment=args.keep_first_per_segment,
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        meta["key"] = key
        np.save(visidx_path, visidx)
        np.save(out_dir / "frame_ids.npy", frame_ids)
        with open(out_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        return key, True, f"ok(L={visidx.size})"
    except Exception as e:
        return key, False, f"{type(e).__name__}: {e}\n{traceback.format_exc()}"


def _read_jsonl(path: str) -> List[Dict]:
    jobs: List[Dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if "video" not in d:
                continue
            key = d.get("key") or Path(str(d["video"])).stem
            jobs.append({"video": d["video"], "key": key})
    return jobs


def main():
    ap = argparse.ArgumentParser("Precompute codec visidx for CoVisco video eval")
    ap.add_argument("--jsonl", required=True, help="jsonl with each line containing {\"video\":..., \"key\":...}")
    ap.add_argument("--out_root", required=True, help="output root directory for visidx assets")
    ap.add_argument("--num_video_frames", type=int, default=64)
    ap.add_argument("--segment_t_size", type=int, default=16)
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--patch_size", type=int, default=14)
    ap.add_argument("--keep_ratio", type=float, default=0.5,
                    help="fraction of patches kept per segment (as the candidate pool fed into the ViT; under query_and_vit a learned selector refines it further)")
    ap.add_argument("--mv_unit_div", type=float, default=4.0)
    ap.add_argument("--mv_pct", type=float, default=95.0)
    ap.add_argument("--res_pct", type=float, default=95.0)
    ap.add_argument("--w_mv", type=float, default=1.0)
    ap.add_argument("--w_res", type=float, default=1.0)
    ap.add_argument("--mv_compensate", type=str, default="median", choices=["median", "mean", "none"])
    ap.add_argument("--res_use_grad", action="store_true")
    ap.add_argument("--keep_first_per_segment", action=argparse.BooleanOptionalAction, default=True,
                    help="each segment forcibly keeps all patches of the first frame, then top-K fills the rest (each segment's total stays equal). Enabled by default; disable with --no-keep_first_per_segment")
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if not _HAS_CV_READER:
        print("[FATAL] cv_reader not available, please run bash Compressed_Video_Reader/install.sh first", file=sys.stderr)
        sys.exit(2)

    jobs = _read_jsonl(args.jsonl)
    print(f"[INFO] jobs={len(jobs)} out_root={args.out_root} "
          f"grid={args.num_video_frames}f x ({args.image_size}/{args.patch_size})^2 "
          f"seg_t={args.segment_t_size} keep_ratio={args.keep_ratio}")
    Path(args.out_root).mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    ok = fail = 0
    if args.num_workers and args.num_workers > 1:
        import multiprocessing as mp
        # cv_reader is a C extension, spawn is safer
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=int(args.num_workers)) as pool:
            results = pool.starmap(_process_one, [(j, args) for j in jobs])
    else:
        results = [_process_one(j, args) for j in jobs]

    for key, success, msg in results:
        if success:
            ok += 1
        else:
            fail += 1
            print(f"[FAIL] {key}: {msg}", file=sys.stderr)

    print(f"[DONE] ok={ok} fail={fail} elapsed={time.time() - t0:.1f}s -> {args.out_root}")


if __name__ == "__main__":
    main()
