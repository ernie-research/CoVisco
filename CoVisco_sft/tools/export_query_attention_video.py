#!/usr/bin/env python3
"""Export per-query attention videos for OneVision ViT (uniform sampling only).

For each (sample, segment, query_token), produces a short video/GIF where
each frame is a 16x16 heatmap of that query token's attention to the ViT
patch tokens in that video frame.

Resolution is fixed at 16x16 patches (= 224/14 x 224/14), i.e., one pixel
per ViT patch.  Use --upscale to enlarge (default 16 -> 256x256 output).
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.visualize_vit_query_attention import (  # noqa: E402
    DEFAULT_CKPT,
    DEFAULT_CONFIG,
    DEFAULT_IMAGE_DATA,
    DEFAULT_VIDEO_DATA,
    build_vit_and_selector,
    collect_inputs,
    force_eager_attention,
    forward_vit_last_query_attention,
    load_checkpoint,
    load_model_config,
    positions_by_segment,
    prepare_video_frames,
    resolve_device,
    resolve_dtype,
)

import torch  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export per-query attention videos (uniform frame sampling). "
            "Each output = one segment x one query token. "
            "Each frame = 16x16 attention heatmap over ViT patches."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--sample-cache-dir", type=Path, default=None)
    parser.add_argument("--video-data-path", type=Path, default=DEFAULT_VIDEO_DATA)
    parser.add_argument("--image-data-path", type=Path, default=DEFAULT_IMAGE_DATA)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "query_attention_video",
    )
    parser.add_argument("--num-videos", type=int, default=1)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--dtype", choices=("auto", "fp32", "fp16", "bf16"), default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-shards", type=int, default=64)
    parser.add_argument("--shuffle-shards", action="store_true")
    # Uniform frame sampling (the only supported video mode here)
    parser.add_argument(
        "--uniform-video-frames", type=int, default=64,
        help="Total frames to uniformly sample from the video.",
    )
    parser.add_argument(
        "--uniform-segment-t-size", type=int, default=16,
        help="Frames per segment; 64/16 = 4 segments.",
    )
    # Which queries to export
    parser.add_argument(
        "--query-indices",
        type=int,
        nargs="+",
        default=None,
        help="Indices of query tokens to export (0-based). Default: all 100.",
    )
    parser.add_argument(
        "--head-index", type=int, default=-1,
        help="-1 = average all heads.",
    )
    # Output format
    parser.add_argument(
        "--format",
        choices=("mp4", "gif", "both"),
        default="gif",
        help="Output format.",
    )
    parser.add_argument(
        "--fps", type=float, default=8.0,
        help="Frames per second for mp4 and gif.",
    )
    parser.add_argument(
        "--upscale", type=int, default=16,
        help="Scale factor applied to each 16x16 frame (default 16 -> 256x256).",
    )
    parser.add_argument(
        "--normalize",
        choices=("global", "per-frame", "none"),
        default="global",
        help=(
            "global: shared min/max over all frames in the segment; "
            "per-frame: each frame independently; "
            "none: raw [0,1] clamp."
        ),
    )
    parser.add_argument(
        "--segments",
        type=int,
        nargs="+",
        default=None,
        help="Segment indices to export. Default: all.",
    )
    return parser.parse_args()


def viridis_frame(values: np.ndarray) -> np.ndarray:
    """Map (H, W) float32 [0,1] to (H, W, 3) uint8 using viridis-like colormap."""
    values = np.clip(values, 0.0, 1.0)
    anchors_x = np.array([0.0, 0.25, 0.50, 0.75, 1.0], dtype=np.float32)
    anchors_rgb = np.array(
        [[68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37]],
        dtype=np.float32,
    )
    flat = values.reshape(-1)
    rgb = np.empty((flat.size, 3), dtype=np.float32)
    for ch in range(3):
        rgb[:, ch] = np.interp(flat, anchors_x, anchors_rgb[:, ch])
    return rgb.reshape(values.shape + (3,)).astype(np.uint8)


def attn_to_frames(
    attn_seg: np.ndarray,
    positions_seg: np.ndarray,
    num_frames: int,
    grid_h: int,
    grid_w: int,
    query_idx: int,
    normalize: str,
    upscale: int,
) -> List[np.ndarray]:
    """Build a list of (H*upscale, W*upscale, 3) uint8 frames for one query token."""
    from PIL import Image as PILImage

    patches_per_frame = grid_h * grid_w
    vec = attn_seg[query_idx]  # (P,)

    frame_grids: List[np.ndarray] = [
        np.full((grid_h, grid_w), np.nan, dtype=np.float32)
        for _ in range(num_frames)
    ]
    for val, pos in zip(vec.tolist(), positions_seg.tolist()):
        pos = int(pos)
        total = num_frames * patches_per_frame
        if pos < 0 or pos >= total:
            continue
        f = pos // patches_per_frame
        rem = pos % patches_per_frame
        y = rem // grid_w
        x = rem % grid_w
        frame_grids[f][y, x] = float(val)

    valid_vals = np.concatenate([g[np.isfinite(g)] for g in frame_grids])
    if valid_vals.size == 0:
        blank = np.zeros((grid_h * upscale, grid_w * upscale, 3), dtype=np.uint8)
        return [blank] * num_frames

    if normalize == "global":
        g_min, g_max = float(valid_vals.min()), float(valid_vals.max())
    else:
        g_min, g_max = None, None

    frames: List[np.ndarray] = []
    for fg in frame_grids:
        if normalize == "per-frame":
            vals = fg[np.isfinite(fg)]
            lo = float(vals.min()) if vals.size else 0.0
            hi = float(vals.max()) if vals.size else 1.0
        else:
            lo, hi = g_min, g_max

        span = hi - lo if (hi - lo) > 1e-8 else 1.0
        normed = np.where(np.isfinite(fg), (fg - lo) / span, 0.0).astype(np.float32)
        rgb = viridis_frame(normed)  # (grid_h, grid_w, 3)

        if upscale > 1:
            pil = PILImage.fromarray(rgb, mode="RGB")
            pil = pil.resize((grid_w * upscale, grid_h * upscale), PILImage.NEAREST)
            rgb = np.asarray(pil)

        frames.append(rgb)
    return frames


def write_mp4(frames: List[np.ndarray], path: Path, fps: float) -> None:
    import cv2
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def write_gif(frames: List[np.ndarray], path: Path, fps: float) -> None:
    import imageio.v2 as iio
    duration_ms = int(1000.0 / fps)
    iio.mimsave(str(path), frames, format="GIF", loop=0, duration=duration_ms)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)

    print(f"[env] repo={REPO_ROOT}", flush=True)
    print(f"[env] device={device} dtype={dtype}", flush=True)

    cfg = load_model_config(args.config)
    vit, selector = build_vit_and_selector(cfg)
    load_checkpoint(vit, selector, args.ckpt)
    force_eager_attention(vit)
    del selector
    gc.collect()
    vit.to(device=device, dtype=dtype).eval()

    args.num_images = 0
    _, video_samples = collect_inputs(args)

    summaries: List[dict] = []
    for vid_idx, sample in enumerate(video_samples):
        print(f"[run] video {vid_idx}: key={sample.key}", flush=True)
        pixel_values, frames, visidx, note, forward_kwargs = prepare_video_frames(
            sample,
            image_size=cfg.vit.image_size,
            patch_size=cfg.vit.patch_size,
            segment_t_size=cfg.vit.segment_t_size,
            video_mode="uniform",
            uniform_video_frames=args.uniform_video_frames,
            uniform_segment_t_size=args.uniform_segment_t_size,
        )
        print(f"[video] {note}", flush=True)
        _, _, attn = forward_vit_last_query_attention(
            vit, pixel_values, device=device, dtype=dtype,
            visidx=visidx, head_index=args.head_index, **forward_kwargs,
        )
        # attn: (1, S, Q, P)
        _, num_segments, num_query, vit_per_segment = attn.shape
        print(f"[shape] attn={tuple(attn.shape)}", flush=True)

        grid_h = cfg.vit.image_size // cfg.vit.patch_size
        grid_w = grid_h
        pos = positions_by_segment(
            modality="video",
            num_frames=len(frames),
            grid_h=grid_h,
            grid_w=grid_w,
            num_segments=num_segments,
            tokens_per_segment=vit_per_segment,
            visidx=visidx,
        )

        frames_per_segment = max(1, len(frames) // num_segments)
        query_indices = (
            list(range(num_query))
            if args.query_indices is None
            else [i for i in args.query_indices if 0 <= i < num_query]
        )
        seg_indices = (
            list(range(num_segments))
            if args.segments is None
            else [s for s in args.segments if 0 <= s < num_segments]
        )

        sample_outputs: List[dict] = []
        for seg_idx in seg_indices:
            start = seg_idx * frames_per_segment
            end = (
                len(frames) if seg_idx == num_segments - 1
                else min(len(frames), (seg_idx + 1) * frames_per_segment)
            )
            attn_seg = attn[0, seg_idx].numpy()   # (Q, P)
            positions_seg = pos[seg_idx]           # (P,)

            seg_paths: List[dict] = []
            for q_idx in query_indices:
                frame_list = attn_to_frames(
                    attn_seg=attn_seg,
                    positions_seg=positions_seg,
                    num_frames=len(frames),
                    grid_h=grid_h,
                    grid_w=grid_w,
                    query_idx=q_idx,
                    normalize=args.normalize,
                    upscale=args.upscale,
                )
                # Only frames belonging to this segment
                seg_frame_list = frame_list[start:end]

                stem = (
                    f"video_{vid_idx:03d}_{sample.key}"
                    f"_seg{seg_idx:02d}_q{q_idx:03d}"
                )
                out_paths: Dict[str, str] = {}
                if args.format in ("mp4", "both"):
                    mp4_path = args.output_dir / (stem + ".mp4")
                    write_mp4(seg_frame_list, mp4_path, args.fps)
                    out_paths["mp4"] = str(mp4_path)
                if args.format in ("gif", "both"):
                    gif_path = args.output_dir / (stem + ".gif")
                    write_gif(seg_frame_list, gif_path, args.fps)
                    out_paths["gif"] = str(gif_path)

                seg_paths.append({
                    "query": q_idx,
                    "num_frames": len(seg_frame_list),
                    "paths": out_paths,
                })
                print(
                    f"  [seg{seg_idx} q{q_idx:03d}] {len(seg_frame_list)} frames → "
                    + " ".join(out_paths.values()),
                    flush=True,
                )

            sample_outputs.append({"segment": seg_idx, "queries": seg_paths})

        summaries.append({
            "kind": "video",
            "key": sample.key,
            "attn_shape": list(attn.shape),
            "uniform_video_frames": args.uniform_video_frames,
            "uniform_segment_t_size": args.uniform_segment_t_size,
            "segments": sample_outputs,
        })
        del attn
        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary_path = args.output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "config": str(args.config),
                "ckpt": str(args.ckpt),
                "head_index": args.head_index,
                "normalize": args.normalize,
                "fps": args.fps,
                "upscale": args.upscale,
                "items": summaries,
            },
            handle,
            indent=2,
            ensure_ascii=True,
        )
    print(f"[done] wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
