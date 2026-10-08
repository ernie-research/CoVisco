#!/usr/bin/env python3
"""Visualize flat query-to-patch attention heatmaps for OneVision ViT.

For each sample and each segment, produces one heat matrix of shape
    (num_query_tokens, vit_tokens_per_segment)
where rows = query tokens (100), columns = flat ViT patch candidates
sorted by their original position index (frame × h × w).

The x-axis is the flat token index within the segment, which naturally
corresponds to temporal position when visidx is used: tokens from earlier
frames appear on the left, later frames on the right.

This makes cross-frame attention distribution visible without mapping back
to a spatial grid.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.visualize_vit_query_attention import (  # noqa: E402
    DEFAULT_CKPT,
    DEFAULT_CONFIG,
    DEFAULT_IMAGE_DATA,
    DEFAULT_VIDEO_DATA,
    CoViscoEncoderAttention,
    build_vit_and_selector,
    collect_inputs,
    force_eager_attention,
    forward_vit_last_query_attention,
    load_checkpoint,
    load_model_config,
    positions_by_segment,
    prepare_video_frames,
    preprocess_image,
    resolve_device,
    resolve_dtype,
)

import torch  # noqa: E402 (needed here for type annotations)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Flat query-attention heatmap: one 100×P matrix per segment "
            "showing cross-frame attention distribution without spatial alignment."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--sample-cache-dir", type=Path, default=None)
    parser.add_argument("--image-data-path", type=Path, default=DEFAULT_IMAGE_DATA)
    parser.add_argument("--video-data-path", type=Path, default=DEFAULT_VIDEO_DATA)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "flat_query_attention",
    )
    parser.add_argument("--num-images", type=int, default=1)
    parser.add_argument("--num-videos", type=int, default=1)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--dtype", choices=("auto", "fp32", "fp16", "bf16"), default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-shards", type=int, default=64)
    parser.add_argument("--shuffle-shards", action="store_true")
    parser.add_argument(
        "--video-mode",
        choices=("visidx", "uniform"),
        default="visidx",
    )
    parser.add_argument("--uniform-video-frames", type=int, default=64)
    parser.add_argument("--uniform-segment-t-size", type=int, default=16)
    parser.add_argument(
        "--head-index",
        type=int,
        default=-1,
        help="-1 = average all heads; otherwise select one head.",
    )
    parser.add_argument(
        "--sort-cols-by-position",
        action="store_true",
        help="Sort flat token columns by original position index (frame×hw) before rendering.",
    )
    parser.add_argument(
        "--normalize",
        choices=("global", "per-row", "per-col", "none"),
        default="per-row",
        help=(
            "How to normalize the attention matrix before colorizing. "
            "'per-row' = each query token independently; 'global' = shared min/max."
        ),
    )
    parser.add_argument(
        "--add-frame-boundary-lines",
        action="store_true",
        default=True,
        help="Draw vertical lines at frame boundaries on the heatmap (visidx mode).",
    )
    parser.add_argument("--no-frame-boundary-lines", dest="add_frame_boundary_lines", action="store_false")
    parser.add_argument(
        "--cell-w", type=int, default=4,
        help="Width in pixels of each flat-token column.",
    )
    parser.add_argument(
        "--cell-h", type=int, default=8,
        help="Height in pixels of each query-token row.",
    )
    parser.add_argument(
        "--save-npy",
        action="store_true",
        help="Also save the raw attention matrix as a .npy file.",
    )
    return parser.parse_args()


def normalize_matrix(mat: np.ndarray, mode: str) -> np.ndarray:
    mat = mat.copy().astype(np.float32)
    if mode == "none":
        return np.clip(mat, 0.0, 1.0)
    if mode == "global":
        lo, hi = mat.min(), mat.max()
        if hi - lo > 1e-8:
            return (mat - lo) / (hi - lo)
        return np.zeros_like(mat)
    if mode == "per-row":
        lo = mat.min(axis=1, keepdims=True)
        hi = mat.max(axis=1, keepdims=True)
        span = hi - lo
        span[span < 1e-8] = 1.0
        return (mat - lo) / span
    if mode == "per-col":
        lo = mat.min(axis=0, keepdims=True)
        hi = mat.max(axis=0, keepdims=True)
        span = hi - lo
        span[span < 1e-8] = 1.0
        return (mat - lo) / span
    raise ValueError(f"unknown normalize mode: {mode}")


def viridis_like(values: np.ndarray) -> np.ndarray:
    """Simple dark-blue→yellow colormap similar to viridis."""
    values = np.clip(values, 0.0, 1.0)
    anchors_x = np.array([0.0, 0.25, 0.50, 0.75, 1.0], dtype=np.float32)
    anchors_rgb = np.array(
        [
            [68, 1, 84],
            [59, 82, 139],
            [33, 145, 140],
            [94, 201, 98],
            [253, 231, 37],
        ],
        dtype=np.float32,
    )
    flat = values.reshape(-1)
    rgb = np.empty((flat.size, 3), dtype=np.float32)
    for ch in range(3):
        rgb[:, ch] = np.interp(flat, anchors_x, anchors_rgb[:, ch])
    return rgb.reshape(values.shape + (3,)).astype(np.uint8)


def render_flat_heatmap(
    attn: np.ndarray,          # (Q, P) float32
    frame_boundaries: List[int],  # column indices where new frames begin (for vert lines)
    normalize: str,
    cell_w: int,
    cell_h: int,
    add_frame_lines: bool,
) -> Image.Image:
    """Render the (Q, P) attention matrix as an RGB image.

    Each row = one query token; each column = one flat patch token.
    """
    normed = normalize_matrix(attn, normalize)
    rgb = viridis_like(normed)  # (Q, P, 3) uint8

    Q, P = rgb.shape[:2]
    img_w = P * cell_w
    img_h = Q * cell_h

    # Nearest-neighbour upscale
    canvas = Image.fromarray(rgb, mode="RGB").resize((img_w, img_h), Image.NEAREST)

    if add_frame_lines and frame_boundaries:
        draw = ImageDraw.Draw(canvas)
        for col in frame_boundaries:
            x = col * cell_w
            draw.line([(x, 0), (x, img_h)], fill=(255, 80, 80), width=1)

    return canvas


def draw_axis_labels(img: Image.Image, num_queries: int, frame_boundaries: List[int]) -> Image.Image:
    """Add minimal axis labels: row indices on left, frame numbers on top."""
    label_w = 40
    label_h = 20
    total_w = img.width + label_w
    total_h = img.height + label_h

    canvas = Image.new("RGB", (total_w, total_h), (255, 255, 255))
    canvas.paste(img, (label_w, label_h))
    draw = ImageDraw.Draw(canvas)

    # Frame numbers along top
    for frame_idx, col_start in enumerate(frame_boundaries):
        x = label_w + col_start * (img.width // max(1, img.width // img.width))
        draw.text((label_w + col_start * (img.width // img.width) if frame_idx == 0 else label_w + col_start, 2),
                  f"f{frame_idx}", fill=(0, 0, 0))

    # Query indices along left
    row_h = img.height // max(1, num_queries)
    for q_idx in range(0, num_queries, max(1, num_queries // 10)):
        y = label_h + q_idx * row_h
        draw.text((2, y), f"q{q_idx}", fill=(0, 0, 0))

    return canvas


def frame_boundaries_from_positions(
    positions: np.ndarray,
    grid_h: int,
    grid_w: int,
) -> List[int]:
    """Column indices in the sorted position array where the frame id changes."""
    patches_per_frame = grid_h * grid_w
    frame_ids = positions // patches_per_frame
    boundaries: List[int] = [0]
    for i in range(1, len(frame_ids)):
        if frame_ids[i] != frame_ids[i - 1]:
            boundaries.append(i)
    return boundaries


def save_flat_heatmap(
    key: str,
    prefix: str,
    attn_mat: np.ndarray,          # (Q, P) float32 for this segment
    positions: np.ndarray,         # (P,) int64, original patch positions
    grid_h: int,
    grid_w: int,
    output_dir: Path,
    args: argparse.Namespace,
    seg_idx: Optional[int] = None,
) -> Dict:
    if args.sort_cols_by_position:
        order = np.argsort(positions)
        attn_mat = attn_mat[:, order]
        positions = positions[order]

    boundaries = (
        frame_boundaries_from_positions(positions, grid_h, grid_w)
        if args.add_frame_boundary_lines
        else []
    )

    img = render_flat_heatmap(
        attn_mat,
        frame_boundaries=boundaries,
        normalize=args.normalize,
        cell_w=args.cell_w,
        cell_h=args.cell_h,
        add_frame_lines=args.add_frame_boundary_lines,
    )

    seg_tag = f"_segment_{seg_idx:02d}" if seg_idx is not None else ""
    stem = f"{prefix}_{key}{seg_tag}_flat_heatmap"
    png_path = output_dir / (stem + ".png")
    img.save(png_path)

    result: Dict = {
        "png": str(png_path),
        "shape": list(attn_mat.shape),
        "frame_boundaries": boundaries,
        "num_frames_in_segment": len(boundaries),
    }

    if args.save_npy:
        npy_path = output_dir / (stem + ".npy")
        np.save(npy_path, attn_mat.astype(np.float32))
        result["npy"] = str(npy_path)

    return result


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

    image_samples, video_samples = collect_inputs(args)
    summaries = []

    # ---- images ----
    for idx, sample in enumerate(image_samples):
        print(f"[run] image {idx}: key={sample.key}", flush=True)
        pixel_values, _ = preprocess_image(sample.image, cfg.vit.image_size)
        _, _, attn = forward_vit_last_query_attention(
            vit, pixel_values, device=device, dtype=dtype, head_index=args.head_index,
        )
        # attn: (1, 1, Q, P)
        attn_np = attn[0, 0].numpy()  # (Q, P)
        grid_h = cfg.vit.image_size // cfg.vit.patch_size
        positions = np.arange(grid_h * grid_h, dtype=np.int64)
        print(f"[shape] image q={tuple(attn.shape)}", flush=True)

        seg_result = save_flat_heatmap(
            key=sample.key,
            prefix=f"image_{idx:03d}",
            attn_mat=attn_np,
            positions=positions,
            grid_h=grid_h,
            grid_w=grid_h,
            output_dir=args.output_dir,
            args=args,
        )
        summaries.append({"kind": "image", "key": sample.key, "segment_0": seg_result})
        del attn
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ---- videos ----
    for idx, sample in enumerate(video_samples):
        print(f"[run] video {idx}: key={sample.key}", flush=True)
        pixel_values, frames, visidx, note, forward_kwargs = prepare_video_frames(
            sample,
            image_size=cfg.vit.image_size,
            patch_size=cfg.vit.patch_size,
            segment_t_size=cfg.vit.segment_t_size,
            video_mode=args.video_mode,
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
        print(f"[shape] video attn={tuple(attn.shape)}", flush=True)

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

        seg_results = []
        for seg_idx in range(num_segments):
            attn_np = attn[0, seg_idx].numpy()  # (Q, P)
            positions = pos[seg_idx]             # (P,) int64

            result = save_flat_heatmap(
                key=sample.key,
                prefix=f"video_{idx:03d}",
                attn_mat=attn_np,
                positions=positions,
                grid_h=grid_h,
                grid_w=grid_w,
                output_dir=args.output_dir,
                args=args,
                seg_idx=seg_idx,
            )
            seg_results.append(result)
            print(
                f"[seg {seg_idx}] frames={result['num_frames_in_segment']} → {result['png']}",
                flush=True,
            )

        summaries.append({
            "kind": "video",
            "key": sample.key,
            "video_mode": args.video_mode,
            "segments": seg_results,
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
                "items": summaries,
            },
            handle,
            indent=2,
            ensure_ascii=True,
        )
    print(f"[done] wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
