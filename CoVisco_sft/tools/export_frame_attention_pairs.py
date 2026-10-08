#!/usr/bin/env python3
"""Export paper-ready figures pairing each original video frame with its ViT
query-attention map (one-to-one frame correspondence).

For a chosen segment, the ViT last-layer query->patch attention is aggregated
over query tokens (mean/max) or restricted to selected queries, mapped back to
its (frame, patch-grid) position, then rendered as:

  row 0: original frames (aligned, one column per frame)
  row 1: the same frames with the attention heatmap overlaid
  row 2 (optional): the pure attention heatmap

Columns are one-to-one with the sampled frames, so a reader can read the figure
top-to-bottom to see "what the model attends to on this exact frame".

Unlike ``export_query_attention_video.py`` (which renders bare 16x16 heatmap
videos with no original frame), this tool keeps the original frame beside the
attention and emits high-resolution static PNGs for a paper.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import warnings
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
    vector_to_grid,
)
from tools.visualize_token_selector import (  # noqa: E402
    add_label,
    make_contact_sheet,
    make_heatmap_image,
    overlay_heatmap,
)

import torch  # noqa: E402
from PIL import Image  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export paper-ready figures pairing each original frame with its "
            "aggregated ViT query-attention map (one-to-one frame correspondence)."
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
        default=REPO_ROOT / "outputs" / "frame_attention_pairs",
    )
    parser.add_argument("--num-videos", type=int, default=1)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--dtype", choices=("auto", "fp32", "fp16", "bf16"), default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-shards", type=int, default=64)
    parser.add_argument("--shuffle-shards", action="store_true")
    # Uniform frame sampling (only supported video mode here, same as the video tool)
    parser.add_argument("--uniform-video-frames", type=int, default=64)
    parser.add_argument("--uniform-segment-t-size", type=int, default=16)
    # Which segments / queries to visualize
    parser.add_argument(
        "--segments", type=int, nargs="+", default=None,
        help="Segment indices to export. Default: all.",
    )
    parser.add_argument(
        "--query-indices", type=int, nargs="+", default=None,
        help="Query token indices to aggregate. Default: all query tokens.",
    )
    parser.add_argument(
        "--query-reduce", choices=("mean", "max"), default="mean",
        help="How to aggregate attention over the selected query tokens.",
    )
    parser.add_argument(
        "--per-query", action="store_true",
        help="Save one figure per query token instead of aggregating over queries.",
    )
    parser.add_argument("--head-index", type=int, default=-1, help="-1 = average all heads.")
    # Figure layout
    parser.add_argument(
        "--max-frames", type=int, default=8,
        help="Max frames shown per segment figure (uniformly subsampled within the segment).",
    )
    parser.add_argument("--tile-size", type=int, default=224, help="Per-frame tile size in px.")
    parser.add_argument("--overlay-alpha", type=float, default=0.5)
    parser.add_argument(
        "--normalize", choices=("global", "per-frame"), default="global",
        help="global: shared color scale across all frames in the segment; per-frame: independent.",
    )
    parser.add_argument(
        "--include-heatmap-row", action="store_true",
        help="Add a third row showing the pure attention heatmap (no original frame).",
    )
    parser.add_argument(
        "--save-individual", action="store_true",
        help="Also save one [original | overlay] pair PNG per frame.",
    )
    return parser.parse_args()


def aggregate_query_grids(
    attn_seg: np.ndarray,
    positions_seg: np.ndarray,
    num_frames: int,
    grid_h: int,
    grid_w: int,
    query_indices: List[int],
    reduce: str,
) -> np.ndarray:
    """Return (num_frames, grid_h, grid_w) attention aggregated over query tokens.

    attn_seg: (Q, P) attention for one segment.
    positions_seg: (P,) global (frame*grid + patch) position of each ViT token.
    """
    grids = np.stack(
        [
            vector_to_grid(attn_seg[q], positions_seg, num_frames, grid_h, grid_w)
            for q in query_indices
        ],
        axis=0,
    )  # (Qsel, F, gh, gw), NaN where no token maps to that cell
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        if reduce == "max":
            agg = np.nanmax(grids, axis=0)
        else:
            agg = np.nanmean(grids, axis=0)
    return agg.astype(np.float32)


def normalize_global(frame_grids: np.ndarray) -> np.ndarray:
    """Scale finite values across all frames to [0, 1] with a shared min/max."""
    valid = np.isfinite(frame_grids)
    if not np.any(valid):
        return np.zeros_like(frame_grids, dtype=np.float32)
    vals = frame_grids[valid]
    lo, hi = float(vals.min()), float(vals.max())
    span = hi - lo if (hi - lo) > 1e-8 else 1.0
    out = np.full_like(frame_grids, np.nan, dtype=np.float32)
    out[valid] = (frame_grids[valid] - lo) / span
    return out


def build_segment_figure(
    seg_frames: List[Image.Image],
    seg_grids: List[np.ndarray],
    frame_labels: List[int],
    tile_size: int,
    overlay_alpha: float,
    per_map_normalize: str,
    include_heatmap_row: bool,
) -> Image.Image:
    """Stack rows so columns line up one-to-one with frames.

    per_map_normalize is passed to overlay_heatmap ("fixed" when we already
    globally normalized, otherwise "per-map").
    """
    orig_tiles: List[Image.Image] = []
    overlay_tiles: List[Image.Image] = []
    heat_tiles: List[Image.Image] = []

    for frame, grid, f_idx in zip(seg_frames, seg_grids, frame_labels):
        base = frame.convert("RGB").resize((tile_size, tile_size), Image.BICUBIC)
        orig_tiles.append(add_label(base.copy(), f"frame {f_idx:03d}"))

        overlay = overlay_heatmap(
            base,
            grid,
            np.zeros_like(grid, dtype=bool),
            alpha=overlay_alpha,
            normalize=per_map_normalize,
            draw_grid_lines=False,
        )
        overlay_tiles.append(add_label(overlay, f"attn {f_idx:03d}"))

        if include_heatmap_row:
            heat = make_heatmap_image(grid, per_map_normalize, (tile_size, tile_size))
            heat_tiles.append(add_label(heat, f"heat {f_idx:03d}"))

    cols = len(orig_tiles)
    row_sheets = [
        make_contact_sheet(orig_tiles, cols=cols),
        make_contact_sheet(overlay_tiles, cols=cols),
    ]
    if include_heatmap_row:
        row_sheets.append(make_contact_sheet(heat_tiles, cols=cols))

    width = max(s.size[0] for s in row_sheets)
    pad = 8
    height = sum(s.size[1] for s in row_sheets) + pad * (len(row_sheets) - 1)
    figure = Image.new("RGB", (width, height), (245, 245, 245))
    y = 0
    for sheet in row_sheets:
        figure.paste(sheet, (0, y))
        y += sheet.size[1] + pad
    return figure


def render_field(
    frame_grids: np.ndarray,
    frames: List[Image.Image],
    seg_frame_idx: List[int],
    stem: str,
    args: argparse.Namespace,
) -> dict:
    """Slice the (already-selected) frames, normalize, build & save a figure."""
    seg_frames = [frames[i] for i in seg_frame_idx]
    seg_grids_arr = frame_grids[seg_frame_idx]  # (n, gh, gw)

    if args.normalize == "global":
        seg_grids_arr = normalize_global(seg_grids_arr)
        per_map_normalize = "fixed"
    else:
        per_map_normalize = "per-map"

    seg_grids = [seg_grids_arr[i] for i in range(seg_grids_arr.shape[0])]

    figure = build_segment_figure(
        seg_frames=seg_frames,
        seg_grids=seg_grids,
        frame_labels=seg_frame_idx,
        tile_size=args.tile_size,
        overlay_alpha=args.overlay_alpha,
        per_map_normalize=per_map_normalize,
        include_heatmap_row=args.include_heatmap_row,
    )
    fig_path = args.output_dir / (stem + "_frame_attention.png")
    figure.save(fig_path)

    individual: List[str] = []
    if args.save_individual:
        indiv_dir = args.output_dir / (stem + "_frames")
        indiv_dir.mkdir(parents=True, exist_ok=True)
        for frame, grid, f_idx in zip(seg_frames, seg_grids, seg_frame_idx):
            base = frame.convert("RGB").resize(
                (args.tile_size, args.tile_size), Image.BICUBIC
            )
            overlay = overlay_heatmap(
                base, grid, np.zeros_like(grid, dtype=bool),
                alpha=args.overlay_alpha,
                normalize=per_map_normalize,
                draw_grid_lines=False,
            )
            pair = make_contact_sheet(
                [add_label(base.copy(), f"frame {f_idx:03d}"),
                 add_label(overlay, f"attn {f_idx:03d}")],
                cols=2,
            )
            p = indiv_dir / f"frame_{f_idx:03d}.png"
            pair.save(p)
            individual.append(str(p))

    return {
        "figure": str(fig_path),
        "frame_indices": seg_frame_idx,
        "individual": individual,
    }


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
            attn_seg = attn[0, seg_idx].numpy()  # (Q, P)
            positions_seg = pos[seg_idx]          # (P,)

            # Restrict to this segment's frames, uniformly subsampling columns.
            seg_frame_idx = list(range(start, end))
            if len(seg_frame_idx) > args.max_frames:
                pick = np.linspace(0, len(seg_frame_idx) - 1, args.max_frames)
                seg_frame_idx = [seg_frame_idx[int(round(p))] for p in pick]

            stem_base = f"video_{vid_idx:03d}_{sample.key}_seg{seg_idx:02d}"

            if args.per_query:
                # One figure per query token.
                fields = [
                    (
                        [q],
                        f"{stem_base}_q{q:03d}",
                    )
                    for q in query_indices
                ]
            else:
                # Single aggregated figure over all selected query tokens.
                fields = [(query_indices, stem_base)]

            seg_records: List[dict] = []
            for q_subset, stem in fields:
                frame_grids = aggregate_query_grids(
                    attn_seg=attn_seg,
                    positions_seg=positions_seg,
                    num_frames=len(frames),
                    grid_h=grid_h,
                    grid_w=grid_w,
                    query_indices=q_subset,
                    reduce=args.query_reduce,
                )  # (F, gh, gw)
                record = render_field(
                    frame_grids=frame_grids,
                    frames=frames,
                    seg_frame_idx=seg_frame_idx,
                    stem=stem,
                    args=args,
                )
                if args.per_query:
                    record["query"] = q_subset[0]
                seg_records.append(record)

            sample_outputs.append({
                "segment": seg_idx,
                "per_query": args.per_query,
                "frame_indices": seg_frame_idx,
                "outputs": seg_records,
            })
            print(
                f"  [seg{seg_idx}] {len(seg_frame_idx)} frames x "
                f"{len(seg_records)} figure(s)",
                flush=True,
            )

        summaries.append({
            "kind": "video",
            "key": sample.key,
            "attn_shape": list(attn.shape),
            "query_reduce": args.query_reduce,
            "num_queries_used": len(query_indices),
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
                "query_reduce": args.query_reduce,
                "normalize": args.normalize,
                "overlay_alpha": args.overlay_alpha,
                "tile_size": args.tile_size,
                "items": summaries,
            },
            handle,
            indent=2,
            ensure_ascii=True,
        )
    print(f"[done] wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
