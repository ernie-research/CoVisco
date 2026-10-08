#!/usr/bin/env python3
"""Visualize final-layer ViT attention maps for OneVision query tokens.

For each sample, this script extracts the last ViT layer attention from every
query token to the ViT patch tokens, then writes contact sheets of per-query
attention overlays.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models._covisco_encoder_src import CoViscoEncoderAttention  # noqa: E402
from tools.visualize_token_selector import (  # noqa: E402
    DEFAULT_CKPT,
    DEFAULT_CONFIG,
    DEFAULT_IMAGE_DATA,
    DEFAULT_VIDEO_DATA,
    add_label,
    build_vit_and_selector,
    collect_samples,
    load_cached_samples,
    load_checkpoint,
    load_model_config,
    make_contact_sheet,
    overlay_heatmap,
    positions_by_segment,
    preprocess_image,
    prepare_video_frames,
    resize_rgb,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize final ViT-layer attention maps for every query token."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--sample-cache-dir", type=Path, default=None)
    parser.add_argument("--image-data-path", type=Path, default=DEFAULT_IMAGE_DATA)
    parser.add_argument("--video-data-path", type=Path, default=DEFAULT_VIDEO_DATA)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs" / "vit_query_attention")
    parser.add_argument("--num-images", type=int, default=1)
    parser.add_argument("--num-videos", type=int, default=1)
    parser.add_argument("--device", type=str, default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument(
        "--dtype",
        choices=("auto", "fp32", "fp16", "bf16"),
        default="auto",
        help="Model compute dtype. auto uses bf16 on CUDA and fp32 on CPU.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-shards", type=int, default=64)
    parser.add_argument("--shuffle-shards", action="store_true")
    parser.add_argument(
        "--video-mode",
        choices=("visidx", "uniform"),
        default="visidx",
        help="Video path: use visidx sparse candidates, or ignore visidx and uniformly sample frames.",
    )
    parser.add_argument("--uniform-video-frames", type=int, default=64)
    parser.add_argument("--uniform-segment-t-size", type=int, default=16)
    parser.add_argument(
        "--video-viz-mode",
        choices=("temporal-reduce", "per-frame"),
        default="per-frame",
        help=(
            "temporal-reduce: collapse all frames in a segment into one spatial attention map. "
            "per-frame: show each frame's attention map as separate tiles (one row per query token)."
        ),
    )
    parser.add_argument(
        "--video-temporal-reduce",
        choices=("max", "mean", "sum"),
        default="max",
        help="Used with --video-viz-mode temporal-reduce.",
    )
    parser.add_argument("--video-frames-per-query-row", type=int, default=16,
                        help="Max frames shown per query token row when --video-viz-mode per-frame.")
    parser.add_argument("--head-index", type=int, default=-1, help="-1 averages all heads; otherwise select one head.")
    parser.add_argument("--max-query-tokens", type=int, default=100, help="Limit query tokens visualized per segment.")
    parser.add_argument("--query-cols", type=int, default=10)
    parser.add_argument("--tile-size", type=int, default=160)
    parser.add_argument("--overlay-alpha", type=float, default=0.35)
    parser.add_argument("--normalize", choices=("fixed", "per-map"), default="per-map")
    parser.add_argument("--save-individual", action="store_true", help="Also save one PNG per query token.")
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda":
        return torch.device("cuda:0")
    return torch.device(device_arg)


def resolve_dtype(dtype_arg: str, device: torch.device) -> torch.dtype:
    if dtype_arg == "fp32":
        return torch.float32
    if dtype_arg == "fp16":
        return torch.float16
    if dtype_arg == "bf16":
        return torch.bfloat16
    return torch.bfloat16 if device.type == "cuda" else torch.float32


def force_eager_attention(vit) -> None:
    """Ensure the ViT can return attention weights.

    Flash attention implementations usually do not materialize attention maps.
    The eager module has the same projection parameters, so it can receive the
    existing state dict directly.
    """
    replaced = 0
    for layer in vit.encoder.encoder.layers:
        if layer.self_attn.__class__.__name__ == "CoViscoEncoderAttention":
            continue
        eager = CoViscoEncoderAttention(vit.encoder.config)
        eager.load_state_dict(layer.self_attn.state_dict())
        layer.self_attn = eager
        replaced += 1
    if replaced:
        print(f"[ViT] replaced {replaced} attention modules with eager attention", flush=True)


@torch.inference_mode()
def forward_vit_last_query_attention(
    vit,
    pixel_values: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    visidx: Optional[np.ndarray] = None,
    uniform_sample_frames: bool = False,
    uniform_sample_n: int = 16,
    uniform_segment_t_size: int = 4,
    head_index: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run ViT and return query->ViT-patch attention from the last layer.

    Returns:
        query_tokens: (B, S, Q, D)
        vit_tokens: (B, S, P, D)
        query_to_vit: (B, S, Q, P), averaged over heads unless head_index >= 0
    """
    encoder = vit.encoder
    pixel_values = pixel_values.to(device=device, dtype=dtype)
    visible_indices = None
    if visidx is not None:
        visible_indices = torch.as_tensor(visidx, dtype=torch.long, device=device).reshape(1, -1)

    if uniform_sample_frames:
        hidden_states, freqs_visible = encoder.get_uniform_frame_segments(
            pixel_values,
            seg_offset=0,
            sample_frames=uniform_sample_n,
            segment_t_size=uniform_segment_t_size,
        )
    else:
        hidden_states, freqs_visible = encoder.get_segments(pixel_values, visible_indices, seg_offset=0)

    batch_size, num_segments, vit_per_segment, hidden_size = hidden_states.shape
    num_query = encoder.num_query_per_seg
    query_tokens = encoder.query_tokens.expand(batch_size, num_segments, -1, -1)
    hidden_states = torch.cat([query_tokens, hidden_states], dim=2)

    query_rope = encoder.get_query_tokens_rope(
        batch_size=batch_size,
        num_seg=num_segments,
        device=hidden_states.device,
        seg_offset=0,
    )
    freqs_visible = torch.cat([query_rope, freqs_visible], dim=2)
    hidden_states = encoder.layernorm_pre(hidden_states)

    last_attn = None
    layers = encoder.encoder.layers
    for layer_idx, layer in enumerate(layers):
        want_attention = layer_idx == len(layers) - 1
        layer_outputs = layer(
            hidden_states,
            attention_mask=None,
            rotary_pos_emb=freqs_visible,
            output_attentions=want_attention,
            layer_idx=layer_idx,
        )
        hidden_states = layer_outputs[0]
        if want_attention:
            last_attn = layer_outputs[1]

    if last_attn is None:
        raise RuntimeError(
            "Last layer attention is None. The ViT attention implementation must support output_attentions=True."
        )

    # last_attn: (B*S, H, Q_len, K_len). Last layer is odd for the 24-layer
    # config, so keys are [all segment query tokens, local segment ViT tokens].
    if head_index >= 0:
        if head_index >= last_attn.shape[1]:
            raise ValueError(f"head_index={head_index} out of range for {last_attn.shape[1]} heads")
        attn = last_attn[:, head_index, :num_query, :]
    else:
        attn = last_attn[:, :, :num_query, :].mean(dim=1)

    global_query_key_count = num_segments * num_query
    if attn.shape[-1] >= global_query_key_count + vit_per_segment:
        vit_key_start = global_query_key_count
    else:
        vit_key_start = num_query
    attn = attn[..., vit_key_start : vit_key_start + vit_per_segment]
    attn = attn.reshape(batch_size, num_segments, num_query, vit_per_segment)

    query_tokens_out = hidden_states[..., :num_query, :]
    vit_tokens_out = hidden_states[..., num_query:, :]
    return query_tokens_out, vit_tokens_out, attn.float().cpu()


def vector_to_grid(values: np.ndarray, positions: np.ndarray, num_frames: int, grid_h: int, grid_w: int) -> np.ndarray:
    grid = np.full((num_frames, grid_h, grid_w), np.nan, dtype=np.float32)
    total = num_frames * grid_h * grid_w
    for value, pos in zip(values.tolist(), positions.tolist()):
        pos = int(pos)
        if pos < 0 or pos >= total:
            continue
        frame = pos // (grid_h * grid_w)
        rem = pos % (grid_h * grid_w)
        y = rem // grid_w
        x = rem % grid_w
        grid[frame, y, x] = float(value)
    return grid


def reduce_video_grid(grid: np.ndarray, start: int, end: int, method: str) -> np.ndarray:
    sub = grid[start:end]
    valid = np.isfinite(sub)
    valid_any = valid.any(axis=0)
    out = np.full(sub.shape[1:], np.nan, dtype=np.float32)
    if not np.any(valid_any):
        return out
    if method == "max":
        filled = np.where(valid, sub, -np.inf)
        out[valid_any] = filled.max(axis=0)[valid_any]
    elif method == "mean":
        total = np.where(valid, sub, 0.0).sum(axis=0)
        count = valid.sum(axis=0)
        out[valid_any] = (total[valid_any] / count[valid_any]).astype(np.float32)
    else:  # sum
        total = np.where(valid, sub, 0.0).sum(axis=0)
        out[valid_any] = total[valid_any]
    return out


def make_query_sheet(
    base: Image.Image,
    query_maps: List[np.ndarray],
    cols: int,
    tile_size: int,
    overlay_alpha: float,
    normalize: str,
    title_prefix: str,
) -> Image.Image:
    base = base.convert("RGB").resize((tile_size, tile_size), Image.BICUBIC)
    tiles: List[Image.Image] = []
    for query_idx, score_map in enumerate(query_maps):
        empty_mask = np.zeros_like(score_map, dtype=bool)
        overlay = overlay_heatmap(
            base,
            score_map,
            empty_mask,
            alpha=overlay_alpha,
            normalize=normalize,
            draw_grid_lines=False,
        )
        tiles.append(add_label(overlay, f"{title_prefix} q{query_idx:03d}"))
    return make_contact_sheet(tiles, cols=cols)


def save_image_attention(
    sample,
    resized_image: Image.Image,
    query_to_vit: torch.Tensor,
    output_dir: Path,
    sample_idx: int,
    args: argparse.Namespace,
    patch_size: int,
) -> Dict:
    grid_h = resized_image.size[1] // patch_size
    grid_w = resized_image.size[0] // patch_size
    positions = np.arange(grid_h * grid_w, dtype=np.int64)
    query_count = min(args.max_query_tokens, query_to_vit.shape[2])
    query_maps = []
    for query_idx in range(query_count):
        vec = query_to_vit[0, 0, query_idx].numpy()
        query_maps.append(vector_to_grid(vec, positions, 1, grid_h, grid_w)[0])

    sheet = make_query_sheet(
        resized_image,
        query_maps,
        cols=args.query_cols,
        tile_size=args.tile_size,
        overlay_alpha=args.overlay_alpha,
        normalize=args.normalize,
        title_prefix="img",
    )
    out_path = output_dir / f"image_{sample_idx:03d}_{sample.key}_query_attention_sheet.png"
    sheet.save(out_path)

    individual = []
    if args.save_individual:
        indiv_dir = output_dir / f"image_{sample_idx:03d}_{sample.key}_queries"
        indiv_dir.mkdir(parents=True, exist_ok=True)
        base = resized_image.convert("RGB").resize((args.tile_size, args.tile_size), Image.BICUBIC)
        for query_idx, score_map in enumerate(query_maps):
            overlay = overlay_heatmap(
                base,
                score_map,
                np.zeros_like(score_map, dtype=bool),
                alpha=args.overlay_alpha,
                normalize=args.normalize,
                draw_grid_lines=False,
            )
            path = indiv_dir / f"query_{query_idx:03d}.png"
            overlay.save(path)
            individual.append(str(path))

    return {
        "kind": "image",
        "key": sample.key,
        "shard": sample.shard,
        "sheet": str(out_path),
        "individual": individual,
        "query_count": query_count,
        "attention_shape": list(query_to_vit.shape),
    }


def make_query_per_frame_sheet(
    frame_list: List[Image.Image],
    frame_grids: List[np.ndarray],
    query_idx: int,
    tile_size: int,
    overlay_alpha: float,
    normalize: str,
    max_frames: int = 16,
) -> Image.Image:
    """One row per query token: tile each frame with its per-frame attention overlay."""
    step = max(1, len(frame_list) // max_frames)
    chosen_frames = list(range(0, len(frame_list), step))[:max_frames]
    tiles: List[Image.Image] = []
    for f_idx in chosen_frames:
        score_map = frame_grids[f_idx]
        base = frame_list[f_idx].convert("RGB").resize((tile_size, tile_size), Image.BICUBIC)
        overlay = overlay_heatmap(
            base,
            score_map,
            np.zeros_like(score_map, dtype=bool),
            alpha=overlay_alpha,
            normalize=normalize,
            draw_grid_lines=False,
        )
        tiles.append(add_label(overlay, f"q{query_idx:03d} f{f_idx:03d}"))
    return make_contact_sheet(tiles, cols=len(tiles))


def save_video_attention(
    sample,
    frames: List[Image.Image],
    query_to_vit: torch.Tensor,
    visidx: Optional[np.ndarray],
    output_dir: Path,
    sample_idx: int,
    args: argparse.Namespace,
    patch_size: int,
) -> Dict:
    grid_h = frames[0].size[1] // patch_size
    grid_w = frames[0].size[0] // patch_size
    _, num_segments, num_query, vit_per_segment = query_to_vit.shape
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
    query_count = min(args.max_query_tokens, num_query)
    segments = []

    for seg_idx in range(num_segments):
        start = seg_idx * frames_per_segment
        end = len(frames) if seg_idx == num_segments - 1 else min(len(frames), (seg_idx + 1) * frames_per_segment)
        base_frame_idx = min(len(frames) - 1, start + max(0, (end - start) // 2))
        base = frames[base_frame_idx]

        # Build per-query, per-frame attention grids
        per_query_frame_grids = []
        for query_idx in range(query_count):
            vec = query_to_vit[0, seg_idx, query_idx].numpy()
            per_query_frame_grids.append(
                vector_to_grid(vec, pos[seg_idx], len(frames), grid_h, grid_w)
            )

        out_paths = []
        if args.video_viz_mode == "per-frame":
            # Per-query rows: each tile shows one frame with its per-frame spatial attention
            max_f = args.video_frames_per_query_row
            for query_idx, frame_grids in enumerate(per_query_frame_grids):
                seg_frames = frames[start:end]
                step = max(1, len(seg_frames) // max_f)
                chosen = list(range(0, len(seg_frames), step))[:max_f]
                tiles: List[Image.Image] = []
                for fi in chosen:
                    fm = frame_grids[start + fi]
                    base_f = seg_frames[fi].convert("RGB").resize((args.tile_size, args.tile_size), Image.BICUBIC)
                    overlay = overlay_heatmap(
                        base_f, fm,
                        np.zeros_like(fm, dtype=bool),
                        alpha=args.overlay_alpha,
                        normalize=args.normalize,
                        draw_grid_lines=False,
                    )
                    tiles.append(add_label(overlay, f"q{query_idx:03d} f{start+fi:03d}"))
                row_path = output_dir / (
                    f"video_{sample_idx:03d}_{sample.key}_segment_{seg_idx:02d}_q{query_idx:03d}_frames.png"
                )
                make_contact_sheet(tiles, cols=len(tiles)).save(row_path)
                out_paths.append(str(row_path))

            # Compact overview: one tile per query, temporal-max collapsed
            overview_maps = [
                reduce_video_grid(fg, start, end, "max") for fg in per_query_frame_grids
            ]
            overview_sheet = make_query_sheet(
                base,
                overview_maps,
                cols=args.query_cols,
                tile_size=args.tile_size,
                overlay_alpha=args.overlay_alpha,
                normalize=args.normalize,
                title_prefix=f"s{seg_idx}",
            )
            overview_path = output_dir / (
                f"video_{sample_idx:03d}_{sample.key}_segment_{seg_idx:02d}_query_overview_sheet.png"
            )
            overview_sheet.save(overview_path)
            out_paths.insert(0, str(overview_path))
        else:
            # temporal-reduce: collapse all frames → one spatial map per query
            query_maps = [
                reduce_video_grid(fg, start, end, args.video_temporal_reduce)
                for fg in per_query_frame_grids
            ]
            sheet = make_query_sheet(
                base,
                query_maps,
                cols=args.query_cols,
                tile_size=args.tile_size,
                overlay_alpha=args.overlay_alpha,
                normalize=args.normalize,
                title_prefix=f"s{seg_idx}",
            )
            out_path = output_dir / (
                f"video_{sample_idx:03d}_{sample.key}_segment_{seg_idx:02d}_query_attention_sheet.png"
            )
            sheet.save(out_path)
            out_paths.append(str(out_path))

        individual = []
        if args.save_individual:
            indiv_dir = output_dir / f"video_{sample_idx:03d}_{sample.key}_segment_{seg_idx:02d}_queries"
            indiv_dir.mkdir(parents=True, exist_ok=True)
            for query_idx, frame_grids in enumerate(per_query_frame_grids):
                score_map = reduce_video_grid(frame_grids, start, end, "max")
                base_r = base.convert("RGB").resize((args.tile_size, args.tile_size), Image.BICUBIC)
                overlay = overlay_heatmap(
                    base_r, score_map,
                    np.zeros_like(score_map, dtype=bool),
                    alpha=args.overlay_alpha,
                    normalize=args.normalize,
                    draw_grid_lines=False,
                )
                path = indiv_dir / f"query_{query_idx:03d}.png"
                overlay.save(path)
                individual.append(str(path))

        segments.append(
            {
                "segment": seg_idx,
                "outputs": out_paths,
                "individual": individual,
                "base_frame": base_frame_idx,
                "frame_range": [start, end],
            }
        )

    return {
        "kind": "video",
        "key": sample.key,
        "shard": sample.shard,
        "query_count": query_count,
        "attention_shape": list(query_to_vit.shape),
        "video_viz_mode": args.video_viz_mode,
        "segments": segments,
    }


def collect_inputs(args: argparse.Namespace):
    if args.sample_cache_dir is not None:
        images = load_cached_samples(args.sample_cache_dir, "image", args.num_images)
        videos = load_cached_samples(args.sample_cache_dir, "video", args.num_videos)
    else:
        images = collect_samples(
            args.image_data_path,
            "image",
            args.num_images,
            max_shards=args.max_shards,
            shuffle=args.shuffle_shards,
            seed=args.seed,
        )
        video_path = args.video_data_path if args.video_data_path.exists() else args.image_data_path
        videos = collect_samples(
            video_path,
            "video",
            args.num_videos,
            max_shards=args.max_shards,
            shuffle=args.shuffle_shards,
            seed=args.seed + 17,
        )
    return images, videos


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

    for idx, sample in enumerate(image_samples):
        print(f"[run] image {idx}: key={sample.key}", flush=True)
        pixel_values, resized = preprocess_image(sample.image, cfg.vit.image_size)
        q, v, attn = forward_vit_last_query_attention(
            vit,
            pixel_values,
            device=device,
            dtype=dtype,
            head_index=args.head_index,
        )
        print(f"[shape] image q={tuple(q.shape)} vit={tuple(v.shape)} query_attn={tuple(attn.shape)}", flush=True)
        summaries.append(save_image_attention(sample, resized, attn, args.output_dir, idx, args, cfg.vit.patch_size))
        del q, v, attn
        if device.type == "cuda":
            torch.cuda.empty_cache()

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
        q, v, attn = forward_vit_last_query_attention(
            vit,
            pixel_values,
            device=device,
            dtype=dtype,
            visidx=visidx,
            head_index=args.head_index,
            **forward_kwargs,
        )
        print(f"[shape] video q={tuple(q.shape)} vit={tuple(v.shape)} query_attn={tuple(attn.shape)}", flush=True)
        summaries.append(save_video_attention(sample, frames, attn, visidx, args.output_dir, idx, args, cfg.vit.patch_size))
        del q, v, attn
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
