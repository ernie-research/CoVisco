#!/usr/bin/env python3
"""Pre-extract local samples for token selector visualization.

This avoids repeatedly scanning large remote WDS directories when running
`tools/visualize_token_selector.py`.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image, ImageOps

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.visualize_token_selector import (  # noqa: E402
    DEFAULT_IMAGE_DATA,
    DEFAULT_VIDEO_DATA,
    VisualSample,
    collect_samples,
)

DEFAULT_CACHE_DIR = REPO_ROOT / "outputs" / "token_selector_sample_cache"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract local image/video samples for token selector visualization.")
    parser.add_argument("--image-data-path", type=Path, default=DEFAULT_IMAGE_DATA)
    parser.add_argument("--video-data-path", type=Path, default=DEFAULT_VIDEO_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--num-images", type=int, default=2)
    parser.add_argument("--num-videos", type=int, default=2)
    parser.add_argument("--max-shards", type=int, default=32)
    parser.add_argument("--shuffle-shards", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-video-frames", type=int, default=128)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def safe_name(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("_") or "sample"


def save_image(image: Image.Image, path: Path) -> None:
    image = ImageOps.exif_transpose(image.convert("RGB"))
    image.save(path, quality=95)


def save_visidx(sample: VisualSample, sample_dir: Path) -> Optional[str]:
    if sample.visidx is None:
        return None
    path = sample_dir / "visidx.npy"
    np.save(path, np.asarray(sample.visidx, dtype=np.int64).reshape(-1))
    return path.name


def write_sample(sample: VisualSample, output_dir: Path, prefix: str, index: int, max_video_frames: int) -> dict:
    name = f"{prefix}_{index:03d}_{safe_name(sample.key)}"
    sample_dir = output_dir / name
    sample_dir.mkdir(parents=True, exist_ok=True)

    item = {
        "key": sample.key,
        "modality": sample.modality,
        "relative_dir": name,
        "source_shard": sample.shard,
        "messages": sample.messages or [],
    }

    if sample.modality == "image":
        if sample.image is None:
            raise ValueError(f"image sample {sample.key} has no image")
        image_name = "image.jpg"
        save_image(sample.image, sample_dir / image_name)
        item["image"] = image_name
    elif sample.modality == "video":
        if not sample.frames:
            raise ValueError(f"video sample {sample.key} has no frames")
        frame_dir = sample_dir / "frames"
        frame_dir.mkdir(parents=True, exist_ok=True)
        frames = sample.frames[:max_video_frames]
        frame_paths: List[str] = []
        for frame_idx, frame in enumerate(frames):
            rel = Path("frames") / f"frame_{frame_idx:04d}.jpg"
            save_image(frame, sample_dir / rel)
            frame_paths.append(rel.as_posix())
        item["frames"] = frame_paths
        item["num_frames_saved"] = len(frame_paths)
        item["num_frames_original"] = len(sample.frames)
    else:
        raise ValueError(f"unsupported modality: {sample.modality}")

    visidx_name = save_visidx(sample, sample_dir)
    if visidx_name is not None:
        item["visidx"] = visidx_name
    return item


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and args.overwrite:
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    image_samples = collect_samples(
        args.image_data_path,
        "image",
        args.num_images,
        max_shards=args.max_shards,
        shuffle=args.shuffle_shards,
        seed=args.seed,
    )
    video_path = args.video_data_path if args.video_data_path.exists() else args.image_data_path
    video_samples = collect_samples(
        video_path,
        "video",
        args.num_videos,
        max_shards=args.max_shards,
        shuffle=args.shuffle_shards,
        seed=args.seed + 17,
    )

    manifest = {
        "format": "onevision_siglip_token_selector_sample_cache/v1",
        "image_data_path": str(args.image_data_path),
        "video_data_path": str(video_path),
        "samples": [],
    }
    for idx, sample in enumerate(image_samples):
        print(f"[write] image {idx}: key={sample.key}", flush=True)
        manifest["samples"].append(write_sample(sample, args.output_dir, "image", idx, args.max_video_frames))
    for idx, sample in enumerate(video_samples):
        print(f"[write] video {idx}: key={sample.key}", flush=True)
        manifest["samples"].append(write_sample(sample, args.output_dir, "video", idx, args.max_video_frames))

    manifest_path = args.output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=True)
    print(f"[done] wrote {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
