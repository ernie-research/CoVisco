#!/usr/bin/env python3
"""Visualize the CoVisco token selector on WDS image/video samples.

The script loads only the ViT and token selector weights from a training
checkpoint. It does not instantiate the Qwen LLM, which keeps the visualization
path much lighter than a full model forward.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import sys
import tarfile
import tempfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw, ImageOps

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models._covisco_encoder_src import CoViscoEncoderConfig
from models.config import build_model_config
from models.covisco_vit import CoViscoViT
from models.token_selector import LearnableTokenSelector

DEFAULT_CONFIG = REPO_ROOT / "configs" / "covisco_qwen3_1.7b.yaml"
DEFAULT_CKPT = None
DEFAULT_IMAGE_DATA = None
DEFAULT_VIDEO_DATA = None

SIGLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
SIGLIP_STD = (0.26862954, 0.26130258, 0.27577711)


@dataclass
class VisualSample:
    key: str
    shard: str
    modality: str
    image: Optional[Image.Image] = None
    frames: Optional[List[Image.Image]] = None
    visidx: Optional[np.ndarray] = None
    messages: Optional[List[dict]] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize query-guided ViT token selection for CoVisco."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--image-data-path", type=Path, default=DEFAULT_IMAGE_DATA)
    parser.add_argument("--video-data-path", type=Path, default=DEFAULT_VIDEO_DATA)
    parser.add_argument(
        "--sample-cache-dir",
        type=Path,
        default=None,
        help="Use pre-extracted local samples from tools/extract_token_selector_samples.py.",
    )
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs" / "token_selector_viz")
    parser.add_argument("--num-images", type=int, default=1)
    parser.add_argument("--num-videos", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=204)
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
        help="Video path: use dataset visidx sparse candidates, or ignore visidx and uniformly sample frames.",
    )
    parser.add_argument(
        "--uniform-video-frames",
        type=int,
        default=64,
        help="Number of frames to feed in --video-mode uniform.",
    )
    parser.add_argument(
        "--uniform-segment-t-size",
        type=int,
        default=16,
        help="Segment size for --video-mode uniform; 64 frames with 16 gives 4 segments.",
    )
    parser.add_argument(
        "--video-frames-without-visidx",
        type=int,
        default=64,
        help="Deprecated alias for --uniform-video-frames.",
    )
    parser.add_argument(
        "--use-visidx",
        choices=("auto", "always", "never"),
        default="auto",
        help="Deprecated; use --video-mode instead.",
    )
    parser.add_argument(
        "--video-frames-per-segment-viz",
        type=int,
        default=8,
        help="Frames shown in each video segment contact sheet.",
    )
    parser.add_argument("--overlay-alpha", type=float, default=0.25)
    parser.add_argument(
        "--normalize",
        choices=("fixed", "per-map"),
        default="fixed",
        help="fixed maps sigmoid scores with vmin=0,vmax=1; per-map stretches each heatmap.",
    )
    parser.add_argument("--draw-grid", action="store_true")
    parser.add_argument("--save-heatmap-only", action="store_true")
    parser.add_argument(
        "--export-video",
        choices=("gif", "mp4", "both", "none"),
        default="none",
        help="Export per-segment frame sequence as GIF and/or MP4 in addition to the contact-sheet PNG.",
    )
    parser.add_argument(
        "--image-resolution",
        type=int,
        default=None,
        help="Image resolution (height=width) for preprocessing. Overrides cfg.vit.image_size when set.",
    )
    parser.add_argument(
        "--video-resolution",
        type=int,
        default=None,
        help="Video frame resolution (height=width) for preprocessing. Overrides cfg.vit.image_size when set.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=4.0,
        help="Frame rate used when writing GIF / MP4 animations.",
    )
    parser.add_argument(
        "--analyze-norm-bias",
        action="store_true",
        help=(
            "Compute per-token L2 norm of ViT tokens and report Pearson / Spearman correlation "
            "with selector scores. Prints a summary table and saves norm_bias_analysis.json."
        ),
    )
    parser.add_argument(
        "--zero-query-tokens",
        action="store_true",
        help=(
            "Zero out the query tokens before feeding them into the token selector. "
            "Use this to ablate the effect of query tokens on selector scores."
        ),
    )
    parser.add_argument(
        "--mmr",
        action="store_true",
        help=(
            "Apply Maximal Marginal Relevance reranking after token selection. "
            "Replaces the top-K indices with a more diverse set of the same size, "
            "trading off relevance (selector score) against redundancy with already-selected tokens."
        ),
    )
    parser.add_argument(
        "--mmr-lambda",
        type=float,
        default=0.5,
        help=(
            "Lambda trade-off for MMR: 1.0 = pure relevance (same as top-K), "
            "0.0 = pure diversity. Default 0.5."
        ),
    )
    parser.add_argument(
        "--first-frame-max-ratio",
        type=float,
        default=0.0,
        help=(
            "Inference-only (video only): cap the fraction of top-K tokens that can come "
            "from the first frame. 0.0 = no cap (default). E.g. 0.6 means at most 60%% of "
            "patches_per_frame tokens may be selected from the first frame."
        ),
    )
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


def load_model_config(config_path: Path):
    with config_path.open("r", encoding="utf-8") as handle:
        return build_model_config(yaml.safe_load(handle))


def build_vit_and_selector(cfg):
    vit_cfg = CoViscoEncoderConfig(
        hidden_size=cfg.vit.hidden_size,
        intermediate_size=cfg.vit.intermediate_size,
        num_hidden_layers=cfg.vit.num_layers,
        num_attention_heads=cfg.vit.num_attention_heads,
        num_channels=cfg.vit.num_channels,
        image_size=cfg.vit.image_size,
        patch_size=cfg.vit.patch_size,
        num_query_per_seg=cfg.vit.num_query_per_seg,
        segment_t_size=cfg.vit.segment_t_size,
        rope_theta=cfg.vit.rope_theta,
        layer_norm_eps=cfg.vit.layer_norm_eps,
        initializer_range=cfg.vit.initializer_range,
    )
    vit = CoViscoViT(vit_cfg)
    selector = LearnableTokenSelector(
        hidden_size=cfg.vit.hidden_size,
        num_layers=cfg.token_selector.num_layers,
        num_heads=cfg.token_selector.num_heads,
        selection_mode=cfg.token_selector.method,
        score_activation=cfg.token_selector.score_activation,
        gumbel_temperature=cfg.token_selector.gumbel_temperature,
        mmr_lambda=cfg.token_selector.mmr_lambda,
        logit_scale=cfg.token_selector.logit_scale,
    )
    return vit, selector


def _strip_module_prefix(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = {}
    for key, value in state.items():
        if key.startswith("module."):
            key = key[len("module.") :]
        out[key] = value
    return out


def _sub_state(state: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    needle = prefix + "."
    return {key[len(needle) :]: value for key, value in state.items() if key.startswith(needle)}


def load_checkpoint(vit, selector, ckpt_path: Path) -> None:
    print(f"[load] checkpoint: {ckpt_path}")
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    state = ckpt.get("model_state", ckpt)
    state = _strip_module_prefix(state)

    vit_state = _sub_state(state, "vit")
    selector_state = _sub_state(state, "token_selector")
    if not vit_state:
        raise RuntimeError("No vit.* weights found in checkpoint")
    if not selector_state:
        raise RuntimeError("No token_selector.* weights found in checkpoint")

    vit_missing, vit_unexpected = vit.load_state_dict(vit_state, strict=False)
    sel_missing, sel_unexpected = selector.load_state_dict(selector_state, strict=False)
    print(
        f"[load] vit tensors={len(vit_state)} missing={len(vit_missing)} unexpected={len(vit_unexpected)}"
    )
    if vit_missing or vit_unexpected:
        print(f"[warn] vit missing={vit_missing[:8]} unexpected={vit_unexpected[:8]}")
    print(
        f"[load] selector tensors={len(selector_state)} missing={len(sel_missing)} unexpected={len(sel_unexpected)}"
    )
    if sel_missing or sel_unexpected:
        print(f"[warn] selector missing={sel_missing[:8]} unexpected={sel_unexpected[:8]}")
    del ckpt, state, vit_state, selector_state
    gc.collect()


def expand_shards(path: Path, max_shards: int, shuffle: bool, seed: int) -> List[Path]:
    if not path.exists():
        print(f"[warn] data path does not exist: {path}")
        return []
    if path.is_file() and path.suffix.lower() == ".tar":
        shards = [path]
    elif path.is_file():
        with path.open("r", encoding="utf-8") as handle:
            shards = [Path(line.strip()) for line in handle if line.strip()]
    elif shuffle:
        shards = sorted(path.glob("*.tar"))
    else:
        # Large mounted dataset directories can contain many shards. For the
        # default deterministic path, stop as soon as enough tar files are seen.
        shards = []
        with os.scandir(path) as entries:
            for entry in entries:
                if entry.is_file() and entry.name.endswith(".tar"):
                    shards.append(Path(entry.path))
                    if len(shards) >= max_shards:
                        break
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(shards)
    return shards[:max_shards]


def _member_key(member_name: str) -> str:
    name = Path(member_name).name
    return name.split(".")[0]


def _read_member(tf: tarfile.TarFile, member: tarfile.TarInfo) -> bytes:
    extracted = tf.extractfile(member)
    if extracted is None:
        return b""
    return extracted.read()


def decode_mp4_frames(mp4_bytes: bytes, target_frames: int = 128) -> List[Image.Image]:
    try:
        import cv2
    except Exception as exc:  # pragma: no cover - depends on local env
        print(f"[warn] cv2 is unavailable, cannot decode mp4 video: {exc}")
        return []

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(mp4_bytes)
        tmp_path = tmp.name
    frames: List[Image.Image] = []
    try:
        cap = cv2.VideoCapture(tmp_path)
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(rgb))
        cap.release()
    finally:
        os.unlink(tmp_path)

    if not frames:
        return frames
    if len(frames) < target_frames:
        frames.extend([frames[-1].copy() for _ in range(target_frames - len(frames))])
    elif len(frames) > target_frames:
        frames = frames[:target_frames]
    return frames


def load_group(tf: tarfile.TarFile, key: str, members: Sequence[tarfile.TarInfo]) -> Optional[VisualSample]:
    images: List[Image.Image] = []
    json_data: dict = {}
    mp4_bytes: Optional[bytes] = None
    visidx: Optional[np.ndarray] = None

    for member in sorted(members, key=lambda m: m.name):
        if not member.isfile():
            continue
        ext = member.name.rsplit(".", 1)[-1].lower()
        lower_name = member.name.lower()
        data = _read_member(tf, member)
        if not data:
            continue
        if ext in ("jpg", "jpeg", "png", "webp"):
            try:
                images.append(Image.open(BytesIO(data)).convert("RGB").copy())
            except Exception as exc:
                print(f"[warn] failed to decode image {member.name}: {exc}")
        elif ext == "json":
            try:
                json_data = json.loads(data.decode("utf-8"))
            except Exception as exc:
                print(f"[warn] failed to decode json {member.name}: {exc}")
        elif ext == "mp4":
            mp4_bytes = data
        elif ext == "npy" and "visidx" in lower_name:
            try:
                visidx = np.load(BytesIO(data), allow_pickle=True)
            except Exception as exc:
                print(f"[warn] failed to decode visidx {member.name}: {exc}")

    messages = json_data.get("messages", []) if isinstance(json_data, dict) else []
    modality = json_data.get("modality") if isinstance(json_data, dict) else None
    if modality is None:
        text = "\n".join(str(msg.get("content", "")) for msg in messages)
        if "<video>" in text or mp4_bytes is not None:
            modality = "video"
        elif "<image>" in text or images:
            modality = "image"
        else:
            modality = "text"

    if modality == "video":
        frames = images
        if mp4_bytes is not None:
            frames = decode_mp4_frames(mp4_bytes)
        if not frames:
            return None
        return VisualSample(
            key=key,
            shard=tf.name,
            modality="video",
            frames=frames,
            visidx=visidx,
            messages=messages,
        )
    if modality == "image" and images:
        return VisualSample(
            key=key,
            shard=tf.name,
            modality="image",
            image=images[0],
            visidx=visidx,
            messages=messages,
        )
    return None


def iter_samples_from_shard(shard: Path) -> Iterable[VisualSample]:
    try:
        with tarfile.open(str(shard), "r:*") as tf:
            groups: Dict[str, List[tarfile.TarInfo]] = {}
            for member in tf.getmembers():
                if member.isfile():
                    groups.setdefault(_member_key(member.name), []).append(member)
            for key, members in groups.items():
                sample = load_group(tf, key, members)
                if sample is not None:
                    yield sample
    except Exception as exc:
        print(f"[warn] failed to read shard {shard}: {exc}")


def collect_samples(
    data_path: Path,
    modality: str,
    count: int,
    max_shards: int,
    shuffle: bool,
    seed: int,
) -> List[VisualSample]:
    if count <= 0:
        return []
    samples: List[VisualSample] = []
    for shard in expand_shards(data_path, max_shards=max_shards, shuffle=shuffle, seed=seed):
        print(f"[data] scanning {modality} shard: {shard}", flush=True)
        for sample in iter_samples_from_shard(shard):
            if sample.modality != modality:
                continue
            samples.append(sample)
            print(f"[data] picked {modality}: key={sample.key} shard={Path(sample.shard).name}", flush=True)
            if len(samples) >= count:
                return samples
    return samples


def load_cached_samples(cache_dir: Path, modality: str, count: int) -> List[VisualSample]:
    if count <= 0:
        return []
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"sample cache manifest not found: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    samples: List[VisualSample] = []
    for item in manifest.get("samples", []):
        if item.get("modality") != modality:
            continue
        sample_dir = cache_dir / item["relative_dir"]
        visidx = None
        visidx_rel = item.get("visidx")
        if visidx_rel:
            visidx_path = sample_dir / visidx_rel
            if visidx_path.exists():
                visidx = np.load(visidx_path, allow_pickle=True)

        if modality == "image":
            image_path = sample_dir / item["image"]
            sample = VisualSample(
                key=item["key"],
                shard=item.get("source_shard", "cache"),
                modality="image",
                image=Image.open(image_path).convert("RGB"),
                visidx=visidx,
                messages=item.get("messages"),
            )
        else:
            frames = [
                Image.open(sample_dir / rel_path).convert("RGB")
                for rel_path in item.get("frames", [])
            ]
            sample = VisualSample(
                key=item["key"],
                shard=item.get("source_shard", "cache"),
                modality="video",
                frames=frames,
                visidx=visidx,
                messages=item.get("messages"),
            )
        samples.append(sample)
        print(f"[cache] picked {modality}: key={sample.key} dir={sample_dir}", flush=True)
        if len(samples) >= count:
            break
    return samples


def resize_rgb(image: Image.Image, size: int) -> Image.Image:
    image = ImageOps.exif_transpose(image.convert("RGB"))
    return image.resize((size, size), Image.BICUBIC)


def pil_to_normalized_tensor(image: Image.Image) -> torch.Tensor:
    arr = np.asarray(image).astype(np.float32) / 255.0
    mean = np.asarray(SIGLIP_MEAN, dtype=np.float32).reshape(1, 1, 3)
    std = np.asarray(SIGLIP_STD, dtype=np.float32).reshape(1, 1, 3)
    arr = (arr - mean) / std
    arr = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(arr)


def preprocess_image(image: Image.Image, image_size: int) -> Tuple[torch.Tensor, Image.Image]:
    resized = resize_rgb(image, image_size)
    tensor = pil_to_normalized_tensor(resized).unsqueeze(0)
    return tensor, resized


def _even_indices(total: int, wanted: int) -> List[int]:
    if total <= 0:
        return []
    if wanted <= 0:
        raise ValueError("wanted frame count must be positive")
    if total == 1:
        return [0] * wanted
    return [int(round(x)) for x in np.linspace(0, total - 1, wanted)]


def _valid_segment_frame_count(n: int, segment_t_size: int) -> None:
    if n >= segment_t_size and n % segment_t_size != 0:
        raise ValueError(
            f"video frame count {n} must be < {segment_t_size} or divisible by {segment_t_size}"
        )


def _frames_needed_by_visidx(visidx: np.ndarray, patches_per_frame: int, segment_t_size: int) -> int:
    max_idx = int(np.max(visidx)) if visidx.size else 0
    needed = max_idx // patches_per_frame + 1
    if needed >= segment_t_size and needed % segment_t_size != 0:
        needed = int(math.ceil(needed / segment_t_size) * segment_t_size)
    return max(1, needed)


def prepare_video_frames(
    sample: VisualSample,
    image_size: int,
    patch_size: int,
    segment_t_size: int,
    video_mode: str,
    uniform_video_frames: int,
    uniform_segment_t_size: int,
) -> Tuple[torch.Tensor, List[Image.Image], Optional[np.ndarray], str, dict]:
    if sample.frames is None:
        raise ValueError("video sample has no frames")
    frames = sample.frames
    grid = image_size // patch_size
    patches_per_frame = grid * grid

    if video_mode == "visidx":
        if sample.visidx is None:
            raise ValueError(f"sample {sample.key} has no visidx.npy for --video-mode visidx")
        visidx = np.asarray(sample.visidx, dtype=np.int64).reshape(-1)
        needed = _frames_needed_by_visidx(visidx, patches_per_frame, segment_t_size)
        chosen = list(range(min(len(frames), needed)))
        if len(chosen) < needed:
            if not chosen:
                raise ValueError(f"sample {sample.key} has no decodable frames")
            chosen.extend([chosen[-1]] * (needed - len(chosen)))
        selected_frames = [resize_rgb(frames[i], image_size) for i in chosen]
        note = f"mode=visidx candidates={len(visidx)} frames={len(selected_frames)} segment_t_size={segment_t_size}"
        _valid_segment_frame_count(len(selected_frames), segment_t_size)
        forward_kwargs = {
            "uniform_sample_frames": False,
            "uniform_sample_n": 16,
            "uniform_segment_t_size": 4,
        }
    elif video_mode == "uniform":
        _valid_segment_frame_count(uniform_video_frames, uniform_segment_t_size)
        chosen = _even_indices(len(frames), uniform_video_frames)
        if not chosen:
            raise ValueError(f"sample {sample.key} has no decodable frames")
        selected_frames = [resize_rgb(frames[i], image_size) for i in chosen]
        visidx = None
        num_segments = len(selected_frames) // uniform_segment_t_size
        note = (
            f"mode=uniform sampled_frames={len(selected_frames)} "
            f"segment_t_size={uniform_segment_t_size} num_segments={num_segments}"
        )
        forward_kwargs = {
            "uniform_sample_frames": True,
            "uniform_sample_n": uniform_video_frames,
            "uniform_segment_t_size": uniform_segment_t_size,
        }
    else:
        raise ValueError(f"unsupported video_mode: {video_mode}")

    frame_tensors = [pil_to_normalized_tensor(frame) for frame in selected_frames]
    tensor = torch.stack(frame_tensors, dim=1).unsqueeze(0)  # (1, C, T, H, W)
    return tensor, selected_frames, visidx, note, forward_kwargs


def selector_forward(
    vit: CoViscoViT,
    selector: LearnableTokenSelector,
    pixel_values: torch.Tensor,
    modality: str,
    top_k: int,
    device: torch.device,
    dtype: torch.dtype,
    visidx: Optional[np.ndarray] = None,
    uniform_sample_frames: bool = False,
    uniform_sample_n: int = 16,
    uniform_segment_t_size: int = 4,
    zero_query_tokens: bool = False,
    use_mmr: bool = False,
    mmr_lambda: float = 0.5,
    first_frame_max_ratio: float = 0.0,
    patches_per_frame: Optional[int] = None,
):
    # Temporarily override selector.mmr_lambda so MMR runs inside the selector
    # itself rather than as a post-processing step.
    original_mmr_lambda = selector.mmr_lambda
    if use_mmr:
        selector.mmr_lambda = mmr_lambda
    try:
        pixel_values = pixel_values.to(device=device, dtype=dtype)
        visidx_tensor = None
        if visidx is not None:
            visidx_tensor = torch.as_tensor(visidx, dtype=torch.long, device=device).reshape(1, -1)
        with torch.inference_mode():
            query_tokens, vit_tokens, _ = vit(
                pixel_values=pixel_values,
                visidx=visidx_tensor,
                modality=modality,
                uniform_sample_frames=uniform_sample_frames,
                uniform_sample_n=uniform_sample_n,
                uniform_segment_t_size=uniform_segment_t_size,
            )
            if zero_query_tokens:
                query_tokens = torch.zeros_like(query_tokens)
            out = selector(
                query_tokens, vit_tokens, top_k=top_k,
                patches_per_frame=patches_per_frame if modality == "video" else None,
                first_frame_max_ratio=first_frame_max_ratio if modality == "video" else 0.0,
            )
    finally:
        selector.mmr_lambda = original_mmr_lambda
    return query_tokens, vit_tokens, out


def turbo_like_colormap(values: np.ndarray) -> np.ndarray:
    values = np.nan_to_num(values, nan=0.0, posinf=1.0, neginf=0.0)
    values = np.clip(values, 0.0, 1.0)
    anchors_x = np.array([0.0, 0.20, 0.45, 0.70, 1.0], dtype=np.float32)
    anchors_rgb = np.array(
        [
            [35, 55, 145],
            [40, 170, 225],
            [75, 210, 115],
            [245, 205, 55],
            [180, 35, 35],
        ],
        dtype=np.float32,
    )
    flat = values.reshape(-1)
    rgb = np.empty((flat.size, 3), dtype=np.float32)
    for channel in range(3):
        rgb[:, channel] = np.interp(flat, anchors_x, anchors_rgb[:, channel])
    return rgb.reshape(values.shape + (3,)).astype(np.uint8)


def normalize_scores(score_map: np.ndarray, mode: str) -> np.ndarray:
    valid = np.isfinite(score_map)
    if not np.any(valid):
        return np.zeros_like(score_map, dtype=np.float32)
    if mode == "fixed":
        return np.clip(score_map, 0.0, 1.0).astype(np.float32)
    vals = score_map[valid]
    lo = float(vals.min())
    hi = float(vals.max())
    if hi - lo < 1e-8:
        out = np.zeros_like(score_map, dtype=np.float32)
        out[valid] = 1.0
        return out
    out = (score_map - lo) / (hi - lo)
    out[~valid] = 0.0
    return out.astype(np.float32)


def make_heatmap_image(score_map: np.ndarray, normalize: str, size: Tuple[int, int]) -> Image.Image:
    norm = normalize_scores(score_map, normalize)
    valid = np.isfinite(score_map)
    rgb = turbo_like_colormap(norm)
    rgb[~valid] = np.array([245, 245, 245], dtype=np.uint8)
    return Image.fromarray(rgb, mode="RGB").resize(size, Image.NEAREST)


def draw_patch_boxes(
    image: Image.Image,
    selected_mask: np.ndarray,
    grid_size: Tuple[int, int],
    color: Tuple[int, int, int] = (255, 185, 0),
    width: int = 2,
) -> None:
    draw = ImageDraw.Draw(image)
    gh, gw = grid_size
    cell_w = image.size[0] / gw
    cell_h = image.size[1] / gh
    ys, xs = np.where(selected_mask)
    for y, x in zip(ys.tolist(), xs.tolist()):
        x0 = int(round(x * cell_w))
        y0 = int(round(y * cell_h))
        x1 = int(round((x + 1) * cell_w)) - 1
        y1 = int(round((y + 1) * cell_h)) - 1
        for offset in range(width):
            draw.rectangle([x0 + offset, y0 + offset, x1 - offset, y1 - offset], outline=color)


def draw_grid(image: Image.Image, grid_size: Tuple[int, int]) -> None:
    draw = ImageDraw.Draw(image)
    gh, gw = grid_size
    cell_w = image.size[0] / gw
    cell_h = image.size[1] / gh
    color = (255, 255, 255)
    for x in range(1, gw):
        px = int(round(x * cell_w))
        draw.line([(px, 0), (px, image.size[1])], fill=color, width=1)
    for y in range(1, gh):
        py = int(round(y * cell_h))
        draw.line([(0, py), (image.size[0], py)], fill=color, width=1)


def overlay_heatmap(
    base: Image.Image,
    score_map: np.ndarray,
    selected_mask: np.ndarray,
    alpha: float,
    normalize: str,
    draw_grid_lines: bool,
) -> Image.Image:
    base = base.convert("RGB")
    heat = make_heatmap_image(score_map, normalize, base.size)
    valid = np.isfinite(score_map).astype(np.uint8) * 255
    alpha_img = Image.fromarray(valid, mode="L").resize(base.size, Image.NEAREST)
    out = Image.composite(Image.blend(base, heat, alpha), base, alpha_img)
    if draw_grid_lines:
        draw_grid(out, score_map.shape)
    draw_patch_boxes(out, selected_mask, score_map.shape)
    return out


def selected_and_score_grids(
    scores: np.ndarray,
    indices: np.ndarray,
    positions: np.ndarray,
    num_frames: int,
    grid_h: int,
    grid_w: int,
) -> Tuple[np.ndarray, np.ndarray]:
    score_grid = np.full((num_frames, grid_h, grid_w), np.nan, dtype=np.float32)
    selected_grid = np.zeros((num_frames, grid_h, grid_w), dtype=bool)
    total = num_frames * grid_h * grid_w
    for cand_idx, pos in enumerate(positions.tolist()):
        pos = int(pos)
        if pos < 0 or pos >= total:
            continue
        frame = pos // (grid_h * grid_w)
        rem = pos % (grid_h * grid_w)
        y = rem // grid_w
        x = rem % grid_w
        score_grid[frame, y, x] = float(scores[cand_idx])
    for cand_idx in indices.tolist():
        if cand_idx < 0 or cand_idx >= len(positions):
            continue
        pos = int(positions[cand_idx])
        if pos < 0 or pos >= total:
            continue
        frame = pos // (grid_h * grid_w)
        rem = pos % (grid_h * grid_w)
        y = rem // grid_w
        x = rem % grid_w
        selected_grid[frame, y, x] = True
    return score_grid, selected_grid


def positions_by_segment(
    modality: str,
    num_frames: int,
    grid_h: int,
    grid_w: int,
    num_segments: int,
    tokens_per_segment: int,
    visidx: Optional[np.ndarray],
) -> np.ndarray:
    if modality == "image":
        all_positions = np.arange(grid_h * grid_w, dtype=np.int64)
    elif visidx is not None:
        all_positions = np.asarray(visidx, dtype=np.int64).reshape(-1)
    else:
        all_positions = np.arange(num_frames * grid_h * grid_w, dtype=np.int64)

    needed = num_segments * tokens_per_segment
    if all_positions.size < needed:
        raise ValueError(
            f"not enough positions to map scores: have {all_positions.size}, need {needed}"
        )
    if all_positions.size > needed:
        all_positions = all_positions[:needed]
    return all_positions.reshape(num_segments, tokens_per_segment)


def save_image_visualization(
    sample: VisualSample,
    image: Image.Image,
    scores: np.ndarray,
    indices: np.ndarray,
    out_dir: Path,
    sample_idx: int,
    args: argparse.Namespace,
    patch_size: int,
) -> dict:
    grid_h = image.size[1] // patch_size
    grid_w = image.size[0] // patch_size
    positions = np.arange(grid_h * grid_w, dtype=np.int64).reshape(1, -1)
    score_grid, selected_grid = selected_and_score_grids(
        scores=scores[0],
        indices=indices[0],
        positions=positions[0],
        num_frames=1,
        grid_h=grid_h,
        grid_w=grid_w,
    )
    prefix = out_dir / f"image_{sample_idx:03d}_{sample.key}"
    overlay = overlay_heatmap(
        image,
        score_grid[0],
        selected_grid[0],
        alpha=args.overlay_alpha,
        normalize=args.normalize,
        draw_grid_lines=args.draw_grid,
    )
    overlay_path = prefix.with_name(prefix.name + "_overlay.png")
    overlay.save(overlay_path)
    heatmap_path = None
    if args.save_heatmap_only:
        heatmap = make_heatmap_image(score_grid[0], args.normalize, image.size)
        heatmap_path = prefix.with_name(prefix.name + "_heatmap.png")
        heatmap.save(heatmap_path)

    return {
        "kind": "image",
        "key": sample.key,
        "shard": sample.shard,
        "overlay": str(overlay_path),
        "heatmap": str(heatmap_path) if heatmap_path else None,
        "scores_shape": list(scores.shape),
        "selected_indices": indices[0].astype(int).tolist(),
        "selected_positions": positions[0][indices[0]].astype(int).tolist(),
        "score_min": float(np.nanmin(scores)),
        "score_max": float(np.nanmax(scores)),
        "score_mean": float(np.nanmean(scores)),
    }


def write_gif(frames: List[Image.Image], path: Path, fps: float) -> None:
    """Write a list of PIL Images as an animated GIF."""
    import imageio.v2 as iio

    duration_ms = int(1000.0 / fps)
    np_frames = [np.asarray(f.convert("RGB")) for f in frames]
    iio.mimsave(str(path), np_frames, format="GIF", loop=0, duration=duration_ms)


def write_mp4(frames: List[Image.Image], path: Path, fps: float) -> None:
    """Write a list of PIL Images as an MP4 video."""
    import cv2

    h, w = frames[0].size[1], frames[0].size[0]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    for frame in frames:
        arr = np.asarray(frame.convert("RGB"))
        writer.write(cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
    writer.release()


def add_label(image: Image.Image, label: str) -> Image.Image:
    label_h = 20
    canvas = Image.new("RGB", (image.size[0], image.size[1] + label_h), (255, 255, 255))
    canvas.paste(image, (0, label_h))
    draw = ImageDraw.Draw(canvas)
    draw.text((4, 3), label, fill=(0, 0, 0))
    return canvas


def make_contact_sheet(images: Sequence[Image.Image], cols: int = 4, pad: int = 8) -> Image.Image:
    if not images:
        raise ValueError("cannot make contact sheet from no images")
    cols = max(1, min(cols, len(images)))
    rows = int(math.ceil(len(images) / cols))
    w = max(img.size[0] for img in images)
    h = max(img.size[1] for img in images)
    sheet = Image.new("RGB", (cols * w + (cols - 1) * pad, rows * h + (rows - 1) * pad), (245, 245, 245))
    for idx, img in enumerate(images):
        row = idx // cols
        col = idx % cols
        sheet.paste(img, (col * (w + pad), row * (h + pad)))
    return sheet


def save_video_visualization(
    sample: VisualSample,
    frames: List[Image.Image],
    scores: np.ndarray,
    indices: np.ndarray,
    visidx: Optional[np.ndarray],
    out_dir: Path,
    sample_idx: int,
    args: argparse.Namespace,
    patch_size: int,
) -> dict:
    grid_h = frames[0].size[1] // patch_size
    grid_w = frames[0].size[0] // patch_size
    num_segments, tokens_per_segment = scores.shape
    pos = positions_by_segment(
        modality="video",
        num_frames=len(frames),
        grid_h=grid_h,
        grid_w=grid_w,
        num_segments=num_segments,
        tokens_per_segment=tokens_per_segment,
        visidx=visidx,
    )

    segment_summaries = []
    prefix = out_dir / f"video_{sample_idx:03d}_{sample.key}"
    frames_per_segment = max(1, len(frames) // num_segments)
    for seg in range(num_segments):
        score_grid, selected_grid = selected_and_score_grids(
            scores=scores[seg],
            indices=indices[seg],
            positions=pos[seg],
            num_frames=len(frames),
            grid_h=grid_h,
            grid_w=grid_w,
        )
        start = seg * frames_per_segment
        end = len(frames) if seg == num_segments - 1 else min(len(frames), (seg + 1) * frames_per_segment)
        candidate_frames = list(range(start, end))
        if len(candidate_frames) > args.video_frames_per_segment_viz:
            rel = _even_indices(len(candidate_frames), args.video_frames_per_segment_viz)
            show_frames = [candidate_frames[i] for i in rel]
        else:
            show_frames = candidate_frames

        tiles: List[Image.Image] = []
        for frame_idx in show_frames:
            overlay = overlay_heatmap(
                frames[frame_idx],
                score_grid[frame_idx],
                selected_grid[frame_idx],
                alpha=args.overlay_alpha,
                normalize=args.normalize,
                draw_grid_lines=args.draw_grid,
            )
            tiles.append(add_label(overlay, f"seg {seg} frame {frame_idx}"))
        sheet = make_contact_sheet(tiles, cols=min(4, len(tiles)))
        out_path = prefix.with_name(prefix.name + f"_segment_{seg:02d}_overlay.png")
        sheet.save(out_path)

        if args.export_video != "none" and tiles:
            anim_base = prefix.with_name(prefix.name + f"_segment_{seg:02d}")
            if args.export_video in ("gif", "both"):
                gif_path = anim_base.with_suffix(".gif")
                write_gif(tiles, gif_path, args.fps)
                print(f"[anim] wrote {gif_path}")
            if args.export_video in ("mp4", "both"):
                mp4_path = anim_base.with_suffix(".mp4")
                write_mp4(tiles, mp4_path, args.fps)
                print(f"[anim] wrote {mp4_path}")

        selected_positions = pos[seg][indices[seg]].astype(int).tolist()
        segment_summaries.append(
            {
                "segment": seg,
                "overlay": str(out_path),
                "shown_frames": show_frames,
                "selected_indices": indices[seg].astype(int).tolist(),
                "selected_positions": selected_positions,
                "score_min": float(np.nanmin(scores[seg])),
                "score_max": float(np.nanmax(scores[seg])),
                "score_mean": float(np.nanmean(scores[seg])),
            }
        )
    return {
        "kind": "video",
        "key": sample.key,
        "shard": sample.shard,
        "num_frames_used": len(frames),
        "used_visidx": visidx is not None,
        "scores_shape": list(scores.shape),
        "segments": segment_summaries,
    }


def analyze_norm_bias(
    vit_tokens: torch.Tensor,
    scores: np.ndarray,
    label: str,
) -> dict:
    """Compute Pearson and Spearman correlation between per-token L2 norm and selector score.

    Args:
        vit_tokens: (B, S, P, D) or (S, P, D) raw ViT token embeddings.
        scores:     (S, P) or (P,) numpy array of sigmoid selector scores.
        label:      human-readable tag for printing.

    Returns a dict with keys: label, n_tokens, pearson_r, pearson_p, spearman_r, spearman_p.
    """
    from scipy.stats import pearsonr, spearmanr

    # collapse batch dim if present
    t = vit_tokens.float()
    if t.ndim == 4:
        t = t[0]  # (S, P, D)
    norms = t.norm(dim=-1).cpu().numpy().reshape(-1)   # (S*P,)
    flat_scores = scores.reshape(-1).astype(np.float64)
    flat_norms = norms.reshape(-1).astype(np.float64)

    pr, pp = pearsonr(flat_norms, flat_scores)
    sr, sp = spearmanr(flat_norms, flat_scores)

    result = {
        "label": label,
        "n_tokens": int(flat_norms.size),
        "norm_mean": float(np.mean(flat_norms)),
        "norm_std": float(np.std(flat_norms)),
        "score_mean": float(np.mean(flat_scores)),
        "score_std": float(np.std(flat_scores)),
        "pearson_r": float(pr),
        "pearson_p": float(pp),
        "spearman_r": float(sr),
        "spearman_p": float(sp),
    }
    print(
        f"[norm-bias] {label:40s}  n={result['n_tokens']:6d}"
        f"  norm={result['norm_mean']:.3f}±{result['norm_std']:.3f}"
        f"  score={result['score_mean']:.3f}±{result['score_std']:.3f}"
        f"  Pearson r={pr:+.4f} (p={pp:.2e})"
        f"  Spearman r={sr:+.4f} (p={sp:.2e})"
    )
    return result


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[env] repo={REPO_ROOT}")
    print(f"[env] device={device} dtype={dtype}")
    cfg = load_model_config(args.config)
    vit, selector = build_vit_and_selector(cfg)
    load_checkpoint(vit, selector, args.ckpt)
    vit.to(device=device, dtype=dtype).eval()
    selector.to(device=device, dtype=dtype).eval()

    image_res = args.image_resolution if args.image_resolution is not None else cfg.vit.image_size
    video_res = args.video_resolution if args.video_resolution is not None else cfg.vit.image_size
    print(f"[env] image_resolution={image_res}  video_resolution={video_res}  (cfg.vit.image_size={cfg.vit.image_size})")

    if args.sample_cache_dir is not None:
        image_samples = load_cached_samples(args.sample_cache_dir, "image", args.num_images)
        video_samples = load_cached_samples(args.sample_cache_dir, "video", args.num_videos)
    else:
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

    if not image_samples and args.num_images:
        print("[warn] no image samples found")
    if not video_samples and args.num_videos:
        print("[warn] no video samples found")

    summaries = []
    for idx, sample in enumerate(image_samples):
        print(f"[run] image {idx}: key={sample.key}")
        pixel_values, image = preprocess_image(sample.image, image_res)
        q, v, out = selector_forward(
            vit,
            selector,
            pixel_values=pixel_values,
            modality="image",
            top_k=args.top_k,
            device=device,
            dtype=dtype,
            zero_query_tokens=args.zero_query_tokens,
            use_mmr=args.mmr,
            mmr_lambda=args.mmr_lambda,
        )
        scores = out["scores"][0].float().cpu().numpy()
        indices = out["indices"][0].long().cpu().numpy()
        summaries.append(
            save_image_visualization(
                sample,
                image,
                scores,
                indices,
                args.output_dir,
                idx,
                args,
                patch_size=cfg.vit.patch_size,
            )
        )
        print(f"[shape] image q={tuple(q.shape)} vit={tuple(v.shape)} scores={scores.shape}")
        if args.analyze_norm_bias:
            if "norm_bias_records" not in locals():
                norm_bias_records: list = []
            norm_bias_records.append(analyze_norm_bias(v, scores, label=f"image_{idx:03d}_{sample.key[:20]}"))

    for idx, sample in enumerate(video_samples):
        print(f"[run] video {idx}: key={sample.key}")
        uniform_video_frames = args.uniform_video_frames
        if args.video_mode == "uniform" and args.video_frames_without_visidx != 64:
            uniform_video_frames = args.video_frames_without_visidx
        pixel_values, frames, visidx, note, forward_kwargs = prepare_video_frames(
            sample,
            image_size=video_res,
            patch_size=cfg.vit.patch_size,
            segment_t_size=cfg.vit.segment_t_size,
            video_mode=args.video_mode,
            uniform_video_frames=uniform_video_frames,
            uniform_segment_t_size=args.uniform_segment_t_size,
        )
        print(f"[video] {note}")
        ppf = (video_res // cfg.vit.patch_size) ** 2
        print(f"[first-frame-cap] ratio={args.first_frame_max_ratio}  patches_per_frame={ppf}  max_allowed={int(ppf*args.first_frame_max_ratio)}")
        q, v, out = selector_forward(
            vit,
            selector,
            pixel_values=pixel_values,
            modality="video",
            top_k=args.top_k,
            device=device,
            dtype=dtype,
            visidx=visidx,
            zero_query_tokens=args.zero_query_tokens,
            use_mmr=args.mmr,
            mmr_lambda=args.mmr_lambda,
            first_frame_max_ratio=args.first_frame_max_ratio,
            patches_per_frame=(video_res // cfg.vit.patch_size) ** 2,
            **forward_kwargs,
        )
        scores = out["scores"][0].float().cpu().numpy()
        indices = out["indices"][0].long().cpu().numpy()
        # debug: report actual first-frame token count per segment
        ppf_dbg = (video_res // cfg.vit.patch_size) ** 2
        for seg_i in range(indices.shape[0]):
            ff_cnt = int((indices[seg_i] < ppf_dbg).sum())
            print(f"  [cap-check] seg {seg_i}: first-frame tokens = {ff_cnt} / {ppf_dbg}  (cap={int(ppf_dbg*args.first_frame_max_ratio)})")
        summaries.append(
            save_video_visualization(
                sample,
                frames,
                scores,
                indices,
                visidx,
                args.output_dir,
                idx,
                args,
                patch_size=cfg.vit.patch_size,
            )
        )
        print(f"[shape] video q={tuple(q.shape)} vit={tuple(v.shape)} scores={scores.shape}")
        if args.analyze_norm_bias:
            if "norm_bias_records" not in locals():
                norm_bias_records = []
            norm_bias_records.append(analyze_norm_bias(v, scores, label=f"video_{idx:03d}_{sample.key[:20]}"))

    if args.analyze_norm_bias and "norm_bias_records" in locals() and norm_bias_records:
        all_pr = np.array([r["pearson_r"] for r in norm_bias_records])
        all_sr = np.array([r["spearman_r"] for r in norm_bias_records])
        print(
            f"\n[norm-bias] AGGREGATE over {len(norm_bias_records)} samples: "
            f"mean Pearson r={all_pr.mean():+.4f} (std={all_pr.std():.4f})  "
            f"mean Spearman r={all_sr.mean():+.4f} (std={all_sr.std():.4f})"
        )
        norm_bias_path = args.output_dir / "norm_bias_analysis.json"
        with norm_bias_path.open("w", encoding="utf-8") as fh:
            json.dump({"samples": norm_bias_records, "aggregate": {
                "n_samples": len(norm_bias_records),
                "pearson_r_mean": float(all_pr.mean()),
                "pearson_r_std": float(all_pr.std()),
                "spearman_r_mean": float(all_sr.mean()),
                "spearman_r_std": float(all_sr.std()),
            }}, fh, indent=2)
        print(f"[norm-bias] wrote {norm_bias_path}")

    summary_path = args.output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "config": str(args.config),
                "ckpt": str(args.ckpt),
                "top_k": args.top_k,
                "normalize": args.normalize,
                "items": summaries,
            },
            handle,
            indent=2,
            ensure_ascii=True,
        )
    print(f"[done] wrote {summary_path}")


if __name__ == "__main__":
    main()
