"""
MMEB-V2 unified evaluation using the CoVisco SigLIP model.

This extends `eval_mmeb_video_retrieval.py` to run several MMEB-V2 meta-tasks
with one command, reusing the same vision model + Qwen3-VL text encoder:

  - video_ret   Video Retrieval        (text -> video, R@1/5/10, MdR)
  - video_cls   Video Classification   (video -> class label, Accuracy)
  - video_mret  Moment Retrieval       (text -> correct clip within a video, R@1)
  - image_cls   Image Classification   (image -> class label, Accuracy; MMEB-V1)
  - visdoc      Visual Document Retrieval (text -> page image, nDCG@5/@10, R@5;
                MMEB-V2, ViDoRe / VisRAG)

Architectural note
------------------
This checkpoint is a CLIP-style *dual encoder* (separate vision tower +
Qwen3-VL text encoder) whose image AND video embeddings are aligned to the same
Qwen3-VL text space. It produces one embedding per modality and matches them by
cosine similarity, so it natively supports retrieval and zero-shot
classification. It does NOT fuse a video/image with a text question into a
single query embedding, so:
  - Video QA (activitynetqa) and the *interleaved* MMEB-V1 image tasks (VQA,
    grounding, multimodal retrieval that fuse an image WITH a text question into
    one query) are out of scope.
  - `image_cls` covers only the pure image->label classification subset of
    MMEB-V1, where the query is the image alone.
  - `visdoc` (MMEB-V2 Visual Document Retrieval) IS supported: it is pure
    text->page-image retrieval (no fused query), which the dual encoder handles
    natively -- the query text goes to the Qwen3-VL encoder, each document page
    image to the vision tower, ranked by cosine.

Text prompting (two independent layers)
---------------------------------------
  1. `_VISION_INSTRUCTION` - the training-time system prompt the vision
     embeddings were aligned to (applied to every text side).
  2. label templates - CLIP-style prompt ensembling of the class name
     ("a photo of a {}.", ...); embeddings over templates are averaged and
     re-normalized. Templates are task/dataset specific and CLI-overridable.

Data layout (local, under --data_root):
  video_ret :  video-tasks/frames/video_ret/data/your_user/video_retrieval/<DS>/frames/<vid>/
               annotations: local jsonl or HuggingFace (see reference script)
  video_cls :  video-tasks/frames/video_cls/<DS>/<vid>/<frames>.jpeg|jpg|png
               annotations: video-tasks/data/<name>.jsonl
  video_mret:  video-tasks/frames/video_mret/video_mret/<DS>/<vid>/{positive_clip,negative_clip_*}/
               annotations: video-tasks/data/<name>.jsonl
  image_cls :  image-tasks/mmeb_v1/MMEB/<qry_img_path>   (images local)
               annotations: HuggingFace TIGER-Lab/MMEB-eval
  visdoc    :  loaded entirely from HuggingFace (ViDoRe / VisRAG BEIR repos);
               page images are read inline from the `corpus` split, so no local
               files are needed

Frame files use a mix of .jpg / .jpeg / .png extensions across datasets, so
this script globs all of them (the reference script only handled .jpg/.png).

HuggingFace access (video_ret annotations, image_cls annotations) can go through
an optional HTTP(S) proxy:
    export http_proxy=http://your-proxy-host:port
    export https_proxy=http://your-proxy-host:port
    export no_proxy=localhost,127.0.0.1

Usage:
    python eval_mme_v2_all.py --tasks video_cls video_mret video_ret image_cls \
        --ckpt /path/to/checkpoint.pt \
        --data_root /path/to/data/mme_v2

Distributed evaluation:
    torchrun --standalone --nproc_per_node=8 eval_mme_v2_all.py \
        --tasks video_ret --ckpt /path/to/checkpoint.pt

Each rank encodes a shard of the visual samples. Retrieval tasks gather the
sharded gallery embeddings and evaluate every query against the full gallery;
rank 0 is the only process that writes --output.
"""

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Reuse the proven pieces from the retrieval reference script. These helpers are
# extension-agnostic / side-effect free at import time. `load_model` is
# re-implemented locally so the input resolution is configurable.
from eval_mmeb_video_retrieval import (  # noqa: E402
    build_preprocess,
    recall_at_k,
    DATASETS as RET_DATASETS,
    load_annotations as ret_load_annotations,
)

# Input resolution (square). Set from --image_size in main(); the zero-padding
# fallbacks for missing frames/images read it so stacked tensors stay uniform.
_IMAGE_SIZE = 224

# Video temporal config. Set from --num_frames / --segment_t_size in main().
# Each clip is uniformly sampled to exactly _NUM_FRAMES frames, and the encoder's
# get_segments() path splits that into _NUM_FRAMES // _SEGMENT_T_SIZE segments, so
# _NUM_FRAMES MUST be a positive multiple of _SEGMENT_T_SIZE.
_NUM_FRAMES = 64
_SEGMENT_T_SIZE = 16

# Normalize class labels before text encoding (first synonym + underscores->spaces).
# Set from --raw_labels in main(). See _clean_label.
_CLEAN_LABELS = True

# ViT patch size (must match the checkpoint config). Native-resolution inputs are
# rounded to a multiple of this so extract_patches() does not drop border pixels.
_PATCH_SIZE = 14
_VISION_MODALITY = "video"
_VISION_OUTPUT_KEY = "to_video_caption"

# Distributed evaluation state. Direct ``python``/``bash`` invocation remains
# single-process; ``torchrun`` initializes this state in the entry point below.
_DIST_ENABLED = False
_DIST_RANK = 0
_DIST_WORLD_SIZE = 1


def _dist_indices(length):
    """Return this rank's strided slice while preserving global indices."""
    return list(range(_DIST_RANK, length, _DIST_WORLD_SIZE))


def _gather_indexed_embeddings(local_embs, local_indices, total):
    """Gather sharded CPU embeddings and restore their original global order."""
    if not _DIST_ENABLED:
        return local_embs
    parts = [None] * _DIST_WORLD_SIZE
    dist.all_gather_object(parts, (local_indices, local_embs.cpu()))
    if not parts:
        return local_embs
    result = torch.empty(total, local_embs.shape[-1], dtype=local_embs.dtype)
    for indices, embs in parts:
        if indices:
            result[indices] = embs
    return result


def _all_reduce_counts(*values):
    """Sum integer counters across ranks and return Python ints."""
    if not _DIST_ENABLED:
        return tuple(int(v) for v in values)
    use_cuda_collective = (dist.get_backend() == "nccl"
                           if _DIST_ENABLED else torch.cuda.is_available())
    tensor = torch.tensor(values, dtype=torch.long,
                          device='cuda' if use_cuda_collective else 'cpu')
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tuple(int(v) for v in tensor.cpu().tolist())


def _broadcast_result(result):
    if not _DIST_ENABLED:
        return result
    obj = [result if _DIST_RANK == 0 else None]
    dist.broadcast_object_list(obj, src=0)
    return obj[0]


def _init_distributed(device):
    """Initialize torchrun state and select the process-local CUDA device."""
    global _DIST_ENABLED, _DIST_RANK, _DIST_WORLD_SIZE
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return device
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl" if torch.cuda.is_available()
                                else "gloo", init_method="env://")
    _DIST_ENABLED = True
    _DIST_RANK = dist.get_rank()
    _DIST_WORLD_SIZE = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", _DIST_RANK))
    if torch.cuda.is_available() and device.startswith("cuda"):
        torch.cuda.set_device(local_rank)
        return f"cuda:{local_rank}"
    return device


def build_native_preprocess(patch_size=_PATCH_SIZE, max_side=1400, min_side=None):
    """Aspect-ratio-preserving preprocessing at (near-)native resolution.

    Unlike `build_preprocess` (Resize + CenterCrop to a fixed square, which
    distorts/crops away detail -- fatal for text-heavy document pages), this
    keeps the original aspect ratio and only:
      * scales so the long side <= `max_side` (memory cap),
      * rounds each side to a multiple of `patch_size` (>= `min_side`).

    The encoder is RoPE-only with no learned position table and derives its
    patch grid as h=H//patch_size, w=W//patch_size at runtime, so it accepts any
    such H x W (including non-square). Because a batch is a single dense tensor
    (no resolution packing), callers must encode one image at a time when using
    this -- see main(), which forces batch size 1 under --native_resolution.
    """
    min_side = min_side or patch_size
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                     (0.26862954, 0.26130258, 0.27577711))

    def _round(x):
        return max(min_side, int(round(x / patch_size)) * patch_size)

    def preprocess(img):
        w, h = img.size
        scale = min(1.0, max_side / max(w, h))
        nw, nh = _round(w * scale), _round(h * scale)
        img = img.resize((nw, nh), Image.BICUBIC)
        return normalize(to_tensor(img))

    return preprocess


def load_model(ckpt_path, video_caption_embed_dim, device, image_size=224,
               segment_t_size=16):
    """Load CoViscoModel (mirrors the reference loader, but `image_size`
    and `segment_t_size` are configurable)."""
    from open_clip.covisco_model import CoViscoModel
    from open_clip.covisco_vit import (
        CoViscoEncoderConfig,
    )

    config = CoViscoEncoderConfig(
        hidden_size=1024, num_hidden_layers=24, num_attention_heads=16,
        num_channels=3, image_size=image_size, patch_size=14,
        num_query_per_seg=100, segment_t_size=segment_t_size,
        use_head=True, output_dim=1024,
    )
    model = CoViscoModel(
        covisco_config=config,
        image_embed_dim=1536,
        image_caption_embed_dim=1536,
        video_caption_embed_dim=video_caption_embed_dim,
        use_reconstruction=False,
    )
    if not ckpt_path:
        raise ValueError(
            "no checkpoint provided: pass --ckpt /path/to/checkpoint.pt "
            "(or use eval_mme_v2_all_hf.py to load the released HF encoder)"
        )
    sd = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    sd = sd.get('state_dict', sd)
    sd = {k.replace('module.', '', 1): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"[WARN] {len(missing)} model params missing from checkpoint, "
              f"left at RANDOM init: {missing[:10]}")
    if unexpected:
        print(f"[WARN] {len(unexpected)} unexpected checkpoint keys ignored: {unexpected[:10]}")
    return model.eval().to(device)

# Training-time text prompt: the vision (image AND video) embeddings were
# aligned to Qwen3-VL text features encoded with this exact system prompt, so
# every text side here (retrieval captions, class labels, moment queries) uses
# it. Note it says "vision", not "video".
_VISION_INSTRUCTION = "Represent the vision description for cross-modal matching. "

# ---------------------------------------------------------------------------
# Frame IO (broad glob: jpg / jpeg / png, upper and lower case)
# ---------------------------------------------------------------------------

_FRAME_PATTERNS = ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG')


def list_frames(frame_dir: str):
    paths = []
    for pat in _FRAME_PATTERNS:
        paths.extend(glob.glob(os.path.join(frame_dir, pat)))
    return sorted(set(paths))


def _sample_frame_paths(frame_dir: str):
    """Uniformly sample exactly `_NUM_FRAMES` frame paths from a clip.

    Up- or down-samples to the fixed count so every clip's temporal length is a
    multiple of `_SEGMENT_T_SIZE` (required by the encoder's get_segments path).
    """
    paths = list_frames(frame_dir)
    if not paths:
        return None
    target = _NUM_FRAMES
    if len(paths) != target:
        idx = [int(i * len(paths) / target) for i in range(target)]
        paths = [paths[i] for i in idx]
    return paths


def _open_frame(path, retries=4, base_delay=0.25):
    """Open a frame image, retrying transient network-FS errors.

    Frames live on a FUSE object-storage mount (PaddleFlowFS/BOS) which can
    intermittently raise FileNotFoundError/OSError under concurrent reads even
    when the file exists. Retry with backoff before giving up.
    """
    last_err = None
    for attempt in range(retries):
        try:
            return Image.open(path).convert('RGB')
        except (FileNotFoundError, OSError) as e:
            last_err = e
            time.sleep(base_delay * (2 ** attempt))
    raise last_err


def _load_one_video(args):
    frame_dir, preprocess = args
    paths = _sample_frame_paths(frame_dir)
    if not paths:
        return None
    try:
        frames = torch.stack([preprocess(_open_frame(p)) for p in paths])
    except (FileNotFoundError, OSError) as e:
        # Persistent read failure after retries: skip this video (zero embed).
        print(f"[WARN] failed to load frames from {frame_dir}: {e}")
        return None
    return frames.permute(1, 0, 2, 3)  # (C, T, H, W)


@torch.no_grad()
def encode_videos_batch(model, preprocess, frame_dirs, max_frames, device,
                        video_batch_size=1, num_workers=4):
    """Encode a list of video frame-dirs into normalized embeddings (M, D).

    Streams IO in sliding windows to cap peak RAM. Missing/empty dirs get a
    zero embedding so indexing stays aligned with `frame_dirs`.
    """
    from concurrent.futures import ThreadPoolExecutor
    from tqdm import tqdm

    all_embs = [None] * len(frame_dirs)
    emb_dim = None
    window = max(video_batch_size, num_workers * video_batch_size)
    args_all = [(d, preprocess) for d in frame_dirs]

    for win_start in tqdm(range(0, len(frame_dirs), window), desc="  video batches"):
        win_args = args_all[win_start:win_start + window]
        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            loaded = list(pool.map(_load_one_video, win_args))

        for start in range(0, len(loaded), video_batch_size):
            sub = loaded[start:start + video_batch_size]
            idxs = list(range(win_start + start, win_start + start + len(sub)))
            buf = [f if f is not None else torch.zeros(3, _NUM_FRAMES, _IMAGE_SIZE, _IMAGE_SIZE)
                   for f in sub]
            try:
                pixel_values = torch.stack(buf).to(device)
                with torch.autocast(device_type=device.split(':')[0]):
                    # get_segments path: use all _NUM_FRAMES frames, split into
                    # _NUM_FRAMES // _SEGMENT_T_SIZE segments. This must match
                    # how the checkpoint was trained.
                    out = model(pixel_values, modality=_VISION_MODALITY)
                embs = out[_VISION_OUTPUT_KEY].detach().cpu()
                if emb_dim is None:
                    emb_dim = embs.shape[-1]
            except Exception as e:  # pragma: no cover - defensive
                print(f"[WARN] batch encode failed: {e}, falling back to zeros")
                embs = torch.zeros(len(sub), emb_dim or 1)
            for j, idx in enumerate(idxs):
                all_embs[idx] = embs[j] if loaded[start + j] is not None \
                    else torch.zeros(emb_dim or 1)
        del loaded

    return F.normalize(torch.stack(all_embs), dim=-1)  # (M, D)


@torch.no_grad()
def encode_images_batch(model, preprocess, image_paths, device,
                        batch_size=64, num_workers=8):
    """Encode single images into normalized embeddings (N, D).

    Reads the ``to_video_caption`` head (``proj_to_video_caption``,
    video_caption_embed_dim) -- this is the contrastive branch aligned to the
    Qwen3-VL text encoder, so it lives in the same space as the label texts.
    (``to_image_caption`` is aligned to the SigLIP2 text tower instead, a
    different dim, which is why it must NOT be used here.) ``modality`` only
    selects which projection head runs; the pooled features come from the same
    encoder pass, so a (B,C,H,W) image tensor is valid. Unreadable/missing
    images get a zero embedding so indexing stays aligned with `image_paths`.
    """
    from concurrent.futures import ThreadPoolExecutor
    from tqdm import tqdm

    def _load(p):
        try:
            return preprocess(Image.open(p).convert('RGB'))
        except Exception:
            return None

    all_embs = [None] * len(image_paths)
    emb_dim = None
    for start in tqdm(range(0, len(image_paths), batch_size), desc="  image batches"):
        chunk = image_paths[start:start + batch_size]
        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            loaded = list(pool.map(_load, chunk))
        buf = [t if t is not None else torch.zeros(3, _IMAGE_SIZE, _IMAGE_SIZE) for t in loaded]
        pixel_values = torch.stack(buf).to(device)
        with torch.autocast(device_type=device.split(':')[0]):
            # The encoder takes the get_segments path: a single image is one
            # segment of 1 frame, positions derived at runtime (h=H//patch_size,
            # w=W//patch_size) -- RoPE-equivalent to the training-time path.
            out = model(pixel_values, modality=_VISION_MODALITY, run_decoder=False)
        embs = out[_VISION_OUTPUT_KEY].detach().cpu()
        if emb_dim is None:
            emb_dim = embs.shape[-1]
        for j, t in enumerate(loaded):
            all_embs[start + j] = embs[j] if t is not None else torch.zeros(emb_dim)
    return F.normalize(torch.stack(all_embs), dim=-1)


def _load_qwen3vl_backbone_weights(backbone, model_path):
    """Load ForConditionalGeneration checkpoint weights into Qwen3VLModel.

    The embedding checkpoint stores parameters below ``model.*`` because it
    was saved from Qwen3VLForConditionalGeneration. SentenceTransformers
    resolves AutoModel to the bare Qwen3VLModel, whose parameter names omit
    that prefix. Without this remapping, transformers initializes the entire
    text encoder randomly while only emitting a warning.
    """
    from safetensors.torch import load_file

    index_path = os.path.join(model_path, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            weight_map = json.load(f)["weight_map"]
        shard_paths = sorted({
            os.path.join(model_path, shard) for shard in weight_map.values()
        })
    else:
        shard_paths = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))

    expected = set(backbone.state_dict().keys())
    remapped = {}
    for shard_path in shard_paths:
        for key, value in load_file(shard_path).items():
            key = key[len("model."):] if key.startswith("model.") else key
            if key in expected:
                remapped[key] = value

    missing, unexpected = backbone.load_state_dict(remapped, strict=False)
    missing = [key for key in missing if not key.endswith(".inv_freq")]
    if not remapped or missing or unexpected:
        raise RuntimeError(
            "Qwen3-VL backbone weights did not load cleanly: "
            f"matched {len(remapped)}/{len(expected)}, "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}"
        )
    print(
        f"[OK] Loaded {len(remapped)}/{len(expected)} Qwen3-VL backbone "
        "weights after stripping the 'model.' prefix."
    )


def encode_texts(texts, text_encoder, prompt: str, batch_size: int = 64):
    """Encode text with the Qwen3-VL embedding model -> normalized (N, D).

    Retries at a halved batch size on CUDA OOM. The text encoder (8B) is
    co-resident on the GPU with the vision model, so a large label bank can
    spike past free memory; backing off the batch size recovers instead of
    crashing the whole eval.
    """
    bs = batch_size
    while True:
        try:
            embs = text_encoder.encode(
                texts,
                prompt=prompt,
                normalize_embeddings=True,
                show_progress_bar=True,
                batch_size=bs,
                convert_to_numpy=True,
            )
            return torch.from_numpy(np.array(embs))
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if bs <= 1:
                raise
            bs = max(1, bs // 2)
            print(f"[WARN] text encode OOM, retrying at batch_size={bs}")


def _clean_label(s: str) -> str:
    """Normalize a class label for text encoding (the index mapping keeps the
    raw string). Takes the first synonym (before the first comma) and turns
    underscores AND hyphens into spaces, e.g.
      'robin, American robin, Turdus migratorius' -> 'robin'
      'lifting_hat'                               -> 'lifting hat'
      'museum-outdoor'                            -> 'museum outdoor'  (Place365)
    ImageNet-family labels are comma-separated synonym lists, K700/UCF use
    underscores, and Place365 uses hyphenated qualifiers; feeding them verbatim
    produces poor, OOD text embeddings.
    """
    return s.split(',')[0].replace('_', ' ').replace('-', ' ').strip()


def _render_template(tmpl, label):
    """A template is either a format string ('a photo of a {}.') or a callable
    (open_clip's OPENAI_IMAGENET_TEMPLATES are lambdas: c -> 'a photo of a c.')."""
    return tmpl(label) if callable(tmpl) else tmpl.format(label)


def encode_label_bank(labels, text_encoder, prompt, templates, batch_size=64,
                      label_map=None):
    """Encode a class-label vocabulary with CLIP-style template ensembling.

    Every label is rendered through each template, encoded with `prompt`
    (`_VISION_INSTRUCTION`), and the per-template embeddings are averaged then
    re-normalized -> (C, D). A single-element `templates` list degenerates to a
    plain per-label encoding. Labels are normalized via `_clean_label` unless
    `_CLEAN_LABELS` is False; the returned row order still matches `labels`.
    Templates may be format strings or callables (open_clip template lambdas).

    `label_map` optionally rewrites the label *text before* cleaning/templating
    (e.g. HatefulMemes 'Yes'/'No' -> 'hateful'/'not hateful'); the row order and
    external index mapping still key on the original `labels`.
    """
    prep = _clean_label if _CLEAN_LABELS else (lambda x: x)
    disp = (lambda l: label_map.get(l, l)) if label_map else (lambda l: l)
    acc = None
    for tmpl in templates:
        prompts = [_render_template(tmpl, prep(disp(lab))) for lab in labels]
        e = encode_texts(prompts, text_encoder, prompt=prompt, batch_size=batch_size)
        acc = e if acc is None else acc + e
    return F.normalize(acc / len(templates), dim=-1)


# ---------------------------------------------------------------------------
# Task configs
# ---------------------------------------------------------------------------

# Zero-shot label templates, following CLIP's prompt-engineering recipe.
# Each class label is rendered through every template, encoded with
# `_VISION_INSTRUCTION`, and the per-template embeddings are averaged then
# re-normalized (prompt ensembling). Templates may be format strings or
# callables. ImageNet-family uses OpenAI's 80-template set from open_clip.
try:
    from open_clip import OPENAI_IMAGENET_TEMPLATES as IMAGENET_TEMPLATES  # 80 callables
except Exception:  # pragma: no cover
    IMAGENET_TEMPLATES = ["a photo of a {}.", "a bad photo of a {}.", "a photo of the {}."]

# CLIP action-recognition templates (Kinetics/UCF/HMDB style).
ACTION_TEMPLATES = [
    "a photo of {}.", "a photo of a person {}.", "a photo of a person using {}.",
    "a photo of a person doing {}.", "a photo of a person during {}.",
    "a photo of a person performing {}.", "a photo of a person practicing {}.",
    "a video of {}.", "a video of a person {}.", "a video of a person using {}.",
    "a video of a person doing {}.", "a video of a person during {}.",
    "a video of a person performing {}.", "a video of a person practicing {}.",
    "a example of {}.", "a demonstration of {}.",
]
# SSv2 labels are already full action phrases -> keep light templates.
SSV2_TEMPLATES = ["a video of {}.", "a photo of {}.", "{}"]
# Breakfast labels are the dish being prepared.
BREAKFAST_TEMPLATES = [
    "a video of a person making {}.", "a video of preparing {}.", "a video of {}.",
]
# CLIP dataset-specific image templates.
COUNTRY211_TEMPLATES = [
    "a photo i took in {}.", "a photo i took while visiting {}.",
    "a photo from my home country of {}.", "a photo showing the country of {}.",
]
SCENE_TEMPLATES = ["a photo of a {}.", "a photo of the {}."]                 # SUN397
PLACES_TEMPLATES = ["a photo of a {}.", "a photo of the {}.", "a photo i took at a {}."]
VOC_TEMPLATES = ["a photo of a {}."]
N24NEWS_TEMPLATES = ["a news photo about {}.", "a photo from a news article about {}."]
HATEFUL_TEMPLATES = ["{}", "a meme that is {}."]

VIDEO_CLS_TEMPLATES = ACTION_TEMPLATES        # default for video classification
IMG_CLS_TEMPLATES_DEFAULT = list(IMAGENET_TEMPLATES)

# Video classification. `neg_field` names a per-record list that already holds
# the full candidate label set (SSv2 stores the 174 official templates there).
# When absent, the candidate set is the unique set of gold labels in the file.
CLS_DATASETS = {
    "UCF101":        dict(jsonl="ucf101.jsonl",             frames_dir="UCF101",    neg_field=None, templates=ACTION_TEMPLATES),
    "HMDB51":        dict(jsonl="hmdb51.jsonl",             frames_dir="HMDB51",    neg_field=None, templates=ACTION_TEMPLATES),
    "K700":          dict(jsonl="k700.jsonl",               frames_dir="K700",      neg_field=None, templates=ACTION_TEMPLATES),
    "Breakfast":     dict(jsonl="breakfast.jsonl",          frames_dir="Breakfast", neg_field=None, templates=BREAKFAST_TEMPLATES),
    "SSv2":          dict(jsonl="ssv2.jsonl",               frames_dir="SSv2",      neg_field="neg_text", templates=SSV2_TEMPLATES),
    "SSv2-Template": dict(jsonl="ssv2-actiontemplate.jsonl", frames_dir="SSv2",     neg_field=None, templates=SSV2_TEMPLATES),
}

# Moment retrieval.
MRET_DATASETS = {
    "Charades-STA": dict(jsonl="charades_sta.jsonl", frames_dir="Charades-STA"),
    "QVHighlight":  dict(jsonl="qvhighlight.jsonl",  frames_dir="QVHighlight"),
}


def _cls_data_dir(data_root):
    return os.path.join(data_root, "video-tasks", "data")


# Image classification (MMEB-V1). Annotations come from the TIGER-Lab/MMEB-eval
# HuggingFace dataset (loaded over the network); raw images are local under
# <data_root>/image-tasks/mmeb_v1/MMEB/<qry_img_path>. Each example lists
# candidate labels in `tgt_text` with the gold label at index 0 (MMEB
# Precision@1 convention). Only pure image->label classification tasks are
# listed: this dual-encoder model cannot fuse the text instruction into the
# query, so the image alone is the query and is matched to label texts.
# `templates` are task-type-specific label templates (CLIP convention).
IMG_CLS_DATASETS = {
    "ImageNet-1K":  dict(hf="ImageNet-1K",  templates=IMAGENET_TEMPLATES),
    "ImageNet-A":   dict(hf="ImageNet-A",   templates=IMAGENET_TEMPLATES),
    "ImageNet-R":   dict(hf="ImageNet-R",   templates=IMAGENET_TEMPLATES),
    "ObjectNet":    dict(hf="ObjectNet",    templates=IMAGENET_TEMPLATES,
                         label_map={"mixing / salad bowl": "salad bowl",
                                    "coffee/french press": "french press"}),
    "VOC2007":      dict(hf="VOC2007",      templates=VOC_TEMPLATES,
                         label_map={"aeroplane": "airplane", "diningtable": "dining table",
                                    "pottedplant": "potted plant", "tvmonitor": "tv monitor",
                                    "motorbike": "motorcycle"}),
    "SUN397":       dict(hf="SUN397",       templates=SCENE_TEMPLATES),
    "Place365":     dict(hf="Place365",     templates=PLACES_TEMPLATES),
    "Country211":   dict(hf="Country211",   templates=COUNTRY211_TEMPLATES),
    "N24News":      dict(hf="N24News",      templates=N24NEWS_TEMPLATES),
    "HatefulMemes": dict(hf="HatefulMemes",
                         templates=["a meme that is with {}.", "a {} meme.", "{}"],
                         label_map={"Yes": "hateful contents", "No": "no hateful contents"}),
}
IMG_CLS_HF_REPO = "TIGER-Lab/MMEB-eval"
# Images are encoded through the `to_video_caption` head (the Qwen3-VL-aligned
# contrastive branch), so labels go through the same encoder with `_VISION_INSTRUCTION`.
IMG_CLS_LABEL_PROMPT = _VISION_INSTRUCTION


def _img_base(data_root):
    return os.path.join(data_root, "image-tasks", "mmeb_v1", "MMEB")


# Visual Document Retrieval (MMEB-V2). Standard visual-document retrieval: a
# *text* query is ranked against a corpus of document *page images*, globally
# over the whole corpus (eval_type="global" in the official VLM2Vec configs).
# This is pure cross-modal text->image retrieval, so the dual encoder handles it
# natively (query text -> Qwen3-VL text embedding via `_VISION_INSTRUCTION`;
# page image -> vision-tower `to_video_caption` embedding; cosine ranking).
#
# Data is BEIR-format on HuggingFace (loaded over the network). Each entry is
# (hf_repo, language_filter_or_None, split). Every repo exposes three configs:
#   queries: {query-id, query}      corpus: {corpus-id, image}
#   qrels  : {query-id, corpus-id, score}   (graded relevance)
# Page images are read inline from the `corpus` split, so nothing needs to be
# downloaded/extracted locally. The primary metric is nDCG@5 (ViDoRe/MMEB-V2
# convention); we also report nDCG@10 and Recall@5.
VISDOC_DATASETS = {
    # ViDoRe v1
    "ViDoRe_arxivqa":     ("vidore/arxivqa_test_subsampled_beir",   None, "test"),
    "ViDoRe_docvqa":      ("vidore/docvqa_test_subsampled_beir",    None, "test"),
    "ViDoRe_infovqa":     ("vidore/infovqa_test_subsampled_beir",   None, "test"),
    "ViDoRe_tabfquad":    ("vidore/tabfquad_test_subsampled_beir",  None, "test"),
    "ViDoRe_tatdqa":      ("vidore/tatdqa_test_beir",               None, "test"),
    "ViDoRe_shiftproject": ("vidore/shiftproject_test_beir",        None, "test"),
    "ViDoRe_syntheticDocQA_artificial_intelligence": ("vidore/syntheticDocQA_artificial_intelligence_test_beir", None, "test"),
    "ViDoRe_syntheticDocQA_energy":            ("vidore/syntheticDocQA_energy_test_beir", None, "test"),
    "ViDoRe_syntheticDocQA_government_reports": ("vidore/syntheticDocQA_government_reports_test_beir", None, "test"),
    "ViDoRe_syntheticDocQA_healthcare_industry": ("vidore/syntheticDocQA_healthcare_industry_test_beir", None, "test"),
    # ViDoRe v2 (multilingual repos filtered to English unless *_multilingual)
    "ViDoRe_esg_reports_human_labeled_v2": ("vidore/esg_reports_human_labeled_v2", None, "test"),
    "ViDoRe_biomedical_lectures_v2":               ("vidore/biomedical_lectures_v2", "english", "test"),
    "ViDoRe_biomedical_lectures_v2_multilingual":  ("vidore/biomedical_lectures_v2", None, "test"),
    "ViDoRe_economics_reports_v2":                 ("vidore/economics_reports_v2", "english", "test"),
    "ViDoRe_economics_reports_v2_multilingual":    ("vidore/economics_reports_v2", None, "test"),
    "ViDoRe_esg_reports_v2":                       ("vidore/esg_reports_v2", "english", "test"),
    "ViDoRe_esg_reports_v2_multilingual":          ("vidore/esg_reports_v2", None, "test"),
    # VisRAG (test data lives in the `train` split of these repos)
    "VisRAG_ArxivQA":   ("openbmb/VisRAG-Ret-Test-ArxivQA",   None, "train"),
    "VisRAG_ChartQA":   ("openbmb/VisRAG-Ret-Test-ChartQA",   None, "train"),
    "VisRAG_MP-DocVQA": ("openbmb/VisRAG-Ret-Test-MP-DocVQA", None, "train"),
    "VisRAG_SlideVQA":  ("openbmb/VisRAG-Ret-Test-SlideVQA",  None, "train"),
    "VisRAG_InfoVQA":   ("openbmb/VisRAG-Ret-Test-InfoVQA",   None, "train"),
    "VisRAG_PlotQA":    ("openbmb/VisRAG-Ret-Test-PlotQA",    None, "train"),
}
# Query-side instruction. Kept identical to the other text sides so the query
# embedding lives in the same space the vision tower was aligned to.
VISDOC_QUERY_PROMPT = _VISION_INSTRUCTION


# ---------------------------------------------------------------------------
# video_ret
# ---------------------------------------------------------------------------

def eval_video_retrieval(ds_name, cfg, model, preprocess, text_encoder,
                         data_root, max_frames, device,
                         video_batch_size, num_workers):
    frames_root = os.path.join(
        data_root, "video-tasks", "frames", "video_ret",
        "data", "your_user", "video_retrieval", cfg['frames_dir'], "frames",
    )
    if not os.path.isdir(frames_root):
        print(f"  [SKIP] frames dir not found: {frames_root}")
        return None

    records = ret_load_annotations(ds_name, cfg, data_root)
    print(f"  {len(records)} annotations loaded")

    vid_id_to_idx, gallery_dirs = {}, []
    for vid_id, _ in records:
        if vid_id not in vid_id_to_idx:
            vid_id_to_idx[vid_id] = len(gallery_dirs)
            gallery_dirs.append(os.path.join(frames_root, vid_id))

    gallery_indices = _dist_indices(len(gallery_dirs))
    print(f"  Encoding {len(gallery_indices)}/{len(gallery_dirs)} videos ...")
    gallery_embs = encode_videos_batch(
        model, preprocess, [gallery_dirs[i] for i in gallery_indices],
        max_frames, device,
        video_batch_size=video_batch_size, num_workers=num_workers,
    )
    gallery_embs = _gather_indexed_embeddings(
        gallery_embs, gallery_indices, len(gallery_dirs))

    query_indices = _dist_indices(len(records))
    print(f"  Encoding {len(query_indices)}/{len(records)} query texts ...")
    captions = [records[i][1] for i in query_indices]
    query2gallery = [vid_id_to_idx[records[i][0]] for i in query_indices]
    query_embs = encode_texts(captions, text_encoder, prompt=_VISION_INSTRUCTION)
    query_embs = F.normalize(query_embs.to(device), dim=-1)
    query_embs = _gather_indexed_embeddings(
        query_embs.cpu(), query_indices, len(records)).to(device)
    if _DIST_ENABLED:
        gathered = [None] * _DIST_WORLD_SIZE
        dist.all_gather_object(gathered, (query_indices, query2gallery))
        query2gallery = [None] * len(records)
        for indices, targets in gathered:
            for i, target in zip(indices, targets):
                query2gallery[i] = target

    metrics = recall_at_k(query_embs, gallery_embs.to(device), query2gallery)
    print(f"  R@1={metrics['R@1']:.4f}  R@5={metrics['R@5']:.4f}  "
          f"R@10={metrics['R@10']:.4f}  MdR={metrics['MdR']:.1f}")
    return _broadcast_result(metrics)


# ---------------------------------------------------------------------------
# video_cls
# ---------------------------------------------------------------------------

def _load_cls_records(jsonl_path, neg_field):
    """Parse a classification jsonl.

    Returns (records, label_list). Each record is
    ``(video_id, gold_label, candidates_or_None)``:
      - When `neg_field` is set, the dataset ships a *per-record* candidate pool
        (e.g. SSv2 lists 174 instance-specific candidates per sample), so
        `candidates` is that per-record list (with the gold label ensured
        present) and ranking is done within it.
      - Otherwise `candidates` is None and classification is global over the
        unique set of gold labels found in the file.
    `label_list` is the sorted union of every candidate/label string, encoded
    once and shared across records.
    """
    records, labels = [], set()
    with open(jsonl_path) as f:
        for line in f:
            s = json.loads(line)
            vid = str(s['video_id'])
            gold = s['pos_text']
            cands = None
            if neg_field and s.get(neg_field):
                cands = list(s[neg_field])
                if gold not in cands:
                    cands.append(gold)
                labels.update(cands)
            else:
                labels.add(gold)
            records.append((vid, gold, cands))
    return records, sorted(labels)


def eval_video_classification(ds_name, cfg, model, preprocess, text_encoder,
                              data_root, max_frames, device,
                              video_batch_size, num_workers, templates):
    frames_root = os.path.join(data_root, "video-tasks", "frames", "video_cls",
                               cfg['frames_dir'])
    jsonl_path = os.path.join(_cls_data_dir(data_root), cfg['jsonl'])
    if not os.path.isdir(frames_root):
        print(f"  [SKIP] frames dir not found: {frames_root}")
        return None
    if not os.path.exists(jsonl_path):
        print(f"  [SKIP] annotation not found: {jsonl_path}")
        return None

    records, label_list = _load_cls_records(jsonl_path, cfg['neg_field'])
    label_to_idx = {lab: i for i, lab in enumerate(label_list)}
    per_record = cfg['neg_field'] is not None
    mode = "per-record candidates" if per_record else "global labels"
    print(f"  {len(records)} samples, {len(label_list)} candidate labels ({mode})")

    # Encode the candidate labels once, with template ensembling.
    print(f"  Encoding {len(label_list)} labels x {len(templates)} template(s) ...")
    label_embs = encode_label_bank(
        label_list, text_encoder, _VISION_INSTRUCTION, templates).to(device)  # (C, D)

    # Encode only this rank's records. Labels are small and intentionally
    # encoded on every rank; video inference is the expensive sharded part.
    local_records = [records[i] for i in _dist_indices(len(records))]

    # Encode videos (dedup by video_id).
    vid_to_idx, video_dirs = {}, []
    for vid, _, _ in local_records:
        if vid not in vid_to_idx:
            vid_to_idx[vid] = len(video_dirs)
            video_dirs.append(os.path.join(frames_root, vid))
    print(f"  Encoding {len(video_dirs)} videos for {len(local_records)} records ...")
    video_embs = encode_videos_batch(
        model, preprocess, video_dirs, max_frames, device,
        video_batch_size=video_batch_size, num_workers=num_workers,
    ).to(device)  # (V, D)

    # Predict: argmax over the (global or per-record) candidate set; count Hit@1.
    sims = video_embs @ label_embs.T          # (V, C)
    global_pred = sims.argmax(dim=-1).cpu()    # (V,)
    correct = 0
    missing = 0
    for vid, gold, cands in local_records:
        vemb_idx = vid_to_idx[vid]
        # Videos with no frames produced a zero embedding -> unreliable; count as wrong.
        if float(video_embs[vemb_idx].abs().sum()) == 0.0:
            missing += 1
            continue
        if per_record:
            cand_idx = [label_to_idx[c] for c in cands]
            pred = cand_idx[int(sims[vemb_idx, cand_idx].argmax())]
        else:
            pred = int(global_pred[vemb_idx])
        if pred == label_to_idx[gold]:
            correct += 1
    correct, missing, count = _all_reduce_counts(correct, missing, len(local_records))
    acc = correct / count if count else 0.0
    print(f"  Acc={acc:.4f}  ({correct}/{len(records)}"
          + (f", {missing} videos had no frames" if missing else "") + ")")
    return {"Acc": acc, "n": count, "num_labels": len(label_list),
            "missing_frames": missing}


# ---------------------------------------------------------------------------
# video_mret
# ---------------------------------------------------------------------------

def _mret_clip_dirs(video_dir):
    """Return (clip_dirs, positive_index) for a moment-retrieval video dir."""
    subs = sorted(d for d in glob.glob(os.path.join(video_dir, '*'))
                  if os.path.isdir(d))
    pos_idx = next((i for i, d in enumerate(subs)
                    if os.path.basename(d) == 'positive_clip'), None)
    return subs, pos_idx


def eval_moment_retrieval(ds_name, cfg, model, preprocess, text_encoder,
                          data_root, max_frames, device,
                          video_batch_size, num_workers):
    frames_root = os.path.join(data_root, "video-tasks", "frames",
                               "video_mret", "video_mret", cfg['frames_dir'])
    jsonl_path = os.path.join(_cls_data_dir(data_root), cfg['jsonl'])
    if not os.path.isdir(frames_root):
        print(f"  [SKIP] frames dir not found: {frames_root}")
        return None
    if not os.path.exists(jsonl_path):
        print(f"  [SKIP] annotation not found: {jsonl_path}")
        return None

    # Build the flat list of candidate clip dirs across all queries.
    queries, clip_dirs = [], []
    query_spans = []   # (start, end, positive_local_idx) into clip_dirs
    skipped = 0
    with open(jsonl_path) as f:
        for line in f:
            s = json.loads(line)
            vid_key = os.path.basename(str(s['clips_dir_path']).rstrip('/'))
            video_dir = os.path.join(frames_root, vid_key)
            if not os.path.isdir(video_dir):
                skipped += 1
                continue
            subs, pos_idx = _mret_clip_dirs(video_dir)
            if not subs or pos_idx is None:
                skipped += 1
                continue
            start = len(clip_dirs)
            clip_dirs.extend(subs)
            query_spans.append((start, len(clip_dirs), pos_idx))
            queries.append(s['query'])

    if not queries:
        print(f"  [SKIP] no usable queries (skipped {skipped})")
        return None
    local_query_indices = _dist_indices(len(queries))
    local_queries = [queries[i] for i in local_query_indices]
    local_spans = [query_spans[i] for i in local_query_indices]
    local_clip_dirs = []
    local_clip_spans = []
    for start, end, pos_local in local_spans:
        new_start = len(local_clip_dirs)
        local_clip_dirs.extend(clip_dirs[start:end])
        local_clip_spans.append((new_start, len(local_clip_dirs), pos_local))

    print(f"  {len(queries)} queries, {len(clip_dirs)} candidate clips"
          + (f" (skipped {skipped})" if skipped else ""))

    print("  Encoding query texts ...")
    q_embs = encode_texts(local_queries, text_encoder, prompt=_VISION_INSTRUCTION)
    q_embs = F.normalize(q_embs.to(device), dim=-1)          # (Q, D)

    print(f"  Encoding {len(local_clip_dirs)} clips ...")
    clip_embs = encode_videos_batch(
        model, preprocess, local_clip_dirs, max_frames, device,
        video_batch_size=video_batch_size, num_workers=num_workers,
    ).to(device)                                              # (K, D)

    hits = 0
    for qi, (start, end, pos_local) in enumerate(local_clip_spans):
        sims = q_embs[qi] @ clip_embs[start:end].T            # (num_clips,)
        if int(sims.argmax()) == pos_local:
            hits += 1
    hits, _, count = _all_reduce_counts(hits, 0, len(local_queries))
    acc = hits / count if count else 0.0
    print(f"  R@1={acc:.4f}  ({hits}/{len(queries)})")
    return {"R@1": acc, "n": count}


# ---------------------------------------------------------------------------
# image_cls (MMEB-V1)
# ---------------------------------------------------------------------------

def eval_image_classification(ds_name, hf_subset, model, preprocess, text_encoder,
                              data_root, device, batch_size, num_workers,
                              label_prompt, templates, label_map=None):
    from datasets import load_dataset
    img_base = _img_base(data_root)
    try:
        ds = load_dataset(IMG_CLS_HF_REPO, hf_subset, split="test")
    except Exception as e:
        print(f"  [SKIP] cannot load HF {IMG_CLS_HF_REPO}:{hf_subset}: {e}")
        return None

    image_paths, cand_lists, golds = [], [], []
    labels, missing_img = set(), 0
    for ex in ds:
        cands = list(ex['tgt_text'])
        if not cands:
            continue
        p = os.path.join(img_base, ex['qry_img_path'])
        image_paths.append(p)
        cand_lists.append(cands)
        golds.append(cands[0])       # MMEB convention: positive target at index 0
        labels.update(cands)
        if not os.path.exists(p):
            missing_img += 1
    if not image_paths:
        print("  [SKIP] no usable examples")
        return None
    label_list = sorted(labels)
    label_to_idx = {lab: i for i, lab in enumerate(label_list)}
    local_indices = _dist_indices(len(image_paths))
    local_image_paths = [image_paths[i] for i in local_indices]
    local_cand_lists = [cand_lists[i] for i in local_indices]
    local_golds = [golds[i] for i in local_indices]
    print(f"  {len(image_paths)} images, {len(label_list)} candidate labels"
          + (f", {missing_img} images missing on disk" if missing_img else ""))

    print(f"  Encoding {len(label_list)} labels x {len(templates)} template(s) ...")
    label_embs = encode_label_bank(
        label_list, text_encoder, label_prompt, templates, label_map=label_map).to(device)

    print(f"  Encoding {len(local_image_paths)}/{len(image_paths)} images ...")
    img_embs = encode_images_batch(
        model, preprocess, local_image_paths, device, batch_size, num_workers).to(device)

    sims = img_embs @ label_embs.T
    correct, missing = 0, 0
    for i, (cands, gold) in enumerate(zip(local_cand_lists, local_golds)):
        if float(img_embs[i].abs().sum()) == 0.0:
            missing += 1
            continue
        cand_idx = [label_to_idx[c] for c in cands]
        pred = cand_idx[int(sims[i, cand_idx].argmax())]
        if pred == label_to_idx[gold]:
            correct += 1
    correct, missing, count = _all_reduce_counts(
        correct, missing, len(local_image_paths))
    acc = correct / count if count else 0.0
    print(f"  Acc={acc:.4f}  ({correct}/{len(image_paths)}"
          + (f", {missing} unreadable" if missing else "") + ")")
    return {"Acc": acc, "n": count, "num_labels": len(label_list),
            "missing_images": missing}


# ---------------------------------------------------------------------------
# visdoc (MMEB-V2 Visual Document Retrieval)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _encode_visdoc_corpus(model, preprocess, corpus_ds, device,
                          batch_size=64, num_workers=8):
    """Encode BEIR corpus page images -> (normalized (M, D), corpus_id list).

    Images are decoded inline from the HuggingFace `corpus` split (PIL objects),
    preprocessed, and run through the same vision path as `encode_images_batch`
    (single image -> num_frame=1 through the Qwen-aligned `to_video_caption`
    head). Unreadable pages get a zero embedding so indexing stays aligned.
    """
    from concurrent.futures import ThreadPoolExecutor
    from tqdm import tqdm

    corpus_ids = [str(cid) for cid in corpus_ds['corpus-id']]
    n = len(corpus_ids)
    all_embs = [None] * n
    emb_dim = None

    def _prep(img):
        try:
            return preprocess(img.convert('RGB'))
        except Exception:
            return None

    for start in tqdm(range(0, n, batch_size), desc="  visdoc corpus"):
        images = corpus_ds[start:start + batch_size]['image']   # list of PIL
        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            loaded = list(pool.map(_prep, images))
        buf = [t if t is not None else torch.zeros(3, _IMAGE_SIZE, _IMAGE_SIZE)
               for t in loaded]
        pixel_values = torch.stack(buf).to(device)
        with torch.autocast(device_type=device.split(':')[0]):
            out = model(pixel_values, modality=_VISION_MODALITY, run_decoder=False)
        embs = out[_VISION_OUTPUT_KEY].detach().cpu()
        if emb_dim is None:
            emb_dim = embs.shape[-1]
        for j, t in enumerate(loaded):
            all_embs[start + j] = embs[j] if t is not None else torch.zeros(emb_dim)
    return F.normalize(torch.stack(all_embs), dim=-1), corpus_ids


def _dcg(rels):
    return sum(r / np.log2(i + 2) for i, r in enumerate(rels))


def _ndcg_at_k(ranked_ids, rel_map, k):
    """Linear nDCG@k with graded relevance (ViDoRe/MMEB-V2 convention)."""
    rels = [rel_map.get(cid, 0.0) for cid in ranked_ids[:k]]
    ideal = sorted(rel_map.values(), reverse=True)[:k]
    idcg = _dcg(ideal)
    return (_dcg(rels) / idcg) if idcg > 0 else 0.0


def _recall_at_k(ranked_ids, rel_ids, k):
    if not rel_ids:
        return 0.0
    return len(set(ranked_ids[:k]) & rel_ids) / len(rel_ids)


def eval_visdoc_retrieval(ds_name, cfg, model, preprocess, text_encoder,
                          device, batch_size, num_workers):
    from datasets import load_dataset
    hf_repo, lang, split = cfg
    try:
        queries = load_dataset(hf_repo, "queries", split=split)
        corpus = load_dataset(hf_repo, "corpus", split=split)
        qrels = load_dataset(hf_repo, "qrels", split=split)
    except Exception as e:
        print(f"  [SKIP] cannot load {hf_repo}: {e}")
        return None

    if lang is not None and 'language' in queries.column_names:
        queries = queries.filter(lambda ex: ex['language'] == lang)

    # qrels: query-id -> {corpus-id: score} (ids are stringified for matching).
    qrels_map = {}
    for qi, ci, sc in zip(qrels['query-id'], qrels['corpus-id'], qrels['score']):
        qrels_map.setdefault(str(qi), {})[str(ci)] = float(sc)

    q_ids = [str(x) for x in queries['query-id']]
    q_texts = list(queries['query'])
    # Keep only queries that have at least one relevant page.
    keep = [i for i, qid in enumerate(q_ids) if qrels_map.get(qid)]
    q_ids = [q_ids[i] for i in keep]
    q_texts = [q_texts[i] for i in keep]
    if not q_ids:
        print("  [SKIP] no queries with relevance judgements")
        return None
    corpus_indices = _dist_indices(len(corpus))
    local_corpus = corpus.select(corpus_indices) if _DIST_ENABLED else corpus
    print(f"  {len(q_ids)} queries, {len(corpus)} corpus pages")

    print(f"  Encoding {len(corpus_indices)}/{len(corpus)} corpus page images ...")
    corpus_embs, corpus_ids = _encode_visdoc_corpus(
        model, preprocess, local_corpus, device, batch_size, num_workers)
    corpus_embs = _gather_indexed_embeddings(
        corpus_embs, corpus_indices, len(corpus))
    if _DIST_ENABLED:
        corpus_ids = [str(x) for x in corpus['corpus-id']]
    corpus_embs = corpus_embs.to(device)

    print(f"  Encoding {len(q_ids)} query texts ...")
    query_indices = _dist_indices(len(q_ids))
    q_embs = encode_texts(
        [q_texts[i] for i in query_indices],
        text_encoder, prompt=VISDOC_QUERY_PROMPT)
    q_embs = F.normalize(q_embs.to(device), dim=-1)

    # Global retrieval: rank the whole corpus per query. Only the top-10 matter
    # for nDCG@10 / R@5, so take top-k instead of a full argsort.
    sims = q_embs @ corpus_embs.T                       # (Q, M)
    topk = min(10, sims.shape[1])
    top_idx = sims.topk(topk, dim=-1).indices.cpu().numpy()

    ndcg5 = ndcg10 = rec5 = 0.0
    local_q_ids = [q_ids[i] for i in query_indices]
    for qi, qid in enumerate(local_q_ids):
        ranked = [corpus_ids[j] for j in top_idx[qi]]
        rel_map = qrels_map[qid]
        ndcg5 += _ndcg_at_k(ranked, rel_map, 5)
        ndcg10 += _ndcg_at_k(ranked, rel_map, 10)
        rec5 += _recall_at_k(ranked, set(rel_map.keys()), 5)
    ndcg5, ndcg10, rec5, n = _all_reduce_counts(
        round(ndcg5 * 1000000), round(ndcg10 * 1000000),
        round(rec5 * 1000000), len(local_q_ids))
    ndcg5 /= 1000000
    ndcg10 /= 1000000
    rec5 /= 1000000
    res = {"nDCG@5": ndcg5 / n if n else 0.0,
           "nDCG@10": ndcg10 / n if n else 0.0,
           "R@5": rec5 / n if n else 0.0,
           "n": n, "corpus": len(corpus_ids)}
    print(f"  nDCG@5={res['nDCG@5']:.4f}  nDCG@10={res['nDCG@10']:.4f}  "
          f"R@5={res['R@5']:.4f}")
    return res


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def _select(names, available):
    if not names:
        return list(available)
    unknown = [n for n in names if n not in available]
    if unknown:
        print(f"[WARN] unknown datasets ignored: {unknown}")
    return [n for n in names if n in available]


def _resolve_templates(cli_override, cfg_templates, default):
    """CLI single-template override > per-dataset templates > task default."""
    if cli_override:
        return [cli_override]
    return cfg_templates or default


def _load_text_encoder(args, device):
    """Load the default Qwen3-VL text encoder used by this entry point."""
    if not args.text_model_path:
        raise ValueError(
            "no text model provided: pass --text_model_path /path/to/Qwen3-VL-Embedding "
            "(the SigLIP2 variants eval_mme_v2_all_siglip2*.py do not need it)"
        )
    from sentence_transformers import SentenceTransformer
    text_encoder = SentenceTransformer(
        args.text_model_path,
        trust_remote_code=True,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device=device,
    )
    backbone = getattr(text_encoder[0], "auto_model", None)
    if backbone is not None and type(backbone).__name__ == "Qwen3VLModel":
        _load_qwen3vl_backbone_weights(backbone, args.text_model_path)
    text_encoder.max_seq_length = 10000
    return text_encoder


def main(args):
    device = args.device
    global _IMAGE_SIZE, _CLEAN_LABELS, _NUM_FRAMES, _SEGMENT_T_SIZE
    _IMAGE_SIZE = args.image_size
    _CLEAN_LABELS = not args.raw_labels
    if (args.segment_t_size < 1 or args.num_frames < 1
            or args.num_frames % args.segment_t_size != 0):
        raise ValueError(
            f"--num_frames ({args.num_frames}) must be a positive multiple of "
            f"--segment_t_size ({args.segment_t_size}); get_segments splits T into "
            f"T // segment_t_size segments.")
    _NUM_FRAMES = args.num_frames
    _SEGMENT_T_SIZE = args.segment_t_size
    print(f"Loading vision model (input resolution {args.image_size}x{args.image_size}) ...")
    print(f"  video temporal: {_NUM_FRAMES} frames -> "
          f"{_NUM_FRAMES // _SEGMENT_T_SIZE} segment(s) of {_SEGMENT_T_SIZE} (get_segments)")
    model = load_model(args.ckpt, args.video_caption_embed_dim, device,
                       image_size=args.image_size, segment_t_size=args.segment_t_size)
    if args.native_resolution:
        preprocess = build_native_preprocess(max_side=args.max_image_side)
        # Native resolution -> each sample has its own H x W. The encoder takes a
        # single dense tensor with no resolution packing, so a batch cannot mix
        # sizes: force per-sample encoding.
        if args.image_batch_size != 1 or args.video_batch_size != 1:
            print("[INFO] --native_resolution: forcing image/video batch size to 1 "
                  "(batched inference needs a uniform resolution).")
        args.image_batch_size = 1
        args.video_batch_size = 1
        print(f"Native resolution ON: aspect-preserving, long side <= "
              f"{args.max_image_side}px, rounded to multiples of {_PATCH_SIZE}.")
    else:
        preprocess = build_preprocess(args.image_size)

    print("Loading text encoder ...")
    text_encoder = _load_text_encoder(args, device)

    results = {}

    if 'video_ret' in args.tasks:
        for ds in _select(args.datasets, RET_DATASETS):
            print(f"\n===== [video_ret] {ds} =====")
            m = eval_video_retrieval(
                ds, RET_DATASETS[ds], model, preprocess, text_encoder,
                args.data_root, args.max_frames, device,
                args.video_batch_size, args.num_workers)
            if m:
                results[f"video_ret/{ds}"] = m

    if 'video_cls' in args.tasks:
        for ds in _select(args.datasets, CLS_DATASETS):
            print(f"\n===== [video_cls] {ds} =====")
            templates = _resolve_templates(
                args.cls_label_template, CLS_DATASETS[ds].get('templates'),
                VIDEO_CLS_TEMPLATES)
            m = eval_video_classification(
                ds, CLS_DATASETS[ds], model, preprocess, text_encoder,
                args.data_root, args.max_frames, device,
                args.video_batch_size, args.num_workers, templates)
            if m:
                results[f"video_cls/{ds}"] = m

    if 'video_mret' in args.tasks:
        for ds in _select(args.datasets, MRET_DATASETS):
            print(f"\n===== [video_mret] {ds} =====")
            m = eval_moment_retrieval(
                ds, MRET_DATASETS[ds], model, preprocess, text_encoder,
                args.data_root, args.max_frames, device,
                args.video_batch_size, args.num_workers)
            if m:
                results[f"video_mret/{ds}"] = m

    if 'image_cls' in args.tasks:
        for ds in _select(args.datasets, IMG_CLS_DATASETS):
            print(f"\n===== [image_cls] {ds} =====")
            cfg = IMG_CLS_DATASETS[ds]
            templates = _resolve_templates(
                args.img_cls_label_template, cfg.get('templates'),
                IMG_CLS_TEMPLATES_DEFAULT)
            m = eval_image_classification(
                ds, cfg['hf'], model, preprocess, text_encoder,
                args.data_root, device, args.image_batch_size, args.num_workers,
                args.img_cls_label_prompt, templates, cfg.get('label_map'))
            if m:
                results[f"image_cls/{ds}"] = m

    if 'visdoc' in args.tasks:
        for ds in _select(args.datasets, VISDOC_DATASETS):
            print(f"\n===== [visdoc] {ds} =====")
            m = eval_visdoc_retrieval(
                ds, VISDOC_DATASETS[ds], model, preprocess, text_encoder,
                device, args.image_batch_size, args.num_workers)
            if m:
                results[f"visdoc/{ds}"] = m

    print("\n===== Summary =====")
    for name, m in results.items():
        if 'nDCG@5' in m:                       # visdoc retrieval
            print(f"{name:32s}  nDCG@5={m['nDCG@5']:.4f}  "
                  f"nDCG@10={m['nDCG@10']:.4f}  R@5={m['R@5']:.4f}")
            continue
        primary = m.get('R@1', m.get('Acc'))
        extra = ""
        if 'MdR' in m:                          # video_ret retrieval
            extra = f"  R@5={m['R@5']:.4f}  R@10={m['R@10']:.4f}  MdR={m['MdR']:.1f}"
        print(f"{name:32s}  {'R@1' if 'R@1' in m else 'Acc'}={primary:.4f}{extra}")

    if args.output and _DIST_RANK == 0:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default=None,
                        help='Path to the open_clip .pt checkpoint. Required for this '
                             'script; the HF-release variant (eval_mme_v2_all_hf.py) '
                             'ignores it and loads config.json + model.safetensors.')
    parser.add_argument('--data_root', type=str, required=True,
                        help='MMEB-V2 data root.')
    parser.add_argument('--text_model_path', type=str, default=None,
                        help='Path to the Qwen3-VL-Embedding text model. Required for '
                             'this (Qwen3-VL) entry point; the SigLIP2 variants '
                             '(eval_mme_v2_all_siglip2*.py) use --siglip2_dir instead.')
    parser.add_argument('--video_caption_embed_dim', type=int, default=4096)
    parser.add_argument('--max_frames', type=int, default=64,
                        help='[deprecated] superseded by --num_frames; kept for '
                             'backward compat and currently ignored for sampling.')
    parser.add_argument('--num_frames', type=int, default=16,
                        help='Uniformly sample exactly this many frames per clip. '
                             'Must be a positive multiple of --segment_t_size.')
    parser.add_argument('--segment_t_size', type=int, default=16,
                        help='Temporal segment size for the encoder get_segments '
                             'path; frames are split into num_frames // segment_t_size '
                             'segments. Must match the checkpoint training config.')
    parser.add_argument('--image_size', type=int, default=224,
                        help='Square input resolution for frames and images. Must be a '
                             'resolution the checkpoint supports (e.g. multi-reso ckpts).')
    parser.add_argument('--native_resolution', action='store_true',
                        help='Preprocess each image/frame at its native aspect ratio '
                             '(no center crop), sizes rounded to a multiple of the patch '
                             'size and capped by --max_image_side. Forces batch size 1 '
                             '(batched inference needs a uniform resolution). Most useful '
                             'for visdoc document pages; ignores --image_size.')
    parser.add_argument('--max_image_side', type=int, default=1400,
                        help='Long-side cap (pixels) under --native_resolution; the actual '
                             'side is rounded down to a multiple of the patch size (14).')
    parser.add_argument('--video_batch_size', type=int, default=1)
    parser.add_argument('--image_batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--tasks', nargs='*',
                        default=['video_ret', 'video_cls', 'video_mret'],
                        choices=['video_ret', 'video_cls', 'video_mret', 'image_cls', 'visdoc'],
                        help='Which meta-tasks to run. image_cls and visdoc need '
                             'HF access (set the proxy env vars).')
    parser.add_argument('--datasets', nargs='*', default=None,
                        help='Subset of dataset names (across selected tasks). Default: all.')
    parser.add_argument('--cls_label_template', type=str, default=None,
                        help='Single template to FORCE for video-cls labels (e.g. '
                             '"a video of {}."). Default: per-task template ensemble.')
    parser.add_argument('--img_cls_label_template', type=str, default=None,
                        help='Single template to FORCE for image-cls labels. '
                             'Default: per-dataset template ensemble.')
    parser.add_argument('--img_cls_label_prompt', type=str, default=IMG_CLS_LABEL_PROMPT,
                        help='Qwen encode prompt/instruction for image-cls label texts.')
    parser.add_argument('--raw_labels', action='store_true',
                        help='Encode class labels verbatim (skip _clean_label: first-synonym + '
                             'underscore->space normalization). Default: cleaning ON.')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output', type=str, default='mme_v2_all_results.json')
    return parser


def run_cli(parser=None):
    parser = parser or build_parser()
    args = parser.parse_args()
    args.device = _init_distributed(args.device)
    if _DIST_ENABLED:
        print(f"[distributed] rank {_DIST_RANK}/{_DIST_WORLD_SIZE}, "
              f"device={args.device}")
    main(args)
    if _DIST_ENABLED:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == '__main__':
    run_cli()
