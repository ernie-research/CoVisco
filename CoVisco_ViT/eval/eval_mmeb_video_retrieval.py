"""
MMEB-V2 Video-Level Retrieval Evaluation using CoVisco SigLIP model.

Datasets: MSR-VTT, MSVD, DiDeMo, VATEX, YouCook2
Annotation fields per sample:
  MSR-VTT: video_id, video, caption
  MSVD:    video_id, video, caption (caption is a list, use caption[0])
  DiDeMo:  video, caption   (video_id = basename without ext)
  VATEX:   videoID, video,  enCap (list, use enCap[0])
  YouCook2: id, video, sentence

Frame root: video-tasks/frames/video_ret/data/your_user/video_retrieval/<Dataset>/frames/<video_id>/

Usage:
    MMEB_V2_DATA_DIR=/path/to/mme_v2 python eval_mmeb_video_retrieval.py \
        --ckpt /path/to/checkpoint.pt \
        --data_root /path/to/data/mme_v2
"""

import argparse
import glob
import json
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# ---------------------------------------------------------------------------
# Dataset config: (local_jsonl_name, hf_repo, hf_subset, hf_split,
#                  video_id_field, caption_field, frames_subdir)
# ---------------------------------------------------------------------------
DATASETS = {
    "MSR-VTT": dict(
        jsonl="msr-vtt.jsonl",
        hf=("VLM2Vec/MSR-VTT", "test_1k", "test"),
        id_field="video_id", cap_field="caption",
        frames_dir="MSR-VTT",
    ),
    "MSVD": dict(
        jsonl="msvd.jsonl",
        hf=("VLM2Vec/MSVD", None, "test"),
        id_field="video_id", cap_field="caption",
        frames_dir="MSVD",
    ),
    "DiDeMo": dict(
        jsonl="didemo.jsonl",
        hf=("VLM2Vec/DiDeMo", None, "test"),
        id_field=None,          # derive from 'video' basename
        cap_field="caption",
        frames_dir="DiDeMo",
    ),
    "VATEX": dict(
        jsonl="vatex.jsonl",
        hf=("VLM2Vec/VATEX", None, "test"),
        id_field="videoID", cap_field="enCap",
        frames_dir="VATEX",
    ),
    "YouCook2": dict(
        jsonl="youcook2-val.jsonl",
        hf=("lmms-lab/YouCook2", None, "val"),
        id_field="id", cap_field="sentence",
        frames_dir="YouCook2",
    ),
}

_QWEN_INSTRUCTION = "Instruct: Represent the video description for cross-modal matching.\nQuery: "

# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def build_preprocess(image_size: int = 224):
    return transforms.Compose([
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                             (0.26862954, 0.26130258, 0.27577711)),
    ])


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(ckpt_path: str, video_caption_embed_dim: int, device: str):
    from open_clip.covisco_model import CoViscoModel
    from open_clip.covisco_vit import CoViscoEncoderConfig

    config = CoViscoEncoderConfig(
        hidden_size=1024, num_hidden_layers=24, num_attention_heads=16,
        num_channels=3, image_size=224, patch_size=14,
        num_query_per_seg=100, segment_t_size=16,
        use_head=True, output_dim=1024,
    )
    model = CoViscoModel(
        covisco_config=config,
        image_embed_dim=1536,
        image_caption_embed_dim=1536,
        video_caption_embed_dim=video_caption_embed_dim,
        use_reconstruction=False,
    )
    if ckpt_path:
        sd = torch.load(ckpt_path, map_location='cpu', weights_only=True)
        sd = sd.get('state_dict', sd)
        sd = {k.replace('module.', '', 1): v for k, v in sd.items()}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing:
            print(f"[WARN] Missing keys ({len(missing)}): {missing[:3]} ...")
        if unexpected:
            print(f"[WARN] Unexpected keys ({len(unexpected)}): {unexpected[:3]} ...")
    return model.eval().to(device)


# ---------------------------------------------------------------------------
# Encoding helpers
# ---------------------------------------------------------------------------

_SEGMENT_T_SIZE = 32  # must match training config

def _align_to_segment(n: int) -> int:
    """Return the largest multiple of _SEGMENT_T_SIZE that is <= n, min 1 segment."""
    return max(_SEGMENT_T_SIZE, (n // _SEGMENT_T_SIZE) * _SEGMENT_T_SIZE)


def encode_video(model, preprocess, frame_dir: str, max_frames: int, device: str) -> torch.Tensor:
    paths = sorted(
        glob.glob(os.path.join(frame_dir, '*.jpg')) +
        glob.glob(os.path.join(frame_dir, '*.png'))
    )
    if not paths:
        raise FileNotFoundError(f"No frames in {frame_dir}")
    # Align to nearest multiple of segment_t_size (e.g. 35 -> 32, 64 -> 64, 96 -> 96)
    target = min(_align_to_segment(len(paths)), max_frames)
    if len(paths) != target:
        idx = [int(i * len(paths) / target) for i in range(target)]
        paths = [paths[i] for i in idx]
    frames = torch.stack([preprocess(Image.open(p).convert('RGB')) for p in paths])
    # frames: (T, C, H, W) -> (B=1, C, T, H, W)
    pixel_values = frames.permute(1, 0, 2, 3).unsqueeze(0).to(device)
    with torch.no_grad(), torch.autocast(device_type=device.split(':')[0]):
        out = model(pixel_values, modality='video')
    return out['to_video_caption'].squeeze(0)   # (D,)


def _load_one_video(args):
    frame_dir, max_frames, preprocess = args
    paths = sorted(
        glob.glob(os.path.join(frame_dir, '*.jpg')) +
        glob.glob(os.path.join(frame_dir, '*.png'))
    )
    if not paths:
        return None
    target = min(_align_to_segment(len(paths)), max_frames)
    if len(paths) != target:
        idx = [int(j * len(paths) / target) for j in range(target)]
        paths = [paths[j] for j in idx]
    frames = torch.stack([preprocess(Image.open(p).convert('RGB')) for p in paths])
    return frames.permute(1, 0, 2, 3)  # (C, T, H, W)


@torch.no_grad()
def encode_videos_batch(model, preprocess, frame_dirs: list, max_frames: int,
                        device: str, video_batch_size: int = 2,
                        num_workers: int = 4) -> torch.Tensor:
    """Encode all videos with batched GPU inference; stream IO to avoid RAM OOM."""
    from concurrent.futures import ThreadPoolExecutor
    from tqdm import tqdm

    all_embs = [None] * len(frame_dirs)
    emb_dim  = None

    # Process in sliding windows of (num_workers * video_batch_size) to cap peak RAM usage
    window = max(video_batch_size, num_workers * video_batch_size)
    args_all = [(d, max_frames, preprocess) for d in frame_dirs]

    for win_start in tqdm(range(0, len(frame_dirs), window), desc="  video batches"):
        win_args = args_all[win_start:win_start + window]
        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            loaded = list(pool.map(_load_one_video, win_args))

        # GPU inference in video_batch_size sub-batches within the window
        for start in range(0, len(loaded), video_batch_size):
            sub   = loaded[start:start + video_batch_size]
            idxs  = list(range(win_start + start, win_start + start + len(sub)))
            buf   = [f if f is not None else torch.zeros(3, _SEGMENT_T_SIZE, 224, 224)
                     for f in sub]
            try:
                pixel_values = torch.stack(buf).to(device)
                with torch.autocast(device_type=device.split(':')[0]):
                    out = model(pixel_values, modality='video')
                embs = out['to_video_caption'].detach().cpu()
                if emb_dim is None:
                    emb_dim = embs.shape[-1]
            except Exception as e:
                print(f"[WARN] batch encode failed: {e}, falling back to zeros")
                embs = torch.zeros(len(sub), emb_dim or 1)
            for j, idx in enumerate(idxs):
                all_embs[idx] = embs[j] if loaded[start + j] is not None else torch.zeros(emb_dim or 1)

        # Release this window's frames immediately
        del loaded

    return F.normalize(torch.stack(all_embs), dim=-1)  # (M, D)


def encode_texts_batch(texts: list, text_encoder, batch_size: int = 64) -> torch.Tensor:
    embs = text_encoder.encode(
        texts,
        prompt=_QWEN_INSTRUCTION,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=batch_size,
        convert_to_numpy=True,
    )
    return torch.from_numpy(np.array(embs))


# ---------------------------------------------------------------------------
# Annotation loading
# ---------------------------------------------------------------------------

def load_annotations(ds_name: str, cfg: dict, data_root: str):
    """Returns list of (video_id, caption)."""
    jsonl_path = os.path.join(data_root, "video-tasks", "data", cfg['jsonl'])
    records = []

    if os.path.exists(jsonl_path):
        print(f"  Loading {ds_name} from local {jsonl_path}")
        with open(jsonl_path) as f:
            for line in f:
                s = json.loads(line.strip())
                vid_id = _get_id(s, cfg)
                cap    = _get_caption(s, cfg)
                records.append((vid_id, cap))
    else:
        print(f"  {jsonl_path} not found, loading from HuggingFace ...")
        from datasets import load_dataset
        repo, subset, split = cfg['hf']
        ds = load_dataset(repo, subset, split=split)
        for s in ds:
            vid_id = _get_id(s, cfg)
            cap    = _get_caption(s, cfg)
            records.append((vid_id, cap))

    return records


def _get_id(sample: dict, cfg: dict) -> str:
    if cfg['id_field']:
        return str(sample[cfg['id_field']])
    # DiDeMo: derive from video path
    return os.path.splitext(os.path.basename(sample['video']))[0]


def _get_caption(sample: dict, cfg: dict) -> str:
    val = sample[cfg['cap_field']]
    if isinstance(val, list):
        return val[0]
    return val


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def recall_at_k(query_embs: torch.Tensor, gallery_embs: torch.Tensor,
                query2gallery: list) -> dict:
    """
    query_embs:   (N, D)
    gallery_embs: (M, D)
    query2gallery: list of length N, each entry is the gallery index for that query
    """
    sims = query_embs @ gallery_embs.T   # (N, M)
    gt = torch.tensor(query2gallery, device=sims.device)
    ranks = (sims.argsort(dim=-1, descending=True) == gt.unsqueeze(1)).nonzero()[:, 1]
    return {
        'R@1':  (ranks < 1).float().mean().item(),
        'R@5':  (ranks < 5).float().mean().item(),
        'R@10': (ranks < 10).float().mean().item(),
        'MdR':  (ranks.float().median() + 1).item(),
    }


# ---------------------------------------------------------------------------
# Per-dataset evaluation
# ---------------------------------------------------------------------------

def eval_dataset(ds_name: str, cfg: dict, model, preprocess, text_encoder,
                 data_root: str, max_frames: int, device: str,
                 args_video_batch_size: int = 2, args_num_workers: int = 4) -> dict | None:
    frames_root = os.path.join(
        data_root,
        "video-tasks", "frames", "video_ret",
        "data", "your_user", "video_retrieval",
        cfg['frames_dir'], "frames"
    )
    if not os.path.isdir(frames_root):
        print(f"  [SKIP] frames dir not found: {frames_root}")
        return None

    records = load_annotations(ds_name, cfg, data_root)
    print(f"  {len(records)} annotations loaded")

    # Build deduplicated video gallery
    vid_id_to_idx = {}
    gallery_dirs  = []
    for vid_id, _ in records:
        if vid_id not in vid_id_to_idx:
            vid_id_to_idx[vid_id] = len(gallery_dirs)
            gallery_dirs.append(os.path.join(frames_root, vid_id))

    # Encode gallery videos (batched)
    print(f"  Encoding {len(gallery_dirs)} videos ...")
    gallery_embs = encode_videos_batch(
        model, preprocess, gallery_dirs, max_frames, device,
        video_batch_size=args_video_batch_size, num_workers=args_num_workers
    )  # (M, D)

    # Encode all query texts in one batch
    print(f"  Encoding {len(records)} query texts ...")
    captions      = [cap for _, cap in records]
    query2gallery = [vid_id_to_idx[vid_id] for vid_id, _ in records]
    query_embs    = encode_texts_batch(captions, text_encoder, batch_size=64)
    query_embs    = F.normalize(query_embs.to(device), dim=-1)   # (N, D)

    metrics = recall_at_k(query_embs, gallery_embs.to(device), query2gallery)
    print(f"  R@1={metrics['R@1']:.4f}  R@5={metrics['R@5']:.4f}  "
          f"R@10={metrics['R@10']:.4f}  MdR={metrics['MdR']:.1f}")
    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    device = args.device

    print("Loading vision model ...")
    model      = load_model(args.ckpt, args.video_caption_embed_dim, device)
    preprocess = build_preprocess()

    print("Loading text encoder (Qwen3-VL-Embedding-8B) ...")
    from sentence_transformers import SentenceTransformer
    text_encoder = SentenceTransformer(
        args.text_model_path,
        trust_remote_code=True,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device=device,
    )
    text_encoder.max_seq_length = 10000

    all_results = {}
    for ds_name, cfg in DATASETS.items():
        if args.datasets and ds_name not in args.datasets:
            continue
        print(f"\n===== {ds_name} =====")
        m = eval_dataset(ds_name, cfg, model, preprocess, text_encoder,
                         args.data_root, args.max_frames_ret, device,
                         args_video_batch_size=args.video_batch_size,
                         args_num_workers=args.num_workers)
        if m:
            all_results[ds_name] = m

    print("\n===== Summary =====")
    for name, m in all_results.items():
        print(f"{name:12s}  R@1={m['R@1']:.4f}  R@5={m['R@5']:.4f}  "
              f"R@10={m['R@10']:.4f}  MdR={m['MdR']:.1f}")

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default=None,
                        help='Checkpoint path. Leave empty for random init.')
    parser.add_argument('--data_root', type=str, required=True,
                        help='MMEB-V2 data root.')
    parser.add_argument('--text_model_path', type=str, required=True,
                        help='Path to the Qwen3-VL-Embedding text model.')
    parser.add_argument('--video_caption_embed_dim', type=int, default=4096)
    parser.add_argument('--max_frames_ret', type=int, default=64,
                        help='Max frames per video (data has 64 frames/video).')
    parser.add_argument('--video_batch_size', type=int, default=1,
                        help='Batch size for video encoding (increase if GPU has enough memory).')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='DataLoader workers for parallel frame loading.')
    parser.add_argument('--datasets', nargs='*', default=None,
                        help='Subset of datasets to eval. Default: all.')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output', type=str, default='mme_v2_vret_results.json')
    args = parser.parse_args()
    main(args)
