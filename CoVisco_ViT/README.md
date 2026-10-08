# CoVisco ViT — Vision Encoder Pretraining

This directory is **Stage 1** of the two-stage **CoVisco** pipeline (Codec-Native Vision Encoder with Native Token Compression for Efficient Unified Visual Understanding): pretraining the CoVisco vision encoder that handles images and videos in a unified way. The codebase is built on [OpenCLIP](https://github.com/mlfoundations/open_clip) (see Acknowledgments).

The pretrained `CoVisco-L-14` weights are consumed by Stage 2, [`../CoVisco_sft`](../CoVisco_sft), which attaches them to a Qwen3 LLM for instruction tuning.

## Core Design

- **Unified image/video encoding**: a single encoder handles both single images and video frame sequences; videos are split into segments of `segment_t_size` and frames are uniformly sampled.
- **Native token compression at two levels**: each segment introduces `num_query_per_seg=100` learnable **abstract tokens**; inside the ViT they aggregate the segment's information and pass information across segments, acting as a compression mechanism in themselves. On top of that, a **token selector** compresses the fine-grained **patch tokens**, keeping only the key ones.
- **SigLIP-style distillation against two teacher branches**: pre-extracted teacher embeddings serve as the supervision signal. `CoViscoModel` carries multiple projection heads, each contrast pair using an **independent** `logit_scale` / `logit_bias`:
  - **SigLIP2 branch** (`to_image_caption`, embed_dim `1536`): distilled from the **SigLIP2** text tower (`ViT-gopt-16-SigLIP2-384`, `pretrained="webli"`). This is the branch used for **image classification / retrieval** tasks.
  - **Qwen3-VL-Embedding branch** (`to_video_caption`, embed_dim `4096`): distilled from **Qwen3-VL-Embedding-8B**. This is the branch used for **video classification / retrieval** tasks.
- **4D RoPE**: positional encoding that respects spatio-temporal structure.

## Model Spec (`CoVisco-L-14`)

- Backbone: ViT-L/14, `image_size=224`, `patch_size=14`, `layers=24`, `width=1024`, `head_width=64`
- `num_query_per_seg=100`, `segment_t_size=32`, `rope_theta=10000`
- Config file: `src/open_clip/model_configs/CoVisco-L-14.json`

## Key Files

```
CoVisco_ViT/
├── src/open_clip/
│   ├── covisco_model.py                 # CoViscoModel: encoder + projection heads (+ optional reconstruction)
│   ├── covisco_vit.py                   # unified vision encoder (unified image/video)
│   ├── token_selector.py                # token selector (compresses fine-grained patch tokens)
│   └── model_configs/CoVisco-L-14.json  # model config (identifier = CoVisco-L-14)
├── src/open_clip_train/
│   └── main_covisco.py                  # training entry (SigLIP + multi contrast pairs + reconstruction)
├── scripts/                             # training launch scripts (MODEL_NAME=CoVisco-L-14)
├── eval/                                # evaluation scripts + shell wrappers
│   ├── eval_zeroshot_siglip2.py         # zero-shot classification / retrieval (SigLIP2 branch)
│   ├── eval_mme_v2_all*.py              # unified MMEB-V2 suite (SigLIP2 + Qwen3-VL-Embedding branches)
│   ├── eval_mmeb_video_retrieval*.py    # video retrieval (Qwen3-VL-Embedding / SigLIP2 branches)
│   ├── covisco_hf.py                    # shared loader for the *_hf.py evaluators (HF release)
│   ├── eval_*_hf.py                     # HF-release variants (load config.json + model.safetensors)
│   └── run_*.sh                         # shell wrappers for the evaluators
├── hf_release/CoVisco-L-14/             # self-contained HF release (config.json, model.safetensors, loader)
└── docs/                                # preprocessing and data notes
```

## Quick Start

### Install

```bash
python3 -m venv .env && source .env/bin/activate
pip install -U pip
pip install -e '.[training]'
```

> **Evaluation uses a separate (newer) environment.** The Qwen3-VL-Embedding
> evaluators need `transformers>=4.57` (see the note under [Evaluate](#evaluate)),
> which conflicts with the training pin. Use a dedicated venv for eval:
>
> ```bash
> python3 -m venv .env-eval && source .env-eval/bin/activate
> pip install -U pip
> pip install -e .                      # base package (torch / torchvision / timm / ...)
> pip install -r requirements-eval.txt  # or: pip install -e '.[eval]'
> ```

### Train

```bash
# Joint image + video pretraining (qwen3vl teacher; defaults to MODEL_NAME=CoVisco-L-14)
bash scripts/run_covisco_train.sh
```

The model is loaded by name `CoVisco-L-14` via `open_clip.create_model_and_transforms`; the factory in `src/open_clip/factory.py` routes this name to the `CoViscoModel` branch.

### Evaluate

Which teacher branch an evaluation uses depends on the task family:

- **Image classification / retrieval → SigLIP2 branch.** Image embeddings come from the `to_image_caption` head and are matched against the SigLIP2 text tower (`ViT-gopt-16-SigLIP2-384`). The checkpoint must have been trained with `image_caption_embed_dim=1536`. Scripts carry the `_siglip2` suffix.
- **Video classification / retrieval/VisDoc → Qwen3-VL-Embedding branch.** Embeddings come from the `to_video_caption` head (`video_caption_embed_dim=4096`) and are matched against the `Qwen3-VL-Embedding-8B` text encoder. `eval/eval_mme_v2_all.py` (wrapper: `eval/run_mme_v2_all_qwen3-vl.sh`) is the unified MMEB-V2 evaluator on this branch and runs the full meta-task suite (`video_ret` / `video_cls` / `video_mret` plus `image_cls` / `visdoc`).

**MMEB-V2 is evaluated with both branches.** We provide two MMEB-V2 evaluators, one per teacher text encoder:

- **SigLIP2 branch** — `eval/eval_mme_v2_all_siglip2.py` (wrapper `eval/run_mme_v2_all_siglip2.sh`), matched against the SigLIP2 text tower via the `to_image_caption` head. The SigLIP2 text encoder only supports **short text** (a fixed, small token budget, ~64 tokens), so it is appropriate for tasks whose queries/captions are short; longer prompts get truncated.
- **Qwen3-VL-Embedding branch** — `eval/eval_mme_v2_all.py` (wrapper `eval/run_mme_v2_all_qwen3-vl.sh`), matched against `Qwen3-VL-Embedding-8B` via the `to_video_caption` head. This LLM-based text encoder supports **long text**, so it handles long instructions/queries without truncation and covers the full MMEB-V2 meta-task suite.

So on MMEB-V2, pick the SigLIP2 evaluator for short-text tasks and the Qwen3-VL-Embedding evaluator for long-text (and the full suite). Both read the same `CoVisco-L-14` checkpoint; they differ only in which teacher text encoder and projection head are used.

**Pre-download the text encoders.** Both branches match against a frozen text model that must be present before evaluating:

- SigLIP2 branch → `timm/ViT-gopt-16-SigLIP2-384` (open_clip auto-downloads this via `pretrained=webli`; pre-download is only needed for offline runs, then pass `--siglip2_dir`).
- Qwen3-VL-Embedding branch → `Qwen/Qwen3-VL-Embedding-8B` (loaded from a local path via `--text_model_path` / `TEXT_MODEL`).

Use the repo-root helper, then point the scripts at the local dirs (set `HF_TOKEN` / `HF_ENDPOINT` / proxy if needed):

```bash
bash ../download_eval_models.sh vit   # -> ./models/{Qwen3-VL-Embedding-8B, ViT-gopt-16-SigLIP2-384}
```

Your own `CoVisco-L-14.pt` checkpoint is a separate trained artifact and is not downloaded by the helper.

> **transformers version:** `Qwen3-VL-Embedding-8B` has `model_type: qwen3_vl` in its
> `config.json`, an architecture added in **transformers 4.57**. Install the eval deps
> (`pip install -r requirements-eval.txt`, which pins `transformers>=4.57.1` and
> `tokenizers>=0.22.2`) before running the Qwen3-VL-Embedding evaluators; otherwise
> `AutoConfig.from_pretrained` raises `KeyError: 'qwen3_vl'`. The SigLIP2-branch
> evaluators do not require this.

```bash
# --- Image tasks: SigLIP2 branch ---
# Zero-shot classification (incl. ImageNet) + text<->image retrieval
# (ImageNet / COCO / Flickr / XM3600); modes: smoke | cls | retrieval | imagenet | all
CKPT=/path/to/CoVisco-L-14.pt IMAGE_SIZE=224 \
    IMAGENET1K_WDS="/path/to/imagenet/val/{0..6}.tar" \
    bash eval/run_zeroshot_siglip2.sh all

# MME-V2 in the SigLIP2 text space
CKPT=/path/to/CoVisco-L-14.pt bash eval/run_mme_v2_all_siglip2.sh

# --- Video tasks: Qwen3-VL-Embedding branch ---
# Unified MMEB-V2 suite (video_ret / video_cls / video_mret / image_cls / visdoc);
# modes: smoke | video | image | visdoc | all
CKPT=/path/to/CoVisco-L-14.pt TEXT_MODEL=/path/to/Qwen3-VL-Embedding-8B \
    bash eval/run_mme_v2_all_qwen3-vl.sh all

# MMEB-V2 video retrieval — no shell wrapper; run the evaluator directly
python eval/eval_mmeb_video_retrieval.py --ckpt /path/to/CoVisco-L-14.pt \
    --text_model_path /path/to/Qwen3-VL-Embedding-8B --video_caption_embed_dim 4096
```

#### Loading from the Hugging Face release (`*_hf.py`)

Each evaluator has a `_hf.py` twin that loads the **released** vision encoder —
the self-contained `config.json` + `model.safetensors` + `covisco_*.py` under
`hf_release/CoVisco-L-14` (or any Hugging Face Hub repo) — instead of an
open_clip `.pt` training checkpoint. This removes the `open_clip` dependency for
the vision tower; the teacher text encoders (SigLIP2 / Qwen3-VL) are unchanged,
so the same eval environment and text-model pre-download apply.

The twins are thin wrappers (shared loader in `eval/covisco_hf.py`) that reuse all
dataset / metric / text-encoder logic from the base evaluators. They replace
`--ckpt` with `--hf_model`, which accepts either a local release directory or a
Hub repo id (auto `snapshot_download`; set `HF_TOKEN` / proxy for a private
repo). `--hf_model` defaults to the Hub repo `ernie-research/CoVisco-L-14` (set
`HF_TOKEN` for the private repo); pass a local release dir (e.g.
`hf_release/CoVisco-L-14`) to load from disk instead.

```bash
# Image tasks, SigLIP2 branch — load from a local release dir
python eval/eval_zeroshot_siglip2_hf.py --tasks classification retrieval \
    --hf_model hf_release/CoVisco-L-14 --image_size 224 \
    --imagenet1k_wds "/path/to/imagenet/val/{0..6}.tar"

# Video tasks, Qwen3-VL branch — default Hub repo (ernie-research/CoVisco-L-14; set HF_TOKEN)
python eval/eval_mme_v2_all_hf.py \
    --data_root /path/to/mme_v2 --text_model_path /path/to/Qwen3-VL-Embedding-8B

# Other twins: eval/eval_mmeb_video_retrieval_hf.py,
#              eval/eval_mmeb_video_retrieval_siglip2_hf.py,
#              eval/eval_mme_v2_all_siglip2_hf.py
```

The three shell wrappers have `_hf` counterparts too (same modes
`smoke|video|image|visdoc|all`, env-knob driven, `torchrun`-aware); they swap
the `CKPT` knob for `HF_MODEL` (defaults to the Hub repo
`ernie-research/CoVisco-L-14`; set a local dir or another Hub repo id to
override):

```bash
# SigLIP2 branch, zero-shot (HF_MODEL defaults to ernie-research/CoVisco-L-14; set HF_TOKEN)
bash eval/run_zeroshot_siglip2_hf.sh all

# SigLIP2 branch, MME-V2 — override with a local release dir
HF_MODEL=hf_release/CoVisco-L-14 bash eval/run_mme_v2_all_siglip2_hf.sh all

# Qwen3-VL-Embedding branch, MME-V2 (default Hub repo)
TEXT_MODEL=/path/to/Qwen3-VL-Embedding-8B bash eval/run_mme_v2_all_qwen3-vl_hf.sh all
```

> The released `config.json` sets `segment_t_size=32`; the `*_hf.py` video
> evaluators override it to `16` to match the base scripts. This is a
> runtime-only setting and does not change any weight shapes, so the same
> `model.safetensors` loads cleanly.

#### Extracting representations directly

To get features in your own code (not through an evaluator), load the release
with `covisco_hf.load_vision_model` and call the model. The loader lives at
`eval/covisco_hf.py`, so run these snippets from the `eval/` directory (or add
it to `sys.path` / `PYTHONPATH` before importing). The forward returns the
**pooled** embedding by default; pass `return_intermediates=True` to also get
the **pre-pooling** token representations. The input layout differs between
images and videos.

**Image** — `pixel_values` is `(B, C, H, W)`:

```python
import torch
from PIL import Image
from covisco_hf import load_vision_model, build_preprocess

model, cfg, _ = load_vision_model("ernie-research/CoVisco-L-14", device="cuda")
preprocess = build_preprocess(image_size=224)
px = preprocess(Image.open("img.jpg").convert("RGB")).unsqueeze(0).cuda()  # (B, C, H, W)

with torch.no_grad():
    out = model(px, modality="image", return_intermediates=True, run_decoder=False)

pooled = out["pooled_output"]       # (B, D)        pooled representation
query_tokens = out["query_tokens"]  # (B, S, Q, D)  pre-pooling abstract/query tokens
vit_tokens = out["vit_tokens"]      # (B, S, N-Q, D) pre-pooling ViT patch tokens
```

**Video** — uniformly sample a fixed number of frames from an `.mp4`,
preprocess each frame, and stack into `(B, C, T, H, W)` (channel before time).
The frame count `T` must be divisible by `segment_t_size`, which yields
`S = T / segment_t_size` segments; the released config defaults to `32`, so
override it (the video evaluators use `16`) to match your clip length. Decoding
uses PyAV (`pip install av`), the same backend the training pipeline prefers:

```python
import av, numpy as np, torch
from covisco_hf import load_vision_model, build_preprocess

NUM_FRAMES = 16  # must be divisible by segment_t_size

def read_video_frames(path, num_frames):
    """Uniformly sample `num_frames` RGB frames (PIL) from an mp4 via np.linspace."""
    container = av.open(path)
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"
    frames = [f.to_image() for f in container.decode(stream)]  # all frames as PIL RGB
    container.close()
    idx = np.linspace(0, len(frames) - 1, num_frames, dtype=int)
    return [frames[i] for i in idx]  # exactly num_frames frames

model, cfg, _ = load_vision_model("ernie-research/CoVisco-L-14", device="cuda", segment_t_size=16)
preprocess = build_preprocess(image_size=224)

frames = read_video_frames("video.mp4", NUM_FRAMES)   # NUM_FRAMES PIL frames
clip = torch.stack([preprocess(f) for f in frames])   # (T, C, H, W)
px = clip.permute(1, 0, 2, 3).unsqueeze(0).cuda()      # (B, C, T, H, W)

with torch.no_grad():
    out = model(px, modality="video",
                return_intermediates=True, run_decoder=False)

video_emb = out["to_video_caption"]  # (B, 4096)      L2-normalized video embedding (retrieval)
pooled = out["pooled_output"]        # (B, D)         pooled representation
query_tokens = out["query_tokens"]   # (B, S, Q, D)   pre-pooling abstract/query tokens
vit_tokens = out["vit_tokens"]       # (B, S, N-Q, D) pre-pooling ViT patch tokens
```

> `read_video_frames` decodes the whole clip then subsamples by `np.linspace`
> (mirrors the OpenCV path in `src/open_clip_train/data_wds.py`); for very long
> videos use PyAV seeking instead to avoid decoding every frame.

`pooled_output` is what the projection heads (`to_image_caption` /
`to_video_caption`) consume for contrastive matching; `query_tokens` /
`vit_tokens` are the uncompressed sequence outputs if you need token-level
features (see `covisco_model.py:217`). Use `modality="image"` or
`modality="video"` to compute only the relevant projection head(s).

## Data Format (WebDataset)

Pretraining uses WebDataset tar shards with **pre-extracted teacher embeddings** as the supervision signal (enabled by `--dataset-type webdataset_embedding`). Three independent dataloaders are built — image, image-vidcap, and video — each reading its own shard set. The loader lives in `src/open_clip_train/data_wds.py`; see `docs/covisco_data_preprocess.md` for the full preprocessing/augmentation details.

Within a tar, files sharing the same basename (the stem before the first `.`) form one sample; the multi-part extension after the first `.` is the field key. All `*.npy` feature files are 1-D float32 vectors (one per sample).

Image shards (`image_shard_*.tar`), decoded by `decode_image_with_features`:

- `*.jpg` / `*.png` — raw image, preprocessed to a `(3, 224, 224)` tensor
- `*.image_features.npy` — SigLIP2 teacher **image** embedding, `[1536]` (image↔image contrast, head `to_image`)
- `*.text_features.npy` — SigLIP2 teacher **caption** embedding, `[1536]` (image↔caption contrast, head `to_image_caption`)
- `*.text_emb.npy` — *(optional)* caption embedding in the **video** text-encoder space, `[4096]` (shared `to_video_caption` head)
- A sample is kept only if it has image bytes **and** at least one of the three feature files (`filter_no_features`).

Video shards (`video_shard_*.tar`), decoded by `decode_video_with_features`:

- `*.mp4` — raw video, pre-sampled offline to ≤128 frames (decoded via PyAV → imageio → OpenCV fallback), padded/truncated to 128 and stacked to `(C, T, H, W)`
- `*.text_emb.npy` — **required** video-caption embedding, `[4096]` (video↔caption contrast, head `to_video_caption`)
- `*.visidx.npy` — *(optional)* codec-filtered sparse ViT patch indices, int64 `[L]` (L=4096), values in `[0, 32767]` for the 128-frame × (224/14)²=256-patch grid; global index `idx = t*(h*w) + row*w + col`. If any sample in a batch lacks `visidx`, the whole batch falls back to uniform frame sampling.
- A sample is kept only if it has `mp4` **and** `text_emb.npy`.

Shards are globbed by prefix (`image_shard_*` / `image_vidcap_shard_*` / `video_shard_*`), falling back to all `*.tar` in the directory. There is no per-shard size file in this path: totals come from `samples_per_shard × num_shards` unless overridden by `--image-num-samples` / `--video-num-samples` etc. in `scripts/run_covisco_train.sh`.

## Handoff to Downstream SFT

The pretrained `CoVisco-L-14` checkpoint is loaded in `../CoVisco_sft` through the config's `vit.pretrained_path`, reusing its vision encoder (`CoViscoViT`) and token selector, then attached to Qwen3 for multimodal alignment and instruction tuning.

## Acknowledgments

The vision training framework is built on [OpenCLIP](https://github.com/mlfoundations/open_clip) (the modeling and training code under `src/open_clip` and `src/open_clip_train` are adapted from it). 
