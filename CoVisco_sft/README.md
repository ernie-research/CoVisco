# CoVisco LLM SFT — Multimodal Alignment & Instruction Tuning

This directory is **Stage 2** of the two-stage **CoVisco** pipeline (Codec-Native Vision Encoder with Native Token Compression for Efficient Unified Visual Understanding): it attaches the CoVisco vision encoder pretrained in [`../CoVisco_ViT`](../CoVisco_ViT) to a Qwen3 LLM,  We do **not** run a separate mid-training stage — the pretrained encoder goes directly into instruction tuning(with data from  mvp-lab/LLaVA-NeXT-780k-webdataset and  lmms-lab/LLaVA-Video-178K).

## Key Features

- **Reuses the CoVisco_ViT vision encoder** (`CoViscoViT`, with learnable query tokens, 4D RoPE 2:4:5:5)
- **Reuses the query-guided `LearnableTokenSelector`** (selects key visual tokens from ViT patch tokens — native token compression)
- **Dynamic token strategy**: randomly picks query_only / vit_only / query_and_vit at training time
- **Configurable sms**: currently sms=1 (no spatial merge), with sms=2/3 interfaces reserved
- **2-layer MLP projector** (consistent with OneVision-Encoder)
- **SFT data reuse OneVision-Encoder directly**, adapting only query/vit token handling at the model boundary

## Directory Layout

```
CoVisco_sft/
├── configs/                    # model/data/training yaml configs (covisco_qwen3_*.yaml)
├── data/                       # LLaVA batch adapter, fallback WDS, dynamic strategy
├── models/                     # CoVisco ViT, projector, token selector, top-level model
├── train/                      # training loop
├── scripts/                    # launch and evaluation scripts
├── docs/                       # data format, dynamic strategy, sms notes
├── third_party/lmms-eval/      # evaluation backend (includes the llava_covisco adapter)
└── tests/                      # end-to-end smoke tests
```

## Quick Start

```bash
# 1. Edit the config (LLM path, ViT pretrained weights path, etc.)
vim configs/covisco_qwen3_4b.yaml

# 2. Train the token selector with a lightweight Qwen3-1.7B LLM
#    (trains projector + token_selector + llm)
NPROC_PER_NODE=8 NNODES=1 bash scripts/run_sft_1.7b.sh

# 3. Instruction tuning on Qwen3-4B-Instruct2507, reusing the trained token selector
#    (trains projector + llm; set --ckpt to the step-2 checkpoint inside the script)
NPROC_PER_NODE=8 NNODES=1 bash scripts/run_sft_4b_instruct2507.sh
```

In the config, `vit.pretrained_path` points to the `CoVisco-L-14` weights produced by CoVisco_ViT; `vit.name` is `CoVisco-L-14`.

### Loading the ViT from Hugging Face

`vit.pretrained_path` accepts four source types, auto-detected by `models/covisco_vit_checkpoint.py::load_vit_weights_direct`:

- a local training checkpoint `*.pt`,
- a local `*.safetensors` file,
- a local Hugging Face release directory (containing `model.safetensors`),
- a Hugging Face Hub repo id — e.g. the released encoder [`ernie-research/CoVisco-L-14`](https://huggingface.co/ernie-research/CoVisco-L-14) (fetched via `snapshot_download`; set `HF_TOKEN` for a private/gated repo).

```yaml
vit:
  name: CoVisco-L-14
  # pick ONE of:
  pretrained_path: ernie-research/CoVisco-L-14        # Hub repo id
  # pretrained_path: /path/to/CoVisco-L-14            # local HF release dir
  # pretrained_path: /path/to/CoVisco-L-14.safetensors
  # pretrained_path: /path/to/CoVisco-L-14.pt
```

Loading matches encoder tensors by exact name after stripping `module.`/`model.`/`encoder.` prefixes; the HF release's extra top-level weights (`head.*`, `visual.*` alias, `logit_*`, `proj_to_*`) are reported as *unexpected* and skipped — this is expected. The architecture fields under `vit.*` must match the release's `covisco_encoder_cfg` in `config.json` (hidden_size, num_layers, patch_size, etc.); mismatches are skipped as shape errors. Needs `safetensors` + `huggingface_hub` (both already in `requirements.txt`).

## Data Flow

Training reads WebDataset tar shards through `data/wds_dataset.py::CoViscoWDSDataset` — this is the **actual training path** (`--data-backend fallback_wds`, the default used by all SFT scripts); it is a self-contained streaming tar reader with no dependency on the `webdataset` package. The alternative `--data-backend llava_onevision` is a placeholder that currently raises `NotImplementedError`; it is meant for plugging in LLaVA-OneVision-2's external dataloader, in which case `data/llava_batch_adapter.py` converts a LLaVA/Energon batch into this model's fields at the model boundary. See `docs/data_alignment.md`.

### Data Format (WebDataset)

Within a tar, members sharing the same `__key__` (everything before the first `.`) form one sample. Decoding/dispatch is in `TarShardReader._fill_sample`. Image and video are **not** mixed inside a shard — they come from separate dataset instances (`DATA_PATH` for images, `VIDEO_DATA_PATH` for videos); per-sample modality is taken from the JSON `modality` field. Shards are a directory of `*.tar` (converter output is named `shard_{N:06d}.tar` under a `train/` subdir) or a newline-delimited shard-list file.

Per-sample members:

- `*.json` — LLaVA-OneVision-style annotation: `{"messages": [{"role": "user"/"assistant", "content": ...}], "modality": "image"|"video"|"text", "source_id": ...}`. `<image>` / `<video>` placeholders in `content` are expanded by `data/plugin.py` into `<|vision_start|> + N×<|image_pad|> + <|vision_end|>`; the collator later resizes the pad count per the sampled `token_plan`.
- `*.jpg` / `*.jpeg` / `*.png` — one image (image sample), or multiple frames `key0.jpg, key1.jpg, …` in tar order (video "frame format" sample, ≤32 frames in the first SFT version).
- `*.mp4` — video "mp4 format"; decoded to `video_target_frames` (default 128) uniformly-sampled frames (decord → cv2 fallback).
- `*.visidx.npy` — *(optional, mp4 videos)* codec-filtered sparse ViT patch indices, int64, 1-D ascending, shape `(num_seg * k_per_segment,)`; global index `idx = t*(h*w) + row*w + col` with `h=w=image_size/patch_size` (224/14=16 → 256 patches/frame). Produced offline by `scripts/precompute_codec_visidx/precompute_video_visidx.py`. The loader's `segment_t_size` (default 32) must match `vit.segment_t_size` in the config.

Sample variants: **image** = `key.jpg` + `key.json`; **video (frames)** = `key0.jpg…keyK.jpg` + `key.json`; **video (mp4)** = `key.mp4` + `key.json` (+ optional `key.visidx.npy`); **text** = `key.json` only. See `docs/data_format.md`.

A one-step converter from LLaVA-NeXT format is provided: `tools/convert_llava_next_to_wds.py` (maps `conversations` `human/gpt` → `user/assistant`, writes `shard_*.tar` plus `manifest.json` with `total_samples` / `shard_size`). The `--num-samples-per-epoch` used for epoch→step conversion is the **image** sample count (e.g. 780K = 390 shards × 2000), not the video count.

## Training

We skip mid-training: the `CoVisco-L-14` encoder pretrained in `CoVisco_ViT` is loaded via the config's `vit.pretrained_path` and goes straight into instruction tuning. Training is a two-step process, both using the `sft` stage:

1. **Token selector training** (`run_sft_1.7b.sh`) — on a lightweight Qwen3-1.7B LLM, train `projector + token_selector + llm` (lr 1e-4). This step learns the query-guided `LearnableTokenSelector`.
2. **Instruction tuning** (`run_sft_4b_instruct2507.sh`) — on Qwen3-4B-Instruct2507, load the step-1 checkpoint (`--ckpt`) and train `projector + llm` (lr 5e-5), reusing the token selector from step 1.

## Architecture

```
Input (images / video frame sequences)
    ↓
CoViscoViT (24 layers, 1024 hidden, 100 query/segment, 4D+3D RoPE)
    ↓
(query_tokens: (B, S, 100, D), vit_tokens: (B, S, P, D))
    ↓
LearnableTokenSelector (query-guided top-K vit selection)
    ↓
selected_vit: (B, S, K, D)
    ↓
concat by strategy: [q_seg0, v_seg0, q_seg1, v_seg1, ...]
    ↓
all_vision: (B, total_tokens, 1024)
    ↓
TwoLayerMLPProjector (sms=1, 2-layer MLP, 1024 → 5120 → 2560)
    ↓
vision_embeds: (B, total_tokens, 2560)
    ↓
masked_scatter into the <|image_pad|> positions of input_ids
    ↓
Qwen3 LLM (4B / 8B / ...)
    ↓
LM loss (assistant tokens only)
```

The top-level model is `LlavaCoViscoModel` in `models/llava_covisco.py`.

## Scripts (`scripts/`)

All scripts `cd` into the repo root themselves, so they can be launched from anywhere (e.g. `bash scripts/run_sft_4b_instruct2507.sh`). Paths like `DATA_PATH` / `CHECKPOINT_PATH` inside the scripts are examples from the authors' cluster — edit them (or override via environment variables) before running.

### Training

A two-step instruction-tuning pipeline (no alignment / mid-training). Step 1 trains the token selector on a small LLM; step 2 does the actual SFT on the larger instruct LLM, reusing the selector.

| Step | Script | Model | Launcher | Trainable modules |
|---|---|---|---|---|
| 1. Token selector | `run_sft_1.7b.sh` | Qwen3-1.7B | multi-node `torchrun` | `projector token_selector llm` |
| 2. SFT | `run_sft_4b_instruct2507.sh` | Qwen3-4B-Instruct2507 | multi-node `torchrun` | `projector llm` |

Both scripts are driven by environment variables so you rarely edit the file itself (step 2 still needs `--ckpt` set to the step-1 checkpoint inside the script):

```bash
# Step 1 — single node, 8 GPUs
NPROC_PER_NODE=8 NNODES=1 bash scripts/run_sft_1.7b.sh

# Step 2 — single node, 8 GPUs
NPROC_PER_NODE=8 NNODES=1 bash scripts/run_sft_4b_instruct2507.sh

# Multi node (set per node, or use launch_multinode.sh below)
NNODES=8 NODE_RANK=0 MASTER_ADDR=10.0.0.1 bash scripts/run_sft_4b_instruct2507.sh
```

Common environment variables (see the comment blocks at the top of each script for the full list):

| Variable | Meaning |
|---|---|
| `NPROC_PER_NODE` / `NNODES` / `NODE_RANK` | torchrun topology (GPUs per node / node count / this node's rank) |
| `MASTER_ADDR` / `MASTER_PORT` | rendezvous address of rank-0 (default `127.0.0.1:29500`) |
| `NUM_EPOCHS` / `NUM_SAMPLES_PER_EPOCH` | epoch count and dataset size used for epoch→step conversion |
| `NATIVE_RESOLUTION` | `1` = per-image native resolution (aspect-ratio preserving, forces image micro-batch 1); `0` = fixed `IMAGE_SIZE_*` |
| `NATIVE_MIN_PATCHES` / `NATIVE_MAX_PATCHES` | patch-count bounds under native resolution (256 ≈ 224², 4096 ≈ 896²) |
| `IMAGE_SIZE_IMAGE` / `IMAGE_SIZE_VIDEO` | fixed per-modality input resolution when native resolution is off |
| `UNIFORM_SAMPLE_N` / `UNIFORM_SEGMENT_T_SIZE` / `UNIFORM_TRAIN_PROB` | uniform frame-sampling branch for video (vs. the visidx sparse path) |

### Multi-node launcher

`launch_multinode.sh` parses a `hostfile`, then SSHes into every node and runs a training script with `NODE_RANK` / `MASTER_ADDR` filled in automatically. Logs land in `logs/node<i>_<script>.log`.

```bash
# Usage: bash scripts/launch_multinode.sh <train_script.sh> [KEY=VAL ...]
HOSTFILE=/path/to/hostfile bash scripts/launch_multinode.sh scripts/run_sft_4b_instruct2507.sh NUM_EPOCHS=2
```

Requirements: passwordless SSH between nodes and a shared filesystem so the script path and data are identical on every host.

### Evaluation

**Environment setup.** Evaluation uses the vendored `third_party/lmms-eval` plus `accelerate`, which need a larger dependency set than training. `scripts/setup_eval_env.sh` installs everything in the correct order and protects the training `transformers` pin:

```bash
bash scripts/setup_eval_env.sh                 # create .env-eval and install
CREATE_VENV=0 bash scripts/setup_eval_env.sh   # install into the current env
```

It installs `requirements.txt` (pins `transformers>=4.51,<4.52` for Qwen3), then `requirements-eval.txt` (accelerate / opencv / datasets / lmms-eval runtime deps, **without** transformers) under a constraint, then `pip install -e third_party/lmms-eval --no-deps`. The sft model uses Qwen3 (`model_type=qwen3`), so — unlike `CoVisco_ViT` eval — it must stay on `transformers` 4.51.x and does **not** need 4.57. Not handled by the script: a CUDA-enabled torch build + GPU(s), a system `ffmpeg` (OpenCV decodes `.mp4` benchmarks), pre-downloaded benchmark data (`HF_HOME` / `HF_DATASETS_CACHE`), `HF_TOKEN`, and — for `eval_covisco_codec.sh` only — the compiled `cv_reader` extension (see the codec visidx precompute section below; its source is vendored in this repo).

**Pre-download the base LLMs.** The eval configs load a base LLM from a local path (`llm.path`); it must be present before evaluating. Fetch both and set `llm.path` in the matching config:

```bash
bash ../download_eval_models.sh sft   # -> ./models/{Qwen3-4B-Instruct-2507, Qwen3-1.7B}
# then set llm.path in the config you evaluate:
#   configs/covisco_qwen3_4b_instruct2507_instruction_tuning.yaml -> /abs/path/to/models/Qwen3-4B-Instruct-2507
#   configs/covisco_qwen3_1.7b.yaml                               -> /abs/path/to/models/Qwen3-1.7B
```

`Qwen3-4B-Instruct-2507` backs the main 4B eval config; `Qwen3-1.7B` backs `covisco_qwen3_1.7b.yaml` (the token-selector model, also used for its caption / token-selector eval). Your own CoVisco encoder (`vit.pretrained_path`) and the trained SFT `.pt` checkpoint are separate artifacts, not downloaded by the helper.

Evaluation runs through the lmms-eval backend; the model is registered as `llava_covisco` (see `third_party/lmms-eval/.../simple/llava_covisco.py`). All three scripts accept `CONFIG_PATH`, `CHECKPOINT_PATH`, `TASKS`, `NUM_GPUS`, `TOKEN_STRATEGY`, `VIT_RATIO` etc. as environment variables.

| Script | Purpose |
|---|---|
| `eval_covisco.sh` | image / general benchmarks and standard (≤64-frame) video benchmarks |
| `eval_covisco_longvideo.sh` | long-video benchmarks (256 frames/video, tasks run one at a time) |
| `eval_covisco_codec.sh` | video evaluation with codec token filtering via precomputed visidx |

```bash
# Image / general benchmarks (comma-separated tasks)
TASKS="ai2d,chartqa,docvqa_val" bash scripts/eval_covisco.sh

# Standard video benchmark (run ONE at a time)
TASKS="videomme" bash scripts/eval_covisco.sh

# Long video
TASKS="mlvu_dev" bash scripts/eval_covisco_longvideo.sh

# Codec-filtered video (requires the precompute step below)
TASKS="videomme" bash scripts/eval_covisco_codec.sh
```

`TOKEN_STRATEGY` must match the training distribution: `query_only` / `vit_only` / `query_and_vit`. Use `vit_only` for OCR / document / chart tasks (full patches, `VIT_RATIO` ignored). Results are written to `eval_results/<MODEL_NAME>/`.

Hugging Face / proxy note: set `HF_TOKEN` for gated datasets. Point `HF_HOME` at the directory holding the (pre-downloaded) evaluation datasets and keep the (small) arrow cache on a local disk via `HF_DATASETS_CACHE` — some network/FUSE mounts do not support `filelock`, so the cache must live on a local filesystem. Adjust both to your environment.

### Codec visidx precompute (`scripts/precompute_codec_visidx/`)

A one-time preprocessing step required before `eval_covisco_codec.sh`. It resolves each benchmark's video paths through lmms-eval (triggering download) and precomputes per-video `visidx.npy` by scoring H.264/H.265 motion vectors + residuals.

```bash
# One run per benchmark; grid params must match eval_covisco_codec.sh
bash scripts/precompute_codec_visidx/preprocess_video_benchmark.sh videomme
```

**Environment.** The scoring functions and the `cv_reader` extension source are vendored in `scripts/precompute_codec_visidx/Compressed_Video_Reader/` (from [AcherStyx/Compressed-Video-Reader](https://github.com/AcherStyx/Compressed-Video-Reader)), so no external repository is needed. The extension itself still requires a one-time build: it compiles `src/cv_reader/h264_api.cpp` against a **patched FFmpeg** (upstream FFmpeg does not export DCT residuals; the patch lives in `ffmpeg/ffmpeg_patch/`). Build once per machine:

```bash
cd scripts/precompute_codec_visidx/Compressed_Video_Reader
bash install.sh          # downloads FFmpeg source, applies the patch, compiles both
cv_reader -h             # verify the install
```

Requirements for the build: a C++ toolchain, `pkg-config`, `numpy < 2`, and `opencv-python` (Python deps listed in its `setup.py`). `precompute_video_visidx.py` imports the scoring functions from the vendored `tool/` directory by default; set `OV_ENCODER_TOOL_DIR` only to override with a separately installed copy.

- `precompute_video_visidx.py` — core worker invoked by the shell wrapper (also runnable directly with `--jsonl --out_root --num_video_frames ...`). Needs the compiled `cv_reader` (see above).
- `retry_failed_h264.py` — transcodes videos that `cv_reader` rejected (non-H264/HEVC, e.g. MPEG-4 in mvbench) to H.264 via `ffmpeg`, then recomputes their visidx. Pass the same grid params as the main precompute.

### Inference / debugging

```bash
# Full inference smoke test (ViT → selector → projector → LLM); optionally run a task
python scripts/test_inference.py --config configs/covisco_qwen3_4b_instruct2507.yaml
python scripts/test_inference.py --config configs/covisco_qwen3_4b_instruct2507.yaml --task ai2d

# Single-image / image-dir caption test (env-var driven)
IMAGE=/path/to/img.jpg bash scripts/run_test_caption.sh
IMAGE_DIR=/path/to/images STRATEGY=query_and_vit bash scripts/run_test_caption.sh
```

`run_test_caption.sh` wraps `tools/test_caption.py`; override `CONFIG`, `CHECKPOINT`, `PROMPT`, `STRATEGY`, `VIT_RATIO`, and `NATIVE_RESOLUTION` via environment variables. Keep `NATIVE_RESOLUTION` consistent with how the checkpoint was trained.

### Resume helper (skip already-trained shards)

To resume a run and skip image shards consumed before the last checkpoint, pass a
newline-separated list of shard filenames via `SKIP_SHARDS=/path/to/skip_list.txt` to
`run_sft_4b_instruct2507.sh` (videos loop and are not skipped).

## Dependencies

Training (`requirements.txt`):

- torch >= 2.0, torchvision
- transformers >= 4.51, < 4.52 (Qwen3, `model_type=qwen3`)
- timm >= 1.0.17, webdataset, safetensors, fsspec
- numpy / pandas / pyyaml / pillow, braceexpand, regex, ftfy, tqdm
- av, imageio, imageio-ffmpeg (video decoding)

Evaluation adds the lmms-eval backend deps on top of the above — see `requirements-eval.txt` and `scripts/setup_eval_env.sh` (`accelerate`, `opencv-python-headless`, `datasets`, task metrics, etc.). `transformers` stays on the training pin (4.51.x).

## License

This project is released under the MIT License. See the [LICENSE](../LICENSE) file at the repository root.

The vendored `third_party/lmms-eval` retains its own upstream MIT license (see `third_party/lmms-eval/LICENSE`).
