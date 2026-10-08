# CoVisco Training: Image & Video Preprocessing and Data Augmentation

This document summarizes the image/video preprocessing used to train the CoVisco vision encoder (`CoVisco-L-14`, patch_size=14, image_size=224), together with the data augmentation applied both at the data-loading stage and in the model forward pass.

Entry point: `src/open_clip_train/main_covisco.py`
Training script (recommended): `scripts/run_covisco_train.sh` (WebDataset form, the one actually used).

---

## 1. Overall Flow

Training consumes two data streams simultaneously: **image** and **video**, each with its own batch size, worker count, and gradient-accumulation frequency, but sharing the same image-transform construction logic.

`main_covisco.py` calls `create_model_and_transforms(...)` once, which returns:

- `preprocess_train`: image training transform (with augmentation)
- `preprocess_val`: image validation transform (no augmentation, used for ImageNet zero-shot)
- `video_preprocess_train`: if `--video-size` is given, a separate per-frame video transform is built via `image_transform(..., is_train=True, ...)`, **inheriting** `preprocess_train`'s mean/std/interpolation/resize_mode and aug_cfg
  - see `main_covisco.py:238-268`
  - if `--video-size` is not given, the video stream falls back to `preprocess_train`

The dataset type is selected by `--dataset-type`:

- `embedding` (CSV + filesystem) → `EmbeddingDataset` (`src/open_clip_train/data.py:56`)
- `webdataset_embedding` (tar shards, **the one actually used**) → `get_image_wds_dataset` / `get_video_wds_dataset` (`src/open_clip_train/data_wds.py:448, 563`)

---

## 2. Image Preprocessing (image_transform, training branch)

Implemented in `src/open_clip/transform.py:324`, invoked indirectly by `create_model_and_transforms`.

### 2.1 Default `PreprocessCfg` (`transform.py:18-25`)

- `size`: 224 (inherited from `CoVisco-L-14.json`'s `image_size`, or forced via `--force-image-size 224`)
- `mode`: `'RGB'`
- `mean`: `OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)`
- `std`: `OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)`
- `interpolation`: `'bicubic'`
- `resize_mode`: `'shortest'`
- `fill_color`: 0

These can be overridden via `--image-mean / --image-std / --image-interpolation / --image-resize-mode`; the current training script does **not** override them and uses the defaults above.

### 2.2 Default `AugmentationCfg` (`transform.py:62-73`)

- `scale = (0.9, 1.0)` (random area ratio for RandomResizedCrop)
- `ratio = None` (uses torchvision default `(3/4, 4/3)`)
- `color_jitter = None` (off by default)
- `re_prob = None` (random erasing off)
- `color_jitter_prob = None`
- `gray_scale_prob = None`
- `use_timm = False`

`scripts/run_covisco_train.sh` does not pass `--aug-cfg`, so the default AugmentationCfg is used.

### 2.3 Training transform composition (`transform.py:357-408`)

With `is_train=True` and `use_timm=False`, in order:

1. `RandomResizedCrop(image_size=224, scale=(0.9, 1.0), interpolation=BICUBIC)`
   - **the only random geometric augmentation currently enabled**
2. `MaybeConvertMode()` (convert to RGB if needed)
3. (optional) `color_jitter(...)`: enabled when `aug_cfg.color_jitter_prob` is set — **currently off**
4. (optional) `gray_scale(...)`: enabled when `aug_cfg.gray_scale_prob` is set — **currently off**
5. `MaybeToTensor()` (PIL→Tensor, [0,1])
6. `Normalize(mean=OPENAI_DATASET_MEAN, std=OPENAI_DATASET_STD)`

Explicitly disabled: horizontal flip `hflip=0` (that parameter only exists in the `use_timm=True` branch) — current training has **no horizontal flip**.

### 2.4 Validation / Zero-shot transform (`transform.py:410-440`)

When `resize_mode='shortest'` and `image_size` is square:

1. `Resize(224, interpolation=BICUBIC)` (shorter side to 224)
2. `CenterCrop(224)`
3. `MaybeConvertMode()`
4. `MaybeToTensor()`
5. `Normalize(mean, std)`

---

## 3. Video Preprocessing

### 3.1 Video frame loading (WebDataset path)

Defined in `src/open_clip_train/data_wds.py:280`: `load_video_frames_from_bytes`.

- Source: the `mp4` binary data inside the tar
- **No further sampling**: each mp4 is offline-sampled to 128 frames and all of them are used
- Loading order (fallback chain):
  1. **PyAV** (`av.open(...).decode(...)`, with `thread_type='AUTO'`) — preferred
  2. **imageio** (FFmpeg) — fallback 1
  3. **OpenCV** (write to a temp file then `cv2.VideoCapture`) — fallback 2
- Output: `List[PIL.Image]` of length ≤ 128

The CSV/EmbeddingDataset path (`data.py:117 _load_video`) uses OpenCV only and samples just 32 frames; **the WebDataset path is the current training path**.

### 3.2 Per-frame transform

Provided by `video_preprocess_train` (`main_covisco.py:258-266`), applying the same training transform as images to each frame:

- The current `scripts/run_covisco_train.sh` sets `--video-size 224`, so video frames also go through the full 224 train transform:
  `RandomResizedCrop(224, scale=(0.9, 1.0), BICUBIC) → MaybeConvertMode → ToTensor → Normalize(OPENAI mean/std)`
- **Note**: `RandomResizedCrop` is sampled **independently per frame** in `apply_video_transform` (`data_wds.py:215`), so different frames of the same video may use different crop windows.

### 3.3 Frame-count alignment (pad / truncate)

`apply_video_transform` (`data_wds.py:215-243`):

- Target `target_frames = 128` (`get_video_wds_dataset` default, or overridden by `--video-num-frames`)
- If frames < 128: **clone the last frame** up to 128
- If frames > 128: **truncate** to the first 128
- Finally stacked and permuted to `(C, T, H, W)`, becoming `(B, C, T, H, W)` after batching

### 3.4 visible_indices

Each mp4 also carries a `.visidx.npy` holding the filtered video ViT token indices, shape `[L]`, L=4096, value range `[0, 32767]`. Decoded to `torch.long` by `decode_video_with_features` (`data_wds.py:181`) and fed into the encoder as the indices for sparse token selection (the image branch is always `None`).

---

## 4. Supervision Signal (not pixel augmentation, but part of the "data")

Besides pixel values / video frames, each sample needs pre-extracted embeddings:

- Image tar: `*.image_features.npy` (SigLIP2 image features), `*.text_features.npy` (SigLIP2 text features)
- Video tar: `*.text_emb.npy` (video caption text embedding), `*.visidx.npy`

Pre-extracted teacher features are the core of the supervision — the "other side" of the contrastive (SigLIP) loss is provided by these embeddings, so the data pipeline must guarantee both file types are present (`filter_no_features`, `data_wds.py:428`).

---

## 5. Frame-level Video Augmentation in the Model Forward Pass

Beyond the data-loading transforms, **two additional random augmentations are applied to video frames during the model forward pass**, implemented across the three forward paths (normal, accumulation cache, accumulation fresh) in `src/open_clip_train/train.py:721-734, 749-758, 800-805`.

Each forward re-samples randomly:

```python
segment_offset = randint(0, 8)                                       # segment-position jitter (fed to RoPE)
use_uniform   = args.random_uniform_frame_sample and modality == "video" and random() < 0.5
use_frame_concat = args.frame_concat and use_uniform and random() < 0.5
```

These are then passed to `model(..., segment_offset=..., uniform_sample_frames=..., frame_concat=...)`.

### 5.1 `segment_offset` (always on)

Value `randint(0, 8)`, passed to the encoder as the segment-index offset along the RoPE time dimension (`get_query_tokens_rope(seg_offset=...)` in `covisco_vit.py`), implementing a random shift of the temporal-segment positional encoding — a kind of **temporal RoPE jitter augmentation**. It is passed for both images and videos, but is semantically meaningless for images.

### 5.2 `uniform_sample_frames` (video only, 50% probability)

Controlled by `--random-uniform-frame-sample` (`params.py:711`, **default True**, explicitly passed by the current training script).

When triggered, the encoder calls `uniform_sample_frames(pixel_values, num_frames=16)` (`covisco_vit.py:25-40`):

- Input: `(B, C, 128, H, W)`
- Output: `(B, C, 16, H, W)`, **uniformly sampling 16 frames** along T via `torch.linspace(0, T-1, 16)`
- Then `get_uniform_frame_segments` (`segment_t_size=4`, `sample_frames=16`) splits into 4 segments of 4 frames each for the ViT

Effect: across different steps the same video is seen either as a **dense 128-frame representation** or a **sparse 16-frame representation**, learning both granularities in alternation.

### 5.3 `frame_concat` (video only, requires uniform already triggered, then 50%)

Controlled by `--frame-concat` (`params.py:723`, default False, not enabled by the current training script).

When triggered, calls `uniform_sample_frames_and_concat(..., concat_mode='horizontal')`
(`covisco_vit.py:41-91`):

- First uniformly sample 16 frames
- Then reshape to `(B, C, 4, 4, H, W)` and concatenate every 4 frames horizontally along width
- Output: `(B, C, 4, H, W*4)`, 4 "wide frames"
- Handled by `get_uniform_frame_segments_then_concat_frames` (`segment_t_size=1`)

Effect: compresses the time dimension by spatially tiling frames, so a single segment contains information from 4 time steps.

> In summary, each video may hit one of the following four forward paths during training:
> 1. Full 128 frames (50%)
> 2. Uniformly sampled 16 frames (25%)
> 3. 16 frames horizontally tiled into 4 wide frames (only when `--frame-concat` is on)
> 4. Only when the full-128-frame path is combined with visible_indices, tokens are taken by sparse index
>
> (Each is sampled independently per forward, including every micro-batch within gradient accumulation.)

---

## 6. Current Actual Training Parameters (`scripts/run_covisco_train.sh`)

Data:

- `IMAGE_DATA_PATH`: image shard directory (~6400 images per shard)
- `VIDEO_DATA_PATH`: video shard directory (~1000 clips per shard)

Input resolution:

- `--force-image-size 224`
- `--video-size 224`

Batch / loading concurrency:

- `IMAGE_BATCH_SIZE=48`, `IMG_ACCUM_FREQ=11`
- `VIDEO_BATCH_SIZE=3`, `VID_ACCUM_FREQ=24`
- `IMAGE_WORKERS=8`, `VIDEO_WORKERS=8`

Augmentation switches:

- `--random-uniform-frame-sample` (explicitly enabled; video takes the uniform-16-frame branch with 50% probability)
- `--frame-concat` not passed (kept off)
- `--aug-cfg` not passed (defaults: `scale=(0.9,1.0)`, no ColorJitter / no Gray / no RandomErasing / no horizontal flip)

---

## 7. Quick Reference

- **Decode** — Image: PIL `Image.open`; Video: PyAV → imageio → OpenCV fallback chain
- **Frame count** — Image: —; Video: fixed 128 (pad with last frame if fewer, truncate if more)
- **Geometric aug** — Image: `RandomResizedCrop(224, (0.9,1.0), BICUBIC)` per image; Video: same, sampled **independently per frame**
- **Color aug** — Image: off; Video: off
- **Horizontal flip** — Image: off; Video: off
- **Tensor** — Image: `ToTensor` → `Normalize(OPENAI mean/std)`; Video: same per frame, then `stack→permute(C,T,H,W)`
- **Model forward aug** — Image: `segment_offset=randint(0,8)`; Video: + `uniform_sample_frames` 50%, `frame_concat` currently off
- **Supervision features** — Image: `image_features.npy`, `text_features.npy`; Video: `text_emb.npy`, `visidx.npy`

---

## 8. Key Source Files

- `src/open_clip/transform.py:324` — `image_transform` (shared transform builder for image/video frames)
- `src/open_clip/transform.py:62` — `AugmentationCfg` defaults
- `src/open_clip/constants.py:1-2` — `OPENAI_DATASET_MEAN/STD`
- `src/open_clip/factory.py:1007-1017` — `create_model_and_transforms` building `preprocess_train/val`
- `src/open_clip_train/main_covisco.py:241-268` — `video_preprocess_train` construction
- `src/open_clip_train/data_wds.py:215-243` — `apply_video_transform` (pad/truncate to 128 frames)
- `src/open_clip_train/data_wds.py:280-382` — `load_video_frames_from_bytes`
- `src/open_clip_train/data_wds.py:448, 563` — `get_image_wds_dataset` / `get_video_wds_dataset` pipeline
- `src/open_clip_train/train.py:721-734` — forward-stage video augmentation sampling
- `src/open_clip/covisco_vit.py:25` — `uniform_sample_frames`
- `src/open_clip/covisco_vit.py:41` — `uniform_sample_frames_and_concat`
- `src/open_clip_train/params.py:711-728` — frame-sampling CLI parameter definitions
- `scripts/run_covisco_train.sh` — the current actual training command
