"""DDP training entry point.

Standard LLaVA-OneVision-2 style arguments:
  --do-valid / --val-data-path / --eval-interval / --eval-iters / --eval-strategy

Usage:
  python -m train.train --config configs/covisco_qwen3_4b.yaml --do-valid \
      --val-data-path /path/to/val_wds --eval-interval 500 --eval-iters 20

Validation loss is disabled by default (matching llava-onevision2's mid_training.sh / sft.sh).

To use the HF Trainer instead: replace train_loop() with Trainer.train().
"""
from __future__ import annotations

import argparse
import os
import time
from contextlib import nullcontext as _nullcontext
from datetime import timedelta
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from data.collator import CollatorConfig, CoViscoCollator
from data.dynamic_strategy import DynamicTokenConfig
from data.llava_batch_adapter import LlavaOneVisionBatchAdapter, is_llava_onevision_batch
from data.plugin import CoViscoPlugin
from data.wds_dataset import DEFAULT_VIDEO_TARGET_FRAMES, CoViscoWDSDataset
from models.config import ModelConfig, build_model_config
from models.llava_covisco import LlavaCoViscoModel


# Under GPU memory pressure, failures are not always raised as torch.cuda.OutOfMemoryError:
# when cuBLAS/cuDNN cannot allocate workspace or submit a kernel they raise plain RuntimeError
# (e.g. ViT attention's QK^T reporting CUBLAS_STATUS_EXECUTION_FAILED). If uncaught, such an
# exception kills this rank while the remaining ranks block in all_reduce(skip_flag) until the
# 30-minute NCCL timeout, dragging the whole job down.
_OOM_LIKE_MARKERS = (
    "out of memory",
    "cublas_status_execution_failed",
    "cublas_status_alloc_failed",
    "cublas_status_not_initialized",
    "cudnn_status_alloc_failed",
    "cudnn_status_execution_failed",
    "cuda error: out of memory",
)

# The RuntimeErrors above may also be delayed reports of an already-broken CUDA context
# (sticky error), in which case every subsequent kernel fails. After this many consecutive
# skips the exception is re-raised so the job fails explicitly instead of spinning and
# flooding skip logs. True OOM does not count (it is recoverable transient noise).
_MAX_CONSECUTIVE_CUDA_ERROR_SKIPS = 10


def _is_oom_like(exc: BaseException) -> bool:
    """Return whether the exception is GPU-memory/BLAS noise that can be downgraded to a skipped micro-batch."""
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    if isinstance(exc, RuntimeError):
        msg = str(exc).lower()
        return any(m in msg for m in _OOM_LIKE_MARKERS)
    return False


def _release_cuda_cache() -> None:
    """empty_cache may itself raise once the CUDA context is broken; guard it separately."""
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass


def load_config(path: str) -> ModelConfig:
    with open(path) as f:
        d = yaml.safe_load(f)
    return build_model_config(d)


def setup_distributed():
    if "RANK" in os.environ:
        # Enlarge the timeout to 30min: while rank0 saves a checkpoint (bf16 weights + AdamW
        # state written to shared storage), other ranks block on subsequent collective calls,
        # and the default 10min watchdog can report a false timeout.
        dist.init_process_group(backend="nccl", timeout=timedelta(minutes=30))
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", rank % 8))
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank
    return 0, 1, 0


def _unwrap(model):
    return model.module if isinstance(model, DDP) else model


@torch.no_grad()
def run_validation(
    model,
    loader,
    device,
    eval_iters: int,
    strategy: str = "query_and_vit",
    vit_ratio: float = 1.0,
    batch_adapter: Optional[LlavaOneVisionBatchAdapter] = None,
) -> float:
    """Run eval_iters batches of validation loss and return per-token CE.

    The metric matches training (global-batch token-mean): accumulate the CE sum and the
    valid token count, then divide and reduce across ranks. Otherwise val loss drifts with
    the answer-length distribution inside a batch and cannot be compared across steps.
    """
    model.eval()
    ce_sum = 0.0
    tok_sum = 0
    for i, batch in enumerate(loader):
        if i >= eval_iters:
            break
        if batch_adapter is not None and is_llava_onevision_batch(batch):
            batch = batch_adapter.adapt(batch)
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        # Keep the token strategy fixed during validation (query_and_vit by default) so loss is comparable
        modality = batch.get("modality", "text")
        if modality in ("image", "video"):
            plan = {"strategy": strategy, "vit_ratio": vit_ratio,
                    "arrangement": "interleave", "num_segments": 1}
            # A video may have multiple segments, already computed by the collator
            token_plan = batch.get("token_plan", plan)
        else:
            token_plan = batch.get("token_plan", None)
        outputs = model(**{k: v for k, v in batch.items() if k != "token_plan"}, token_plan=token_plan)
        labels = batch.get("labels")
        n_tok = 0 if labels is None else int((labels[:, 1:] != -100).sum().item())
        # In eval mode sparsity_loss is always None (see the self.training check in the model
        # forward), so outputs.loss is pure CE.
        if outputs.loss is not None and n_tok > 0 and torch.isfinite(outputs.loss.detach()):
            ce_sum += outputs.loss.item() * n_tok
            tok_sum += n_tok
    model.train()
    if dist.is_available() and dist.is_initialized():
        buf = torch.tensor([ce_sum, float(tok_sum)], dtype=torch.float64, device=device)
        dist.all_reduce(buf, op=dist.ReduceOp.SUM)
        ce_sum, tok_sum = buf[0].item(), buf[1].item()
    return ce_sum / max(tok_sum, 1.0)


def main():
    parser = argparse.ArgumentParser(description="DDP trainer for LLaVA-style training")
    parser.add_argument("--config", type=str, required=True, help="path to model config yaml")
    parser.add_argument("--data-path", type=str, required=True, help="LLaVA-OneVision-2 data path or fallback WDS shards path")
    parser.add_argument("--output-dir", type=str, default="./output")
    parser.add_argument("--stage", type=str, default="sft", choices=["alignment", "mid_training", "sft"])
    parser.add_argument("--trainable-modules", type=str, nargs="+", default=None)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--num-train-steps", type=int, default=0,
                        help="total micro-batch steps; mutually exclusive with --num-epochs")
    parser.add_argument("--num-epochs", type=int, default=0,
                        help="training epochs; requires --num-samples-per-epoch when set")
    parser.add_argument("--num-samples-per-epoch", type=int, default=0,
                        help="dataset samples per epoch (used with --num-epochs to derive total steps)")
    parser.add_argument("--warmup-steps", type=int, default=20,
                        help="LR warmup steps; use --warmup-ratio to specify as a fraction instead")
    parser.add_argument("--warmup-ratio", type=float, default=0.0,
                        help="LR warmup fraction of total steps (overrides --warmup-steps when > 0)")
    parser.add_argument("--lr-scheduler", type=str, default="cosine",
                        choices=["cosine", "linear", "constant"],
                        help="LR decay schedule after warmup")
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--micro-batch", type=int, default=1,
                        help="image micro-batch size per step")
    parser.add_argument("--video-data-path", type=str, default="",
                        help="path to video WDS shards (dir or shard list file); leave empty to disable video")
    parser.add_argument("--video-micro-batch", type=int, default=1,
                        help="video micro-batch size per step (independent of --micro-batch)")
    parser.add_argument("--video-micro-batch-every", type=int, default=1,
                        help="insert one video micro-batch every N image micro-batches (default 1 = "
                             "one video micro-batch after every image micro-batch). Raise it to keep "
                             "the image/video sample ratio when lowering --micro-batch.")
    parser.add_argument("--video-target-frames", type=int, default=DEFAULT_VIDEO_TARGET_FRAMES,
                        help="frames decoded per mp4 sample (uniform sampling over the whole clip; "
                             f"default {DEFAULT_VIDEO_TARGET_FRAMES}). Only affects mp4 shards; "
                             "jpg-frame datasets keep the frames stored on disk.")
    parser.add_argument("--uniform-sample-n", type=int, default=32,
                        help="frames to keep after uniform subsampling in uniform-mode video path (default 32)")
    parser.add_argument("--uniform-segment-t-size", type=int, default=8,
                        help="segment_t_size for uniform video path, must divide --uniform-sample-n (default 8)")
    parser.add_argument("--uniform-train-prob", type=float, default=0.5,
                        help="probability of using uniform frame sampling per video batch (0.0 = always visidx, 1.0 = always uniform)")
    parser.add_argument("--frame-cat-prob", type=float, default=0.0,
                        help="probability of applying frame concat augmentation in uniform video path "
                             "(only effective when uniform_train_prob > 0; 0.0 = disabled)")
    parser.add_argument("--image-size-image", type=int, default=0,
                        help="ViT input resolution for images (0 = use model config image_size)")
    parser.add_argument("--image-size-video", type=int, default=0,
                        help="ViT input resolution for video frames (0 = use model config image_size)")
    parser.add_argument("--candidate-resolutions", type=int, nargs="+", default=None,
                        help="candidate image resolutions for dynamic resolution training "
                             "(e.g. 224 336 448). Each batch randomly picks one. Only for image modality.")
    parser.add_argument("--native-resolution", action="store_true",
                        help="image native resolution mode: resize each image to the nearest "
                             "patch_size-divisible size keeping aspect ratio. Images only (videos "
                             "keep --image-size-video). Only active when the image batch has exactly "
                             "1 sample, i.e. --micro-batch 1; otherwise falls back to "
                             "--candidate-resolutions / fixed size.")
    parser.add_argument("--native-min-patches", type=int, default=256,
                        help="lower bound on patch count in native resolution mode (256 = 224x224 @ patch14)")
    parser.add_argument("--native-max-patches", type=int, default=1296,
                        help="upper bound on patch count in native resolution mode (1296 = 504x504 @ patch14)")
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--save-interval", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for reproducibility")
    parser.add_argument("--num-workers", type=int, default=8,
                        help="DataLoader num_workers for image/video loaders")
    parser.add_argument("--shard-interleave", type=int, default=4,
                        help="Number of shards each DataLoader worker opens and polls concurrently. "
                             "Shards are packed in blocks by data source, so sequential reading makes "
                             "the task mix of the global batch switch as a whole every "
                             "(shard_size*num_workers/samples per step) steps, producing stair-step loss. "
                             "1 = legacy sequential reading; larger values add concurrent sequential read "
                             "streams, which may affect readahead throughput on network filesystems")
    parser.add_argument("--shuffle-buffer", type=int, default=500,
                        help="Sample-level shuffle buffer capacity (0/1 = disabled), same semantics as "
                             "webdataset's .shuffle(bufsize). The buffer holds raw samples, so memory is "
                             "roughly capacity * raw bytes per sample * num_workers; each worker must fill "
                             "the buffer before producing its first sample at startup")
    parser.add_argument("--max-grad-norm", type=float, default=1.0,
                        help="gradient clipping max norm (0 to disable)")
    parser.add_argument("--ckpt", type=str, default="", help="path to load checkpoint")
    parser.add_argument("--skip-shards", type=str, default="",
                        help="text file of image shard names to skip on resume (one basename/path per line). "
                             "empty = process all shards")
    parser.add_argument("--skip-video-shards", type=str, default="",
                        help="text file of video shard names to skip on resume. empty = process all video shards")
    parser.add_argument("--skip-optimizer-state", action="store_true",
                        help="skip loading optimizer state from checkpoint (useful when trainable-modules change)")
    parser.add_argument("--reset-step", type=int, default=-1,
                        help="override the step counter loaded from checkpoint (e.g. 0 to restart step/LR schedule "
                             "from scratch while keeping weights); -1 = keep the checkpoint's step")
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--data-backend", type=str, default="fallback_wds",
                        choices=["fallback_wds", "llava_onevision"],
                        help="fallback_wds is the local minimal reader; llava_onevision expects an external LLaVA dataloader batch")

    # === Validation (LLaVA-OneVision-2 style standard arguments) ===
    parser.add_argument("--do-valid", action="store_true",
                        help="enable validation loss eval (default off, like llava-onevision2)")
    parser.add_argument("--val-data-path", type=str, default="",
                        help="validation WDS shards path")
    parser.add_argument("--eval-interval", type=int, default=0,
                        help="run validation every N steps (0 = disabled)")
    parser.add_argument("--eval-iters", type=int, default=20,
                        help="number of validation batches per eval")
    parser.add_argument("--eval-strategy", type=str, default="query_and_vit",
                        choices=["query_only", "vit_only", "query_and_vit"],
                        help="token strategy for validation (default query_and_vit)")
    parser.add_argument("--eval-vit-ratio", type=float, default=1.0,
                        help="vit token ratio (0~1) for query_and_vit validation; applied to actual patch count")
    parser.add_argument("--save-best-on-val", action="store_true",
                        help="save best.pt when val loss improves")
    parser.add_argument("--sparsity-lambda", type=float, default=None,
                        help="override sparsity_lambda in config (token selector regularization; 0=off)")

    args = parser.parse_args()

    # --- Batch composition: image / video samples actually consumed per optimizer step ---
    # The epoch -> step conversion below and the [batch] log share this metric; it must
    # account for grad_accum and the micro-batch slots occupied by videos.
    _ws = int(os.environ.get("WORLD_SIZE", 1))
    _v_every = max(1, args.video_micro_batch_every)
    if args.video_data_path:
        # Out of every (_v_every + 1) micro-batch slots, _v_every are images and 1 is a video
        _img_mb = args.grad_accum * _v_every / (_v_every + 1)
        _vid_mb = args.grad_accum / (_v_every + 1)
    else:
        _img_mb, _vid_mb = float(args.grad_accum), 0.0
    _img_per_step = _img_mb * args.micro_batch * _ws
    _vid_per_step = _vid_mb * args.video_micro_batch * _ws

    # Convert the epoch configuration into a step count (based on actual throughput, see the metric above)
    _explicit_steps, _steps_per_epoch = args.num_train_steps, 0
    if args.num_epochs > 0:
        if args.num_samples_per_epoch <= 0:
            raise ValueError("--num-epochs requires --num-samples-per-epoch")
        if _img_per_step <= 0:
            raise ValueError("--num-epochs requires a positive image throughput per step "
                             "(check --grad-accum / --micro-batch)")
        _steps_per_epoch = int(args.num_samples_per_epoch / _img_per_step)
        args.num_train_steps = _steps_per_epoch * args.num_epochs
    elif args.num_train_steps <= 0:
        raise ValueError("Specify either --num-train-steps or --num-epochs + --num-samples-per-epoch")

    # warmup ratio takes precedence over warmup steps
    if args.warmup_ratio > 0:
        args.warmup_steps = int(args.num_train_steps * args.warmup_ratio)

    rank, world_size, local_rank = setup_distributed()
    is_main = rank == 0
    device = torch.device(f"cuda:{local_rank}")

    # --- Actual batch composition / epoch conversion (metric defined after argparse) ---
    if is_main:
        print(f"[batch] per optimizer step: {_img_per_step:.0f} images"
              f"{f' + {_vid_per_step:.0f} videos' if _vid_per_step else ''} "
              f"(grad_accum={args.grad_accum}, micro_batch={args.micro_batch}, "
              f"video_micro_batch={args.video_micro_batch}, video_every={_v_every}, "
              f"world_size={_ws})", flush=True)
        if world_size != _ws:
            print(f"[batch] WARNING: WORLD_SIZE env var ({_ws}) differs from the actual process-group "
                  f"size ({world_size}); throughput and the epoch->step conversion above use {_ws} "
                  f"and may deviate from reality.", flush=True)
        if args.num_epochs > 0:
            print(f"[batch] epoch->step: {_steps_per_epoch} step/epoch x {args.num_epochs} epoch "
                  f"= {args.num_train_steps} step (warmup={args.warmup_steps})", flush=True)
            if _explicit_steps > 0 and _explicit_steps != args.num_train_steps:
                print(f"[batch] NOTE: explicitly passed --num-train-steps {_explicit_steps} was overridden "
                      f"by --num-epochs to {args.num_train_steps}; remove --num-epochs to train by explicit steps.",
                      flush=True)
        print(f"[data] shard_interleave={args.shard_interleave}, "
              f"shuffle_buffer={args.shuffle_buffer}, num_workers={args.num_workers}; "
              f"concurrent shards ~= {args.shard_interleave * args.num_workers * _ws}", flush=True)
        print("[loss] normalization = global-batch token-mean "
              "(sum(CE) / global valid label tokens, aligned with per-token loss)",
              flush=True)

    # TensorBoard writer (rank 0 only)
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "tb_logs")) if is_main else None

    # 1) Load config + model
    model_cfg = load_config(args.config)
    if args.sparsity_lambda is not None:
        model_cfg.sparsity_lambda = args.sparsity_lambda
    model = LlavaCoViscoModel(model_cfg)
    model.freeze_for_stage(args.stage, args.trainable_modules)
    if is_main:
        model.print_trainable()
    if args.bf16:
        model = model.to(torch.bfloat16)
    model = model.to(device)
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    # 2) Plugin / Collator / Train Dataset
    plugin = CoViscoPlugin(
        sms=model_cfg.projector.sms,
        num_query_per_seg=model_cfg.vit.num_query_per_seg,
    )
    # Under the vit_only strategy, the number of vit tokens per segment is decided by
    # vit_token_counts_video[0] in the yaml. This is also the number of tokens actually sent to
    # the ViT on the visidx path, used by the collator to compute the exact pad count.
    _vit_counts_video = getattr(model_cfg.token_strategy, "vit_token_counts_video", None)
    token_strategy_vit_tokens_video = int(_vit_counts_video[0]) if _vit_counts_video else 0
    collator_cfg = CollatorConfig(
        image_size=model_cfg.vit.image_size,
        image_size_image=args.image_size_image or model_cfg.vit.image_size_image,
        image_size_video=args.image_size_video or model_cfg.vit.image_size_video,
        patch_size=model_cfg.vit.patch_size,
        segment_t_size=model_cfg.vit.segment_t_size,
        num_query_per_seg=model_cfg.vit.num_query_per_seg,
        uniform_sample_n=args.uniform_sample_n,
        uniform_segment_t_size=args.uniform_segment_t_size,
        uniform_train_prob=args.uniform_train_prob,
        frame_cat_prob=args.frame_cat_prob,
        vit_tokens_per_seg_video=token_strategy_vit_tokens_video,
        candidate_resolutions=args.candidate_resolutions or [],
        native_resolution=args.native_resolution,
        native_min_patches=args.native_min_patches,
        native_max_patches=args.native_max_patches,
    )
    if args.native_resolution and is_main:
        if args.micro_batch != 1:
            print(f"[native-resolution] WARNING: --micro-batch={args.micro_batch} != 1, "
                  "native resolution will NOT activate (images of different sizes in a batch "
                  "cannot be stacked); please set --micro-batch 1.", flush=True)
        else:
            print(f"[native-resolution] enabled for image modality, "
                  f"patch budget [{args.native_min_patches}, {args.native_max_patches}], "
                  f"patch_size={model_cfg.vit.patch_size}; video keeps fixed resolution.", flush=True)

    # The two video branches must produce the same segment count; otherwise the collator
    # computes pads with the visidx segment count while the encoder outputs the uniform
    # segment count, and the <|image_pad|> mismatch discards the whole video batch.
    if args.video_data_path and is_main:
        visidx_segs = max(args.video_target_frames // model_cfg.vit.segment_t_size, 1)
        uniform_segs = max(args.uniform_sample_n // args.uniform_segment_t_size, 1)
        if visidx_segs != uniform_segs:
            print(f"[video-frames] WARNING: visidx path {args.video_target_frames}/"
                  f"{model_cfg.vit.segment_t_size}={visidx_segs} segments, uniform path "
                  f"{args.uniform_sample_n}/{args.uniform_segment_t_size}={uniform_segs} segments; "
                  "a mismatch makes the collator discard the video batch; "
                  "please tune --video-target-frames / --uniform-sample-n / --uniform-segment-t-size.",
                  flush=True)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_cfg.llm.path, trust_remote_code=True)
    dynamic_cfg = DynamicTokenConfig(
        enabled=model_cfg.token_strategy.enabled,
        p_query_only=model_cfg.token_strategy.p_query_only,
        p_vit_only=model_cfg.token_strategy.p_vit_only,
        p_query_and_vit=model_cfg.token_strategy.p_query_and_vit,
        probs_image=model_cfg.token_strategy.p_image,
        probs_video=model_cfg.token_strategy.p_video,
        vit_token_ratios=model_cfg.token_strategy.vit_token_ratios,
        vit_token_ratios_image=model_cfg.token_strategy.vit_token_ratios_image,
        vit_token_ratios_video=model_cfg.token_strategy.vit_token_ratios_video,
        arrangement=model_cfg.token_strategy.arrangement,
        vit_token_counts_image=getattr(model_cfg.token_strategy, "vit_token_counts_image", None),
        vit_token_counts_video=getattr(model_cfg.token_strategy, "vit_token_counts_video", None),
    )
    collator = CoViscoCollator(
        tokenizer=tokenizer,
        plugin=plugin,
        image_pad_token_id=model_cfg.llm.image_pad_token_id,
        vision_start_token_id=model_cfg.llm.vision_start_token_id,
        vision_end_token_id=model_cfg.llm.vision_end_token_id,
        config=collator_cfg,
        dynamic_token_config=dynamic_cfg,  # Batch-level plan sampling keeps the whole batch consistent
    )
    batch_adapter = LlavaOneVisionBatchAdapter(
        dynamic_token_config=dynamic_cfg,
        num_query_per_seg=model_cfg.vit.num_query_per_seg,
        image_token_id=model_cfg.llm.image_pad_token_id,
    )
    if args.data_backend == "llava_onevision":
        raise NotImplementedError(
            "Please plug in LLaVA-OneVision-2's get_train_loader/Qwen2VLTaskEncoder directly "
            "and keep LlavaOneVisionBatchAdapter.adapt(batch) as the sole adaptation boundary "
            "in the training loop."
        )
    train_dataset = CoViscoWDSDataset(
        shards_path=args.data_path,
        dynamic_token_config=dynamic_cfg,
        plugin=plugin,
        seed=args.seed,
        skip_shards=args.skip_shards or None,
        video_target_frames=args.video_target_frames,
        segment_t_size=model_cfg.vit.segment_t_size,
        shard_interleave=args.shard_interleave,
        shuffle_buffer=args.shuffle_buffer,
    )
    if is_main and train_dataset.skip_shards:
        n_hit = sum(
            1 for s in train_dataset.shards
            if os.path.basename(s) in train_dataset.skip_shards
        )
        print(f"[WDS] skip {n_hit}/{len(train_dataset.shards)} image shards "
              f"from {args.skip_shards}", flush=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.micro_batch,
        collate_fn=collator,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        pin_memory=True,
        # Dynamic resolution requires collator workers on all ranks to pick the same
        # resolution to avoid NCCL deadlock; use a Generator with a fixed seed to
        # synchronize random state.
        generator=torch.Generator().manual_seed(args.seed),
    )

    # 2.3) Video DataLoader (optional, independent batch size)
    video_loader = None
    if args.video_data_path:
        video_dataset = CoViscoWDSDataset(
            shards_path=args.video_data_path,
            dynamic_token_config=dynamic_cfg,
            plugin=plugin,
            seed=args.seed,
            skip_shards=args.skip_video_shards or None,
            video_target_frames=args.video_target_frames,
            segment_t_size=model_cfg.vit.segment_t_size,
            shard_interleave=args.shard_interleave,
            shuffle_buffer=args.shuffle_buffer,
        )

        video_loader = DataLoader(
            video_dataset,
            batch_size=args.video_micro_batch,
            collate_fn=collator,
            num_workers=args.num_workers,
            persistent_workers=args.num_workers > 0,
            pin_memory=True,
            generator=torch.Generator().manual_seed(args.seed),
        )
        if is_main:
            print(f"[video] loader enabled, video_micro_batch={args.video_micro_batch}, "
                  f"shards={args.video_data_path}", flush=True)
            if video_dataset.skip_shards:
                n_hit = sum(
                    1 for s in video_dataset.shards
                    if os.path.basename(s) in video_dataset.skip_shards
                )
                print(f"[WDS] skip {n_hit}/{len(video_dataset.shards)} video shards "
                      f"from {args.skip_video_shards}", flush=True)

        # Startup self-check: every rank first pulls one video batch so ranks that fail are listed.
        # Faults such as inconsistent mounts or unreadable shards silently degrade training to
        # images-only (the training loop keeps "skipping video for all ranks"). The specific rank
        # must be identified at startup rather than by burning GPU-hours on log inspection.
        probe_ok = 1
        try:
            next(iter(video_loader))
        except StopIteration:
            probe_ok = 0
        if world_size > 1:
            flags = torch.zeros(world_size, dtype=torch.int32, device=device)
            flags[rank] = probe_ok
            dist.all_reduce(flags, op=dist.ReduceOp.SUM)
            bad = [i for i, v in enumerate(flags.tolist()) if v == 0]
            if is_main and bad:
                shown = bad[:16]
                print(f"[video] ERROR self-check failed: {len(bad)}/{world_size} ranks cannot fetch "
                      f"a video batch, rank={shown}{' ...' if len(bad) > len(shown) else ''}; "
                      f"their video shards are unreadable and video training will be skipped throughout",
                      flush=True)
            elif is_main:
                print(f"[video] self-check passed: {world_size}/{world_size} ranks can fetch a video batch",
                      flush=True)
        elif probe_ok == 0:
            print("[video] ERROR self-check failed: cannot fetch a video batch", flush=True)

    # 2.5) Val dataset / loader (optional)
    val_loader = None
    if args.do_valid and args.val_data_path:
        val_dynamic_cfg = DynamicTokenConfig(
            enabled=False,  # Strategy is fixed during validation
            p_query_only=0.0,
            p_vit_only=0.0,
            p_query_and_vit=1.0,
            vit_token_ratios=(args.eval_vit_ratio,),
            arrangement=model_cfg.token_strategy.arrangement,
        )
        val_batch_adapter = LlavaOneVisionBatchAdapter(
            dynamic_token_config=val_dynamic_cfg,
            num_query_per_seg=model_cfg.vit.num_query_per_seg,
            image_token_id=model_cfg.llm.image_pad_token_id,
        )
        val_dataset = CoViscoWDSDataset(
            shards_path=args.val_data_path,
            dynamic_token_config=val_dynamic_cfg,
            plugin=plugin,
            seed=args.seed,
            video_target_frames=args.video_target_frames,
            segment_t_size=model_cfg.vit.segment_t_size,
            # No shuffling or interleaving for the val set: each eval reads only the first
            # eval_iters batches, which must be the same samples for val loss to be comparable
            # across steps. shuffle=True would give a different shard order each epoch, so
            # evals would actually measure different subsets.
            shuffle=False,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.micro_batch,
            collate_fn=collator,
            num_workers=2,
            persistent_workers=True,
            pin_memory=True,
            generator=torch.Generator().manual_seed(args.seed),
        )
        if is_main:
            print(f"[val] enabled, {args.eval_iters} iters every {args.eval_interval} steps")

    # 3) Optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.0,
    )

    # 4) Train loop
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    step = 0
    lr_scale = 1.0
    # Loss normalization = global-batch token-mean (see the docstring of _do_optimizer_step).
    # window_tokens: valid label tokens accumulated on this rank within the current grad_accum
    # window; summed across ranks at the end of the window as the gradient denominator.
    window_tokens = 0
    # Log accumulators: record the CE sum and token count, then divide at logging time to get
    # the true per-token CE. Sparsity is a per-batch regularizer, averaged separately over
    # micro-batches rather than mixed into the CE scalar.
    log_ce_sum = 0.0
    log_tok = 0
    log_ce_sum_image = 0.0
    log_tok_image = 0
    log_ce_sum_video = 0.0
    log_tok_video = 0
    log_sparsity_sum = 0.0
    log_sparsity_count = 0
    best_val_loss = float("inf")
    val_batch_adapter = None     # Assigned only when do_valid; pre-initialize to avoid NameError
    optimizer.zero_grad()

    # Resume from checkpoint
    if args.ckpt:
        ckpt = torch.load(args.ckpt, map_location="cpu")
        result = _unwrap(model).load_state_dict(ckpt["model_state"], strict=False)
        if is_main:
            if result.missing_keys:
                print(f"[ckpt] missing keys ({len(result.missing_keys)}): {result.missing_keys[:5]}{'...' if len(result.missing_keys) > 5 else ''}", flush=True)
            if result.unexpected_keys:
                print(f"[ckpt] unexpected keys ({len(result.unexpected_keys)}): {result.unexpected_keys[:5]}{'...' if len(result.unexpected_keys) > 5 else ''}", flush=True)
            if not result.missing_keys and not result.unexpected_keys:
                print("[ckpt] all keys matched", flush=True)
        if "optimizer_state" in ckpt and not args.skip_optimizer_state:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        if "lr_scale" in ckpt:
            lr_scale = ckpt["lr_scale"]
            for pg in optimizer.param_groups:
                pg["lr"] = args.lr * lr_scale
        step = ckpt.get("step", 0)
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        if is_main:
            print(f"resumed from {args.ckpt} at step {step}", flush=True)
        if args.reset_step >= 0:
            if is_main:
                print(f"[ckpt] step reset: {step} -> {args.reset_step} "
                      f"(LR schedule/warmup and save naming follow the new counter)", flush=True)
            step = args.reset_step
        del ckpt
        import gc; gc.collect()
        torch.cuda.empty_cache()

    t0 = time.time()
    # micro_step must be initialized after checkpoint restore, otherwise it always restarts from 0.
    # micro_step: counts all micro-batches (image + video) and marks grad_accum boundaries.
    # step: optimizer step count (one per grad_accum micro-batches); the externally visible "training step".
    micro_step = step * args.grad_accum
    image_mb_count = 0
    consecutive_cuda_error_skips = 0
    _score = _score_max = _score_min = None

    model.train()

    # Run a baseline validation at startup (step 0)
    if val_loader is not None:
        val_loss = run_validation(
            _unwrap(model), val_loader, device,
            args.eval_iters, args.eval_strategy, args.eval_vit_ratio,
            batch_adapter=val_batch_adapter,
        )
        if is_main:
            print(f"step {step:6d} | val_loss {val_loss:.4f} (baseline)", flush=True)
            if writer:
                writer.add_scalar("val/loss", val_loss, step)
        best_val_loss = val_loss

    import math as _math

    def _sync_grads():
        """Average the gradients accumulated over this grad_accum window across ranks once (equivalent to what DDP does in backward).

        Why it is moved here: DDP's autograd hook issues allreduce per bucket during backward.
        Once a rank OOMs mid-backward it exits after sending only some buckets, while non-OOM
        ranks finish sending the rest — the process group's NCCL call sequences are permanently
        misaligned from then on: different ranks stick on the same SeqNum, one waiting for a
        gradient bucket and one for a scalar allreduce, until the watchdog times out and brings
        the whole job down. Therefore every micro-batch runs under no_sync, no collective is
        in flight during backward, and an OOM degrades into a purely local event.

        The allreduce is done per-tensor in place (not flattened into one large buffer) to avoid
        extra GPU memory — this is exactly when memory is tightest; issue all asynchronously first,
        then wait together so NCCL pipelines by itself.
        """
        # The number of collectives must be rank-independent. `p.grad is not None` is rank-local
        # state: when a whole window is discarded, zero_grad(set_to_none=True) resets grads to
        # None, and a parameter may also be untouched on this rank's data within a window. Once
        # ranks have different list lengths, the NCCL call sequences are permanently misaligned —
        # all ranks stick on the same SeqNum waiting for an allreduce that can never complete,
        # until the watchdog timeout kills the job. So missing gradients are zero-filled in place
        # and the full trainable_params list is sent: better to transmit one extra zero tensor
        # than let the call sequence depend on rank-local state.
        for p in trainable_params:
            if p.grad is None:
                p.grad = torch.zeros_like(p)
        handles = [
            dist.all_reduce(p.grad, op=dist.ReduceOp.AVG, async_op=True)
            for p in trainable_params
        ]
        for h in handles:
            h.wait()

    def _do_optimizer_step(n_tokens: int) -> bool:
        """Normalize gradients by the global token count, clip, update LR, and perform one optimizer step.

        Normalization = global-batch token-mean, aligned with per-token loss
        and the HF Trainer's `num_items_in_batch`:

            L = sum_{all micro-batches and all ranks in the global batch} CE_t / T
            T = total valid label tokens across ranks in this window

        Each micro-batch's backward uses the CE sum (out.loss * n_tok, no averaging;
        see _forward_micro_batch), so multiplying by 1/T at the end of the window yields
        the formula above. Every label token then carries the same weight regardless of
        its sample's answer length; the old `out.loss / grad_accum` weighted samples
        equally, giving each token of a one-word answer one to two orders of magnitude
        more weight than a long answer.

        T is obtained via a cross-rank all_reduce, so every rank agrees on "skip this
        step or not" and the NCCL call sequence stays aligned. Returns whether an
        optimizer step actually executed.
        """
        nonlocal lr_scale
        tok = torch.tensor([float(n_tokens)], dtype=torch.float64, device=device)
        if world_size > 1:
            dist.all_reduce(tok, op=dist.ReduceOp.SUM)
        total_tokens = tok.item()
        if total_tokens <= 0:
            # Every micro-batch in the window was skipped (OOM / all-ignored labels); no usable gradient.
            optimizer.zero_grad()
            return False
        # Must sync before clipping: the grad norm must be computed on the globally averaged gradients
        if world_size > 1:
            _sync_grads()
        # _sync_grads uses ReduceOp.AVG; multiply back by world_size to restore SUM,
        # then divide by the global token count to get the global-batch token-mean gradient.
        grad_scale = world_size / total_tokens
        for p in trainable_params:
            if p.grad is not None:
                p.grad.mul_(grad_scale)
        if args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=args.max_grad_norm)
        if step <= args.warmup_steps:
            lr_scale = step / max(1, args.warmup_steps)
        else:
            progress = (step - args.warmup_steps) / max(1, args.num_train_steps - args.warmup_steps)
            if args.lr_scheduler == "cosine":
                lr_scale = 0.5 * (1 + _math.cos(_math.pi * progress))
            elif args.lr_scheduler == "linear":
                lr_scale = 1.0 - progress
            else:  # constant
                lr_scale = 1.0
        for pg in optimizer.param_groups:
            pg["lr"] = args.lr * lr_scale
        optimizer.step()
        optimizer.zero_grad()
        return True

    def _forward_micro_batch(b, label: str) -> bool:
        """Run one micro-batch forward+backward; returns whether it was skipped (True = skipped).

        Gradient sync: the whole forward+backward runs under no_sync, so no cross-rank
        communication happens during backward and gradients only accumulate locally; the
        cross-rank average is deferred to the grad_accum boundary and done once by
        _sync_grads() (see its docstring). The key benefit is that no collective is in
        flight during backward, so a rank that OOMs in backward leaves no half-sent
        buckets behind, and the all_reduce(MAX) below can actually pull states back in sync.

        Skip logic (OOM / cuBLAS-cuDNN failure / non-finite loss): forward first, then one
        all_reduce(MAX) lets all ranks agree on "skip or not" before backward runs. The NCCL
        call sequences of all ranks therefore stay strictly aligned at all times, avoiding
        the SeqNum-misalignment deadlock where one rank does a gradient allreduce and the
        others do not.
        """
        nonlocal window_tokens
        nonlocal log_ce_sum, log_tok
        nonlocal log_ce_sum_image, log_tok_image
        nonlocal log_ce_sum_video, log_tok_video
        nonlocal log_sparsity_sum, log_sparsity_count
        nonlocal consecutive_cuda_error_skips
        # skip_flag: 0 = normal, 1 = OOM/cuBLAS-like, 2 = non-finite loss (all-(-100) labels make HF's CE return NaN)
        skip_flag = torch.zeros(1, dtype=torch.int32, device=device)
        loss = None
        out = None
        # Valid label token count, matching HF's shift convention: after labels shift right by one,
        # position 0 is not part of the prediction (ForCausalLMLoss right-pads labels with ignore
        # and takes [..., 1:]).
        _labels = b.get("labels")
        n_tok = 0 if _labels is None else int((_labels[:, 1:] != -100).sum().item())
        is_ddp = isinstance(model, torch.nn.parallel.DistributedDataParallel)
        ctx = model.no_sync() if is_ddp else _nullcontext()

        def _handle_oom_like(exc: BaseException, where: str) -> None:
            """Record a memory/BLAS transient as a skip; raise when consecutive CUDA RuntimeErrors exceed the cap."""
            nonlocal consecutive_cuda_error_skips
            if not _is_oom_like(exc):
                raise
            if isinstance(exc, torch.cuda.OutOfMemoryError):
                consecutive_cuda_error_skips = 0
            else:
                consecutive_cuda_error_skips += 1
                n = consecutive_cuda_error_skips
                cap = _MAX_CONSECUTIVE_CUDA_ERROR_SKIPS
                if n > cap:
                    raise
                if is_main:
                    print(
                        f"[OOM-{label}] step {step}, {where} cuda/cublas error "
                        f"({n}/{cap}): {exc}",
                        flush=True,
                    )
            skip_flag.fill_(1)
            _release_cuda_cache()

        with ctx:
            try:
                out = model(**b)
                if out.loss is None or not torch.isfinite(out.loss.detach()):
                    skip_flag.fill_(2)
                elif n_tok <= 0:
                    # labels are all ignore_index (e.g. the sample was truncated for being too long): no learnable tokens
                    skip_flag.fill_(2)
                else:
                    # Scale both CE and sparsity back by n_tok to restore the regularization
                    # magnitude matching the global token normalization.
                    sparsity_tensor = getattr(
                        _unwrap(model), "_last_sparsity_loss_tensor", None
                    )
                    if sparsity_tensor is None:
                        loss = out.loss * n_tok
                    else:
                        ce_loss = out.loss - sparsity_tensor
                        loss = ce_loss * n_tok + sparsity_tensor * n_tok
            except Exception as e:
                _handle_oom_like(e, "forward")
            # No rank has run backward yet, so the sequences are strictly aligned
            if world_size > 1:
                dist.all_reduce(skip_flag, op=dist.ReduceOp.MAX)
            reason = int(skip_flag.item())
            if reason > 0:
                if reason == 1:
                    # OOM / cuBLAS-like: gradients accumulated in this window may be incomplete,
                    # so discard the whole window (preserving existing behavior)
                    optimizer.zero_grad()
                    window_tokens = 0
                    if is_main:
                        print(f"[OOM-{label}] step {step}, skip micro-batch", flush=True)
                elif is_main:
                    # Non-finite loss: CE yields all-zero gradients for all-ignore samples and
                    # does not pollute already-accumulated gradients, so only this micro-batch
                    # is skipped and the window is not cleared.
                    print(f"[skip-{label}] step {step}, non-finite loss (labels may be all "
                          f"ignore_index, sample exceeds max_seq_length)", flush=True)
                return True
            try:
                loss.backward()
            except Exception as e:
                _handle_oom_like(e, "backward")
            if world_size > 1:
                dist.all_reduce(skip_flag, op=dist.ReduceOp.MAX)
            if skip_flag.item() > 0:
                optimizer.zero_grad()
                window_tokens = 0
                if is_main:
                    print(f"[OOM-{label}] step {step}, skip micro-batch (backward OOM)", flush=True)
                return True
        consecutive_cuda_error_skips = 0
        window_tokens += n_tok
        # out.loss = CE (per-token mean) + sparsity_loss; log them separately, CE weighted by tokens.
        _sp = getattr(_unwrap(model), "_last_sparsity_loss", None)
        _ce = out.loss.item() - (_sp or 0.0)
        log_ce_sum += _ce * n_tok
        log_tok += n_tok
        if _sp is not None:
            log_sparsity_sum += _sp
            log_sparsity_count += 1
        if label == "image":
            log_ce_sum_image += _ce * n_tok
            log_tok_image += n_tok
        elif label == "video":
            log_ce_sum_video += _ce * n_tok
            log_tok_video += n_tok
        return False

    while step < args.num_train_steps:
        video_iter = iter(video_loader) if video_loader is not None else None
        for batch in train_loader:
            if step >= args.num_train_steps:
                break

            if batch_adapter is not None and is_llava_onevision_batch(batch):
                batch = batch_adapter.adapt(batch)
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}

            # --- Image micro-batch ---
            oom = _forward_micro_batch(batch, "image")
            micro_step += 1
            image_mb_count += 1
            if not oom:
                _score = getattr(_unwrap(model), "_last_selector_score_mean", None)
                _score_max = getattr(_unwrap(model), "_last_selector_score_max", None)
                _score_min = getattr(_unwrap(model), "_last_selector_score_min", None)

            # --- Video micro-batch (inside the same grad_accum window, does not advance step) ---
            # One video micro-batch is inserted every video_micro_batch_every image micro-batches,
            # keeping the image/video sample ratio stable when --micro-batch changes.
            if (
                video_iter is not None
                and step < args.num_train_steps
                and image_mb_count % args.video_micro_batch_every == 0
            ):
                try:
                    video_batch = next(video_iter)
                except StopIteration:
                    video_iter = iter(video_loader)
                    try:
                        video_batch = next(video_iter)
                    except StopIteration:
                        video_batch = None
                # "Whether the video stream yielded a batch" must be consistent across ranks:
                # micro_step is the sole basis for grad_accum window boundaries. If one rank
                # skips the video micro-batch while others do not, the window boundaries drift
                # apart — one side enters _do_optimizer_step() and issues gradient allreduces
                # while the other is still running micro-batches, deadlocking on NCCL SeqNum
                # mismatch. Take a SUM of how many ranks got a batch; if fewer than
                # world_size, all ranks skip, and the count of empty ranks is printed
                # (it should be 0 in normal operation).
                if world_size > 1:
                    n_video = torch.tensor(
                        [0 if video_batch is None else 1], dtype=torch.int32, device=device)
                    dist.all_reduce(n_video, op=dist.ReduceOp.SUM)
                    n_empty = world_size - int(n_video.item())
                    if n_empty > 0:
                        if is_main:
                            print(f"[video] step {step}, {n_empty}/{world_size} ranks have an empty "
                                  f"video stream; all ranks skip the video micro-batch this round", flush=True)
                        video_batch = None
                if video_batch is not None:
                    if batch_adapter is not None and is_llava_onevision_batch(video_batch):
                        video_batch = batch_adapter.adapt(video_batch)
                    video_batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in video_batch.items()}
                    oom_v = _forward_micro_batch(video_batch, "video")
                    micro_step += 1
                    if not oom_v:
                        # The video micro-batch refreshes selector stats (overwrites image, keeps the latest)
                        _score = getattr(_unwrap(model), "_last_selector_score_mean", None)
                        _score_max = getattr(_unwrap(model), "_last_selector_score_max", None)
                        _score_min = getattr(_unwrap(model), "_last_selector_score_min", None)

            # --- grad_accum boundary: one optimizer step per grad_accum micro-batches ---
            if micro_step % args.grad_accum == 0:
                step += 1
                _do_optimizer_step(window_tokens)
                window_tokens = 0

                if step % args.log_interval == 0:
                    # Loss stats must be reduced across ranks: a single rank only covers
                    # 1/world_size of the samples. all_reduce must run on all ranks and cannot
                    # be placed inside the is_main branch, or it deadlocks.
                    stats = torch.tensor(
                        [log_ce_sum, float(log_tok),
                         log_ce_sum_image, float(log_tok_image),
                         log_ce_sum_video, float(log_tok_video),
                         log_sparsity_sum, float(log_sparsity_count)],
                        dtype=torch.float64, device=device,
                    )
                    if world_size > 1:
                        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
                    if is_main:
                        dt = time.time() - t0
                        # per-token CE: same metric as the training objective, comparable across steps / experiments
                        avg = stats[0].item() / max(stats[1].item(), 1.0)
                        avg_img = stats[2].item() / max(stats[3].item(), 1.0)
                        avg_vid = stats[4].item() / max(stats[5].item(), 1.0)
                        avg_sp = stats[6].item() / max(stats[7].item(), 1.0)
                        cur_lr = args.lr * lr_scale
                        score_str = ""
                        if _score is not None:
                            score_str = (f" | score mean/max/min "
                                         f"{_score.item():.4f}/{_score_max.item():.4f}/{_score_min.item():.4f}")
                        # Print image / video losses separately (hidden when the video loader is off)
                        if video_loader is not None:
                            loss_str = (f"loss {avg:.4f} "
                                        f"(img {avg_img:.4f} / vid {avg_vid:.4f})")
                        else:
                            loss_str = f"loss {avg:.4f}"
                        print(f"step {step:6d} | {loss_str} | lr {cur_lr:.2e}{score_str} | {dt:.1f}s", flush=True)
                        if writer:
                            writer.add_scalar("train/loss", avg, step)
                            writer.add_scalar("train/loss_image", avg_img, step)
                            if video_loader is not None:
                                writer.add_scalar("train/loss_video", avg_vid, step)
                            writer.add_scalar("train/lr", cur_lr, step)
                            if stats[7].item() > 0:
                                writer.add_scalar("train/sparsity_loss", avg_sp, step)
                            if _score is not None:
                                writer.add_scalar("train/selector_score_mean", _score.item(), step)
                                writer.add_scalar("train/selector_score_max", _score_max.item(), step)
                                writer.add_scalar("train/selector_score_min", _score_min.item(), step)
                        t0 = time.time()
                    log_ce_sum = 0.0
                    log_tok = 0
                    log_ce_sum_image = 0.0
                    log_tok_image = 0
                    log_ce_sum_video = 0.0
                    log_tok_video = 0
                    log_sparsity_sum = 0.0
                    log_sparsity_count = 0

                # Validation (every eval_interval steps)
                if (
                    val_loader is not None
                    and args.eval_interval > 0
                    and step % args.eval_interval == 0
                ):
                    val_loss = run_validation(
                        _unwrap(model), val_loader, device,
                        args.eval_iters, args.eval_strategy, args.eval_vit_ratio,
                        batch_adapter=val_batch_adapter,
                    )
                    if is_main:
                        print(f"step {step:6d} | val_loss {val_loss:.4f} | best {best_val_loss:.4f}", flush=True)
                        if writer:
                            writer.add_scalar("val/loss", val_loss, step)
                        if args.save_best_on_val and val_loss < best_val_loss:
                            best_val_loss = val_loss
                            best_path = os.path.join(args.output_dir, "best.pt")
                            torch.save({
                                "step": step,
                                "model_state": _unwrap(model).state_dict(),
                                "optimizer_state": optimizer.state_dict(),
                                "lr_scale": lr_scale,
                                "best_val_loss": best_val_loss,
                                "args": vars(args),
                                "val_loss": val_loss,
                            }, best_path)
                            print(f"  saved best to {best_path}", flush=True)
                    # Other ranks must not run ahead while rank0 writes to disk: otherwise they
                    # would enter the next micro-batch's allreduce and keep waiting on rank0,
                    # with the wait counting against the NCCL watchdog. The barrier makes the
                    # wait happen here, with clear semantics.
                    if world_size > 1:
                        dist.barrier()

                # Save points: save_interval cycles + epoch boundaries + end of training.
                # The last two need separate checks: num_train_steps / _steps_per_epoch are
                # derived from epochs (see the conversion after argparse) and are generally not
                # multiples of save_interval, so relying on modulo alone would drop up to
                # save_interval-1 steps of training at an epoch end. The conditions depend only
                # on step / static config, so all ranks decide identically and barriers stay aligned.
                save_reason = ""
                if _steps_per_epoch > 0 and step % _steps_per_epoch == 0:
                    save_reason = f"epoch {step // _steps_per_epoch} end"
                elif step >= args.num_train_steps:
                    save_reason = "training end"
                elif step % args.save_interval == 0:
                    save_reason = "interval"
                if save_reason:
                    if is_main:
                        ckpt_path = os.path.join(args.output_dir, f"step_{step}.pt")
                        torch.save({
                            "step": step,
                            "model_state": _unwrap(model).state_dict(),
                            "optimizer_state": optimizer.state_dict(),
                            "lr_scale": lr_scale,
                            "best_val_loss": best_val_loss,
                            "args": vars(args),
                        }, ckpt_path)
                        print(f"saved checkpoint to {ckpt_path} ({save_reason})", flush=True)
                    if world_size > 1:
                        dist.barrier()

            if step >= args.num_train_steps:
                break

    if is_main:
        print("training done.")
        if writer:
            writer.close()


if __name__ == "__main__":
    main()
