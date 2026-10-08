"""
Main training script for CoVisco SigLIP model.

This is a variant of main.py tailored for CoVisco SigLIP training, which:
1. Creates CoViscoModel instead of standard CLIP models
2. Uses CoViscoSigLIPLoss with multi-pair SigLIP contrastive + reconstruction loss
3. Loads separate image/video embedding datasets (--image-data, --video-data)
4. Uses evaluate_covisco() which temporarily loads the teacher text encoder
   (timm/ViT-gopt-16-SigLIP2-384) for ImageNet zero-shot classification
5. Uses train_one_epoch_covisco_siglip() for the training loop
"""

import copy
import glob
import logging
import multiprocessing as mp
import os
import re
import subprocess
import sys
import random
from datetime import datetime
from functools import partial

import numpy as np
import torch
from torch import optim

try:
    import wandb
except ImportError:
    wandb = None

try:
    import torch.utils.tensorboard as tensorboard
except ImportError:
    tensorboard = None

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None

from open_clip import create_model_and_transforms, trace_model, get_tokenizer, get_input_dtype
from open_clip.transform import image_transform, PreprocessCfg, DynamicResolutionTransform
from open_clip.loss import CoViscoSigLIPLoss
from open_clip_train.data import get_data
from open_clip_train.distributed import is_master, init_distributed_device, broadcast_object
from open_clip_train.logger import setup_logging
from open_clip_train.params import parse_args
from open_clip_train.scheduler import cosine_lr, const_lr, const_lr_cooldown
from open_clip_train.train import train_one_epoch_covisco_siglip, evaluate_covisco, get_optimizer_param_names
from open_clip_train.file_utils import pt_load, check_exists, start_sync_process, remote_sync


LATEST_CHECKPOINT_NAME = "epoch_latest.pt"


def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)


def natural_key(string_):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_.lower())]


def get_latest_checkpoint(path: str, remote: bool):
    if remote:
        result = subprocess.run(["aws", "s3", "ls", path + "/"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(result)
        if result.returncode == 1:
            return None
        checkpoints = [os.path.join(path, x.split(' ')[-1]) for x in result.stdout.decode().split('\n')[:-1]]
    else:
        checkpoints = glob.glob(path + '**/*.pt', recursive=True)
    if checkpoints:
        checkpoints = sorted(checkpoints, key=natural_key)
        return checkpoints[-1]
    return None


def _key_variants(name):
    variants = [name]
    if name.startswith('module.'):
        variants.append(name[len('module.'):])
    else:
        variants.append('module.' + name)
    return variants


def _adapt_state_dict_keys(model, state_dict):
    model_keys = set(model.state_dict().keys())
    candidates = [('as-is', state_dict)]

    if state_dict:
        candidates.append((
            'strip module.',
            {k[len('module.'):] if k.startswith('module.') else k: v for k, v in state_dict.items()},
        ))
        candidates.append((
            'add module.',
            {k if k.startswith('module.') else 'module.' + k: v for k, v in state_dict.items()},
        ))

    best_name, best_state_dict = max(
        candidates,
        key=lambda item: len(model_keys.intersection(item[1].keys()))
    )
    matched = len(model_keys.intersection(best_state_dict.keys()))
    return best_state_dict, best_name, matched, len(model_keys)


def _load_model_state_dict_compatible(model, checkpoint, args):
    raw_state_dict = checkpoint['state_dict']
    state_dict, key_mode, matched_keys, total_model_keys = _adapt_state_dict_keys(model, raw_state_dict)
    model_state_dict = model.state_dict()

    filtered_state_dict = {}
    unexpected_keys = []
    shape_mismatch_keys = []
    for name, value in state_dict.items():
        target = model_state_dict.get(name)
        if target is None:
            unexpected_keys.append(name)
            continue
        if hasattr(value, 'shape') and hasattr(target, 'shape') and value.shape != target.shape:
            shape_mismatch_keys.append((name, tuple(value.shape), tuple(target.shape)))
            continue
        filtered_state_dict[name] = value

    incompatible = model.load_state_dict(filtered_state_dict, strict=False)

    if is_master(args):
        logging.info(
            "Loaded model state_dict with key mode '%s': matched %d/%d model keys, "
            "loaded %d tensors, missing %d, unexpected %d, shape-mismatched %d",
            key_mode,
            matched_keys,
            total_model_keys,
            len(filtered_state_dict),
            len(incompatible.missing_keys),
            len(unexpected_keys),
            len(shape_mismatch_keys),
        )
        if incompatible.missing_keys:
            logging.warning("Missing model keys while loading checkpoint: %s", incompatible.missing_keys[:20])
        if unexpected_keys:
            logging.warning("Unexpected checkpoint keys while loading model: %s", unexpected_keys[:20])
        if shape_mismatch_keys:
            logging.warning("Shape-mismatched checkpoint keys skipped: %s", shape_mismatch_keys[:20])

    return incompatible


def _optimizer_state_is_compatible(param, state):
    for value in state.values():
        if torch.is_tensor(value) and value.ndim > 0 and value.shape != param.shape:
            return False
    return True


def _optimizer_checkpoint_exactly_matches(optimizer, optimizer_checkpoint):
    checkpoint_groups = optimizer_checkpoint.get('param_groups', [])
    checkpoint_state = optimizer_checkpoint.get('state', {})
    if len(checkpoint_groups) != len(optimizer.param_groups):
        return False, f"param group count differs: checkpoint={len(checkpoint_groups)}, current={len(optimizer.param_groups)}"

    for group_idx, (current_group, checkpoint_group) in enumerate(zip(optimizer.param_groups, checkpoint_groups)):
        current_params = current_group.get('params', [])
        checkpoint_params = checkpoint_group.get('params', [])
        if len(current_params) != len(checkpoint_params):
            return False, (
                f"param count differs in group {group_idx}: "
                f"checkpoint={len(checkpoint_params)}, current={len(current_params)}"
            )
        for param_idx, (param, checkpoint_param_id) in enumerate(zip(current_params, checkpoint_params)):
            state = checkpoint_state.get(checkpoint_param_id, {})
            if not _optimizer_state_is_compatible(param, state):
                return False, f"optimizer state shape differs at group {group_idx}, param {param_idx}"

    return True, "exact match"


def _clone_optimizer_state_value(value, param):
    if not torch.is_tensor(value):
        return copy.deepcopy(value)

    cloned = value.detach().clone()
    if cloned.ndim == 0:
        return cloned
    if torch.is_floating_point(cloned):
        return cloned.to(device=param.device, dtype=param.dtype)
    return cloned.to(device=param.device)


def _checkpoint_param_name_to_id(optimizer_checkpoint, optimizer_param_names):
    if isinstance(optimizer_param_names, dict):
        return dict(optimizer_param_names)

    name_to_id = {}
    checkpoint_groups = optimizer_checkpoint.get('param_groups', [])
    for group_names, checkpoint_group in zip(optimizer_param_names or [], checkpoint_groups):
        for name, checkpoint_param_id in zip(group_names or [], checkpoint_group.get('params', [])):
            if name is not None:
                name_to_id[name] = checkpoint_param_id
    return name_to_id


def _restore_optimizer_state_by_name(optimizer, optimizer_checkpoint, optimizer_param_names, named_parameters):
    checkpoint_state = optimizer_checkpoint.get('state', {})
    checkpoint_name_to_id = _checkpoint_param_name_to_id(optimizer_checkpoint, optimizer_param_names)
    current_name_by_param_id = {id(param): name for name, param in named_parameters}

    restored = 0
    skipped_missing_name = 0
    skipped_missing_state = 0
    skipped_shape = 0

    for group in optimizer.param_groups:
        for param in group.get('params', []):
            current_name = current_name_by_param_id.get(id(param))
            if current_name is None:
                skipped_missing_name += 1
                continue

            checkpoint_param_id = None
            for candidate_name in _key_variants(current_name):
                if candidate_name in checkpoint_name_to_id:
                    checkpoint_param_id = checkpoint_name_to_id[candidate_name]
                    break
            if checkpoint_param_id is None:
                skipped_missing_name += 1
                continue

            state = checkpoint_state.get(checkpoint_param_id)
            if state is None:
                skipped_missing_state += 1
                continue
            if not _optimizer_state_is_compatible(param, state):
                skipped_shape += 1
                continue

            optimizer.state[param] = {
                key: _clone_optimizer_state_value(value, param)
                for key, value in state.items()
            }
            restored += 1

    return restored, skipped_missing_name, skipped_missing_state, skipped_shape


def _restore_optimizer_state_compatible(optimizer, checkpoint, named_parameters, args):
    if optimizer is None or 'optimizer' not in checkpoint:
        if is_master(args):
            logging.info("No optimizer state found in checkpoint; optimizer starts fresh.")
        return

    optimizer_checkpoint = checkpoint['optimizer']
    exact_match, mismatch_reason = _optimizer_checkpoint_exactly_matches(optimizer, optimizer_checkpoint)
    if exact_match:
        optimizer.load_state_dict(optimizer_checkpoint)
        if is_master(args):
            logging.info("Restored full optimizer state from checkpoint.")
        return

    optimizer_param_names = checkpoint.get('optimizer_param_names')
    if optimizer_param_names is None:
        if is_master(args):
            logging.warning(
                "Optimizer checkpoint is not exactly compatible (%s) and has no optimizer_param_names; "
                "optimizer state starts fresh.",
                mismatch_reason,
            )
        return

    restored, skipped_missing_name, skipped_missing_state, skipped_shape = _restore_optimizer_state_by_name(
        optimizer,
        optimizer_checkpoint,
        optimizer_param_names,
        named_parameters,
    )
    if is_master(args):
        logging.info(
            "Partially restored optimizer state by parameter name: restored=%d, "
            "missing_name=%d, missing_state=%d, shape_mismatch=%d",
            restored,
            skipped_missing_name,
            skipped_missing_state,
            skipped_shape,
        )


def main(args):
    args = parse_args(args)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    device = init_distributed_device(args)

    # Experiment name
    if args.name is None:
        model_name_safe = args.model.replace('/', '-')
        date_str = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        if args.distributed:
            date_str = broadcast_object(args, date_str)
        args.name = '-'.join([
            date_str,
            f"model_{model_name_safe}",
            f"lr_{args.lr}",
            f"b_{args.batch_size}",
            f"j_{args.workers}",
            f"p_{args.precision}",
        ])

    resume_latest = args.resume == 'latest'
    log_base_path = os.path.join(args.logs, args.name)
    args.log_path = None
    if is_master(args, local=args.log_local):
        os.makedirs(log_base_path, exist_ok=True)
        log_filename = f'out-{args.rank}' if args.log_local else 'out.log'
        args.log_path = os.path.join(log_base_path, log_filename)
        if os.path.exists(args.log_path) and not resume_latest:
            print(
                "Error. Experiment already exists. Use --name {} to specify a new experiment."
            )
            return -1

    # Setup text logger
    args.log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(args.log_path, args.log_level)

    # Setup wandb, tensorboard, checkpoint logging
    args.wandb = 'wandb' in args.report_to or 'all' in args.report_to
    args.tensorboard = 'tensorboard' in args.report_to or 'all' in args.report_to
    args.checkpoint_path = os.path.join(log_base_path, "checkpoints")
    if is_master(args):
        args.tensorboard_path = os.path.join(log_base_path, "tensorboard") if args.tensorboard else ''
        for dirname in [args.tensorboard_path, args.checkpoint_path]:
            if dirname:
                os.makedirs(dirname, exist_ok=True)
    else:
        args.tensorboard_path = ''

    if resume_latest:
        resume_from = None
        checkpoint_path = args.checkpoint_path
        if args.remote_sync is not None:
            checkpoint_path = os.path.join(args.remote_sync, args.name, "checkpoints")
            if args.save_most_recent:
                print('Error. Cannot use save-most-recent with remote_sync and resume latest.')
                return -1
            if args.remote_sync_protocol != 's3':
                print('Error. Sync protocol not supported when using resume latest.')
                return -1
        if is_master(args):
            if args.save_most_recent:
                resume_from = os.path.join(checkpoint_path, LATEST_CHECKPOINT_NAME)
                if not os.path.exists(resume_from):
                    resume_from = None
            else:
                resume_from = get_latest_checkpoint(checkpoint_path, remote=args.remote_sync is not None)
            if resume_from:
                logging.info(f'Found latest resume checkpoint at {resume_from}.')
            else:
                logging.info(f'No latest resume checkpoint found in {checkpoint_path}.')
        if args.distributed:
            resume_from = broadcast_object(args, resume_from)
        args.resume = resume_from

    if args.copy_codebase:
        copy_codebase(args)

    # Start remote sync process if needed
    remote_sync_process = None
    if is_master(args) and args.remote_sync is not None:
        result = remote_sync(
            os.path.join(args.logs, args.name),
            os.path.join(args.remote_sync, args.name),
            args.remote_sync_protocol
        )
        if result:
            logging.info('remote sync successful.')
        else:
            logging.info('Error: remote sync failed. Exiting.')
            return -1
        remote_sync_process = start_sync_process(
            args.remote_sync_frequency,
            os.path.join(args.logs, args.name),
            os.path.join(args.remote_sync, args.name),
            args.remote_sync_protocol
        )
        remote_sync_process.start()

    if args.precision == 'fp16':
        logging.warning(
            'It is recommended to use AMP mixed-precision instead of FP16. '
            'FP16 support needs further verification and tuning, especially for train.')

    if args.distributed:
        logging.info(
            f'Running in distributed mode with multiple processes. Device: {args.device}.'
            f'Process (global: {args.rank}, local {args.local_rank}), total {args.world_size}.')
    else:
        logging.info(f'Running with a single process. Device {args.device}.')

    if isinstance(args.force_image_size, (tuple, list)) and len(args.force_image_size) == 1:
        args.force_image_size = args.force_image_size[0]

    random_seed(args.seed, 0)

    # ---- Create CoVisco SigLIP Model ----
    # We use create_model_and_transforms to get the model + image transforms.
    # The factory detects CoVisco SigLIP via the model name (identifier containing 'CoVisco').
    model_kwargs = {}
    model_kwargs['args'] = args  # pass args so factory can read covisco-specific params

    model, preprocess_train, preprocess_val = create_model_and_transforms(
        args.model,
        args.pretrained,
        precision=args.precision,
        device=device,
        jit=args.torchscript,
        force_quick_gelu=args.force_quick_gelu,
        force_custom_text=args.force_custom_text,
        force_patch_dropout=args.force_patch_dropout,
        force_image_size=args.force_image_size,
        force_context_length=args.force_context_length,
        image_mean=args.image_mean,
        image_std=args.image_std,
        image_interpolation=args.image_interpolation,
        image_resize_mode=args.image_resize_mode,
        aug_cfg=args.aug_cfg,
        pretrained_image=args.pretrained_image,
        output_dict=True,
        cache_dir=args.cache_dir,
        **model_kwargs,
    )

    if args.grad_checkpointing:
        if hasattr(model, 'set_grad_checkpointing'):
            model.set_grad_checkpointing()
        elif hasattr(model, 'encoder') and hasattr(model.encoder, 'set_grad_checkpointing'):
            model.encoder.set_grad_checkpointing()

    # ---- Build a separate video transform if requested ----
    # When image and video use the same resolution and video augmentation is enabled,
    # video_preprocess_train stays None and EmbeddingDataset falls back to preprocess_train.
    video_preprocess_train = None
    raw_video_size = getattr(args, 'video_size', None)
    disable_video_aug = getattr(args, 'disable_video_aug', False)
    if raw_video_size is None and disable_video_aug:
        raw_video_size = args.force_image_size
        if raw_video_size is None:
            for t in getattr(preprocess_train, 'transforms', []):
                if hasattr(t, 'size'):
                    raw_video_size = t.size
                    break
    if raw_video_size is not None:
        if isinstance(raw_video_size, (tuple, list)) and len(raw_video_size) == 1:
            raw_video_size = raw_video_size[0]
        elif isinstance(raw_video_size, (tuple, list)) and len(raw_video_size) == 2:
            raw_video_size = tuple(raw_video_size)
        # Inherit preprocessing settings (mean/std/interpolation) from the image preprocess config.
        # We derive them by inspecting preprocess_train's Normalize transform.
        mean = None
        std = None
        for t in getattr(preprocess_train, 'transforms', []):
            from torchvision.transforms import Normalize as _Normalize
            if isinstance(t, _Normalize):
                mean = t.mean
                std = t.std
                break
        video_preprocess_train = image_transform(
            image_size=raw_video_size,
            is_train=not disable_video_aug,
            mean=mean,
            std=std,
            interpolation=args.image_interpolation,
            resize_mode=args.image_resize_mode,
            aug_cfg=None if disable_video_aug else args.aug_cfg,
        )
        if is_master(args):
            logging.info(
                f"Using separate video transform with size={raw_video_size}, "
                f"augmentation={'disabled' if disable_video_aug else 'enabled'}"
            )

    # ---- Dynamic Resolution Setup ----
    # When enabled, the DataLoader always produces images at the LARGEST candidate
    # resolution.  The training loop then dynamically resizes each batch on GPU
    # to a randomly chosen candidate resolution.  This avoids cross-process
    # synchronization issues with DataLoader worker prefetching.
    if getattr(args, 'dynamic_resolution', False):
        dynamic_resolutions = getattr(args, 'dynamic_resolutions', None)
        if not dynamic_resolutions or len(dynamic_resolutions) < 2:
            raise ValueError(
                "--dynamic-resolution requires --dynamic-resolutions "
                "with at least 2 candidate sizes."
            )
        for res in dynamic_resolutions:
            if res % 14 != 0:
                logging.warning(
                    f"Dynamic resolution {res} is not divisible by patch_size 14. "
                    f"This may cause errors if the model's actual patch_size is 14."
                )
        max_res = max(dynamic_resolutions)
        # Ensure force_image_size is at least the max candidate resolution
        if args.force_image_size is None or args.force_image_size < max_res:
            args.force_image_size = max_res
            if is_master(args):
                logging.info(
                    f"Dynamic resolution: set --force-image-size to {max_res} "
                    f"(largest candidate) for the DataLoader."
                )
        if is_master(args):
            logging.info(
                f"Dynamic resolution enabled with candidates: {dynamic_resolutions}. "
                f"DataLoader will use {max_res}, training loop resizes on GPU."
            )

    random_seed(args.seed, args.rank)

    if is_master(args):
        logging.info("Model:")
        logging.info(f"{str(model)}")
        logging.info("Params:")
        params_file = os.path.join(args.logs, args.name, "params.txt")
        with open(params_file, "w") as f:
            for name in sorted(vars(args)):
                val = getattr(args, name)
                logging.info(f"  {name}: {val}")
                f.write(f"{name}: {val}\n")

    if args.distributed:
        if args.use_bn_sync:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        ddp_args = {}
        # find_unused_parameters causes "marked ready twice" when a parameter is unused
        # in one modality's backward but used in the other's. We do manual all-reduce
        # in the training loop instead, so disable DDP's gradient sync entirely via
        # find_unused_parameters=False and gradient_as_bucket_view=True.
        ddp_args['find_unused_parameters'] = False
        ddp_args['bucket_cap_mb'] = 100  # Increase the broadcast bucket size to reduce communication rounds
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device], **ddp_args)

    # ---- Create Optimizer ----
    optimizer = None
    scaler = None

    has_train_data = (
        getattr(args, 'image_data', None) is not None or
        getattr(args, 'video_data', None) is not None or
        getattr(args, 'image_data_path', None) is not None or
        getattr(args, 'video_data_path', None) is not None or
        getattr(args, 'image_vidcap_data_path', None) is not None
    )
    if has_train_data:
        exclude = lambda n, p: p.ndim < 2 or "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n or 'logit_bias' in n
        include = lambda n, p: not exclude(n, p)

        named_parameters = list(model.named_parameters())
        gain_or_bias_params = [p for n, p in named_parameters if exclude(n, p) and p.requires_grad]
        rest_params = [p for n, p in named_parameters if include(n, p) and p.requires_grad]

        opt = getattr(args, 'opt', 'adamw').lower()
        if opt.startswith('timm/'):
            from timm.optim import create_optimizer_v2
            timm_opt = opt.split('timm/')[-1]
            opt_kwargs = {}
            if args.beta1 is not None:
                opt_kwargs['betas'] = (args.beta1, args.beta2)
            if args.momentum is not None:
                opt_kwargs['momentum'] = args.momentum
            optimizer = create_optimizer_v2(
                model,
                timm_opt,
                lr=args.lr,
                weight_decay=args.wd,
                eps=args.eps,
                **opt_kwargs,
            )
        elif opt == 'adamw':
            optimizer = optim.AdamW(
                [
                    {"params": gain_or_bias_params, "weight_decay": 0.},
                    {"params": rest_params, "weight_decay": args.wd},
                ],
                lr=args.lr,
                betas=(args.beta1, args.beta2),
                eps=args.eps,
            )
        else:
            assert False, f'Unknown optimizer {opt}'

        if is_master(args):
            defaults = copy.deepcopy(optimizer.defaults)
            defaults['weight_decay'] = args.wd
            defaults = ', '.join([f'{k}: {v}' for k, v in defaults.items()])
            logging.info(f'Created {type(optimizer).__name__} ({args.opt}) optimizer: {defaults}')

        scaler = None
        if args.precision == "amp":
            try:
                scaler = torch.amp.GradScaler(device=device)
            except (AttributeError, TypeError):
                scaler = torch.cuda.amp.GradScaler()

    # ---- Resume from checkpoint ----
    start_epoch = 0
    if args.resume is not None:
        checkpoint = pt_load(args.resume, map_location='cpu')
        if 'epoch' in checkpoint:
            start_epoch = checkpoint["epoch"]
            _load_model_state_dict_compatible(model, checkpoint, args)
            _restore_optimizer_state_compatible(optimizer, checkpoint, named_parameters if has_train_data else [], args)
            if scaler is not None and 'scaler' in checkpoint:
                scaler.load_state_dict(checkpoint['scaler'])
            logging.info(f"=> resuming checkpoint '{args.resume}' (epoch {start_epoch})")
        else:
            checkpoint = {'state_dict': checkpoint}
            _load_model_state_dict_compatible(model, checkpoint, args)
            logging.info(f"=> loaded checkpoint '{args.resume}' (epoch {start_epoch})")

    # ---- Initialize datasets ----
    # For CoVisco SigLIP, we don't need a tokenizer for training data (uses pre-extracted embeddings),
    # but we still load preprocess_fns for ImageNet validation.
    data = get_data(
        args,
        (preprocess_train, preprocess_val, video_preprocess_train),
        epoch=start_epoch,
        tokenizer=None,
    )
    assert len(data), 'At least one train or eval dataset must be specified.'

    # ---- Create scheduler ----
    scheduler = None
    # Determine the primary training dataloader for scheduler step count
    train_data_key = None
    for key in ('image_train', 'video_train', 'image_vidcap_train'):
        if key in data:
            train_data_key = key
            break

    if train_data_key is not None and optimizer is not None:
        # Per-source accumulation frequency (consistent with train_one_epoch_covisco_siglip)
        accum_freq_image = max(1, getattr(args, 'accum_freq_image', None) or args.accum_freq)
        accum_freq_video = max(1, getattr(args, 'accum_freq_video', None) or args.accum_freq)
        accum_freq_image_vidcap = max(1, getattr(args, 'accum_freq_image_vidcap', None) or args.accum_freq)

        image_num_batches = data['image_train'].dataloader.num_batches if 'image_train' in data else 0
        video_num_batches = data['video_train'].dataloader.num_batches if 'video_train' in data else 0
        image_vidcap_num_batches = data['image_vidcap_train'].dataloader.num_batches if 'image_vidcap_train' in data else 0
        # Number of accumulation cycles per epoch for each source.
        # An optim step covers one cycle of image AND video AND image-vidcap.
        image_cycles = (image_num_batches // accum_freq_image) if ('image_train' in data and image_num_batches > 0) else 0
        video_cycles = (video_num_batches // accum_freq_video) if ('video_train' in data and video_num_batches > 0) else 0
        image_vidcap_cycles = (image_vidcap_num_batches // accum_freq_image_vidcap) if ('image_vidcap_train' in data and image_vidcap_num_batches > 0) else 0
        num_optim_steps_per_epoch = max(image_cycles, video_cycles, image_vidcap_cycles)
        total_steps = num_optim_steps_per_epoch * args.epochs

        if args.lr_scheduler == "cosine":
            scheduler = cosine_lr(optimizer, args.lr, args.warmup, total_steps)
        elif args.lr_scheduler == "const":
            scheduler = const_lr(optimizer, args.lr, args.warmup, total_steps)
        elif args.lr_scheduler == "const-cooldown":
            assert args.epochs_cooldown is not None, \
                "Please specify the number of cooldown epochs for this lr schedule."
            cooldown_steps = num_optim_steps_per_epoch * args.epochs_cooldown
            scheduler = const_lr_cooldown(
                optimizer, args.lr, args.warmup, total_steps,
                cooldown_steps, args.lr_cooldown_power, args.lr_cooldown_end)
        else:
            logging.error(
                f'Unknown scheduler, {args.lr_scheduler}. Available options are: cosine, const, const-cooldown.')
            exit(1)

    # ---- Logging setup ----
    args.save_logs = args.logs and args.logs.lower() != 'none' and is_master(args)
    writer = None
    if args.save_logs and args.tensorboard:
        assert tensorboard is not None, "Please install tensorboard."
        writer = tensorboard.SummaryWriter(args.tensorboard_path)

    if args.wandb and is_master(args):
        assert wandb is not None, 'Please install wandb.'
        logging.debug('Starting wandb.')
        wandb.init(
            project=args.wandb_project_name,
            name=args.name,
            id=args.name,
            notes=args.wandb_notes,
            tags=[],
            resume='auto' if args.resume == "latest" else None,
            config=vars(args),
        )
        if args.debug:
            wandb.watch(model, log='all')
        wandb.save(params_file)
        logging.debug('Finished loading wandb.')

    # Save original model ref for checkpointing (torch.compile adds prefix)
    original_model = model
    if args.torchcompile:
        logging.info('Compiling model...')
        if args.grad_checkpointing and args.distributed:
            logging.info('Disabling DDP dynamo optimizer when grad checkpointing enabled.')
            torch._dynamo.config.optimize_ddp = False
        model = torch.compile(original_model)

    # ---- Eval-only mode ----
    if train_data_key is None:
        evaluate_covisco(model, data, start_epoch, args, tb_writer=writer)
        return

    # ---- Create CoVisco SigLIP Loss ----
    loss = CoViscoSigLIPLoss(
        reconstruction_weight=getattr(args, 'reconstruction_weight', 0.1),
        image_caption_weight=getattr(args, 'image_caption_weight', 1.0),
        image_image_weight=getattr(args, 'image_image_weight', 1.0),
        video_caption_weight=getattr(args, 'video_caption_weight', 1.0),
        image_video_caption_weight=getattr(args, 'image_video_caption_weight', 1.0),
        rank=args.rank,
        world_size=args.world_size,
        dist_impl=getattr(args, 'loss_dist_impl', 'reduce'),
    )

    # ---- Training loop ----
    for epoch in range(start_epoch, args.epochs):
        if is_master(args):
            logging.info(f'Start epoch {epoch}')

        train_one_epoch_covisco_siglip(
            model, data, loss, epoch, optimizer, scaler, scheduler, args, tb_writer=writer,
            original_model=original_model, checkpoint_path=args.checkpoint_path,
        )
        completed_epoch = epoch + 1

        # Saving checkpoints (save BEFORE evaluation to prevent losing checkpoint if eval fails)
        if args.save_logs:
            checkpoint_dict = {
                "epoch": completed_epoch,
                "name": args.name,
                "state_dict": original_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "optimizer_param_names": get_optimizer_param_names(optimizer, original_model),
            }
            if scaler is not None:
                checkpoint_dict["scaler"] = scaler.state_dict()

            if completed_epoch == args.epochs or (
                args.save_frequency > 0 and (completed_epoch % args.save_frequency) == 0
            ):
                torch.save(
                    checkpoint_dict,
                    os.path.join(args.checkpoint_path, f"epoch_{completed_epoch}.pt"),
                )
            if args.delete_previous_checkpoint:
                previous_checkpoint = os.path.join(args.checkpoint_path, f"epoch_{completed_epoch - 1}.pt")
                if os.path.exists(previous_checkpoint):
                    os.remove(previous_checkpoint)

            if args.save_most_recent:
                tmp_save_path = os.path.join(args.checkpoint_path, "tmp.pt")
                latest_save_path = os.path.join(args.checkpoint_path, LATEST_CHECKPOINT_NAME)
                torch.save(checkpoint_dict, tmp_save_path)
                os.replace(tmp_save_path, latest_save_path)

        # Evaluation (after checkpoint is saved)
        if any(v in data for v in ('imagenet-val', 'imagenet-v2')):
            evaluate_covisco(model, data, completed_epoch, args, tb_writer=writer)
            # Explicitly reclaim CUDA memory after evaluation before next epoch training
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

    if args.wandb and is_master(args):
        wandb.finish()

    # Final remote sync
    if remote_sync_process is not None:
        logging.info('Final remote sync.')
        remote_sync_process.terminate()
        result = remote_sync(
            os.path.join(args.logs, args.name),
            os.path.join(args.remote_sync, args.name),
            args.remote_sync_protocol
        )
        if result:
            logging.info('Final remote sync successful.')
        else:
            logging.info('Final remote sync failed.')


def copy_codebase(args):
    from shutil import copytree, ignore_patterns
    new_code_path = os.path.join(args.logs, args.name, "code")
    if os.path.exists(new_code_path):
        print(
            f"Error. Experiment already exists at {new_code_path}. Use --name to specify a new experiment."
        )
        return -1
    print(f"Copying codebase to {new_code_path}")
    current_code_path = os.path.realpath(__file__)
    for _ in range(3):
        current_code_path = os.path.dirname(current_code_path)
    copytree(current_code_path, new_code_path, ignore=ignore_patterns('log', 'logs', 'wandb'))
    print("Done copying code.")
    return 1


if __name__ == "__main__":
    main(sys.argv[1:])
