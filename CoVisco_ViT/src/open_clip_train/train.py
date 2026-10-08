import itertools
import json
import logging
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.parallel.distributed import DistributedDataParallel
from tqdm import tqdm

try:
    import wandb
except ImportError:
    wandb = None

from open_clip import get_input_dtype, CLIP, CustomTextCLIP
from open_clip_train.distributed import is_master
from open_clip_train.zero_shot import zero_shot_eval
from open_clip_train.precision import get_autocast
from random import randint, random, choice

import torchvision.transforms.functional as TF


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def postprocess_clip_output(model_out):
    return {
        "image_features": model_out[0],
        "text_features": model_out[1],
        "logit_scale": model_out[2]
    }


def unwrap_model(model):
    if hasattr(model, 'module'):
        return model.module
    else:
        return model


def get_optimizer_param_names(optimizer, model):
    if optimizer is None or model is None:
        return None
    param_to_name = {id(p): n for n, p in model.named_parameters()}
    return [
        [param_to_name.get(id(p)) for p in group.get('params', [])]
        for group in optimizer.param_groups
    ]


def backward(total_loss, scaler):
    if scaler is not None:
        scaler.scale(total_loss).backward()
    else:
        total_loss.backward()


def train_one_epoch(model, data, loss, epoch, optimizer, scaler, scheduler, dist_model, args, tb_writer=None):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision, device_type=device.type)
    input_dtype = get_input_dtype(args.precision)

    model.train()
    if args.distill:
        dist_model.eval()

    data['train'].set_epoch(epoch)  # set epoch in process safe manner via sampler or shared_epoch
    dataloader = data['train'].dataloader
    num_batches_per_epoch = dataloader.num_batches // args.accum_freq
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    if args.accum_freq > 1:
        accum_images, accum_texts, accum_features = [], [], {}

    losses_m = {}
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()
    for i, batch in enumerate(dataloader):
        i_accum = i // args.accum_freq
        step = num_batches_per_epoch * epoch + i_accum

        if not args.skip_scheduler:
            scheduler(step)

        images, texts = batch
        images = images.to(device=device, dtype=input_dtype, non_blocking=True)
        texts = texts.to(device=device, non_blocking=True)

        data_time_m.update(time.time() - end)
        optimizer.zero_grad()

        if args.accum_freq == 1:
            with autocast():
                model_out = model(images, texts)
                logit_scale = model_out["logit_scale"]
                if args.distill:
                    with torch.no_grad():
                        dist_model_out = dist_model(images, texts)
                    model_out.update({f'dist_{k}': v for k, v in dist_model_out.items()})
                losses = loss(**model_out, output_dict=True)

                total_loss = sum(losses.values())
                losses["loss"] = total_loss

            backward(total_loss, scaler)
        else:
            # First, cache the features without any gradient tracking.
            with torch.no_grad():
                with autocast():
                    model_out = model(images, texts)

                    for f in ("logit_scale", "logit_bias"):
                        model_out.pop(f, None)

                    for key, val in model_out.items():
                        if key in accum_features:
                            accum_features[key].append(val)
                        else:
                            accum_features[key] = [val]

                accum_images.append(images)
                accum_texts.append(texts)

            # If (i + 1) % accum_freq is not zero, move on to the next batch.
            if ((i + 1) % args.accum_freq) > 0:
                # FIXME this makes data time logging unreliable when accumulating
                continue

            # Now, ready to take gradients for the last accum_freq batches.
            # Re-do the forward pass for those batches, and use the cached features from the other batches as negatives.
            # Call backwards each time, but only step optimizer at the end.
            optimizer.zero_grad()
            for j in range(args.accum_freq):
                images = accum_images[j]
                texts = accum_texts[j]
                with autocast():
                    model_out = model(images, texts)

                    inputs_no_accum = {}
                    inputs_no_accum["logit_scale"] = logit_scale = model_out.pop("logit_scale")
                    if "logit_bias" in model_out:
                        inputs_no_accum["logit_bias"] = model_out.pop("logit_bias")

                    inputs = {}
                    for key, val in accum_features.items():
                        accumulated = accum_features[key]
                        inputs[key] = torch.cat(accumulated[:j] + [model_out[key]] + accumulated[j + 1:])

                    losses = loss(**inputs, **inputs_no_accum, output_dict=True)
                    del inputs
                    del inputs_no_accum
                    total_loss = sum(losses.values())
                    losses["loss"] = total_loss

                backward(total_loss, scaler)

        if scaler is not None:
            if args.horovod:
                optimizer.synchronize()
                scaler.unscale_(optimizer)
                if args.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                with optimizer.skip_synchronize():
                    scaler.step(optimizer)
            else:
                if args.grad_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                scaler.step(optimizer)
            scaler.update()
        else:
            if args.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
            optimizer.step()

        # reset gradient accum, if enabled
        if args.accum_freq > 1:
            accum_images, accum_texts, accum_features = [], [], {}

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
            unwrap_model(model).logit_scale.clamp_(0, math.log(100))

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i_accum + 1
        if is_master(args) and (i_accum % args.log_every_n_steps == 0 or batch_count == num_batches_per_epoch):
            batch_size = len(images)
            num_samples = batch_count * batch_size * args.accum_freq * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # NOTE loss is coarsely sampled, just master node and per log update
            for key, val in losses.items():
                if key not in losses_m:
                    losses_m[key] = AverageMeter()
                losses_m[key].update(val.item(), batch_size)

            logit_scale_scalar = logit_scale.item()
            loss_log = " ".join(
                [
                    f"{loss_name.capitalize()}: {loss_m.val:#.5g} ({loss_m.avg:#.5g})" 
                    for loss_name, loss_m in losses_m.items()
                ]
            )
            samples_per_second = args.accum_freq * args.batch_size * args.world_size / batch_time_m.val
            samples_per_second_per_gpu = args.accum_freq * args.batch_size / batch_time_m.val
            logging.info(
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f}, {samples_per_second:#g}/s, {samples_per_second_per_gpu:#g}/s/gpu "
                f"LR: {optimizer.param_groups[0]['lr']:5f} "
                f"Logit Scale: {logit_scale_scalar:.3f} " + loss_log
            )

            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            log_data = {
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_second": samples_per_second,
                "samples_per_second_per_gpu": samples_per_second_per_gpu,
                "scale": logit_scale_scalar,
                "lr": optimizer.param_groups[0]["lr"]
            }            
            log_data.update({name:val.val for name,val in losses_m.items()})

            log_data = {"train/" + name: val for name, val in log_data.items()}

            if tb_writer is not None:
                for name, val in log_data.items():
                    tb_writer.add_scalar(name, val, step)
            
            if args.wandb:
                assert wandb is not None, 'Please install wandb.'
                log_data['step'] = step  # for backwards compatibility
                wandb.log(log_data, step=step)
            
            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()
    # end for


def evaluate(model, data, epoch, args, tb_writer=None, tokenizer=None):
    metrics = {}
    if not is_master(args):
        return metrics
    device = torch.device(args.device)
    model.eval()

    zero_shot_metrics = zero_shot_eval(model, data, epoch, args, tokenizer=tokenizer)
    metrics.update(zero_shot_metrics)

    autocast = get_autocast(args.precision, device_type=device.type)
    input_dtype = get_input_dtype(args.precision)

    if 'val' in data and (args.val_frequency and ((epoch % args.val_frequency) == 0 or epoch == args.epochs)):
        dataloader = data['val'].dataloader
        num_samples = 0
        samples_per_val = dataloader.num_samples

        # FIXME this does not scale past small eval datasets
        # all_image_features @ all_text_features will blow up memory and compute very quickly
        cumulative_loss = 0.0
        cumulative_gen_loss = 0.0
        all_image_features, all_text_features = [], []
        with torch.inference_mode():
            for i, batch in enumerate(dataloader):
                images, texts = batch
                images = images.to(device=device, dtype=input_dtype, non_blocking=True)
                texts = texts.to(device=device, non_blocking=True)

                with autocast():
                    model_out = model(images, texts)
                    image_features = model_out["image_features"]
                    text_features = model_out["text_features"]
                    logit_scale = model_out["logit_scale"]
                    # features are accumulated in CPU tensors, otherwise GPU memory exhausted quickly
                    # however, system RAM is easily exceeded and compute time becomes problematic
                    all_image_features.append(image_features.cpu())
                    all_text_features.append(text_features.cpu())
                    logit_scale = logit_scale.mean()
                    logits_per_image = logit_scale * image_features @ text_features.t()
                    logits_per_text = logits_per_image.t()

                    batch_size = images.shape[0]
                    labels = torch.arange(batch_size, device=device).long()
                    total_loss = (
                        F.cross_entropy(logits_per_image, labels) +
                        F.cross_entropy(logits_per_text, labels)
                    ) / 2

                    gen_loss = maybe_compute_generative_loss(model_out)

                cumulative_loss += total_loss * batch_size
                num_samples += batch_size
                if is_master(args) and (i % 100) == 0:
                    logging.info(
                        f"Eval Epoch: {epoch} [{num_samples} / {samples_per_val}]\t"
                        f"Clip Loss: {cumulative_loss / num_samples:.6f}\t")

                    if gen_loss is not None:
                        cumulative_gen_loss += gen_loss * batch_size
                        logging.info(
                            f"Generative Loss: {cumulative_gen_loss / num_samples:.6f}\t")

            val_metrics = get_clip_metrics(
                image_features=torch.cat(all_image_features),
                text_features=torch.cat(all_text_features),
                logit_scale=logit_scale.cpu(),
            )
            loss = cumulative_loss / num_samples
            metrics.update(
                {**val_metrics, "clip_val_loss": loss.item(), "epoch": epoch, "num_samples": num_samples}
            )
            if gen_loss is not None:
                gen_loss = cumulative_gen_loss / num_samples
                metrics.update({"val_generative_loss": gen_loss.item()})

    if not metrics:
        return metrics

    logging.info(
        f"Eval Epoch: {epoch} "
        + "\t".join([f"{k}: {round(v, 4):.4f}" for k, v in metrics.items()])
    )

    log_data = {"val/" + name: val for name, val in metrics.items()}

    if args.save_logs:
        if tb_writer is not None:
            for name, val in log_data.items():
                tb_writer.add_scalar(name, val, epoch)

        with open(os.path.join(args.checkpoint_path, "results.jsonl"), "a+") as f:
            f.write(json.dumps(metrics))
            f.write("\n")

    if args.wandb:
        assert wandb is not None, 'Please install wandb.'
        if 'train' in data:
            dataloader = data['train'].dataloader
            num_batches_per_epoch = dataloader.num_batches // args.accum_freq
            step = num_batches_per_epoch * epoch
        else:
            step = None
        log_data['epoch'] = epoch
        wandb.log(log_data, step=step)

    return metrics


def get_clip_metrics(image_features, text_features, logit_scale):
    metrics = {}
    logits_per_image = (logit_scale * image_features @ text_features.t()).detach().cpu()
    logits_per_text = logits_per_image.t().detach().cpu()

    logits = {"image_to_text": logits_per_image, "text_to_image": logits_per_text}
    ground_truth = torch.arange(len(text_features)).view(-1, 1)

    for name, logit in logits.items():
        ranking = torch.argsort(logit, descending=True)
        preds = torch.where(ranking == ground_truth)[1]
        preds = preds.detach().cpu().numpy()
        metrics[f"{name}_mean_rank"] = preds.mean() + 1
        metrics[f"{name}_median_rank"] = np.floor(np.median(preds)) + 1
        for k in [1, 5, 10]:
            metrics[f"{name}_R@{k}"] = np.mean(preds < k)

    return metrics


def maybe_compute_generative_loss(model_out):
    if "logits" in model_out and "labels" in model_out:
        token_logits = model_out["logits"]
        token_labels = model_out["labels"]
        return F.cross_entropy(token_logits.permute(0, 2, 1), token_labels)


def evaluate_covisco(model, data, epoch, args, tb_writer=None):
    """
    Evaluate CoVisco SigLIP model on ImageNet zero-shot classification.

    Since the CoVisco model has no text encoder (it trains with pre-extracted
    text embeddings from timm/ViT-gopt-16-SigLIP2-384), this function temporarily
    loads the full SigLIP2 teacher model's text encoder to build the zero-shot
    classifier, and uses the CoVisco model's image encoder + proj_to_image_caption
    head to extract image features for classification.

    Args:
        model: CoViscoModel (or DDP-wrapped)
        data: Dictionary containing 'imagenet-val' and/or 'imagenet-v2' dataloaders
        epoch: Current epoch number
        args: Training arguments
        tb_writer: TensorBoard writer (optional)

    Returns:
        Dictionary of evaluation metrics
    """
    metrics = {}
    if not is_master(args):
        return metrics

    if 'imagenet-val' not in data and 'imagenet-v2' not in data:
        return metrics
    if args.zeroshot_frequency == 0:
        return metrics
    if (epoch % args.zeroshot_frequency) != 0 and epoch != args.epochs:
        return metrics

    device = torch.device(args.device)
    autocast = get_autocast(args.precision, device_type=device.type)
    input_dtype = get_input_dtype(args.precision)

    # Unwrap DDP if needed
    covisco_model = unwrap_model(model)
    covisco_model.eval()

    # --- Step 1: Load the teacher SigLIP2 text encoder for building the classifier ---
    teacher_model_name = 'ViT-gopt-16-SigLIP2-384'
    logging.info(
        f'CoVisco zero-shot eval: loading teacher text encoder ({teacher_model_name}) '
        f'for building zero-shot classifier...'
    )
    from open_clip import create_model, get_tokenizer, build_zero_shot_classifier, \
        IMAGENET_CLASSNAMES, OPENAI_IMAGENET_TEMPLATES

    # For CPU training, load teacher to CPU to avoid OOM; for GPU training, use the training device
    teacher_device = device if torch.cuda.is_available() else torch.device('cpu')
    teacher_model = create_model(
        teacher_model_name,
        pretrained='webli',
        device=teacher_device,
    )
    teacher_model.eval()
    tokenizer = get_tokenizer(teacher_model_name)

    # --- Step 2: Build the zero-shot classifier using teacher's text encoder ---
    logging.info('Building zero-shot classifier with teacher text encoder...')
    with autocast():
        classifier = build_zero_shot_classifier(
            teacher_model,
            tokenizer=tokenizer,
            classnames=IMAGENET_CLASSNAMES,
            templates=OPENAI_IMAGENET_TEMPLATES,
            num_classes_per_batch=10,
            device=teacher_device,
            use_tqdm=True,
        )
    # classifier shape: (embed_dim, num_classes)

    # Move classifier to training device if it was built on CPU
    if teacher_device != device:
        classifier = classifier.to(device)

    # Free the teacher model to reclaim memory
    del teacher_model
    del tokenizer

    # Properly clean up memory based on device type
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    else:
        import gc
        gc.collect()

    logging.info('Teacher text encoder released. Starting zero-shot evaluation...')

    # --- Step 3: Run zero-shot classification with CoVisco image encoder ---
    def _run_zeroshot(dataloader):
        with torch.inference_mode():
            top1, top5, n = 0., 0., 0.
            for images, target in tqdm(dataloader, unit_scale=args.batch_size):
                images = images.to(device=device, dtype=input_dtype)
                target = target.to(device)

                with autocast():
                    # Forward through CoVisco encoder + image_caption projection
                    model_out = covisco_model(
                        images,
                        modality='image',
                        return_intermediates=False,
                        run_decoder=False,
                    )
                    # Use to_image_caption features (aligned with teacher text embeddings)
                    image_features = model_out['to_image_caption']
                    logits = 100. * image_features @ classifier

                # Measure accuracy
                pred = logits.topk(5, 1, True, True)[1].t()
                correct = pred.eq(target.view(1, -1).expand_as(pred))
                top1 += float(correct[:1].reshape(-1).float().sum(0).cpu().numpy())
                top5 += float(correct[:5].reshape(-1).float().sum(0).cpu().numpy())
                n += images.size(0)

                # Explicit cleanup for CPU training after each batch
                if not torch.cuda.is_available():
                    import gc
                    gc.collect()

        # Handle empty dataloader case
        if n == 0:
            logging.warning('Zero-shot evaluation: no samples processed (dataloader may be empty or data loading failed)')
            return 0.0, 0.0

        return (top1 / n), (top5 / n)

    results = {}
    if 'imagenet-val' in data:
        top1, top5 = _run_zeroshot(data['imagenet-val'].dataloader)
        results['imagenet-zeroshot-val-top1'] = top1
        results['imagenet-zeroshot-val-top5'] = top5
    if 'imagenet-v2' in data:
        top1, top5 = _run_zeroshot(data['imagenet-v2'].dataloader)
        results['imagenetv2-zeroshot-val-top1'] = top1
        results['imagenetv2-zeroshot-val-top5'] = top5

    metrics.update(results)

    if not metrics:
        return metrics

    logging.info(
        f"Eval Epoch: {epoch} "
        + "\t".join([f"{k}: {round(v, 4):.4f}" for k, v in metrics.items()])
    )

    log_data = {"val/" + name: val for name, val in metrics.items()}

    if args.save_logs:
        if tb_writer is not None:
            for name, val in log_data.items():
                tb_writer.add_scalar(name, val, epoch)

        with open(os.path.join(args.checkpoint_path, "results.jsonl"), "a+") as f:
            f.write(json.dumps(metrics))
            f.write("\n")

    if args.wandb:
        assert wandb is not None, 'Please install wandb.'
        # Determine step for wandb logging
        step = None
        for data_key in ('image_train', 'video_train', 'train'):
            if data_key in data:
                dataloader = data[data_key].dataloader
                num_batches_per_epoch = dataloader.num_batches // args.accum_freq
                step = num_batches_per_epoch * epoch
                break
        log_data['epoch'] = epoch
        wandb.log(log_data, step=step)

    # Explicit cleanup of classifier and intermediate results
    del classifier

    # Properly clean up memory based on device type
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    else:
        import gc
        gc.collect()

    # Restore model to train mode before returning
    covisco_model.train()

    logging.info('Finished CoVisco zero-shot imagenet evaluation.')
    return metrics


def train_one_epoch_covisco_siglip(
    model, data, loss, epoch, optimizer, scaler, scheduler, args, tb_writer=None,
    original_model=None, checkpoint_path=None,
):
    """
    Training loop for CoVisco SigLIP training.

    Each optimizer step interleaves image, video, and image-vidcap batches: forward
    +backward for an image batch, then a video batch, then an image-vidcap batch,
    then optimizer.step(). All three sources have independent batch sizes and
    accumulation frequencies.

    Args:
        model: CoViscoModel (possibly wrapped with DDP)
        data: Dictionary containing 'image_train', 'video_train' and/or
            'image_vidcap_train' dataloaders
        loss: CoViscoSigLIPLoss
        epoch: Current epoch
        optimizer: Optimizer
        scaler: GradScaler for mixed precision
        scheduler: Learning rate scheduler
        args: Training arguments
        tb_writer: TensorBoard writer (optional)

    Returns:
        Dictionary of average losses for this epoch
    """
    device = torch.device(args.device)
    autocast = get_autocast(args.precision, device_type=device.type)
    input_dtype = get_input_dtype(args.precision)

    # Get unwrapped model for accessing custom methods
    unwrap_model = model.module if hasattr(model, 'module') else model

    model.train()

    # Set up dataloaders and iterators per modality
    image_dataloader = None
    video_dataloader = None
    image_vidcap_dataloader = None

    if 'image_train' in data:
        data['image_train'].set_epoch(epoch)
        image_dataloader = data['image_train'].dataloader
    if 'video_train' in data:
        data['video_train'].set_epoch(epoch)
        video_dataloader = data['video_train'].dataloader
    if 'image_vidcap_train' in data:
        data['image_vidcap_train'].set_epoch(epoch)
        image_vidcap_dataloader = data['image_vidcap_train'].dataloader

    assert image_dataloader is not None or video_dataloader is not None or image_vidcap_dataloader is not None, \
        "No training data found. Please specify --image-data-path, --video-data-path, or --image-vidcap-data-path."

    # Per-source accumulation frequency. Each source accumulates its own micro-batches
    # independently; one optimizer step covers all accumulated micro-batches of all sources.
    accum_freq_image = getattr(args, 'accum_freq_image', None) or args.accum_freq
    accum_freq_video = getattr(args, 'accum_freq_video', None) or args.accum_freq
    accum_freq_image_vidcap = getattr(args, 'accum_freq_image_vidcap', None) or args.accum_freq
    accum_freq_image = max(1, int(accum_freq_image))
    accum_freq_video = max(1, int(accum_freq_video))
    accum_freq_image_vidcap = max(1, int(accum_freq_image_vidcap))

    image_num_batches = image_dataloader.num_batches if image_dataloader is not None else 0
    video_num_batches = video_dataloader.num_batches if video_dataloader is not None else 0
    image_vidcap_num_batches = image_vidcap_dataloader.num_batches if image_vidcap_dataloader is not None else 0

    # Number of full accumulation cycles per epoch for each source.
    # An "optim step" is one optimizer.step(), which covers accum_freq_image image
    # micro-batches + accum_freq_video video micro-batches + accum_freq_image_vidcap
    # image-vidcap micro-batches (the longest source drives the count; shorter ones cycle).
    image_cycles = (image_num_batches // accum_freq_image) if (image_dataloader is not None and image_num_batches > 0) else 0
    video_cycles = (video_num_batches // accum_freq_video) if (video_dataloader is not None and video_num_batches > 0) else 0
    image_vidcap_cycles = (image_vidcap_num_batches // accum_freq_image_vidcap) if (image_vidcap_dataloader is not None and image_vidcap_num_batches > 0) else 0
    num_optim_steps = max(image_cycles, video_cycles, image_vidcap_cycles)

    # Track which dataloaders need manual cycling (any source with fewer cycles than the max)
    needs_image_cycle = image_dataloader is not None and image_cycles < num_optim_steps
    needs_video_cycle = video_dataloader is not None and video_cycles < num_optim_steps
    needs_image_vidcap_cycle = image_vidcap_dataloader is not None and image_vidcap_cycles < num_optim_steps

    if image_dataloader is not None:
        image_iter = iter(image_dataloader)
    else:
        image_iter = None
    if video_dataloader is not None:
        video_iter = iter(video_dataloader)
    else:
        video_iter = None
    if image_vidcap_dataloader is not None:
        image_vidcap_iter = iter(image_vidcap_dataloader)
    else:
        image_vidcap_iter = None

    sample_digits = math.ceil(math.log(max(
        image_dataloader.num_samples if image_dataloader else 0,
        video_dataloader.num_samples if video_dataloader else 0,
        image_vidcap_dataloader.num_samples if image_vidcap_dataloader else 0,
        1,
    ) + 1, 10))

    logging.info(
        f"Training epoch {epoch}: "
        f"image batches={image_num_batches}, video batches={video_num_batches}, "
        f"image_vidcap batches={image_vidcap_num_batches}, "
        f"accum_freq_image={accum_freq_image}, accum_freq_video={accum_freq_video}, "
        f"accum_freq_image_vidcap={accum_freq_image_vidcap}, "
        f"optim steps per epoch={num_optim_steps}"
    )

    # Initialize loss meters
    losses_m = {}
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()

    # Gradient accumulation buffers (per modality). Initialized unconditionally:
    # the main loop calls _accum_cache/_accum_forward_backward regardless of
    # args.accum_freq (per-source accum counts are clamped to >= 1), so a
    # default --accum-freq 1 must not leave this name unbound.
    accum_buffers = {}  # modality -> {'pixel_values': [], 'embeddings': [], 'cached_outputs': []}

    # Helper function to explicitly clear accumulation buffers to prevent memory leaks
    def _clear_accum_buffers(accum_buffers):
        """Explicitly clear accumulation buffers and trigger garbage collection for CPU training."""
        # Clear all lists in buffers
        for modality in list(accum_buffers.keys()):
            for key in list(accum_buffers[modality].keys()):
                if isinstance(accum_buffers[modality][key], list):
                    accum_buffers[modality][key].clear()
            accum_buffers[modality].clear()
        accum_buffers.clear()

        # Trigger garbage collection for CPU training to free memory
        if not torch.cuda.is_available():
            import gc
            gc.collect()

    # Helper: extract batch data and move to device
    def _prepare_batch(batch):
        pixel_values = batch['pixel_values'].to(device=device, dtype=input_dtype, non_blocking=True)
        embeddings = {k: F.normalize(v.to(device=device, non_blocking=True),dim=-1) for k, v in batch['embeddings'].items()}
        vi = batch.get('visible_indices', None)
        if vi is not None:
            vi = vi.to(device=device, non_blocking=True)
        return pixel_values, embeddings, vi

    # Helper: get next batch from iterator, manually cycling if needed
    def _get_next_batch(iterator, dataloader, needs_cycle):
        """Get next batch from iterator, with manual cycling to prevent memory leaks."""
        try:
            batch = next(iterator)
        except StopIteration:
            if needs_cycle:
                iterator = iter(dataloader)
                batch = next(iterator)
                return batch, iterator
            else:
                raise
        return batch, iterator

    # Helper: gradient accumulation for one modality
    def _accum_cache(pixel_values, embeddings, buffer_key, visible_indices=None, model_modality=None):
        """Cache one micro-batch's inputs and model features (without gradients).

        Like standard CLIP accumulation, we cache model output features to use as
        negatives for other micro-batches, expanding the effective batch size for
        SigLIP contrastive loss.

        buffer_key identifies which accumulation buffer to use (e.g. "image", "video",
        "image_vidcap"); model_modality is what gets passed to model(modality=...) and
        defaults to buffer_key. image_vidcap uses buffer_key="image_vidcap" (its own
        buffer, kept separate from the real image batches) but model_modality="image"
        (same forward pass as image data, since the input is still images).
        """
        model_modality = model_modality or buffer_key
        buf = accum_buffers.setdefault(buffer_key, {
            'pixel_values': [], 'embeddings': [], 'cached_outputs': [], 'visible_indices': [],
        })
        with torch.no_grad():
            with autocast():
                # Randomly apply uniform frame sampling for video with 50% probability
                # Force uniform sampling when no visible_indices (no visidx) to avoid expensive dense path
                use_uniform = args.random_uniform_frame_sample and model_modality == "video" and random() < 0.5
                if model_modality == "video" and visible_indices is None:
                    use_uniform = True
                # If frame_concat is enabled and use_uniform is True, apply with 50% probability
                use_frame_concat = args.frame_concat and use_uniform and random() < 0.5
                model_outputs = model(pixel_values, visible_indices=visible_indices, modality=model_modality, return_intermediates=False, run_decoder=False,segment_offset=randint(0,8), uniform_sample_frames=use_uniform, frame_concat=use_frame_concat)
        # Store inputs on CPU to free GPU memory during accumulation.
        # pixel_values is the largest tensor (~hundreds of MB for large resolutions).
        buf['pixel_values'].append(pixel_values.cpu())
        buf['embeddings'].append(embeddings)
        buf['visible_indices'].append(
            visible_indices.cpu() if visible_indices is not None else None
        )
        # Cache projection features (detached, no grad) for negative accumulation
        buf['cached_outputs'].append({k: v.detach() for k, v in model_outputs.items()})

        # Explicitly free model_outputs to reduce memory footprint before next cache iteration
        del model_outputs

    def _accum_forward_backward(buffer_key, model_modality=None):
        """Re-forward with gradients for each micro-batch, using cached features
        from other micro-batches as negatives (same pattern as train_one_epoch).

        For micro-batch j:
          - student features = cat(cached[0], ..., fresh[j], ..., cached[N-1])
          - teacher embeddings = cat(embeddings[0], ..., embeddings[N-1])
          - Reconstruction loss is computed only on the fresh forward (with gradients),
            and scaled down by 1/num_accum to maintain consistent weighting relative
            to contrastive loss (which is already normalized by the full accumulated
            batch size).
        
        Note: Uses DDP no_sync() to properly handle gradient accumulation with
        find_unused_parameters=True.

        buffer_key identifies which accumulation buffer to drain; model_modality is
        what gets passed to model(modality=...) and to get_logit_scales_and_biases,
        defaulting to buffer_key (see _accum_cache for why image_vidcap differs).
        """
        model_modality = model_modality or buffer_key
        buf = accum_buffers.get(buffer_key)
        if buf is None or len(buf['pixel_values']) == 0:
            return None

        num_accum = len(buf['pixel_values'])
        loss_dict = None

        # Pre-concatenate all teacher embeddings (no grad needed)
        all_embeddings = {}
        for key in buf['embeddings'][0].keys():
            all_embeddings[key] = torch.cat([buf['embeddings'][k][key] for k in range(num_accum)], dim=0)

        # Determine which keys are projection outputs (to be accumulated as negatives)
        proj_keys = [k for k in buf['cached_outputs'][0].keys()
                     if k not in ('query_tokens', 'vit_tokens', 'reconstructed', 'attentions')]

        import contextlib
        no_sync_ctx = model.no_sync() if (args.distributed and hasattr(model, 'no_sync')) else contextlib.nullcontext()

        def _run_micro_batch(j):
            nonlocal loss_dict
            # Move inputs from CPU back to GPU for the with-grad forward
            pixel_values_j = buf['pixel_values'][j].to(device=device, dtype=input_dtype, non_blocking=True)
            visible_indices_j = buf['visible_indices'][j]
            if visible_indices_j is not None:
                visible_indices_j = visible_indices_j.to(device=device, non_blocking=True)
            with autocast():
                use_uniform = args.random_uniform_frame_sample and model_modality == "video" and random() < 0.5
                # Force uniform sampling when no visible_indices (no visidx) to avoid expensive dense path
                if model_modality == "video" and visible_indices_j is None:
                    use_uniform = True
                use_frame_concat = args.frame_concat and use_uniform and random() < 0.5
                fresh_outputs = model(pixel_values_j, visible_indices=visible_indices_j, modality=model_modality, return_intermediates=True, segment_offset=randint(0, 8), uniform_sample_frames=use_uniform, frame_concat=use_frame_concat)

                accumulated_outputs = {}
                for key in proj_keys:
                    parts = []
                    for k in range(num_accum):
                        if k == j:
                            parts.append(fresh_outputs[key])
                        else:
                            parts.append(buf['cached_outputs'][k][key])
                    accumulated_outputs[key] = torch.cat(parts, dim=0)

                for key in fresh_outputs:
                    if key not in accumulated_outputs:
                        accumulated_outputs[key] = fresh_outputs[key]

                logit_scales, logit_biases = unwrap_model.get_logit_scales_and_biases(model_modality)
                loss_dict = loss(accumulated_outputs, all_embeddings, logit_scales, logit_biases)

                total_loss = loss_dict['total']
                if 'reconstruction' in loss_dict and num_accum > 1:
                    recon = loss_dict['reconstruction']
                    recon_weight = loss.reconstruction_weight
                    total_loss = total_loss - recon_weight * recon + recon_weight * recon / num_accum

            if scaler is not None:
                scaler.scale(total_loss).backward()
            else:
                total_loss.backward()

            del fresh_outputs, accumulated_outputs, total_loss

        loss_dict = None
        # All micro-batches run inside no_sync; the caller controls when all-reduce fires.
        with no_sync_ctx:
            for j in range(num_accum):
                _run_micro_batch(j)

        # Reset buffer - clear only the processed buffer's lists
        for key in accum_buffers[buffer_key]:
            if isinstance(accum_buffers[buffer_key][key], list):
                accum_buffers[buffer_key][key].clear()

        # Free local references before returning
        del all_embeddings

        return loss_dict

    # Per-source batch sizes (fallback to --batch-size) for samples/s logging
    image_batch_size = (getattr(args, 'image_batch_size', None) or args.batch_size) if image_dataloader is not None else 0
    video_batch_size = (getattr(args, 'video_batch_size', None) or args.batch_size) if video_dataloader is not None else 0
    image_vidcap_batch_size = (getattr(args, 'image_vidcap_batch_size', None) or args.batch_size) if image_vidcap_dataloader is not None else 0
    samples_per_optim_step = (
        accum_freq_image * image_batch_size
        + accum_freq_video * video_batch_size
        + accum_freq_image_vidcap * image_vidcap_batch_size
    )

    # Main training loop: each iteration is ONE optimizer step.
    # Image, video, and image-vidcap accumulate INDEPENDENTLY to their own counts within this step.
    for optim_step_i in range(num_optim_steps):
        # Dynamic resolution: pick a resolution for this entire optimizer step.
        # All micro-batches within this step will be resized to the same size,
        # ensuring torch.cat in _accum_forward_backward works correctly.
        dynamic_resolutions = getattr(args, 'dynamic_resolutions', None)
        chosen_image_size = None
        if dynamic_resolutions and len(dynamic_resolutions) > 1:
            chosen_image_size = choice(dynamic_resolutions)

        data_time_m.update(time.time() - end)

        if not args.skip_scheduler:
            global_step = epoch * num_optim_steps + optim_step_i
            scheduler(global_step)

        optimizer.zero_grad()
        merged_loss_dict = {}

        # ---- Cache accum_freq_image image micro-batches (no grad) ----
        if image_iter is not None:
            for _ in range(accum_freq_image):
                image_batch, image_iter = _get_next_batch(image_iter, image_dataloader, needs_image_cycle)
                img_pv, img_emb, img_vi = _prepare_batch(image_batch)
                # Dynamic resolution: resize on GPU to the chosen size
                if chosen_image_size is not None and img_pv.shape[-1] != chosen_image_size:
                    img_pv = TF.resize(img_pv, [chosen_image_size, chosen_image_size],
                                       interpolation=TF.InterpolationMode.BICUBIC)
                _accum_cache(img_pv, img_emb, "image", visible_indices=img_vi)

        # ---- Cache accum_freq_video video micro-batches (no grad) ----
        if video_iter is not None:
            for _ in range(accum_freq_video):
                video_batch, video_iter = _get_next_batch(video_iter, video_dataloader, needs_video_cycle)
                vid_pv, vid_emb, vid_vi = _prepare_batch(video_batch)
                _accum_cache(vid_pv, vid_emb, "video", visible_indices=vid_vi)

        # ---- Cache accum_freq_image_vidcap image-vidcap micro-batches (no grad) ----
        # Own buffer, own dataloader, but forwarded through the model as modality="image"
        # (the input is still images; only the caption embeddings come from the video
        # text encoder), and it shares the image_video_caption loss branch.
        if image_vidcap_iter is not None:
            for _ in range(accum_freq_image_vidcap):
                image_vidcap_batch, image_vidcap_iter = _get_next_batch(
                    image_vidcap_iter, image_vidcap_dataloader, needs_image_vidcap_cycle)
                ivc_pv, ivc_emb, ivc_vi = _prepare_batch(image_vidcap_batch)
                if chosen_image_size is not None and ivc_pv.shape[-1] != chosen_image_size:
                    ivc_pv = TF.resize(ivc_pv, [chosen_image_size, chosen_image_size],
                                        interpolation=TF.InterpolationMode.BICUBIC)
                _accum_cache(ivc_pv, ivc_emb, "image_vidcap", visible_indices=ivc_vi, model_modality="image")

        # ---- Forward + backward for accumulated image + video + image-vidcap batches ----
        # All three backwards run inside no_sync (inside _accum_forward_backward).
        # Manual all-reduce fires here once for the entire optimizer step.
        img_loss_dict = _accum_forward_backward("image")
        if img_loss_dict is not None:
            merged_loss_dict.update(img_loss_dict)
            img_total = img_loss_dict['total']
        else:
            img_total = 0.0

        vid_loss_dict = _accum_forward_backward("video")
        if vid_loss_dict is not None:
            merged_loss_dict.update(vid_loss_dict)
            vid_total = vid_loss_dict['total']
        else:
            vid_total = 0.0

        # image_vidcap shares the "siglip_image_video_caption" key with real image
        # batches (when an image tar also carries a video-encoder caption). Since
        # they're now separate accumulation buffers, sum both contributions instead
        # of letting dict.update() silently overwrite one with the other.
        image_vidcap_loss_dict = _accum_forward_backward("image_vidcap", model_modality="image")
        if image_vidcap_loss_dict is not None:
            shared_key = 'siglip_image_video_caption'
            if shared_key in merged_loss_dict and shared_key in image_vidcap_loss_dict:
                merged_loss_dict[shared_key] = merged_loss_dict[shared_key] + image_vidcap_loss_dict[shared_key]
                for k, v in image_vidcap_loss_dict.items():
                    if k not in (shared_key, 'total'):
                        merged_loss_dict[k] = v
            else:
                merged_loss_dict.update({k: v for k, v in image_vidcap_loss_dict.items() if k != 'total'})
            image_vidcap_total = image_vidcap_loss_dict['total']
        else:
            image_vidcap_total = 0.0

        # Manually all-reduce accumulated gradients (bypasses DDP hooks entirely)
        if args.distributed:
            import torch.distributed as dist
            world_size = dist.get_world_size()
            for param in model.parameters():
                if param.grad is not None:
                    dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                    param.grad.div_(world_size)

        # Combined total for logging (gradients are already summed from separate backwards)
        merged_loss_dict['total'] = img_total + vid_total + image_vidcap_total

        # ---- ONE optimizer step covering all accumulated image + video micro-batches ----
        if scaler is not None:
            if args.grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            if args.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
            optimizer.step()

        loss_dict = merged_loss_dict

        # Explicit cleanup for CPU training
        if not torch.cuda.is_available():
            import gc
            gc.collect()

        # Clamp logit_scale to [0, 100] (consistent with SigLIP implementation)
        # Note: SigLIP uses logit_scale directly as temperature (no exp()), so we clamp
        # the temperature value directly, not the log of temperature like CLIP.
        # CLIP:    clamp(0, ln(100)) because logit_scale stores ln(temperature)
        # SigLIP:  clamp(0, 100) because logit_scale is the temperature itself
        with torch.no_grad():
            for attr in ('logit_scale_image_caption', 'logit_scale_image_image', 'logit_scale_video_caption'):
                if hasattr(unwrap_model, attr):
                    getattr(unwrap_model, attr).clamp_(0, 100)

        # Collect logit_scale and logit_bias values for logging
        with torch.no_grad():
            scale_bias_vals = {}
            for attr in (
                'logit_scale_image_caption', 'logit_scale_image_image', 'logit_scale_video_caption',
                'logit_bias_image_caption', 'logit_bias_image_image', 'logit_bias_video_caption',
            ):
                if hasattr(unwrap_model, attr):
                    scale_bias_vals[attr] = getattr(unwrap_model, attr).item()

        # Update loss meters
        for k, v in loss_dict.items():
            if torch.is_tensor(v):
                if k not in losses_m:
                    losses_m[k] = AverageMeter()
                losses_m[k].update(v.item())

        batch_time_m.update(time.time() - end)
        end = time.time()

        # Logging
        global_step = epoch * num_optim_steps + optim_step_i
        effective_step = global_step
        if is_master(args) and (effective_step % args.log_every_n_steps == 0 or optim_step_i == num_optim_steps - 1):
            loss_log = " ".join(
                f"{name}: {m.val:#.5g} ({m.avg:#.5g})"
                for name, m in losses_m.items()
            )
            scale_bias_log = " ".join(
                f"{name}: {val:.4f}" for name, val in scale_bias_vals.items()
            )
            samples_per_second = samples_per_optim_step * args.world_size / batch_time_m.val
            samples_per_second_per_gpu = samples_per_optim_step / batch_time_m.val
            logging.info(
                f"Train Epoch: {epoch} [{optim_step_i + 1}/{num_optim_steps}] "
                f"LR: {optimizer.param_groups[0]['lr']:.6f} "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f}, {samples_per_second:#g}/s, {samples_per_second_per_gpu:#g}/s/gpu "
                f"{loss_log} "
                f"{scale_bias_log}"
            )

            # TensorBoard logging
            if tb_writer is not None:
                for k, v in loss_dict.items():
                    if torch.is_tensor(v):
                        tb_writer.add_scalar(f"train/{k}", v.item(), effective_step)
                tb_writer.add_scalar("train/batch_time", batch_time_m.avg, effective_step)
                tb_writer.add_scalar("train/data_time", data_time_m.avg, effective_step)
                tb_writer.add_scalar("train/lr", optimizer.param_groups[0]['lr'], effective_step)
                for name, val in scale_bias_vals.items():
                    tb_writer.add_scalar(f"train/{name}", val, effective_step)

            if args.wandb:
                assert wandb is not None, 'Please install wandb.'
                log_data = {"train/" + k: m.val for k, m in losses_m.items()}
                log_data["train/lr"] = optimizer.param_groups[0]['lr']
                for name, val in scale_bias_vals.items():
                    log_data[f"train/{name}"] = val
                log_data['step'] = effective_step
                wandb.log(log_data, step=effective_step)

            batch_time_m.reset()
            data_time_m.reset()

        # ---- Save checkpoint by step ----
        save_steps = getattr(args, 'save_steps', 0)
        if (
            save_steps > 0
            and checkpoint_path is not None
            and original_model is not None
            and is_master(args)
            and (global_step + 1) % save_steps == 0
        ):
            step_checkpoint = {
                "epoch": epoch,
                "global_step": global_step + 1,
                "name": args.name,
                "state_dict": original_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "optimizer_param_names": get_optimizer_param_names(optimizer, original_model),
            }
            if scaler is not None:
                step_checkpoint["scaler"] = scaler.state_dict()
            torch.save(
                step_checkpoint,
                os.path.join(checkpoint_path, f"step_{global_step + 1}.pt"),
            )
            logging.info(f"Saved step checkpoint: step_{global_step + 1}.pt")

    # Compute average losses for the epoch
    avg_losses = {k: v.avg for k, v in losses_m.items()}
    return avg_losses
