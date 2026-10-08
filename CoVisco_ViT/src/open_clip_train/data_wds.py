"""
WebDataset-based data loader for CoVisco SigLIP training.

This module provides efficient data loading for tar files containing:
- Images: jpg + .image_features.npy + .text_features.npy
- Images captioned by the video text encoder: jpg + .text_emb.npy
- Videos: mp4 + .text_emb.npy + .visidx.npy
"""

import io
import logging
import math
import os
import warnings

import numpy as np
import torch
import torchvision
import webdataset as wds
from PIL import Image
from torch.utils.data import get_worker_info

# Suppress imageio FFmpeg warnings to keep logs clean
warnings.filterwarnings('ignore', message='.*We had to kill ffmpeg.*')
warnings.filterwarnings('ignore', message='.*imageio.*ffmpeg.*')

# Suppress warnings from the av library
logging.getLogger('av').setLevel(logging.ERROR)


def pytorch_worker_seed(increment=0):
    """Get dataloader worker seed from pytorch."""
    worker_info = get_worker_info()
    if worker_info is not None:
        seed = worker_info.seed
        if increment:
            seed += increment * max(1, worker_info.num_workers)
        return seed
    return wds.utils.pytorch_worker_seed()


def dict_collation_fn(samples):
    """Custom collation function for dictionary samples.
    
    Collates a list of sample dictionaries into a batched dictionary.
    Handles nested dictionaries (like 'embeddings') and optional fields.
    """
    if not samples:
        return samples
    
    # Get all keys from the first sample
    keys = samples[0].keys()
    batch = {}
    
    for key in keys:
        values = [sample[key] for sample in samples]
        
        if key == 'embeddings':
            # Handle nested embeddings dict
            emb_keys = values[0].keys()
            batch[key] = {}
            for emb_key in emb_keys:
                emb_values = [v[emb_key] for v in values]
                batch[key][emb_key] = torch.stack(emb_values)
        elif key == 'visible_indices':
            # If any sample is None (mixed batch), set entire batch to None
            # so the training loop can force uniform sampling for the whole batch
            if all(v is not None for v in values):
                batch[key] = torch.stack(values)
            else:
                batch[key] = None
        elif isinstance(values[0], torch.Tensor):
            batch[key] = torch.stack(values)
        else:
            batch[key] = values
    
    return batch


def log_and_continue(exn):
    """Call in an exception handler to ignore any exception, issue a warning, and continue."""
    logging.warning(f'Handling webdataset error ({repr(exn)}). Ignoring.')
    return True


class SharedEpoch:
    def __init__(self, epoch: int = 0):
        from multiprocessing import Value
        self.shared_epoch = Value('i', epoch)

    def set_value(self, epoch):
        self.shared_epoch.value = epoch

    def get_value(self):
        return self.shared_epoch.value


class detshuffle2(wds.PipelineStage):
    def __init__(self, bufsize=1000, initial=100, seed=0, epoch=-1):
        self.bufsize = bufsize
        self.initial = initial
        self.seed = seed
        self.epoch = epoch

    def run(self, src):
        import random
        from webdataset.filters import _shuffle

        if isinstance(self.epoch, SharedEpoch):
            epoch = self.epoch.get_value()
        else:
            self.epoch += 1
            epoch = self.epoch
        rng = random.Random()
        if self.seed < 0:
            seed = pytorch_worker_seed(epoch)
        else:
            seed = self.seed + epoch
        rng.seed(seed)
        return _shuffle(src, self.bufsize, self.initial, rng)


def decode_image_with_features(sample):
    """Decode image and associated feature files."""
    try:
        # Decode image
        if 'jpg' in sample:
            image_bytes = sample['jpg']
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            sample['image'] = image
        elif 'png' in sample:
            image_bytes = sample['png']
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            sample['image'] = image

        # Load image features (siglip2 encoding)
        if 'image_features.npy' in sample:
            features_bytes = sample['image_features.npy']
            image_features = np.load(io.BytesIO(features_bytes))
            sample['image_features'] = torch.from_numpy(image_features).float()

        # Load text features (caption siglip2 feature)
        if 'text_features.npy' in sample:
            features_bytes = sample['text_features.npy']
            text_features = np.load(io.BytesIO(features_bytes))
            sample['text_features'] = torch.from_numpy(text_features).float()

        # Load caption features produced by the *video* text encoder.
        # Image shards using the video-style key name ('text_emb.npy') are trained
        # against the shared video-caption head instead of the image-caption head.
        if 'text_emb.npy' in sample:
            features_bytes = sample['text_emb.npy']
            video_text_emb = np.load(io.BytesIO(features_bytes))
            sample['video_text_emb'] = torch.from_numpy(video_text_emb).float()

        # Load JSON metadata if available
        # if 'json' in sample:
            # import json
            # json_bytes = sample['json']
            # sample['metadata'] = json.loads(json_bytes.decode('utf-8'))

    except Exception as e:
        logging.warning(f"Error decoding image sample: {e}")
        raise

    return sample


def prepare_image_sample(sample):
    """Prepare image sample for training (transform and format)."""
    # This will be applied after decode_image_with_features and preprocessing
    # We need to create the proper format for training
    result = {
        'pixel_values': sample['image'],  # Already preprocessed
        'embeddings': {},
        'visible_indices': None,
        'idx': 0  # Placeholder
    }

    # Add image features if available (for image-image contrast)
    if 'image_features' in sample and sample['image_features'] is not None:
        result['embeddings']['image'] = sample['image_features']

    # Add text features if available (for image-caption contrast)
    if 'text_features' in sample and sample['text_features'] is not None:
        result['embeddings']['image_caption'] = sample['text_features']

    # Add video-encoder text features if available (contrast against the shared
    # video-caption head)
    if 'video_text_emb' in sample and sample['video_text_emb'] is not None:
        result['embeddings']['image_video_caption'] = sample['video_text_emb']

    return result


def decode_video_with_features(sample):
    """Decode video and associated feature files."""
    try:
        # For videos, load directly from bytes without writing to temp file
        if 'mp4' in sample:
            video_bytes = sample['mp4']
            video_frames = load_video_frames_from_bytes(video_bytes)
            sample['video'] = video_frames

        # Load text embedding
        if 'text_emb.npy' in sample:
            features_bytes = sample['text_emb.npy']
            text_emb = np.load(io.BytesIO(features_bytes))
            sample['text_emb'] = torch.from_numpy(text_emb).float()

        # Load visible indices
        if 'visidx.npy' in sample:
            features_bytes = sample['visidx.npy']
            visidx = np.load(io.BytesIO(features_bytes))
            sample['visidx'] = torch.from_numpy(visidx).long()

        # Load JSON metadata if available
        if 'json' in sample:
            import json
            json_bytes = sample['json']
            sample['metadata'] = json.loads(json_bytes.decode('utf-8'))

    except Exception as e:
        logging.warning(f"Error decoding video sample: {e}")
        raise

    return sample


def apply_video_transform(frames, transform, target_frames=128):
    """Apply image transform to each frame in a video.

    Args:
        frames: List of PIL Images (video frames)
        transform: torchvision transform to apply to each frame
        target_frames: Target number of frames (default: 128). Shorter videos are padded.

    Returns:
        Tensor of shape (C, target_frames, H, W) for model input
    """
    import torch

    transformed_frames = []
    for frame in frames:
        transformed = transform(frame)
        transformed_frames.append(transformed)

    num_frames = len(transformed_frames)
    if num_frames < target_frames:
        last_frame = transformed_frames[-1]
        for _ in range(target_frames - num_frames):
            transformed_frames.append(last_frame.clone())
    elif num_frames > target_frames:
        transformed_frames = transformed_frames[:target_frames]

    frames_tensor = torch.stack(transformed_frames).permute(1, 0, 2, 3)

    return frames_tensor


def preprocess_video_sample(sample, video_preprocess, target_frames=128):
    """Preprocess video frames in a sample.
    
    Args:
        sample: Dictionary containing 'video' key with list of PIL Images
        video_preprocess: torchvision transform to apply to each frame
        target_frames: Target number of frames (default: 128). Shorter videos are padded.
        
    Returns:
        Sample with preprocessed video tensor
    """
    if 'video' in sample and sample['video'] is not None:
        sample['video'] = apply_video_transform(sample['video'], video_preprocess, target_frames)
    return sample


def prepare_video_sample(sample):
    """Prepare video sample for training (transform and format)."""
    # This will be applied after decode_video_with_features and preprocessing
    # We need to create the proper format for training
    result = {
        'pixel_values': sample['video'],  # Already preprocessed (stacked frames)
        'embeddings': {},
        'visible_indices': sample.get('visidx', None),
        'idx': 0  # Placeholder
    }

    # Add text embedding if available (for video-caption contrast)
    if 'text_emb' in sample and sample['text_emb'] is not None:
        result['embeddings']['video_caption'] = sample['text_emb']

    return result


def load_video_frames_from_bytes(video_bytes, num_frames=128):
    """
    Load video frames directly from bytes without writing to temp file.

    Args:
        video_bytes: Video file content as bytes
        num_frames: Number of frames to extract (default: 128)

    Returns:
        List of PIL Images
    """
    # Method 1: Try PyAV (av library) first - most reliable for FFmpeg resource management
    try:
        import av
        import io
        from PIL import Image as PILImage

        video_file = io.BytesIO(video_bytes)
        container = None
        try:
            container = av.open(video_file)
            video_stream = container.streams.video[0]
            video_stream.thread_type = 'AUTO'  # Enable threading for faster decoding

            frames = []
            for i, frame in enumerate(container.decode(video_stream)):
                if i >= num_frames:
                    break
                img_array = frame.to_rgb().to_ndarray()
                pil_image = PILImage.fromarray(img_array)
                frames.append(pil_image)

            if len(frames) == 0:
                raise ValueError("No frames loaded from video bytes using PyAV")

            return frames
        finally:
            # Ensure container is closed even if exception occurs
            if container is not None:
                container.close()
                # Force cleanup of internal resources
                del container

    except ImportError:
        pass
    except Exception as e:
        logging.warning(f"PyAV decoding failed: {e}")

    # Method 2: Try imageio (fallback)
    try:
        import imageio
        import io
        from PIL import Image as PILImage

        video_file = io.BytesIO(video_bytes)
        reader = None
        try:
            reader = imageio.get_reader(video_file, format='mp4')
            frames = []
            for i, frame in enumerate(reader):
                if i >= num_frames:
                    break
                if len(frame.shape) == 3:  # (H, W, C)
                    pil_image = PILImage.fromarray(frame)
                    frames.append(pil_image)
                elif len(frame.shape) == 2:  # (H, W) grayscale
                    pil_image = PILImage.fromarray(frame, mode='L').convert('RGB')
                    frames.append(pil_image)

            if len(frames) == 0:
                raise ValueError("No frames loaded from video bytes using imageio")

            return frames
        finally:
            # Explicitly close reader to prevent FFmpeg process leaks
            if reader is not None:
                try:
                    reader.close()
                except Exception:
                    pass
                del reader

    except ImportError:
        pass
    except Exception as e:
        logging.warning(f"imageio decoding failed: {e}")

    # Method 3: Fallback to temp file with OpenCV
    import tempfile
    import os

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
            tmp_file.write(video_bytes)
            tmp_path = tmp_file.name
        return load_video_frames(tmp_path, num_frames)
    finally:
        if tmp_path is not None and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


def load_video_frames(video_path, num_frames=128):
    """
    Load all frames from a video file.

    Args:
        video_path: Path to video file
        num_frames: Number of frames to extract (default: 128)

    Returns:
        List of PIL Images
    """
    try:
        import cv2
    except ImportError:
        raise ImportError("OpenCV is required for video loading. Install with: pip install opencv-python")

    cap = cv2.VideoCapture(video_path)
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Sample frames uniformly
        if total_frames <= num_frames:
            frame_indices = list(range(total_frames))
        else:
            frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int).tolist()

        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame))

        if len(frames) == 0:
            raise ValueError(f"No frames loaded from video: {video_path}")

        return frames
    finally:
        # Ensure the capture is always released, even on exceptions
        cap.release()


def filter_no_features(sample):
    """Filter out samples without required feature files."""
    # For images: need image_features.npy, text_features.npy or text_emb.npy
    # (text_emb.npy = caption encoded by the video text encoder)
    has_image_features = 'image_features.npy' in sample
    has_text_features = 'text_features.npy' in sample
    has_video_text_emb = 'text_emb.npy' in sample
    has_media = 'jpg' in sample or 'png' in sample or 'jpeg' in sample

    if has_media:
        return has_image_features or has_text_features or has_video_text_emb

    # For videos: need text_emb.npy
    has_video = 'mp4' in sample
    has_text_emb = 'text_emb.npy' in sample

    if has_video:
        return has_text_emb

    return False


def _get_image_like_wds_dataset(
    args, image_preprocess, epoch, data_path_attr, samples_per_shard_attr,
    num_samples_attr, batch_size_attr, workers_attr, shard_glob_prefix, arg_name,
):
    """
    Shared implementation for building an image-style webdataset dataloader
    (used for both --image-data-path and --image-vidcap-data-path). Both sources
    decode with decode_image_with_features/prepare_image_sample and only differ
    in which command-line args and shard filename prefix they use, so they get
    fully independent dataloaders (own shards, batch size, workers, shuffling)
    while sharing the same sample format.

    Args:
        args: Arguments containing data paths and settings
        image_preprocess: Image preprocessing function
        epoch: Current epoch for shared epoch
        data_path_attr: args attribute holding the shard directory/pattern
        samples_per_shard_attr: args attribute holding samples-per-shard
        num_samples_attr: args attribute holding total num samples override
        batch_size_attr: args attribute holding this source's batch size
        workers_attr: args attribute holding this source's worker count
        shard_glob_prefix: tar filename prefix to look for (e.g. "image_shard_")
        arg_name: CLI flag name to use in the error message if unset

    Returns:
        DataInfo containing dataloader
    """
    from .data import DataInfo

    data_path = getattr(args, data_path_attr, None)
    if not data_path:
        raise ValueError(f"--{arg_name} must be specified for this dataset")

    # Check if it's a directory or file list
    if os.path.isdir(data_path):
        # Expand to all tar files
        import glob
        shards = sorted(glob.glob(os.path.join(data_path, f"{shard_glob_prefix}*.tar")))
        if not shards:
            # Datasets that don't follow the expected shard naming (e.g. shards holding
            # captions encoded by the video text encoder) fall back to every tar file.
            shards = sorted(glob.glob(os.path.join(data_path, "*.tar")))
        # Filter out .tmp files
        shards = [s for s in shards if not s.endswith('.tmp')]
    else:
        # Assume it's a file or pattern
        shards = [data_path]

    logging.info(f"Found {len(shards)} shards for --{arg_name}")

    # Get num samples per shard
    samples_per_shard = getattr(args, samples_per_shard_attr, 6400)
    total_samples = len(shards) * samples_per_shard
    # Use calculated total_samples if num_samples override is 0 or not set
    num_samples = getattr(args, num_samples_attr, 0) or total_samples

    # Get batch size
    batch_size = getattr(args, batch_size_attr, None) or args.batch_size

    # Get number of workers (default 8 for images, reduce to 2 for CPU)
    num_workers_default = 2 if not torch.cuda.is_available() else 8
    num_workers = getattr(args, workers_attr, None)
    if num_workers is None:
        num_workers = getattr(args, 'workers', num_workers_default)
    is_cpu_training = not torch.cuda.is_available()

    # Create shared epoch
    shared_epoch = SharedEpoch(epoch)

    # For CPU training, use smaller shuffle buffers and disable persistent workers to prevent memory leaks
    if is_cpu_training:
        shard_bufsize = 200
        shard_initial = 50
        sample_bufsize = 1000
        sample_initial = 200
        persistent_workers = False
    else:
        shard_bufsize = 500
        shard_initial = 200
        sample_bufsize = 5000
        sample_initial = 2000
        persistent_workers = num_workers > 0

    # Build pipeline
    # Use adaptive buffer sizes based on device type to prevent memory accumulation
    pipeline = [
        wds.SimpleShardList(shards),
        detshuffle2(
            bufsize=shard_bufsize,
            initial=shard_initial,
            seed=args.seed,
            epoch=shared_epoch,
        ),
        wds.split_by_node,
        wds.split_by_worker,
        wds.tarfile_to_samples(handler=log_and_continue),
        wds.shuffle(
            bufsize=sample_bufsize,
            initial=sample_initial,
        ),
        wds.select(filter_no_features),
        wds.map(decode_image_with_features, handler=log_and_continue),
        wds.map_dict(image=image_preprocess),
        wds.map(prepare_image_sample),  # Convert to dict format expected by training code
        wds.batched(batch_size, collation_fn=dict_collation_fn, partial=False),
    ]

    dataset = wds.DataPipeline(*pipeline)

    # Calculate number of batches
    global_batch_size = batch_size * args.world_size
    num_batches = math.ceil(num_samples / global_batch_size)
    num_workers = max(1, num_workers)
    num_worker_batches = math.ceil(num_batches / num_workers)
    num_batches = num_worker_batches * num_workers
    num_samples = num_batches * global_batch_size

    dataset = dataset.with_epoch(num_worker_batches)

    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        pin_memory=True, 
    )

    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples

    return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch)


def get_image_wds_dataset(args, image_preprocess, epoch=0):
    """
    Create webdataset for image training data (image-caption / image-image contrast).

    Args:
        args: Arguments containing data paths and settings
        image_preprocess: Image preprocessing function
        epoch: Current epoch for shared epoch

    Returns:
        DataInfo containing dataloader
    """
    return _get_image_like_wds_dataset(
        args, image_preprocess, epoch,
        data_path_attr='image_data_path',
        samples_per_shard_attr='image_samples_per_shard',
        num_samples_attr='image_num_samples',
        batch_size_attr='image_batch_size',
        workers_attr='image_workers',
        shard_glob_prefix='image_shard_',
        arg_name='image-data-path',
    )


def get_image_vidcap_wds_dataset(args, image_preprocess, epoch=0):
    """
    Create webdataset for the image <-> video-text-encoder-caption contrast pair
    (images whose captions were encoded by the video text encoder). This is a fully
    independent dataloader from get_image_wds_dataset: separate shards, batch size,
    worker pool and shuffling, so this data source can be scaled/mixed independently
    from the image-caption/image-image data.

    Args:
        args: Arguments containing data paths and settings
        image_preprocess: Image preprocessing function
        epoch: Current epoch for shared epoch

    Returns:
        DataInfo containing dataloader
    """
    return _get_image_like_wds_dataset(
        args, image_preprocess, epoch,
        data_path_attr='image_vidcap_data_path',
        samples_per_shard_attr='image_vidcap_samples_per_shard',
        num_samples_attr='image_vidcap_num_samples',
        batch_size_attr='image_vidcap_batch_size',
        workers_attr='image_vidcap_workers',
        shard_glob_prefix='image_vidcap_shard_',
        arg_name='image-vidcap-data-path',
    )


def get_video_wds_dataset(args, video_preprocess, epoch=0):
    """
    Create webdataset for video training data.

    Args:
        args: Arguments containing data paths and settings
        video_preprocess: Video preprocessing function
        epoch: Current epoch for shared epoch

    Returns:
        DataInfo containing dataloader
    """
    from .data import DataInfo

    # Get video data path
    video_data_path = getattr(args, 'video_data_path', None)
    if not video_data_path:
        raise ValueError("--video-data-path must be specified for video dataset")

    # Check if it's a directory or file list
    if os.path.isdir(video_data_path):
        # Expand to all tar files
        tar_pattern = os.path.join(video_data_path, "video_shard_*.tar")
        import glob
        shards = sorted(glob.glob(tar_pattern))
        # Filter out .tmp files
        shards = [s for s in shards if not s.endswith('.tmp')]
    else:
        # Assume it's a file or pattern
        shards = [video_data_path]

    logging.info(f"Found {len(shards)} video shards")

    # Get num samples per shard
    samples_per_shard = getattr(args, 'video_samples_per_shard', 1000)
    total_samples = len(shards) * samples_per_shard
    # Use calculated total_samples if video_num_samples is 0 or not set
    num_samples = getattr(args, 'video_num_samples', 0) or total_samples

    # Get batch size
    batch_size = getattr(args, 'video_batch_size', args.batch_size)

    # Create shared epoch
    shared_epoch = SharedEpoch(epoch)

    # Get target frames from args or use default 128
    target_frames = getattr(args, 'video_num_frames', 128)

    # Get number of workers (default 2 for videos to prevent OOM, reduce to 1 for CPU)
    num_workers_default = 1 if not torch.cuda.is_available() else 2
    num_workers = getattr(args, 'video_workers', getattr(args, 'workers', num_workers_default))
    is_cpu_training = not torch.cuda.is_available()

    # For CPU training, use smaller shuffle buffers and disable persistent workers to prevent memory leaks
    if is_cpu_training:
        shard_bufsize = 100
        shard_initial = 30
        sample_bufsize = 50
        sample_initial = 20
        persistent_workers = False
    else:
        shard_bufsize = 500
        shard_initial = 200
        sample_bufsize = 1000
        sample_initial = 500
        persistent_workers = num_workers > 0

    # Build pipeline for videos
    # Adaptive buffer sizes based on device type to prevent memory accumulation
    pipeline = [
        wds.SimpleShardList(shards),
        detshuffle2(
            bufsize=shard_bufsize,
            initial=shard_initial,
            seed=args.seed,
            epoch=shared_epoch,
        ),
        wds.split_by_node,
        wds.split_by_worker,
        wds.tarfile_to_samples(handler=log_and_continue),
        wds.shuffle(
            bufsize=sample_bufsize,
            initial=sample_initial,
        ),
        wds.select(filter_no_features),
        wds.map(decode_video_with_features, handler=log_and_continue),
        wds.map(lambda s: preprocess_video_sample(s, video_preprocess, target_frames)),  # Apply transform to each frame
        wds.map(prepare_video_sample),  # Convert to dict format expected by training code
        wds.batched(batch_size, collation_fn=dict_collation_fn, partial=False),
    ]

    dataset = wds.DataPipeline(*pipeline)

    # Calculate number of batches
    global_batch_size = batch_size * args.world_size
    num_batches = math.ceil(num_samples / global_batch_size)
    num_workers = max(1, num_workers)
    num_worker_batches = math.ceil(num_batches / num_workers)
    num_batches = num_worker_batches * num_workers
    num_samples = num_batches * global_batch_size

    dataset = dataset.with_epoch(num_worker_batches)

    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        pin_memory=True,
    )

    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples

    return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch)