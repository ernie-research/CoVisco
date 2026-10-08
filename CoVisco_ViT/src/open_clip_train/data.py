import ast
import json
import logging
import math
import os
import random
import sys
import braceexpand
from dataclasses import dataclass
from multiprocessing import Value

import numpy as np
import pandas as pd
import torch
import torchvision.datasets as datasets
import webdataset as wds
from PIL import Image
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, IterableDataset, get_worker_info
from torch.utils.data.distributed import DistributedSampler
from webdataset.filters import _shuffle
from webdataset.tariterators import base_plus_ext, url_opener, tar_file_expander, valid_sample

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None


class CsvDataset(Dataset):
    def __init__(self, input_filename, transforms, img_key, caption_key, sep="\t", tokenizer=None):
        logging.debug(f'Loading csv data from {input_filename}.')
        df = pd.read_csv(input_filename, sep=sep)

        self.images = df[img_key].tolist()
        self.captions = df[caption_key].tolist()
        self.transforms = transforms
        logging.debug('Done loading data.')

        self.tokenize = tokenizer

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        images = self.transforms(Image.open(str(self.images[idx])))
        texts = self.tokenize([str(self.captions[idx])])[0]
        return images, texts


class EmbeddingDataset(Dataset):
    """
    Dataset for loading images/videos with pre-extracted embeddings.

    This dataset is designed for CoVisco SigLIP training where pre-extracted
    embeddings from other models serve as supervision signals.

    Expected CSV format:
        filepath, image_emb, image_caption_emb, video_caption_emb, modality[, visible_indices]
        /path/to/img.jpg, img_emb_001.pt, cap_emb_001.pt, , image
        /path/to/vid.mp4, , , vid_cap_emb_001.pt, video, vis_idx_001.pt

    For video samples, ``visible_indices`` is a pre-computed tensor of shape
    ``[L]`` with ``L=4096``, values in ``[0, 32767]``.  The column is optional:
    when absent (or empty for a row) no masking is applied and ``visible_indices``
    is returned as ``None``.

    ``video_caption_emb`` may also be set on image rows: such images are contrasted
    against the shared video-caption head (their captions were encoded by the video
    text encoder) and are exposed under the ``image_video_caption`` embedding key.

    Args:
        csv_path: Path to CSV file containing data info
        transforms: Image/video transforms to apply
        modality: "image" or "video" - determines data loading mode
        embedding_dir: Directory containing embedding files (default: same as csv_path)
        tokenizer: Optional tokenizer for text (not used for pre-extracted embeddings)
    """

    def __init__(
        self,
        csv_path: str,
        transforms,
        modality: str = "image",
        embedding_dir: str = None,
        tokenizer=None,
        video_transforms=None,
    ):
        logging.debug(f'Loading embedding dataset from {csv_path}.')
        df = pd.read_csv(csv_path, sep='\t')
        self.df = df
        self.transforms = transforms
        # If video_transforms is provided, use it for video frames; otherwise fall back to transforms.
        self.video_transforms = video_transforms if video_transforms is not None else transforms
        self.modality = modality
        self.embedding_dir = embedding_dir or os.path.dirname(csv_path)
        self.tokenize = tokenizer

        # Validate modality
        assert modality in ('image', 'video'), f"Invalid modality: {modality}"

        # Filter rows by modality if column exists
        if 'modality' in df.columns:
            self.df = df[df['modality'] == modality].reset_index(drop=True)

        logging.debug(f'Loaded {len(self.df)} samples for modality {modality}.')

    def __len__(self):
        return len(self.df)

    def _load_image(self, filepath):
        """Load and transform an image."""
        image = Image.open(str(filepath)).convert('RGB')
        return self.transforms(image)

    def _load_video(self, filepath, num_frames=32):
        """
        Load and transform video frames.

        Args:
            filepath: Path to video file
            num_frames: Number of frames to sample

        Returns:
            Tensor of shape (T, C, H, W) where T is number of frames
        """
        try:
            import cv2
        except ImportError:
            raise ImportError("OpenCV is required for video loading. Install with: pip install opencv-python")

        cap = cv2.VideoCapture(str(filepath))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Sample frames uniformly
        if total_frames <= num_frames:
            frame_indices = list(range(total_frames))
        else:
            frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)
                frames.append(self.video_transforms(frame))
        cap.release()

        if len(frames) == 0:
            raise ValueError(f"No frames loaded from video: {filepath}")

        # Stack frames: list of (C, H, W) -> (T, C, H, W)
        frames_tensor = torch.stack(frames)
        # Permute to (C, T, H, W) so that batched result is (B, C, T, H, W) as expected by the model
        return frames_tensor.permute(1, 0, 2, 3)

    def _load_embedding(self, filepath):
        """Load pre-extracted embedding from file."""
        if pd.isna(filepath) or filepath == '':
            return None

        full_path = filepath if os.path.isabs(filepath) else os.path.join(self.embedding_dir, filepath)
        return torch.load(full_path, map_location='cpu')

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Load image or video
        filepath = row['filepath']
        if self.modality == "image":
            pixel_values = self._load_image(filepath)
        else:  # video
            pixel_values = self._load_video(filepath)

        # Load pre-extracted embeddings
        embeddings = {}

        # Image embedding (for image-image contrast)
        if 'image_emb' in row.index and pd.notna(row.get('image_emb', None)):
            emb = self._load_embedding(row['image_emb'])
            if emb is not None:
                embeddings['image'] = emb

        # Image caption embedding (for image-caption contrast)
        if 'image_caption_emb' in row.index and pd.notna(row.get('image_caption_emb', None)):
            emb = self._load_embedding(row['image_caption_emb'])
            if emb is not None:
                embeddings['image_caption'] = emb

        # Video caption embedding (for video-caption contrast). For image samples the
        # same column feeds the shared video-caption head under a distinct key, so the
        # loss keeps the two pairs apart while reusing the projection.
        if 'video_caption_emb' in row.index and pd.notna(row.get('video_caption_emb', None)):
            emb = self._load_embedding(row['video_caption_emb'])
            if emb is not None:
                key = 'image_video_caption' if self.modality == 'image' else 'video_caption'
                embeddings[key] = emb

        # visible_indices: only meaningful for video; shape [L] with L=4096, values in [0, 32767].
        # The column is optional — images always return None.
        visible_indices = None
        if self.modality == "video":
            vi_path = row.get('visible_indices', None) if 'visible_indices' in row.index else None
            if vi_path is not None and not pd.isna(vi_path) and vi_path != '':
                visible_indices = self._load_embedding(vi_path)

        return {
            'pixel_values': pixel_values,
            'modality': self.modality,
            'embeddings': embeddings,
            'visible_indices': visible_indices,
            'idx': idx,
        }


class SharedEpoch:
    def __init__(self, epoch: int = 0):
        self.shared_epoch = Value('i', epoch)

    def set_value(self, epoch):
        self.shared_epoch.value = epoch

    def get_value(self):
        return self.shared_epoch.value


@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler = None
    shared_epoch: SharedEpoch = None

    def set_epoch(self, epoch):
        if self.shared_epoch is not None:
            self.shared_epoch.set_value(epoch)
        if self.sampler is not None and isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(epoch)


def expand_urls(urls, weights=None):
    if weights is None:
        if isinstance(urls, list):
            return urls, None
        expanded_urls = wds.shardlists.expand_urls(urls)
        return expanded_urls, None
    if isinstance(urls, str):
        urllist = urls.split("::")
        weights = weights.split('::')
        assert len(weights) == len(urllist),\
            f"Expected the number of data components ({len(urllist)}) and weights({len(weights)}) to match."
        weights = [float(weight) for weight in weights]
        all_urls, all_weights = [], []
        for url, weight in zip(urllist, weights):
            expanded_url = list(braceexpand.braceexpand(url))
            expanded_weights = [weight for _ in expanded_url]
            all_urls.extend(expanded_url)
            all_weights.extend(expanded_weights)
        return all_urls, all_weights
    else:
        all_urls = list(urls)
        return all_urls, weights


def get_dataset_size(shards):
    shards_list, _ = expand_urls(shards)
    dir_path = os.path.dirname(shards_list[0])
    sizes_filename = os.path.join(dir_path, 'sizes.json')
    len_filename = os.path.join(dir_path, '__len__')
    if os.path.exists(sizes_filename):
        sizes = json.load(open(sizes_filename, 'r'))
        total_size = sum([int(sizes[os.path.basename(shard)]) for shard in shards_list])
    elif os.path.exists(len_filename):
        # FIXME this used to be eval(open(...)) but that seemed rather unsafe
        total_size = ast.literal_eval(open(len_filename, 'r').read())
    else:
        total_size = None  # num samples undefined
        # some common dataset sizes (at time of authors last download)
        # CC3M (train): 2905954
        # CC12M: 10968539
        # LAION-400M: 407332084
        # LAION-2B (english): 2170337258
    num_shards = len(shards_list)
    return total_size, num_shards


def get_imagenet(args, preprocess_fns, split):
    assert split in ["train", "val", "v2"]
    is_train = split == "train"
    # Handle case where preprocess_fns has more than 2 elements (e.g., video transform)
    preprocess_train, preprocess_val = preprocess_fns[0], preprocess_fns[1]

    if split == "v2":
        from imagenetv2_pytorch import ImageNetV2Dataset
        dataset = ImageNetV2Dataset(location=args.imagenet_v2, transform=preprocess_val)
    else:
        if is_train:
            data_path = args.imagenet_train
            preprocess_fn = preprocess_train
        else:
            data_path = args.imagenet_val
            preprocess_fn = preprocess_val
        assert data_path

        dataset = datasets.ImageFolder(data_path, transform=preprocess_fn)

    if is_train:
        idxs = np.zeros(len(dataset.targets))
        target_array = np.array(dataset.targets)
        k = 50
        for c in range(1000):
            m = target_array == c
            n = len(idxs[m])
            arr = np.zeros(n)
            arr[:k] = 1
            np.random.shuffle(arr)
            idxs[m] = arr

        idxs = idxs.astype('int')
        sampler = SubsetRandomSampler(np.where(idxs)[0])
    else:
        sampler = None

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        sampler=sampler,
    )

    return DataInfo(dataloader=dataloader, sampler=sampler)


def get_imagenet_wds(args, preprocess_fns, split):
    """Load ImageNet validation set from webdataset format (tar files with jpg+cls)."""
    assert split == "val", "Only val split is supported for webdataset format"
    # Handle case where preprocess_fns has more than 2 elements (e.g., video transform)
    _, preprocess_val = preprocess_fns[0], preprocess_fns[1]

    input_shards = args.imagenet_val_wds
    assert input_shards is not None

    # Fix malformed brace expansion: {0..6.tar} -> {0..6}.tar
    import re
    input_shards = re.sub(r'\{(\d+)\.\.(\d+)\.tar\}', r'{\1..\2}.tar', input_shards)

    # Expand brace expansion (e.g., {0..6}.tar -> [0.tar, 1.tar, ...])
    if '{' in input_shards and '}' in input_shards:
        expanded = list(braceexpand.braceexpand(input_shards))
        if len(expanded) > 1 or expanded[0] != input_shards:
            input_shards = expanded

    num_samples = args.imagenet_val_num_samples or 0

    pipeline = [
        wds.SimpleShardList(input_shards),
        wds.split_by_worker,
        wds.tarfile_to_samples(handler=log_and_continue),
        wds.decode("pilrgb", handler=log_and_continue),
        wds.rename(image="jpg;png;jpeg;webp", target="cls"),
        wds.map_dict(image=preprocess_val, target=lambda x: int(x)),
        wds.to_tuple("image", "target"),
        wds.batched(args.batch_size, partial=True),
    ]

    dataset = wds.DataPipeline(*pipeline)
    num_batches = math.ceil(num_samples / args.batch_size) if num_samples else 0

    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
    )

    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples

    return DataInfo(dataloader=dataloader, sampler=None)


def count_samples(dataloader):
    os.environ["WDS_EPOCH"] = "0"
    n_elements, n_batches = 0, 0
    for images, texts in dataloader:
        n_batches += 1
        n_elements += len(images)
        assert len(images) == len(texts)
    return n_elements, n_batches


def filter_no_caption_or_no_image(sample):
    has_caption = ('txt' in sample)
    has_image = ('png' in sample or 'jpg' in sample or 'jpeg' in sample or 'webp' in sample)
    return has_caption and has_image


def log_and_continue(exn):
    """Call in an exception handler to ignore any exception, issue a warning, and continue."""
    logging.warning(f'Handling webdataset error ({repr(exn)}). Ignoring.')
    return True


def group_by_keys_nothrow(data, keys=base_plus_ext, lcase=True, suffixes=None, handler=None):
    """Return function over iterator that groups key, value pairs into samples.

    :param keys: function that splits the key into key and extension (base_plus_ext)
    :param lcase: convert suffixes to lower case (Default value = True)
    """
    current_sample = None
    for filesample in data:
        assert isinstance(filesample, dict)
        if "fname" not in filesample:
            if handler:
                handler(ValueError(f"Empty or malformed filesample: {filesample.keys()}"))
            continue
        fname, value = filesample["fname"], filesample["data"]
        prefix, suffix = keys(fname)
        if prefix is None:
            continue
        if lcase:
            suffix = suffix.lower()
        # FIXME webdataset version throws if suffix in current_sample, but we have a potential for
        #  this happening in the current LAION400m dataset if a tar ends with same prefix as the next
        #  begins, rare, but can happen since prefix aren't unique across tar files in that dataset
        if current_sample is None or prefix != current_sample["__key__"] or suffix in current_sample:
            if valid_sample(current_sample):
                yield current_sample
            current_sample = dict(__key__=prefix, __url__=filesample["__url__"])
        if suffixes is None or suffix in suffixes:
            current_sample[suffix] = value
    if valid_sample(current_sample):
        yield current_sample


def tarfile_to_samples_nothrow(src, handler=log_and_continue):
    # NOTE this is a re-impl of the webdataset impl with group_by_keys that doesn't throw
    streams = url_opener(src, handler=handler)
    files = tar_file_expander(streams, handler=handler)
    samples = group_by_keys_nothrow(files, handler=handler)
    return samples


def pytorch_worker_seed(increment=0):
    """get dataloader worker seed from pytorch"""
    worker_info = get_worker_info()
    if worker_info is not None:
        # favour using the seed already created for pytorch dataloader workers if it exists
        seed = worker_info.seed
        if increment:
            # space out seed increments so they can't overlap across workers in different iterations
            seed += increment * max(1, worker_info.num_workers)
        return seed
    # fallback to wds rank based seed
    return wds.utils.pytorch_worker_seed()


_SHARD_SHUFFLE_SIZE = 2000
_SHARD_SHUFFLE_INITIAL = 500
_SAMPLE_SHUFFLE_SIZE = 5000
_SAMPLE_SHUFFLE_INITIAL = 1000


class detshuffle2(wds.PipelineStage):
    def __init__(
            self,
            bufsize=1000,
            initial=100,
            seed=0,
            epoch=-1,
    ):
        self.bufsize = bufsize
        self.initial = initial
        self.seed = seed
        self.epoch = epoch

    def run(self, src):
        if isinstance(self.epoch, SharedEpoch):
            epoch = self.epoch.get_value()
        else:
            # NOTE: this is epoch tracking is problematic in a multiprocess (dataloader workers or train)
            # situation as different workers may wrap at different times (or not at all).
            self.epoch += 1
            epoch = self.epoch
        rng = random.Random()
        if self.seed < 0:
            # If seed is negative, we use the worker's seed, this will be different across all nodes/workers
            seed = pytorch_worker_seed(epoch)
        else:
            # This seed to be deterministic AND the same across all nodes/workers in each epoch
            seed = self.seed + epoch
        rng.seed(seed)
        return _shuffle(src, self.bufsize, self.initial, rng)


class ResampledShards2(IterableDataset):
    """An iterable dataset yielding a list of urls."""

    def __init__(
        self,
        urls,
        weights=None,
        nshards=sys.maxsize,
        worker_seed=None,
        deterministic=False,
        epoch=-1,
    ):
        """Sample shards from the shard list with replacement.

        :param urls: a list of URLs as a Python list or brace notation string
        """
        super().__init__()
        urls, weights = expand_urls(urls, weights)
        self.urls = urls
        self.weights = weights
        if self.weights is not None:
            assert len(self.urls) == len(self.weights),\
                f"Number of urls {len(self.urls)} and weights {len(self.weights)} should match."
        assert isinstance(self.urls[0], str)
        self.nshards = nshards
        self.rng = random.Random()
        self.worker_seed = worker_seed
        self.deterministic = deterministic
        self.epoch = epoch

    def __iter__(self):
        """Return an iterator over the shards."""
        if isinstance(self.epoch, SharedEpoch):
            epoch = self.epoch.get_value()
        else:
            # NOTE: this is epoch tracking is problematic in a multiprocess (dataloader workers or train)
            # situation as different workers may wrap at different times (or not at all).
            self.epoch += 1
            epoch = self.epoch
        if self.deterministic:
            # reset seed w/ epoch if deterministic
            if self.worker_seed is None:
                # pytorch worker seed should be deterministic due to being init by arg.seed + rank + worker id
                seed = pytorch_worker_seed(epoch)
            else:
                seed = self.worker_seed() + epoch
            self.rng.seed(seed)
        for _ in range(self.nshards):
            if self.weights is None:
                yield dict(url=self.rng.choice(self.urls))
            else:
                yield dict(url=self.rng.choices(self.urls, weights=self.weights, k=1)[0])


def get_wds_dataset(args, preprocess_img, is_train, epoch=0, floor=False, tokenizer=None):
    input_shards = args.train_data if is_train else args.val_data
    assert input_shards is not None


    # Support reading shards list from a text file
    if isinstance(input_shards, str) and input_shards.endswith('.txt') and os.path.exists(input_shards):
        logging.info(f"Loading shards from {input_shards}")
        with open(input_shards, 'r') as f:
            input_shards = [line.strip() for line in f if line.strip()]
        logging.info(f"Loaded {len(input_shards)} shards.")

    # Support reading shards list from a text file
    if isinstance(input_shards, str) and input_shards.endswith('.txt') and os.path.exists(input_shards):
        logging.info(f"Loading shards from {input_shards}")
        with open(input_shards, 'r') as f:
            input_shards = [line.strip() for line in f if line.strip()]
        logging.info(f"Loaded {len(input_shards)} shards.")

    resampled = getattr(args, 'dataset_resampled', False) and is_train

    num_shards = None
    if is_train:
        if args.train_num_samples is not None:
            num_samples = args.train_num_samples
        else:
            num_samples, num_shards = get_dataset_size(input_shards)
            if not num_samples:
                raise RuntimeError(
                    'Currently, the number of dataset samples must be specified for the training dataset. '
                    'Please specify it via `--train-num-samples` if no dataset length info is present.')
    else:
        # Eval will just exhaust the iterator if the size is not specified.
        num_samples = args.val_num_samples or 0 

    shared_epoch = SharedEpoch(epoch=epoch)  # create a shared epoch store to sync epoch to dataloader worker proc

    if is_train and args.train_data_upsampling_factors is not None:
        assert resampled, "--train_data_upsampling_factors is only supported when sampling with replacement (with --dataset-resampled)."
    
    if resampled:
        pipeline = [ResampledShards2(
            input_shards,
            weights=args.train_data_upsampling_factors,
            deterministic=True,
            epoch=shared_epoch,
        )]
    else:
        pipeline = [wds.SimpleShardList(input_shards)]

    # at this point we have an iterator over all the shards
    if is_train:
        if not resampled:
            pipeline.extend([
                detshuffle2(
                    bufsize=_SHARD_SHUFFLE_SIZE,
                    initial=_SHARD_SHUFFLE_INITIAL,
                    seed=args.seed,
                    epoch=shared_epoch,
                ),
                wds.split_by_node,
                wds.split_by_worker,
            ])
        pipeline.extend([
            # at this point, we have an iterator over the shards assigned to each worker at each node
            tarfile_to_samples_nothrow,  # wds.tarfile_to_samples(handler=log_and_continue),
            wds.shuffle(
                bufsize=_SAMPLE_SHUFFLE_SIZE,
                initial=_SAMPLE_SHUFFLE_INITIAL,
            ),
        ])
    else:
        pipeline.extend([
            wds.split_by_worker,
            # at this point, we have an iterator over the shards assigned to each worker
            wds.tarfile_to_samples(handler=log_and_continue),
        ])
    pipeline.extend([
        wds.select(filter_no_caption_or_no_image),
        wds.decode("pilrgb", handler=log_and_continue),
        wds.rename(image="jpg;png;jpeg;webp", text="txt"),
        wds.map_dict(image=preprocess_img, text=lambda text: tokenizer(text)[0]),
        wds.to_tuple("image", "text"),
        wds.batched(args.batch_size, partial=not is_train)
    ])

    dataset = wds.DataPipeline(*pipeline)

    if is_train:
        if not resampled:
            num_shards = num_shards or len(expand_urls(input_shards)[0])
            assert num_shards >= args.workers * args.world_size, 'number of shards must be >= total workers'
        # roll over and repeat a few samples to get same number of full batches on each node
        round_fn = math.floor if floor else math.ceil
        global_batch_size = args.batch_size * args.world_size
        num_batches = round_fn(num_samples / global_batch_size)
        num_workers = max(1, args.workers)
        num_worker_batches = round_fn(num_batches / num_workers)  # per dataloader worker
        num_batches = num_worker_batches * num_workers
        num_samples = num_batches * global_batch_size
        dataset = dataset.with_epoch(num_worker_batches)  # each worker is iterating over this
    else:
        # last batches are partial, eval is done on single (master) node
        num_batches = math.ceil(num_samples / args.batch_size)

    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
    )

    # FIXME not clear which approach is better, with_epoch before vs after dataloader?
    # hoping to resolve via https://github.com/webdataset/webdataset/issues/169
    # if is_train:
    #     # roll over and repeat a few samples to get same number of full batches on each node
    #     global_batch_size = args.batch_size * args.world_size
    #     num_batches = math.ceil(num_samples / global_batch_size)
    #     num_workers = max(1, args.workers)
    #     num_batches = math.ceil(num_batches / num_workers) * num_workers
    #     num_samples = num_batches * global_batch_size
    #     dataloader = dataloader.with_epoch(num_batches)
    # else:
    #     # last batches are partial, eval is done on single (master) node
    #     num_batches = math.ceil(num_samples / args.batch_size)

    # add meta-data to dataloader instance for convenience
    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples

    return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch)


def get_csv_dataset(args, preprocess_fn, is_train, epoch=0, tokenizer=None):
    input_filename = args.train_data if is_train else args.val_data
    assert input_filename
    dataset = CsvDataset(
        input_filename,
        preprocess_fn,
        img_key=args.csv_img_key,
        caption_key=args.csv_caption_key,
        sep=args.csv_separator,
        tokenizer=tokenizer
    )
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)


def get_embedding_dataset(args, preprocess_fn, is_train, epoch=0, tokenizer=None, modality="image", video_preprocess_fn=None):
    """
    Create an embedding dataset for CoVisco SigLIP training.

    Args:
        args: Arguments containing data paths and settings
        preprocess_fn: Preprocessing function for images (and video fallback)
        is_train: Whether this is training data
        epoch: Current epoch for shared epoch
        tokenizer: Tokenizer (not used for pre-extracted embeddings)
        modality: "image" or "video" — determines which data path and batch size to use
        video_preprocess_fn: Optional separate preprocessing function for video frames.
            When provided, video frames will use this transform instead of preprocess_fn.

    Returns:
        DataInfo containing dataloader and sampler
    """
    if is_train:
        if modality == "image":
            input_filename = getattr(args, 'image_data', None)
        elif modality == "video":
            input_filename = getattr(args, 'video_data', None)
        else:
            raise ValueError(f"Unknown modality: {modality}")
        assert input_filename, f"No data path specified for modality: {modality}"
    else:
        input_filename = args.val_data
        modality = getattr(args, 'val_modality', 'image')

    # Per-modality batch size, fallback to global --batch-size
    if modality == "image":
        batch_size = getattr(args, 'image_batch_size', None) or args.batch_size
    elif modality == "video":
        batch_size = getattr(args, 'video_batch_size', None) or args.batch_size
    else:
        batch_size = args.batch_size

    dataset = EmbeddingDataset(
        csv_path=input_filename,
        transforms=preprocess_fn,
        modality=modality,
        embedding_dir=getattr(args, 'embedding_dir', None),
        tokenizer=tokenizer,
        video_transforms=video_preprocess_fn,
    )

    def _collate_fn(samples):
        """Default collate with special handling for visible_indices.

        When all samples in a batch have ``visible_indices=None`` (image batches),
        the batch-level value is also ``None``.  When all samples have a tensor
        (video batches with masking), they are stacked to ``[B, L]``.
        """
        from torch.utils.data.dataloader import default_collate

        # Separate out visible_indices before default_collate (it may contain None).
        vis_idx_list = [s.pop('visible_indices') for s in samples]
        batch = default_collate(samples)

        if all(v is None for v in vis_idx_list):
            batch['visible_indices'] = None
        else:
            # Replace None with zeros so we can stack; caller should handle accordingly.
            placeholder = torch.zeros_like(
                next(v for v in vis_idx_list if v is not None)
            )
            batch['visible_indices'] = torch.stack(
                [v if v is not None else placeholder for v in vis_idx_list]
            )
        return batch

    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
        collate_fn=_collate_fn,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)


class SyntheticDataset(Dataset):

    def __init__(
            self,
            transform=None,
            image_size=(224, 224),
            caption="Dummy caption",
            dataset_size=100,
            tokenizer=None,
    ):
        self.transform = transform
        self.image_size = image_size
        self.caption = caption
        self.image = Image.new('RGB', image_size)
        self.dataset_size = dataset_size

        self.preprocess_txt = lambda text: tokenizer(text)[0]

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        if self.transform is not None:
            image = self.transform(self.image)
        return image, self.preprocess_txt(self.caption)


def get_synthetic_dataset(args, preprocess_fn, is_train, epoch=0, tokenizer=None):
    image_size = preprocess_fn.transforms[0].size
    dataset = SyntheticDataset(
        transform=preprocess_fn, image_size=image_size, dataset_size=args.train_num_samples, tokenizer=tokenizer)
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)


def get_dataset_fn(data_path, dataset_type):
    if dataset_type == "webdataset":
        return get_wds_dataset
    elif dataset_type == "csv":
        return get_csv_dataset
    elif dataset_type == "synthetic":
        return get_synthetic_dataset
    elif dataset_type == "embedding":
        return get_embedding_dataset
    elif dataset_type == "auto":
        ext = data_path.split('.')[-1]
        if ext in ['csv', 'tsv']:
            return get_csv_dataset
        elif ext in ['tar']:
            return get_wds_dataset
        else:
            raise ValueError(
                f"Tried to figure out dataset type, but failed for extension {ext}.")
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")


def get_data(args, preprocess_fns, epoch=0, tokenizer=None):
    preprocess_train, preprocess_val = preprocess_fns[0], preprocess_fns[1]
    # Optional third element: a separate transform for video frames (different resolution).
    video_preprocess_train = preprocess_fns[2] if len(preprocess_fns) > 2 else None
    data = {}

    # Support for webdataset-based training with pre-extracted embeddings
    if getattr(args, 'dataset_type', None) == "webdataset_embedding":
        from .data_wds import get_image_wds_dataset, get_video_wds_dataset, get_image_vidcap_wds_dataset

        # Handle image data loader (empty string means "no image data")
        if getattr(args, 'image_data_path', None):
            data["image_train"] = get_image_wds_dataset(
                args, preprocess_train, epoch=epoch
            )
            logging.info(f"Created image training dataset from {args.image_data_path}")

        # Handle video data loader (empty string means "no video data")
        if getattr(args, 'video_data_path', None):
            data["video_train"] = get_video_wds_dataset(
                args, video_preprocess_train or preprocess_train, epoch=epoch
            )
            logging.info(f"Created video training dataset from {args.video_data_path}")

        # Handle image-vidcap data loader: independent dataloader for images whose
        # captions were encoded by the video text encoder (empty string means "none")
        if getattr(args, 'image_vidcap_data_path', None):
            data["image_vidcap_train"] = get_image_vidcap_wds_dataset(
                args, preprocess_train, epoch=epoch
            )
            logging.info(f"Created image-vidcap training dataset from {args.image_vidcap_data_path}")

        # ImageNet zero-shot evaluation datasets
        if getattr(args, 'imagenet_val_wds', None) is not None:
            data["imagenet-val"] = get_imagenet_wds(args, preprocess_fns, "val")
        elif args.imagenet_val is not None:
            data["imagenet-val"] = get_imagenet(args, preprocess_fns, "val")

        if args.imagenet_v2 is not None:
            data["imagenet-v2"] = get_imagenet(args, preprocess_fns, "v2")

        return data

    # Support for embedding dataset type (CoVisco SigLIP training)
    if getattr(args, 'dataset_type', None) == "embedding":
        # Handle separate image and video data loaders
        if hasattr(args, 'image_data') and args.image_data is not None:
            data["image_train"] = get_embedding_dataset(
                args, preprocess_train, is_train=True, epoch=epoch, tokenizer=tokenizer,
                modality="image",
            )
            logging.info(f"Created image training dataset from {args.image_data}")

        if hasattr(args, 'video_data') and args.video_data is not None:
            data["video_train"] = get_embedding_dataset(
                args, preprocess_train, is_train=True, epoch=epoch, tokenizer=tokenizer,
                modality="video", video_preprocess_fn=video_preprocess_train,
            )
            logging.info(f"Created video training dataset from {args.video_data}")

        # Validation data
        if args.val_data:
            data["val"] = get_embedding_dataset(
                args, preprocess_val, is_train=False, epoch=epoch, tokenizer=tokenizer
            )

        # ImageNet zero-shot evaluation datasets (also needed for embedding mode)
        if getattr(args, 'imagenet_val_wds', None) is not None:
            data["imagenet-val"] = get_imagenet_wds(args, preprocess_fns, "val")
        elif args.imagenet_val is not None:
            data["imagenet-val"] = get_imagenet(args, preprocess_fns, "val")

        if args.imagenet_v2 is not None:
            data["imagenet-v2"] = get_imagenet(args, preprocess_fns, "v2")

        return data

    # Original data loading logic
    if args.train_data or args.dataset_type == "synthetic":
        data["train"] = get_dataset_fn(args.train_data, args.dataset_type)(
            args, preprocess_train, is_train=True, epoch=epoch, tokenizer=tokenizer)

    if getattr(args, 'imagenet_val_wds', None) is not None:
        data["imagenet-val"] = get_imagenet_wds(args, preprocess_fns, "val")

    if args.val_data:
        data["val"] = get_dataset_fn(args.val_data, args.dataset_type)(
            args, preprocess_val, is_train=False, tokenizer=tokenizer)

    if args.imagenet_val is not None:
        data["imagenet-val"] = get_imagenet(args, preprocess_fns, "val")

    if args.imagenet_v2 is not None:
        data["imagenet-v2"] = get_imagenet(args, preprocess_fns, "v2")

    return data
