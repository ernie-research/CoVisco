"""wds_dataset.py - WDS dataset loader.

Reuses the LLaVA-OneVision-2 tar field format:
  - Image samples: jpg + json (containing messages)
  - Video samples (frame format): img0_0.jpg, img0_1.jpg, ... (multiple frames) + json
  - Video samples (mp4 format): xxx.mp4 + xxx.json
  - Text samples: json

Example JSON fields:
  {
    "messages": [
      {"role": "user", "content": "<image>\nDescribe the image."},
      {"role": "assistant", "content": "A cat."}
    ],
    "modality": "image" | "video" | "text"
  }
"""
from __future__ import annotations

import json
import os
import tempfile
from io import BytesIO
from typing import Any, Dict, Iterable, List, Optional, Set, Union

import numpy as np
from PIL import Image
from torch.utils.data import IterableDataset

from .ocr_patterns import is_ocr_sample


def load_skip_shards(src: Union[str, Iterable[str], None]) -> Set[str]:
    """Convert a skip list to a set of basenames. Empty / None means all shards are processed.

    src can be:
      - A text-file path (one shard per line; lines beginning with # are comments)
      - An iterable of shard paths / basenames
    Matching uses only basenames, so either absolute paths or file names are accepted.
    """
    if not src:
        return set()
    if isinstance(src, str):
        if os.path.isfile(src):
            names: Set[str] = set()
            with open(src) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    names.add(os.path.basename(line))
            return names
        return {os.path.basename(src)}
    return {os.path.basename(str(x)) for x in src if str(x).strip()}


def _dist_info() -> tuple:
    """Return (rank, world_size).

    torch.distributed is not initialized in DataLoader worker processes, so read the
    environment variables first (torchrun injects RANK / WORLD_SIZE), then fall back to
    the process group.
    """
    try:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        if world_size > 0:
            return rank, world_size
    except (KeyError, ValueError):
        pass
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank(), dist.get_world_size()
    except Exception:
        pass
    return 0, 1


DEFAULT_VIDEO_TARGET_FRAMES = 128   # Default mp4 frame count (overridable by the dataset parameter)
DEFAULT_SEGMENT_T_SIZE = 32         # Frames per segment in the visidx path; must match model config vit.segment_t_size


def _decode_mp4_frames(mp4_bytes: bytes,
                       target_frames: int = DEFAULT_VIDEO_TARGET_FRAMES) -> List[Image.Image]:
    """Decode an mp4 byte stream into target_frames uniformly sampled frames.

    Prefer decord (in-memory, no disk I/O, and fastest); fall back to the cv2
    frame-skipping approach when decord is unavailable.
    """
    # --- decord path (recommended, no disk I/O) ---
    try:
        import decord
        decord.bridge.set_bridge("native")
        vr = decord.VideoReader(BytesIO(mp4_bytes), ctx=decord.cpu(0))
        total = len(vr)
        n = min(target_frames, total)
        indices = np.linspace(0, total - 1, n, dtype=int).tolist()
        frames_np = vr.get_batch(indices).asnumpy()  # (n, H, W, 3) uint8
        frames = [Image.fromarray(frames_np[i]) for i in range(len(indices))]
        if len(frames) < target_frames:
            frames.extend([frames[-1]] * (target_frames - len(frames)))
        return frames
    except Exception:
        pass  # decord is unavailable; use the cv2 fallback

    # --- cv2 fallback (frame skipping, not full decoding) ---
    import cv2
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(mp4_bytes)
        tmp_path = tmp.name
    try:
        cap = cv2.VideoCapture(tmp_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total <= 0:
            frames_list: List[Image.Image] = []
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frames_list.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            cap.release()
            frames = frames_list
        else:
            n = min(target_frames, total)
            indices = np.linspace(0, total - 1, n, dtype=int).tolist()
            frames = []
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret:
                    if frames:
                        frames.append(frames[-1])
                    continue
                frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            cap.release()
    finally:
        os.unlink(tmp_path)

    if not frames:
        return frames
    if len(frames) < target_frames:
        frames.extend([frames[-1]] * (target_frames - len(frames)))
    return frames


# Simplified tar loading (does not depend on webdataset, convenient for standalone runs)
class TarShardReader:
    """Tar shard reader that streams in tar order and groups members of the same sample by key.

    Avoid getmembers() followed by extractfile seek-back over the full archive. On network
    filesystems (NFS/FUSE), scanning the fd to EOF and then seeking back can turn into EBADF,
    killing the DataLoader worker; after that rank exits, the remaining ranks can block in
    all_reduce until NCCL timeout.
    """

    def __init__(self, tar_path: str):
        import tarfile
        self.tar_path = tar_path
        # r|*: streaming read without seeking back. On network filesystems (NFS/FUSE),
        # getmembers()+extractfile can scan the fd to EOF and then seek back, which is prone to EBADF.
        self.tar = tarfile.open(tar_path, "r|*")

    def __iter__(self):
        import logging
        import tarfile

        current_key = None
        sample: Optional[Dict[str, Any]] = None
        try:
            for m in self.tar:
                if not m.isfile():
                    continue
                base = m.name.split(".")[0]
                if base != current_key:
                    if sample is not None:
                        yield sample
                    current_key = base
                    sample = {"__key__": base}
                try:
                    f = self.tar.extractfile(m)
                    if f is None:
                        continue
                    data = f.read()
                    self._fill_sample(sample, m.name, data)
                except Exception as e:
                    if isinstance(e, (KeyboardInterrupt, SystemExit, GeneratorExit)):
                        raise
                    logging.warning(
                        f"[WDS] skip member {m.name} shard={self.tar_path}: {e}"
                    )
                    # The fd is dead; subsequent extractfile calls will also hit EBADF, so drop the rest of this shard
                    if isinstance(e, OSError) and getattr(e, "errno", None) == 9:
                        logging.warning(
                            f"[WDS] stale fd, skip rest of shard {self.tar_path}"
                        )
                        return
                    continue
            if sample is not None:
                yield sample
        except (OSError, EOFError, tarfile.TarError) as e:
            logging.warning(f"[WDS] stop reading shard {self.tar_path}: {e}")
            return

    @staticmethod
    def _fill_sample(sample: Dict[str, Any], name: str, data: bytes) -> None:
        ext = name.split(".")[-1].lower()
        if ext in ("jpg", "jpeg", "png"):
            img = Image.open(BytesIO(data))
            img.load()  # Decode eagerly to avoid deferred I/O blocking the worker
            sample.setdefault("images", []).append(img)
        elif ext == "mp4":
            sample["mp4"] = data
        elif ext == "npy":
            if "visidx" in name:
                sample["visidx"] = np.load(BytesIO(data), allow_pickle=True)
            else:
                sample.setdefault("npy", []).append(
                    np.load(BytesIO(data), allow_pickle=True)
                )
        elif ext == "json":
            try:
                sample["json"] = json.loads(data.decode("utf-8"))
            except Exception:
                pass
        elif ext == "txt":
            sample["text"] = data.decode("utf-8", errors="ignore")

    def close(self):
        tar = self.tar
        self.tar = None
        if tar is None:
            return
        try:
            tar.close()
        except Exception:
            pass


def _shuffle_buffered(it, bufsize: int, rng):
    """Sample-level shuffle: maintain a buffer of bufsize, replacing a random item each time.

    This matches the semantics of webdataset's `.shuffle(bufsize)`. The buffer stores raw
    samples whose video frames have not been decoded (images are already PIL objects and
    mp4 values remain bytes), so memory use is approximately bufsize * the raw bytes per
    sample; bufsize should not be too large.
    """
    buf = []
    for x in it:
        if len(buf) < bufsize:
            buf.append(x)
            continue
        i = rng.randrange(bufsize)
        yield buf[i]
        buf[i] = x
    rng.shuffle(buf)
    for x in buf:
        yield x


class CoViscoWDSDataset(IterableDataset):
    """WDS dataset (IterableDataset split by worker/rank)."""

    def __init__(
        self,
        shards_path: str,            # Tar-shard list file or directory (auto-discovers *.tar)
        dynamic_token_config=None,   # DynamicTokenConfig instance
        plugin=None,                 # CoViscoPlugin instance
        shuffle: bool = True,
        seed: int = 0,
        skip_shards: Union[str, Iterable[str], None] = None,
        video_target_frames: int = DEFAULT_VIDEO_TARGET_FRAMES,   # Uniformly sampled frames after mp4 decoding
        segment_t_size: int = DEFAULT_SEGMENT_T_SIZE,             # Frames per segment in the visidx path (= model config vit.segment_t_size)
        shard_interleave: int = 1,   # Shards opened and polled per worker (1 = legacy sequential read)
        shuffle_buffer: int = 0,     # Sample-level shuffle-buffer capacity (0/1 = disabled)
    ):
        super().__init__()
        self.dynamic_token_config = dynamic_token_config
        self.plugin = plugin
        self.shuffle = shuffle
        self.seed = seed
        self.video_target_frames = int(video_target_frames)
        if self.video_target_frames <= 0:
            raise ValueError(f"video_target_frames must be positive, got {video_target_frames}")
        self.segment_t_size = int(segment_t_size)
        if self.segment_t_size <= 0:
            raise ValueError(f"segment_t_size must be positive, got {segment_t_size}")
        self.shard_interleave = max(1, int(shard_interleave))
        self.shuffle_buffer = max(0, int(shuffle_buffer))
        self._epoch = 0   # Increment on each __iter__ so shard order changes across epochs
        # An empty list processes everything; otherwise drop these tars by basename after rank/worker splitting.
        self.skip_shards = load_skip_shards(skip_shards)

        # Resolve shards
        if os.path.isfile(shards_path):
            with open(shards_path) as f:
                self.shards = [line.strip() for line in f if line.strip()]
        else:
            import glob
            self.shards = sorted(glob.glob(os.path.join(shards_path, "*.tar")))
        if not self.shards:
            raise ValueError(f"No tar shards found at {shards_path}")

        # Cross-rank shuffle discriminator: different datasets (train/video) get different
        # orders, while the same dataset has the same order on every rank. zlib.crc32 is
        # stable across processes (hash() is randomized).
        import zlib
        self._shards_key = zlib.crc32(os.path.abspath(shards_path).encode()) & 0xFFFF

        rank, world_size = _dist_info()
        if rank == 0 and world_size > 1 and len(self.shards) < world_size:
            import warnings
            warnings.warn(
                f"[WDS] shard count ({len(self.shards)}) is less than world_size ({world_size}); "
                f"some ranks receive no data after rank splitting: {shards_path}"
            )

    def _iter_shards(self):
        worker_info = None
        try:
            import torch.utils.data
            worker_info = torch.utils.data.get_worker_info()
        except Exception:
            pass

        # epoch_seed changes on each __iter__ to avoid the same shard order across epochs.
        # It must be identical across ranks (otherwise rank splitting no longer produces
        # disjoint partitions), so it can use only values shared by all ranks:
        # self.seed / self._epoch / shards_path. Do not use process-local addresses such as
        # id(self). _shards_key distinguishes train from video.
        epoch_seed = self.seed ^ self._epoch ^ self._shards_key
        self._epoch += 1

        if self.shuffle:
            import random
            rng = random.Random(epoch_seed)
            order = list(range(len(self.shards)))
            rng.shuffle(order)
        else:
            order = list(range(len(self.shards)))

        # Two-level split (matching webdataset split_by_node + split_by_worker):
        # split by rank first so each device gets disjoint shards, then split by worker.
        rank, world_size = _dist_info()
        if world_size > 1:
            order = order[rank::world_size]

        if worker_info is not None:
            wid = worker_info.id
            wnum = worker_info.num_workers
            order = order[wid::wnum]

        # Filter by the skip list after splitting: remaining shards stay on their original
        # (rank, worker) instead of being reassigned because a prefix was removed. An empty list skips filtering.
        if self.skip_shards:
            order = [
                idx for idx in order
                if os.path.basename(self.shards[idx]) not in self.skip_shards
            ]

        if not order:
            # No readable shard remains after two-level splitting, so this (rank, worker)
            # stream immediately raises StopIteration. Emit diagnostic information for the
            # training side, which otherwise appears as an empty video/image stream.
            import warnings
            warnings.warn(
                f"[WDS] no readable shard after splitting: rank={rank}/{world_size} "
                f"worker={worker_info.id if worker_info else '-'}/"
                f"{worker_info.num_workers if worker_info else '-'} "
                f"total shards={len(self.shards)}"
            )

        for idx in order:
            yield self.shards[idx]

    def _iter_raw_samples(self):
        """Interleave reads from multiple shards and yield raw samples one by one.

        Why interleave: shards are packed in blocks by data source, so samples in one tar
        have similar tasks. With sequential reads, each worker produces one task type for
        a while, and all workers on a rank cross shard boundaries at the same time. The
        global batch composition therefore switches as a whole every
        (shard_size * num_workers / samples per step) steps, causing stair-step loss
        changes. Opening K shards and polling them in turn multiplies the concurrent shard
        count by K, keeping the global batch closer to the dataset distribution and stable.

        Streaming tar reads ("r|*") do not cache the archive; K readers are only K sequential
        file descriptors and use almost no memory.
        """
        import logging

        shard_paths = self._iter_shards()

        def _open_next():
            for shard_path in shard_paths:
                # Shard-level fault tolerance: move to the next shard when a bad tar or
                # mount transient causes open to fail. Otherwise the exception kills the
                # DataLoader worker and interrupts the entire data stream.
                try:
                    return TarShardReader(shard_path)
                except Exception as e:
                    logging.warning(f"[WDS] skip bad shard {shard_path}: {e}")
                    continue
            return None

        active = []   # [[reader, iterator], ...]
        for _ in range(self.shard_interleave):
            reader = _open_next()
            if reader is None:
                break
            active.append([reader, iter(reader)])

        i = 0
        while active:
            i %= len(active)
            reader, it = active[i]
            try:
                sample = next(it)
            except StopIteration:
                sample = None
            except Exception as e:
                # I/O errors from extractfile/read (including EBADF) must not escape the
                # DataLoader worker; otherwise this rank exits while other ranks block in
                # all_reduce until the NCCL timeout.
                logging.warning(f"[WDS] skip rest of shard {reader.tar_path}: {e}")
                sample = None
            if sample is None:
                # This shard is exhausted (or failed): replace it with the next shard, or
                # reduce the active reader count when no new shard is available.
                reader.close()
                nxt = _open_next()
                if nxt is None:
                    active.pop(i)
                else:
                    active[i] = [nxt, iter(nxt)]
                continue
            yield sample
            i += 1

    def __iter__(self):
        import logging

        stream = self._iter_raw_samples()
        if self.shuffle and self.shuffle_buffer > 1:
            # Sample-level shuffle buffer (the standard webdataset `.shuffle(bufsize)` approach):
            # randomize the raw order within shards and further mix samples from interleaved readers.
            # Distinguish seeds by (epoch, rank, worker) so workers do not use the same permutation.
            import random
            rank, _ = _dist_info()
            worker_info = None
            try:
                import torch.utils.data
                worker_info = torch.utils.data.get_worker_info()
            except Exception:
                pass
            wid = worker_info.id if worker_info is not None else 0
            rng = random.Random(
                (self.seed ^ self._shards_key)
                + 1000003 * (self._epoch + 1)
                + 9176 * rank
                + wid
            )
            stream = _shuffle_buffered(stream, self.shuffle_buffer, rng)

        for sample in stream:
            try:
                processed = self._process_sample(sample)
            except Exception as e:
                logging.warning(
                    f"[WDS] skip bad sample key={sample.get('__key__')}: {e}"
                )
                continue
            if processed is None:
                continue
            yield processed

    def _process_sample(self, sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Convert a WDS sample to the format expected by the collator.

        Return None for samples that cannot be trained; the caller skips them.
        """
        import logging
        json_data = sample.get("json", {})
        messages = json_data.get("messages", [])
        modality = json_data.get("modality", None)

        # Infer modality from message contents when the JSON has no modality field
        if modality is None:
            has_video_placeholder = any(
                "<video>" in (msg.get("content") or "") for msg in messages
            )
            has_image_placeholder = any(
                "<image>" in (msg.get("content") or "") for msg in messages
            )
            if has_video_placeholder:
                modality = "video"
            elif has_image_placeholder or sample.get("images"):
                modality = "image"
            else:
                modality = "text"

        images = sample.get("images") if modality == "image" else None

        # Video-frame source: a jpg frame list or an mp4 byte stream
        if modality == "video":
            if sample.get("images"):
                videos = [sample["images"]]
            elif sample.get("mp4"):
                frames = _decode_mp4_frames(sample["mp4"], target_frames=self.video_target_frames)
                videos = [frames] if frames else None
            else:
                videos = None
        else:
            videos = None

        # For a video sample with no decoded frames (corrupt mp4 or both decord and cv2
        # failing), plugin expands <video> in messages into an <|image_pad|> block while
        # pixel_values is empty. The sample cannot learn visual information and would make
        # the collator compute loss on pad tokens, so discard it.
        if modality == "video" and not videos:
            logging.warning(
                f"[WDS] skip sample key={sample.get('__key__')}: modality=video "
                "but no frames could be decoded (corrupt mp4 or missing jpg frames)"
            )
            return None

        # visidx: precomputed video patch indices passed to the ViT for sparse token selection
        # shape: (n_vit_candidate,) int32, written as xxx.visidx.npy during dataset creation
        visidx = sample.get("visidx", None)  # np.ndarray or None

        # Compute num_segments
        if modality == "video" and videos is not None:
            num_segments = self.plugin.compute_video_segments(
                n_frames=len(videos[0]),
                segment_t_size=self.segment_t_size,
            ) if self.plugin else 1
        else:
            num_segments = 1

        # Determine whether this is an OCR sample (images only; videos ignore the OCR flag)
        is_ocr = False
        if modality == "image" and messages:
            is_ocr = is_ocr_sample(messages, min_confidence='low')

        # Sample the token plan
        if self.dynamic_token_config is not None:
            import numpy as np
            plan = self.dynamic_token_config.sample(
                num_segments=num_segments,
                modality=modality,
                is_ocr=is_ocr,
            )
        else:
            from .dynamic_strategy import TokenPlan
            plan = TokenPlan(strategy="query_and_vit", vit_per_seg=100, arrangement="interleave", num_segments=num_segments)

        # Use the plugin to expand <image>/<video> into a vision block (<|vision_start|>...<|vision_end|>);
        # the collator later adjusts the pad count according to token_plan
        if self.plugin is not None:
            messages, _ = self.plugin.process_messages(
                messages,
                images=images or [],
                videos=videos or [],
            )

        return {
            "messages": messages,
            "images": images,
            "videos": videos,
            "visidx": visidx,
            "token_plan": plan,
            "modality": modality,
            "is_ocr": is_ocr,
        }
