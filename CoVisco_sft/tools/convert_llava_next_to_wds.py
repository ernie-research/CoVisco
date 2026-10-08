"""convert_llava_next_to_wds.py

Convert the LLaVA-NeXT-Data (SFT) parquet dataset into the WebDataset tar shard
format required by CoVisco_sft.

Source format (parquet fields):
  - id:            str
  - conversations: list of {"from": "human"/"gpt", "value": str}
  - data_source:   str
  - image:         {"bytes": bytes, "path": str} or None

Output format (tar shards):
  <dst_dir>/train/shard_{N:06d}.tar   each shard holds shard_size samples
  <dst_dir>/manifest.json
  <dst_dir>/progress.json             resumable conversion state

  Each sample in a tar:
    sample_{global_idx:010d}.jpg
    sample_{global_idx:010d}.json
      {"messages": [...], "modality": "image"/"text", "source_id": "..."}

  conversations from/value -> messages role/content:
    human -> user,  gpt -> assistant

Usage:
  python tools/convert_llava_next_to_wds.py \\
      --input_dir /path/to/LLaVA-NeXT-Data/data \\
      --output_dir /path/to/output_wds \\
      --shard_size 1000 \\
      --workers 1

Optional:
  --start_parquet  start from the N-th parquet file (0-based, for resuming)
  --end_parquet    end before the N-th parquet file (exclusive)
"""

from __future__ import annotations

import argparse
import io
import json
import os
import tarfile
import time
from pathlib import Path


# ────────────────────────────────────────────────────────────────────────────
# Format-conversion utilities
# ────────────────────────────────────────────────────────────────────────────

def _from_to_role(from_str: str) -> str:
    return "assistant" if from_str == "gpt" else "user"


def _convert_conversations(conversations) -> list:
    if hasattr(conversations, "tolist"):
        conversations = conversations.tolist()
    return [
        {"role": _from_to_role(t.get("from", "human")), "content": t.get("value", "")}
        for t in conversations
    ]


def _has_image_placeholder(messages: list) -> bool:
    return any("<image>" in (m.get("content") or "") for m in messages)


# ────────────────────────────────────────────────────────────────────────────
# Resumable progress (written to both local /tmp and the destination directory)
# ────────────────────────────────────────────────────────────────────────────

def _progress_local_path(dst_dir: Path) -> Path:
    safe = str(dst_dir).replace("/", "_").strip("_")
    return Path(f"/tmp/wds_progress_{safe}.json")


def _load_progress(dst_dir: Path) -> tuple[set[str], int, int]:
    for p in [_progress_local_path(dst_dir), dst_dir / "progress.json"]:
        if p.exists():
            try:
                data = json.loads(p.read_text())
                done = data.get("done_parquets", [])
                if done:
                    return set(done), data.get("shard_count", 0), data.get("total_samples", 0)
            except Exception:
                continue
    return set(), 0, 0


def _save_progress(dst_dir: Path, done_parquets: list[str], shard_count: int, total_samples: int):
    content = json.dumps({
        "done_parquets": done_parquets,
        "shard_count": shard_count,
        "total_samples": total_samples,
    }, indent=2)
    _progress_local_path(dst_dir).write_text(content)
    dst_pf = dst_dir / "progress.json"
    try:
        if dst_pf.exists():
            dst_pf.unlink()
        dst_pf.write_text(content)
    except OSError:
        pass


# ────────────────────────────────────────────────────────────────────────────
# Shard writing (naming convention: shard_{N:06d}.tar)
# ────────────────────────────────────────────────────────────────────────────

def _add_to_tar(tf: tarfile.TarFile, key: str, jpg_bytes: bytes, json_bytes: bytes):
    for name, data in [(f"{key}.jpg", jpg_bytes), (f"{key}.json", json_bytes)]:
        info = tarfile.TarInfo(name=name)
        info.size = len(data)
        tf.addfile(info, io.BytesIO(data))


class ShardWriter:
    """Buffer the whole shard in memory and write it out at once, avoiding tarfile seek issues on network filesystems."""

    def __init__(self, out_dir: Path, shard_size: int,
                 start_shard_idx: int = 0, start_global_idx: int = 0):
        out_dir.mkdir(parents=True, exist_ok=True)
        self.out_dir = out_dir
        self.shard_size = shard_size
        self._shard_idx = start_shard_idx
        self._sample_in_shard = 0
        self._global_idx = start_global_idx
        self._buf: io.BytesIO | None = None
        self._tf: tarfile.TarFile | None = None
        self._completed: list[Path] = []
        self._open_shard()

    def _open_shard(self):
        if self._tf is not None:
            self._flush()
        self._buf = io.BytesIO()
        self._tf = tarfile.open(fileobj=self._buf, mode="w")
        self._sample_in_shard = 0
        self._shard_idx += 1

    def _flush(self):
        self._tf.close()
        data = self._buf.getvalue()
        dest = self.out_dir / f"shard_{self._shard_idx - 1:06d}.tar"
        for attempt in range(30):
            try:
                if dest.exists():
                    dest.unlink()
                with open(str(dest), "wb") as f:
                    f.write(data)
                break
            except OSError:
                time.sleep(2)
        else:
            raise RuntimeError(f"Cannot write {dest} after 30 retries")
        self._completed.append(dest)
        self._buf = None
        self._tf = None

    def write(self, jpg_bytes: bytes | None, messages: list, modality: str, source_id: str):
        json_obj = {"messages": messages, "modality": modality, "source_id": source_id}
        json_bytes = json.dumps(json_obj, ensure_ascii=False).encode("utf-8")
        key = f"sample_{self._global_idx:010d}"
        if modality == "image" and jpg_bytes:
            _add_to_tar(self._tf, key, jpg_bytes, json_bytes)
        else:
            # Text-only samples only write the json
            info = tarfile.TarInfo(name=f"{key}.json")
            info.size = len(json_bytes)
            self._tf.addfile(info, io.BytesIO(json_bytes))
        self._sample_in_shard += 1
        self._global_idx += 1
        if self._sample_in_shard >= self.shard_size:
            self._open_shard()

    def close(self) -> list[Path]:
        if self._tf is not None and self._sample_in_shard > 0:
            self._flush()
        elif self._tf is not None:
            self._tf.close()
        return self._completed

    @property
    def total(self) -> int:
        return self._global_idx

    @property
    def shard_count(self) -> int:
        return self._shard_idx


# ────────────────────────────────────────────────────────────────────────────
# Main conversion logic
# ────────────────────────────────────────────────────────────────────────────

def convert(input_dir: Path, output_dir: Path, shard_size: int,
            start_parquet: int = 0, end_parquet: int | None = None):
    import pandas as pd

    parquet_files = sorted(input_dir.glob("train-*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {input_dir}")

    parquet_files = parquet_files[start_parquet:end_parquet]
    print(f"Found {len(parquet_files)} parquet files to process")

    train_dir = output_dir / "train"

    done_set, start_shard_idx, start_global_idx = _load_progress(output_dir)
    if done_set:
        print(f"Resuming: {len(done_set)} parquets done, "
              f"{start_shard_idx} shards, {start_global_idx} samples")

    writer = ShardWriter(train_dir, shard_size,
                         start_shard_idx=start_shard_idx,
                         start_global_idx=start_global_idx)

    done_parquets = list(done_set)
    total_skip = 0

    for i, pq_path in enumerate(parquet_files):
        if str(pq_path) in done_set:
            print(f"  [{i+1}/{len(parquet_files)}] skip (already done): {pq_path.name}")
            continue

        try:
            df = pd.read_parquet(str(pq_path), engine="pyarrow")
        except Exception as e:
            print(f"  [{i+1}/{len(parquet_files)}] WARN skip {pq_path.name}: {e}")
            continue

        n_written = 0
        n_skip = 0
        for _, row in df.iterrows():
            conversations = row.get("conversations")
            if conversations is None:
                n_skip += 1
                continue

            messages = _convert_conversations(conversations)
            has_img = _has_image_placeholder(messages)

            img_field = row.get("image")
            jpg_bytes = None
            if isinstance(img_field, dict):
                jpg_bytes = img_field.get("bytes")
            elif isinstance(img_field, bytes):
                jpg_bytes = img_field

            if has_img and not jpg_bytes:
                n_skip += 1
                continue

            modality = "image" if (has_img and jpg_bytes) else "text"
            source_id = str(row.get("id") or "")

            writer.write(jpg_bytes, messages, modality, source_id)
            n_written += 1

        total_skip += n_skip
        done_parquets.append(str(pq_path))
        _save_progress(output_dir, done_parquets, writer.shard_count, writer.total)
        print(f"  [{i+1}/{len(parquet_files)}] {pq_path.name}: "
              f"written={n_written}, skip={n_skip}, total_so_far={writer.total}")

    all_shards = writer.close()
    print(f"\nDone. total_samples={writer.total}, total_shards={len(list(train_dir.glob('shard_*.tar')))}, "
          f"total_skipped={total_skip}")

    # Write manifest.json (aligned with tools_convert_parquet_to_wds.py)
    manifest = {
        "src_root": str(input_dir),
        "total_samples": writer.total,
        "train_shards": len(list(train_dir.glob("shard_*.tar"))),
        "val_shards": 0,
        "shard_size": shard_size,
    }
    mf = output_dir / "manifest.json"
    if mf.exists():
        mf.unlink()
    mf.write_text(json.dumps(manifest, indent=2))

    # Clear the progress files (all done)
    for p in [_progress_local_path(output_dir), output_dir / "progress.json"]:
        try:
            if p.exists():
                p.unlink()
        except OSError:
            pass

    print(f"manifest: {mf}")


# ────────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Convert LLaVA-NeXT-Data parquet to WebDataset tar shards")
    parser.add_argument(
        "--input_dir",
        required=True,
        help="directory containing train-*.parquet",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="output root directory (the script creates a train/ subdirectory under it)",
    )
    parser.add_argument("--shard_size", type=int, default=2000, help="samples per tar (default 2000)")
    parser.add_argument("--workers", type=int, default=32, help="not used yet, kept for interface compatibility")
    parser.add_argument("--start_parquet", type=int, default=0, help="start from the N-th parquet (0-based)")
    parser.add_argument("--end_parquet", type=int, default=None, help="end before the N-th parquet (exclusive)")
    args = parser.parse_args()

    convert(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        shard_size=args.shard_size,
        start_parquet=args.start_parquet,
        end_parquet=args.end_parquet,
    )


if __name__ == "__main__":
    main()
