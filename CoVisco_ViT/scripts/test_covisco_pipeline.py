#!/usr/bin/env python3
"""
Generate a fake dataset (image + video) and run a smoke test for the CoVisco SigLIP pipeline.

Usage:
    # Standard smoke test (same resolution for images and videos)
    python scripts/test_covisco_pipeline.py [--num-samples 64] [--epochs 1]

    # Test separate image/video resolutions
    python scripts/test_covisco_pipeline.py --test-video-size

This script:
1. Creates a temporary directory with fake images, videos, embeddings, and TSV manifests
2. Launches main_covisco.main() with minimal settings (small batch, few steps)
3. Verifies that training + evaluation complete without errors
4. Cleans up the temporary directory on exit

When --test-video-size is given, a second training run is performed with
--force-image-size 32 --video-size 16, verifying that:
  a) EmbeddingDataset applies the correct per-modality transform
  b) The full training loop runs end-to-end with mismatched resolutions
"""

import argparse
import os
import sys
import tempfile
import shutil

import numpy as np
import torch
from PIL import Image

# Add src directory to Python path
_current_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_current_dir)
sys.path.insert(0, os.path.join(_parent_dir, 'src'))


def _make_fake_video(path, num_frames=8, height=224, width=224):
    """Create a minimal AVI video file using OpenCV."""
    import cv2
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(path, fourcc, 10, (width, height))
    for _ in range(num_frames):
        frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()


def _save_normalized_embedding(path, dim):
    """Save a random L2-normalized embedding to a .pt file."""
    emb = torch.randn(dim)
    emb = emb / emb.norm()
    torch.save(emb, path)


def _save_visible_indices(path, L=3600, max_idx=2048):
    """Save a fake visible_indices tensor of shape [L] with values in [0, max_idx]."""
    indices = torch.randint(0, max_idx + 1, (L,), dtype=torch.long)
    torch.save(indices, path)


def create_fake_dataset(root_dir, num_samples=64, image_size=224, embed_dim=1024,
                     image_embed_dim=1536, image_caption_embed_dim=1536, video_caption_embed_dim=1024,
                     num_video_frames=64):
    """
    Create fake image + video embedding datasets for CoVisco SigLIP training.

    Directory layout:
        root_dir/
            images/              # fake JPEG images
            videos/              # fake AVI videos
            embeddings/          # fake .pt embedding files
            image_data.tsv       # manifest for image training data
            video_data.tsv       # manifest for video training data
            imagenet_val/        # fake ImageNet-style val set

    Args:
        root_dir: Root directory for fake data
        num_samples: Number of fake samples per modality
        image_size: Size of fake images
        embed_dim: Deprecated (kept for backward compatibility)
        image_embed_dim: Dimension of pre-extracted image embeddings
        image_caption_embed_dim: Dimension of pre-extracted image caption embeddings
        video_caption_embed_dim: Dimension of pre-extracted video caption embeddings
        num_video_frames: Number of frames per fake video
    """
    img_dir = os.path.join(root_dir, "images")
    vid_dir = os.path.join(root_dir, "videos")
    emb_dir = os.path.join(root_dir, "embeddings")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(vid_dir, exist_ok=True)
    os.makedirs(emb_dir, exist_ok=True)

    # ---- Image data ----
    image_rows = []
    for i in range(num_samples):
        # fake image
        img_path = os.path.join(img_dir, f"img_{i:04d}.jpg")
        img = Image.fromarray(
            np.random.randint(0, 255, (image_size, image_size, 3), dtype=np.uint8)
        )
        img.save(img_path)

        # image embedding (image-image contrast)
        img_emb_name = f"img_emb_{i:04d}.pt"
        _save_normalized_embedding(os.path.join(emb_dir, img_emb_name), image_embed_dim)

        # image caption embedding (image-caption contrast)
        img_cap_emb_name = f"img_cap_emb_{i:04d}.pt"
        _save_normalized_embedding(os.path.join(emb_dir, img_cap_emb_name), image_caption_embed_dim)

        image_rows.append(f"{img_path}\t{img_emb_name}\t{img_cap_emb_name}\t\timage")

    image_tsv = os.path.join(root_dir, "image_data.tsv")
    header = "filepath\timage_emb\timage_caption_emb\tvideo_caption_emb\tmodality"
    with open(image_tsv, "w") as f:
        f.write(header + "\n")
        f.write("\n".join(image_rows) + "\n")

    # ---- Video data ----
    video_rows = []
    for i in range(num_samples):
        # fake video
        vid_path = os.path.join(vid_dir, f"vid_{i:04d}.avi")
        _make_fake_video(vid_path, num_frames=num_video_frames, height=image_size, width=image_size)

        # video caption embedding (video-caption contrast)
        vid_cap_emb_name = f"vid_cap_emb_{i:04d}.pt"
        _save_normalized_embedding(os.path.join(emb_dir, vid_cap_emb_name), video_caption_embed_dim)

        # visible_indices: shape [4096], values in [0, 32767]
        vis_idx_name = f"vis_idx_{i:04d}.pt"
        _save_visible_indices(os.path.join(emb_dir, vis_idx_name))

        video_rows.append(f"{vid_path}\t\t\t{vid_cap_emb_name}\tvideo\t{vis_idx_name}")

    video_tsv = os.path.join(root_dir, "video_data.tsv")
    video_header = "filepath\timage_emb\timage_caption_emb\tvideo_caption_emb\tmodality\tvisible_indices"
    with open(video_tsv, "w") as f:
        f.write(video_header + "\n")
        f.write("\n".join(video_rows) + "\n")

    # ---- Fake ImageNet val set (2 classes) ----
    imagenet_val_dir = os.path.join(root_dir, "imagenet_val")
    num_classes = 2
    samples_per_class = max(num_samples // num_classes, 2)
    for cls_idx in range(num_classes):
        cls_dir = os.path.join(imagenet_val_dir, f"class_{cls_idx:04d}")
        os.makedirs(cls_dir, exist_ok=True)
        for j in range(samples_per_class):
            img = Image.fromarray(
                np.random.randint(0, 255, (image_size, image_size, 3), dtype=np.uint8)
            )
            img.save(os.path.join(cls_dir, f"img_{j:04d}.jpg"))

    print(f"Created fake dataset at {root_dir}")
    print(f"  - {num_samples} training images  ({image_tsv})")
    print(f"  - {num_samples} training videos  ({video_tsv}), {num_video_frames} frames each")
    print(f"  - {num_classes} ImageNet-val classes x {samples_per_class} images")
    return image_tsv, video_tsv, emb_dir, imagenet_val_dir


def _test_embedding_dataset_resolutions(image_tsv, video_tsv, emb_dir, image_size, video_size):
    """
    Unit-level check: verify that EmbeddingDataset returns tensors at the expected
    spatial resolution when separate image/video transforms are provided.
    """
    from open_clip.transform import image_transform
    from open_clip_train.data import EmbeddingDataset

    img_transform = image_transform(image_size=image_size, is_train=False)
    vid_transform = image_transform(image_size=video_size, is_train=False)

    # --- Image dataset: pixel_values should be (C, image_size, image_size) ---
    img_ds = EmbeddingDataset(
        csv_path=image_tsv,
        transforms=img_transform,
        modality="image",
        embedding_dir=emb_dir,
        video_transforms=vid_transform,   # should NOT be used for images
    )
    sample = img_ds[0]
    pv = sample["pixel_values"]
    assert pv.shape == (3, image_size, image_size), (
        f"Image pixel_values shape mismatch: expected (3, {image_size}, {image_size}), got {tuple(pv.shape)}"
    )
    print(f"  [OK] Image sample shape: {tuple(pv.shape)}")

    # --- Video dataset: each frame should be (C, video_size, video_size) ---
    vid_ds = EmbeddingDataset(
        csv_path=video_tsv,
        transforms=img_transform,         # image transform (wrong size) — should NOT be used for video frames
        modality="video",
        embedding_dir=emb_dir,
        video_transforms=vid_transform,   # should be used for video frames
    )
    sample = vid_ds[0]
    pv = sample["pixel_values"]           # shape: ( C,T, H, W)
    assert pv.ndim == 4, f"Expected 4-D video tensor, got shape {tuple(pv.shape)}"
    assert pv.shape[0] == 3, f"Expected 3 channels, got {pv.shape[0]}"
    assert pv.shape[2] == video_size and pv.shape[3] == video_size, (
        f"Video frame shape mismatch: expected (*, 3, {video_size}, {video_size}), got {tuple(pv.shape)}"
    )
    print(f"  [OK] Video sample shape:  {tuple(pv.shape)}")

    # visible_indices should be a 1-D LongTensor of length 4096 with values in [0, 32767]
    vi = sample["visible_indices"]
    assert vi is not None, "Expected visible_indices to be a tensor for video samples, got None"
    assert vi.ndim == 1 and vi.shape[0] == 4096, (
        f"visible_indices shape mismatch: expected [4096], got {tuple(vi.shape)}"
    )
    assert vi.dtype == torch.long, f"visible_indices dtype should be long, got {vi.dtype}"
    assert vi.min() >= 0 and vi.max() <= 32767, (
        f"visible_indices values out of range [0, 32767]: min={vi.min()}, max={vi.max()}"
    )
    print(f"  [OK] Video visible_indices shape: {tuple(vi.shape)}, dtype: {vi.dtype}")


def _test_video_size_transform(
    image_tsv, video_tsv, emb_dir, imagenet_val_dir,
    batch_size, logs_dir,
    image_size=32, video_size=16,
):
    """
    Full end-to-end smoke test with --force-image-size and --video-size set to
    different values.  Verifies both the unit-level tensor shape and the training loop.
    """
    print("\n" + "=" * 60)
    print(f"Testing separate image/video resolution")
    print(f"  image_size={image_size}, video_size={video_size}")
    print("=" * 60)

    # 1. Unit check: tensor shapes from EmbeddingDataset
    print("\n[1/2] Checking EmbeddingDataset tensor shapes ...")
    _test_embedding_dataset_resolutions(
        image_tsv=image_tsv,
        video_tsv=video_tsv,
        emb_dir=emb_dir,
        image_size=image_size,
        video_size=video_size,
    )

    # 2. End-to-end training run with --video-size
    print("\n[2/2] Running training loop with --force-image-size and --video-size ...")
    from open_clip_train.main_covisco import main as train_main
    train_args = [
        "--dataset-type", "embedding",
        "--image-data", image_tsv,
        "--video-data", video_tsv,
        "--embedding-dir", emb_dir,
        "--image-embed-dim", str(1536),
        "--image-caption-embed-dim", str(1536),
        "--video-caption-embed-dim", str(1024),
        "--model", "CoVisco-L-14",
        "--force-image-size", str(image_size),
        "--video-size", str(224),
        "--use-reconstruction",
        "--reconstruction-weight", "0.1",
        "--decoder-layers", "2",
        "--decoder-num-image-queries", "64",
        "--decoder-num-video-queries", "64",
        "--decoder-num-heads", "8",
        "--init-logit-scale-image-caption", "2.659",
        "--init-logit-scale-image-image", "2.659",
        "--init-logit-scale-video-caption", "2.659",
        "--init-logit-bias-image-caption", "10.0",
        "--init-logit-bias-image-image", "10.0",
        "--init-logit-bias-video-caption", "10.0",
        "--imagenet-val", imagenet_val_dir,
        "--zeroshot-frequency", "1",
        "--batch-size", str(batch_size),
        "--epochs", "1",
        "--lr", "1e-4",
        "--wd", "0.0",
        "--warmup", "0",
        "--workers", "0",
        "--precision", "amp_bfloat16",
        "--logs", logs_dir,
        "--report-to", "",
        "--save-frequency", "0",
        "--beta1", "0.9",
        "--beta2", "0.95",
        "--eps", "1e-8",
        "--log-every-n-steps", "1",
        "--lr-scheduler", "const",
        "--name", "smoke_test_video_size",
    ]
    train_main(train_args)

    print("\n" + "=" * 60)
    print("Separate image/video resolution test PASSED!")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Smoke test for CoVisco SigLIP pipeline")
    parser.add_argument("--num-samples", type=int, default=64, help="Number of fake samples per modality")
    parser.add_argument("--num-video-frames", type=int, default=32, help="Frames per fake video")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for testing")
    parser.add_argument("--embed-dim", type=int, default=512, help="Embedding dimension")
    parser.add_argument("--keep-tmp", action="store_true", help="Don't clean up temp dir")
    parser.add_argument(
        "--test-video-size", action="store_true",
        help="Run an extra test with --force-image-size 32 --video-size 16 to verify "
             "that separate image/video transforms produce the correct frame resolution",
    )
    cli_args = parser.parse_args()

    tmp_dir = tempfile.mkdtemp(prefix="covisco_test_")
    logs_dir = os.path.join(tmp_dir, "logs")

    try:
        image_tsv, video_tsv, emb_dir, imagenet_val_dir = create_fake_dataset(
            tmp_dir,
            num_samples=cli_args.num_samples,
            image_size=448,
            image_embed_dim=1536,
            image_caption_embed_dim=1536,
            video_caption_embed_dim=1024,
            num_video_frames=12,
        )

        train_args = [
            "--dataset-type", "embedding",
            "--accum-freq", "1",
            # Image + Video data
            "--image-data", image_tsv,
            "--video-data", video_tsv,
            "--video-size", "224",
            "--force-image-size","224",
            "--video-batch-size", "4",
            "--image-batch-size", "36",
            "--grad-checkpointing",
            "--embedding-dir", emb_dir,
            # Embedding dimensions
            "--image-embed-dim", str(1536),
            "--image-caption-embed-dim", str(1536),
            "--video-caption-embed-dim", str(1024),
            # Model
            "--model", "CoVisco-L-14",
            # Reconstruction decoder
            "--use-reconstruction",
            "--reconstruction-weight", "0.1",
            "--decoder-layers", "8",
            "--decoder-num-image-queries", "256",
            "--decoder-num-video-queries", "256",
            "--decoder-num-heads", "8",
            # Logit scale / bias init
            "--init-logit-scale-image-caption", "2.659",
            "--init-logit-scale-image-image", "2.659",
            "--init-logit-scale-video-caption", "2.659",
            "--init-logit-bias-image-caption", "-10.0",
            "--init-logit-bias-image-image", "-10.0",
            "--init-logit-bias-video-caption", "-10.0",
            # ImageNet zero-shot eval
            "--imagenet-val", imagenet_val_dir,
            "--zeroshot-frequency", "1",
            # Training hyperparams (minimal for smoke test)
            # "--batch-size", str(cli_args.batch_size),
            "--epochs", str(cli_args.epochs),
            "--lr", "1e-3",
            "--wd", "0.0",
            "--warmup", "0",
            "--workers", "0",
            "--precision", "amp_bfloat16",
            "--logs", logs_dir,
            "--report-to", "",
            "--save-frequency", "1",
            "--beta1", "0.9",
            "--beta2", "0.95",
            "--eps", "1e-8",
            "--log-every-n-steps", "1",
            "--lr-scheduler", "const",
            "--name", "smoke_test",
        ]

        print("\n" + "=" * 60)
        print("Starting CoVisco SigLIP pipeline smoke test")
        print("  (image + video training, ImageNet zero-shot eval)")
        print("=" * 60)

        from open_clip_train.main_covisco import main as train_main
        train_main(train_args)

        print("\n" + "=" * 60)
        print("Smoke test PASSED!")
        print("=" * 60)

        # ------------------------------------------------------------------ #
        # Optional: separate image/video resolution test                      #
        # ------------------------------------------------------------------ #
        if cli_args.test_video_size:
            _test_video_size_transform(
                image_tsv=image_tsv,
                video_tsv=video_tsv,
                emb_dir=emb_dir,
                imagenet_val_dir=imagenet_val_dir,
                batch_size=cli_args.batch_size,
                logs_dir=logs_dir,
            )

    except Exception as e:
        print(f"\nSmoke test FAILED with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    finally:
        if cli_args.keep_tmp:
            print(f"\nTemp directory kept at: {tmp_dir}")
        else:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            print(f"\nTemp directory cleaned up.")


if __name__ == "__main__":
    main()
