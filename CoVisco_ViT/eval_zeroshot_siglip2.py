"""
Zero-shot evaluation for the CoVisco SigLIP dual-encoder using the SigLIP2
teacher text encoder (the ``to_image_caption`` branch).

Image side : CoViscoModel, output key 'to_image_caption' (dim 1536,
             already L2-normalized inside the model).
Text side  : open_clip 'ViT-gopt-16-SigLIP2-384' (pretrained='webli'), encoded
             with the same Gemma tokenizer + 'canonicalize' cleaning used at
             training time (see eval_imagenet_zeroshot.py).

Two task families:
  classification  top-1 / top-5 zero-shot classification
      - imagenet1k  (local class-sorted WebDataset, or clip-benchmark WDS)
      - imagenetv2  (clip-benchmark/wds_imagenetv2)
      - objectnet   (clip-benchmark/wds_objectnet, 113-class protocol)
      - imagenet_real  (ImageNet val in ORIGINAL filename order + ReaL
                        multi-label labels; needs --imagenet_real_dir)
  retrieval       text<->image Recall@{1,5,10}
      - coco    (clip-benchmark/wds_mscoco_captions, Karpathy 5k test)
      - flickr  (clip-benchmark/wds_flickr30k, 1k test)
      - xm3600  (floschne/xm3600, Crossmodal-3600, all 36 languages, averaged)

classnames / prompt templates for the classification datasets are pulled
directly from the clip-benchmark WDS repos (classnames.txt +
zeroshot_classification_templates.txt), so ObjectNet's 113-class name/template
set is used as-is.

Usage (see run_zeroshot_siglip2.sh):
    python eval_zeroshot_siglip2.py --tasks classification retrieval \
        --ckpt /path/step.pt --image_size 224
"""

import argparse
import io
import json
import os
import sys
import tarfile
from collections import defaultdict

import braceexpand
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------
CLS_HF = {
    'imagenet1k': 'clip-benchmark/wds_imagenet1k',
    'imagenetv2': 'clip-benchmark/wds_imagenetv2',
    'objectnet':  'clip-benchmark/wds_objectnet',
}
RET_HF = {
    'coco':   'clip-benchmark/wds_mscoco_captions',
    'flickr': 'clip-benchmark/wds_flickr30k',
}
REAL_JSON_URL = (
    'https://raw.githubusercontent.com/google-research/'
    'reassessed-imagenet/master/real.json'
)
XM3600_REPO = 'floschne/xm3600'
XM3600_LANGS = [
    'ar', 'bn', 'cs', 'da', 'de', 'el', 'en', 'es', 'fa', 'fi', 'fil', 'fr',
    'hi', 'hr', 'hu', 'id', 'it', 'he', 'ja', 'ko', 'mi', 'nl', 'no', 'pl',
    'pt', 'quz', 'ro', 'ru', 'sv', 'sw', 'te', 'th', 'tr', 'uk', 'vi', 'zh',
]


# ---------------------------------------------------------------------------
# Model loading  (mirrors eval_imagenet_zeroshot.py / eval_mmeb_..._siglip2.py)
# ---------------------------------------------------------------------------
def load_vision_model(ckpt_path, video_caption_embed_dim, image_size, device):
    from open_clip.covisco_model import CoViscoModel
    from open_clip.covisco_vit import (
        CoViscoEncoderConfig,
    )
    config = CoViscoEncoderConfig(
        hidden_size=1024, num_hidden_layers=24, num_attention_heads=16,
        num_channels=3, image_size=image_size, patch_size=14,
        num_query_per_seg=100, segment_t_size=32,
        use_head=True, output_dim=1024,
    )
    model = CoViscoModel(
        covisco_config=config,
        image_embed_dim=1536,
        image_caption_embed_dim=1536,   # matches SigLIP2-gopt-384 text embed dim
        video_caption_embed_dim=video_caption_embed_dim,
        use_reconstruction=False,
    )
    if not ckpt_path:
        raise ValueError(
            "no checkpoint provided: pass --ckpt /path/to/checkpoint.pt "
            "(or use eval_zeroshot_siglip2_hf.py to load the released HF encoder)"
        )
    sd = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    sd = sd.get('state_dict', sd)
    sd = {k.replace('module.', '', 1): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"[WARN] {len(missing)} model params missing from checkpoint, "
              f"left at RANDOM init: {missing[:10]}")
    if unexpected:
        print(f"[WARN] {len(unexpected)} unexpected checkpoint keys ignored: {unexpected[:10]}")
    return model.eval().to(device)


class _GemmaTokWrapper:
    """Callable wrapper around HF AutoTokenizer mimicking open_clip HFTokenizer.

    Applies the same 'canonicalize' cleaning used at training time so text
    embeddings match training (SigLIP2 tokenization is case/punctuation
    sensitive). Returns a LongTensor of input_ids (N, context_length).
    """

    def __init__(self, tokenizer, context_length=64):
        self.tokenizer = tokenizer
        self.context_length = context_length
        from open_clip.tokenizer import _clean_canonicalize
        self.clean_fn = _clean_canonicalize

    def __call__(self, texts, context_length=None):
        if isinstance(texts, str):
            texts = [texts]
        ctx = context_length or self.context_length
        texts = [self.clean_fn(t) for t in texts]
        enc = self.tokenizer(
            texts, return_tensors='pt',
            max_length=ctx, padding='max_length', truncation=True,
        )
        return enc.input_ids


def load_text_model(device, siglip2_dir=None):
    """Load ViT-gopt-16-SigLIP2-384 teacher + Gemma tokenizer wrapper.

    If ``siglip2_dir`` is given and contains the open_clip weights, load them
    from local disk (avoids re-downloading to the HF cache); otherwise fall
    back to open_clip's pretrained='webli'.
    """
    from open_clip import create_model
    from transformers import AutoTokenizer

    local_ckpt = None
    tok_src = 'timm/ViT-gopt-16-SigLIP2-384'
    if siglip2_dir and os.path.isdir(siglip2_dir):
        for fname in ('open_clip_model.safetensors', 'open_clip_pytorch_model.bin'):
            cand = os.path.join(siglip2_dir, fname)
            if os.path.exists(cand):
                local_ckpt = cand
                break
        if os.path.exists(os.path.join(siglip2_dir, 'tokenizer.json')):
            tok_src = siglip2_dir

    if local_ckpt:
        print(f"Loading teacher text encoder from local: {local_ckpt}")
        teacher = create_model('ViT-gopt-16-SigLIP2-384', pretrained=local_ckpt, device=device)
    else:
        print("Loading teacher text encoder (ViT-gopt-16-SigLIP2-384, pretrained=webli) ...")
        teacher = create_model('ViT-gopt-16-SigLIP2-384', pretrained='webli', device=device)
    teacher.eval()
    hf_tok = AutoTokenizer.from_pretrained(tok_src, use_fast=True)
    tokenizer = _GemmaTokWrapper(hf_tok, context_length=64)
    return teacher, tokenizer


def build_preprocess(image_size):
    """CoVisco-L-14 val transform (matches training-time transform)."""
    from open_clip import create_model_and_transforms
    _, _, preprocess_val = create_model_and_transforms(
        'CoVisco-L-14', pretrained=None,
        load_weights=False, force_image_size=image_size,
    )
    return preprocess_val


# ViT patch size (must match the checkpoint config). Native-resolution inputs
# are rounded to a multiple of this so the encoder's patch grid drops no pixels.
_PATCH_SIZE = 14


def build_native_preprocess(patch_size=_PATCH_SIZE, max_side=1400, min_side=None):
    """Aspect-ratio-preserving preprocessing at (near-)native resolution.

    Unlike `build_preprocess` (shortest-edge Resize + CenterCrop to a fixed
    square, which distorts/crops away detail), this keeps the original aspect
    ratio and only:
      * scales so the long side <= `max_side` (memory cap),
      * rounds each side to a multiple of `patch_size` (>= `min_side`).

    The CoVisco vision tower is RoPE-only (no learned position table) and
    derives its patch grid as h=H//patch_size, w=W//patch_size at runtime, so it
    accepts any such H x W (including non-square). A batch is a single dense
    tensor with no resolution packing, so callers must encode one image at a
    time -- main() forces batch size 1 under --native_resolution. Uses the same
    OpenAI-CLIP normalization as build_preprocess.
    """
    from torchvision import transforms
    min_side = min_side or patch_size
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                     (0.26862954, 0.26130258, 0.27577711))

    def _round(x):
        return max(min_side, int(round(x / patch_size)) * patch_size)

    def preprocess(img):
        w, h = img.size
        scale = min(1.0, max_side / max(w, h))
        nw, nh = _round(w * scale), _round(h * scale)
        img = img.resize((nw, nh), Image.BICUBIC)
        return normalize(to_tensor(img))

    return preprocess


# ---------------------------------------------------------------------------
# HuggingFace helpers
# ---------------------------------------------------------------------------
def hf_wds_shards(repo, split='test'):
    """Resolve + download all .tar shards of a clip-benchmark WDS split."""
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi()
    files = api.list_repo_files(repo, repo_type='dataset')
    tars = sorted(f for f in files if f.endswith('.tar') and f.split('/')[0] == split)
    if not tars:
        tars = sorted(f for f in files if f.endswith('.tar'))
    local = [hf_hub_download(repo, t, repo_type='dataset')
             for t in tqdm(tars, desc=f"  dl {repo.split('/')[-1]}")]
    return local


def hf_text_lines(repo, name):
    from huggingface_hub import hf_hub_download
    return open(hf_hub_download(repo, name, repo_type='dataset')).read().splitlines()


# ---------------------------------------------------------------------------
# Zero-shot classifier
# ---------------------------------------------------------------------------
def build_classifier(teacher, tokenizer, classnames, templates, device):
    from open_clip import build_zero_shot_classifier
    # clip-benchmark templates use the named placeholder '{c}', but open_clip's
    # build_zero_shot_classifier applies str.format(c) positionally ('{}').
    # Normalize '{c}' -> '{}' so template.format(classname) works.
    templates = [t.replace('{c}', '{}') for t in templates]
    dev_type = torch.device(device).type
    with torch.autocast(device_type=dev_type, dtype=torch.bfloat16):
        return build_zero_shot_classifier(
            teacher, tokenizer=tokenizer,
            classnames=classnames, templates=templates,
            num_classes_per_batch=10, device=device, use_tqdm=True,
        )  # (embed_dim, num_classes)


# ---------------------------------------------------------------------------
# Classification data + loop
# ---------------------------------------------------------------------------
def build_cls_loader(shards, batch_size, num_workers, preprocess):
    import webdataset as wds

    def _cont(exn):
        print(f"[WARN] wds handler: {repr(exn)}")
        return True

    pipeline = [
        wds.SimpleShardList(shards),
        wds.split_by_worker,
        wds.tarfile_to_samples(handler=_cont),
        wds.decode("pilrgb", handler=_cont),
        wds.rename(image="jpg;png;jpeg;webp", target="cls"),
        wds.map_dict(image=preprocess, target=lambda x: int(x)),
        wds.to_tuple("image", "target"),
        wds.batched(batch_size, partial=True),
    ]
    return wds.WebLoader(
        wds.DataPipeline(*pipeline), batch_size=None, shuffle=False,
        num_workers=num_workers, persistent_workers=num_workers > 0,
    )


@torch.inference_mode()
def eval_classification(model, classifier, loader, device):
    top1 = top5 = n = 0
    for images, target in tqdm(loader, desc="  cls eval"):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        with torch.autocast(device_type=device.split(':')[0], dtype=torch.bfloat16):
            feats = model(images, modality='image', run_decoder=False)['to_image_caption']
            logits = 100.0 * feats @ classifier
        pred = logits.topk(5, dim=1).indices.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        top1 += correct[:1].reshape(-1).float().sum().item()
        top5 += correct[:5].reshape(-1).float().sum().item()
        n += images.size(0)
    if n == 0:
        raise RuntimeError("No samples processed; check dataset path/format.")
    return {'top1': top1 / n, 'top5': top5 / n, 'n': n}


def run_cls_dataset(name, teacher, tokenizer, model, preprocess, device,
                    batch_size, num_workers, imagenet1k_wds=None):
    repo = CLS_HF[name]
    print(f"  classnames/templates <- {repo}")
    classnames = hf_text_lines(repo, 'classnames.txt')
    templates = hf_text_lines(repo, 'zeroshot_classification_templates.txt')
    print(f"  {len(classnames)} classes x {len(templates)} templates")
    classifier = build_classifier(teacher, tokenizer, classnames, templates, device)

    if name == 'imagenet1k' and imagenet1k_wds:
        shards = list(braceexpand.braceexpand(imagenet1k_wds))
        print(f"  local imagenet1k shards: {len(shards)}")
    else:
        shards = hf_wds_shards(repo, split='test')
    loader = build_cls_loader(shards, batch_size, num_workers, preprocess)
    return eval_classification(model, classifier, loader, device)


# ---------------------------------------------------------------------------
# ImageNet-ReaL  (original-order val + multi-label real.json)
# ---------------------------------------------------------------------------
def _download_real_json(cache_dir):
    import urllib.request
    os.makedirs(cache_dir, exist_ok=True)
    dst = os.path.join(cache_dir, 'imagenet_real.json')
    if not os.path.exists(dst):
        print(f"  downloading real.json -> {dst}")
        urllib.request.urlretrieve(REAL_JSON_URL, dst)
    return json.load(open(dst))


@torch.inference_mode()
def run_imagenet_real(model, teacher, tokenizer, preprocess, device,
                      batch_size, cache_dir, real_wds=None, real_dir=None):
    """ImageNet-ReaL: score val predictions against the multi-label real.json.

    Alignment is by ORIGINAL validation index i (0-based, i.e. ILSVRC2012_val_
    {i+1:08d}.JPEG). Two input modes:
      real_wds : dir or glob of timm/imagenet-1k-wds validation .tar shards
                 (samples keyed by 'ILSVRC2012_val_XXXXXXXX'); index parsed
                 from the filename, so shard/sample order does not matter.
      real_dir : dir of the raw 50000 val JPEGs; sorting by filename yields the
                 canonical order (index = sorted position).
    Score: top-1 prediction correct if in the sample's ReaL label set; samples
    with an empty ReaL set are skipped (standard protocol).
    """
    import glob
    real = _download_real_json(cache_dir)  # list of 50000 lists
    # 1000-way classifier from imagenet1k names/templates
    classnames = hf_text_lines(CLS_HF['imagenet1k'], 'classnames.txt')
    templates = hf_text_lines(CLS_HF['imagenet1k'], 'zeroshot_classification_templates.txt')
    classifier = build_classifier(teacher, tokenizer, classnames, templates, device)

    correct = evaluated = 0
    buf, idxs = [], []

    def _flush():
        nonlocal correct, evaluated
        if not buf:
            return
        images = torch.stack(buf).to(device)
        with torch.autocast(device_type=device.split(':')[0], dtype=torch.bfloat16):
            feats = model(images, modality='image', run_decoder=False)['to_image_caption']
            preds = (100.0 * feats @ classifier).argmax(dim=1).tolist()
        for p, ri in zip(preds, idxs):
            evaluated += 1
            correct += int(p in real[ri])
        buf.clear()
        idxs.clear()

    if real_wds:
        if any(c in real_wds for c in '*?['):
            shards = sorted(glob.glob(real_wds))
        else:
            shards = sorted(glob.glob(os.path.join(real_wds, '*.tar')))
        if not shards:
            raise RuntimeError(f"No .tar shards found under {real_wds}")
        print(f"  ReaL from {len(shards)} timm val shards")
        for sh in tqdm(shards, desc="  real shards"):
            t = tarfile.open(sh)
            grouped = defaultdict(dict)
            for m in t.getmembers():
                if not m.isfile():
                    continue
                key, ext = os.path.splitext(m.name)
                grouped[key][ext.lower()] = m
            for key in grouped:
                ri = int(key.split('_')[-1]) - 1   # ILSVRC2012_val_XXXXXXXX -> 0-based
                if not real[ri]:
                    continue
                e = grouped[key]
                img_m = next((e[x] for x in ('.jpg', '.jpeg', '.png', '.webp') if x in e), None)
                if img_m is None:
                    continue
                img = Image.open(io.BytesIO(t.extractfile(img_m).read())).convert('RGB')
                buf.append(preprocess(img))
                idxs.append(ri)
                if len(buf) == batch_size:
                    _flush()
        _flush()
    else:
        paths = sorted(
            glob.glob(os.path.join(real_dir, '*.JPEG')) +
            glob.glob(os.path.join(real_dir, '*.jpeg')) +
            glob.glob(os.path.join(real_dir, '*.jpg'))
        )
        if len(paths) != len(real):
            print(f"  [WARN] found {len(paths)} images but real.json has {len(real)} "
                  f"entries; alignment requires the full raw val set in order.")
        for gi, path in enumerate(tqdm(paths, desc="  real eval")):
            if not real[gi]:
                continue
            buf.append(preprocess(Image.open(path).convert('RGB')))
            idxs.append(gi)
            if len(buf) == batch_size:
                _flush()
        _flush()

    if evaluated == 0:
        raise RuntimeError("No ReaL samples evaluated; check --imagenet_real_wds / --imagenet_real_dir.")
    return {'top1': correct / evaluated, 'n': evaluated}


# ---------------------------------------------------------------------------
# Retrieval  (text <-> image Recall@K)
# ---------------------------------------------------------------------------
@torch.inference_mode()
def encode_image_list(model, preprocess, pil_imgs, device, bs):
    embs = []
    for i in tqdm(range(0, len(pil_imgs), bs), desc="  img enc"):
        batch = torch.stack([preprocess(im) for im in pil_imgs[i:i + bs]]).to(device)
        with torch.autocast(device_type=device.split(':')[0], dtype=torch.bfloat16):
            e = model(batch, modality='image', run_decoder=False)['to_image_caption']
        embs.append(e.float().cpu())
    return F.normalize(torch.cat(embs), dim=-1)


@torch.inference_mode()
def encode_text_list(teacher, tokenizer, texts, device, bs=256):
    embs = []
    for i in tqdm(range(0, len(texts), bs), desc="  txt enc"):
        tokens = tokenizer(texts[i:i + bs]).to(device)
        with torch.autocast(device_type=device.split(':')[0], dtype=torch.bfloat16):
            e = teacher.encode_text(tokens, normalize=True)
        embs.append(e.float().cpu())
    return F.normalize(torch.cat(embs), dim=-1)


def retrieval_metrics(image_embs, text_embs, txt2img):
    """image_embs (N,D), text_embs (T,D), txt2img[t] = image index of text t."""
    gt = torch.tensor(txt2img)
    # text -> image
    sims_t2i = text_embs @ image_embs.T                       # (T, N)
    ranks_t2i = (sims_t2i.argsort(dim=1, descending=True) == gt.unsqueeze(1)).nonzero()[:, 1]
    t2i = {f'R@{k}': (ranks_t2i < k).float().mean().item() for k in (1, 5, 10)}
    # image -> text (correct if any gt caption in top-k)
    img2txt = defaultdict(set)
    for ti, ii in enumerate(txt2img):
        img2txt[ii].add(ti)
    sims_i2t = image_embs @ text_embs.T                       # (N, T)
    order = sims_i2t.argsort(dim=1, descending=True)
    N = image_embs.size(0)
    i2t = {f'R@{k}': 0 for k in (1, 5, 10)}
    for ii in range(N):
        row = order[ii]
        for k in (1, 5, 10):
            if img2txt[ii] & set(row[:k].tolist()):
                i2t[f'R@{k}'] += 1
    i2t = {k: v / N for k, v in i2t.items()}
    return {'text2image': t2i, 'image2text': i2t}


def load_ret_wds(shards):
    """Load a clip-benchmark retrieval WDS (jpg + txt captions per sample)."""
    pil_imgs, cap_lists = [], []
    for sh in shards:
        t = tarfile.open(sh)
        grouped = defaultdict(dict)
        for m in t.getmembers():
            if not m.isfile():
                continue
            key, ext = os.path.splitext(m.name)
            grouped[key][ext.lower()] = m
        for key in sorted(grouped):
            e = grouped[key]
            img_m = next((e[x] for x in ('.jpg', '.jpeg', '.png', '.webp') if x in e), None)
            if img_m is None or '.txt' not in e:
                continue
            img = Image.open(io.BytesIO(t.extractfile(img_m).read())).convert('RGB')
            caps = [c for c in t.extractfile(e['.txt']).read().decode().splitlines() if c.strip()]
            pil_imgs.append(img)
            cap_lists.append(caps)
    return pil_imgs, cap_lists


def run_ret_wds(name, model, teacher, tokenizer, preprocess, device, bs):
    shards = hf_wds_shards(RET_HF[name], split='test')
    pil_imgs, cap_lists = load_ret_wds(shards)
    print(f"  {len(pil_imgs)} images, {sum(len(c) for c in cap_lists)} captions")
    texts, txt2img = [], []
    for i, caps in enumerate(cap_lists):
        for c in caps:
            texts.append(c)
            txt2img.append(i)
    image_embs = encode_image_list(model, preprocess, pil_imgs, device, bs)
    text_embs = encode_text_list(teacher, tokenizer, texts, device)
    return retrieval_metrics(image_embs, text_embs, txt2img)


# ---------------------------------------------------------------------------
# XM3600 (Crossmodal-3600, all 36 languages, images shared across languages)
# ---------------------------------------------------------------------------
def run_xm3600(model, teacher, tokenizer, preprocess, device, bs, langs):
    from datasets import load_dataset

    # Encode the shared 3600-image gallery once, from the English split.
    print("  loading image gallery (en split) ...")
    en = load_dataset(XM3600_REPO, split='en')
    id2idx, pil_imgs = {}, []
    for ex in en:
        iid = ex['image_id']
        if iid not in id2idx:
            id2idx[iid] = len(pil_imgs)
            pil_imgs.append(Image.open(io.BytesIO(ex['image']['bytes'])).convert('RGB'))
    print(f"  {len(pil_imgs)} unique images")
    image_embs = encode_image_list(model, preprocess, pil_imgs, device, bs)

    per_lang = {}
    for lang in langs:
        try:
            ds = load_dataset(XM3600_REPO, split=lang)
        except Exception as exc:
            print(f"  [SKIP] {lang}: {exc}")
            continue
        texts, txt2img = [], []
        for ex in ds:
            idx = id2idx.get(ex['image_id'])
            if idx is None:
                continue
            for c in ex['captions']:
                if c and c.strip():
                    texts.append(c)
                    txt2img.append(idx)
        text_embs = encode_text_list(teacher, tokenizer, texts, device)
        m = retrieval_metrics(image_embs, text_embs, txt2img)
        per_lang[lang] = m
        print(f"  {lang}: t2i R@1={m['text2image']['R@1']:.4f}  "
              f"i2t R@1={m['image2text']['R@1']:.4f}")

    # Average across languages.
    avg = {'text2image': {}, 'image2text': {}}
    for direction in avg:
        for k in ('R@1', 'R@5', 'R@10'):
            vals = [per_lang[l][direction][k] for l in per_lang]
            avg[direction][k] = sum(vals) / len(vals) if vals else 0.0
    return {'average': avg, 'per_language': per_lang, 'num_languages': len(per_lang)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tasks', nargs='+', default=['classification', 'retrieval'],
                   choices=['classification', 'retrieval'])
    p.add_argument('--cls_datasets', nargs='+',
                   default=['imagenet1k', 'imagenetv2', 'objectnet', 'imagenet_real'])
    p.add_argument('--ret_datasets', nargs='+',
                   default=['coco', 'flickr', 'xm3600'])
    p.add_argument('--ckpt', type=str, default=None,
                   help='Path to the open_clip .pt checkpoint. Required for this '
                        'script; the HF-release variant (eval_zeroshot_siglip2_hf.py) '
                        'ignores it and loads config.json + model.safetensors.')
    p.add_argument('--imagenet1k_wds', type=str, default='',
                   help='Local ImageNet-1k val WDS pattern (class-sorted). '
                        'Set empty to download clip-benchmark/wds_imagenet1k instead.')
    p.add_argument('--imagenet_real_wds', type=str, default='',
                   help='Dir or glob of timm/imagenet-1k-wds validation .tar shards '
                        '(samples keyed by ILSVRC2012_val_XXXXXXXX). Preferred source '
                        'for imagenet_real.')
    p.add_argument('--imagenet_real_dir', type=str, default='',
                   help='Alt: dir with the raw ImageNet val JPEGs in original filename '
                        'order (ILSVRC2012_val_*.JPEG).')
    p.add_argument('--image_size', type=int, default=224)
    p.add_argument('--native_resolution', action='store_true',
                   help='Preprocess each image at its native aspect ratio (no '
                        'center crop), sizes rounded to a multiple of the patch '
                        'size (14) and capped by --max_image_side. Forces batch '
                        'size 1 (batched inference needs a uniform resolution); '
                        'ignores --image_size for preprocessing.')
    p.add_argument('--max_image_side', type=int, default=1400,
                   help='Long-side cap (pixels) under --native_resolution; the '
                        'actual side is rounded to a multiple of the patch size (14).')
    p.add_argument('--video_caption_embed_dim', type=int, default=4096)
    p.add_argument('--siglip2_dir', type=str, default='',
                   help='Local dir with ViT-gopt-16-SigLIP2-384 open_clip weights + '
                        'tokenizer. Falls back to pretrained=webli if absent.')
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--num_workers', type=int, default=8)
    p.add_argument('--xm3600_langs', nargs='+', default=XM3600_LANGS)
    p.add_argument('--device', type=str,
                   default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--cache_dir', type=str,
                   default=os.path.join(os.path.dirname(__file__), '.zeroshot_cache'))
    p.add_argument('--output', type=str, default='zeroshot_siglip2_results.json')
    args = p.parse_args()

    device = args.device
    print(f"Device: {device} | ckpt: {args.ckpt} | res: {args.image_size}")

    model = load_vision_model(args.ckpt, args.video_caption_embed_dim,
                              args.image_size, device)
    teacher, tokenizer = load_text_model(device, siglip2_dir=args.siglip2_dir)

    if args.native_resolution:
        if args.batch_size != 1:
            print("[INFO] --native_resolution: forcing batch size to 1 "
                  "(batched inference needs a uniform resolution).")
        args.batch_size = 1
        preprocess = build_native_preprocess(max_side=args.max_image_side)
        print(f"Native resolution ON: aspect-preserving, long side <= "
              f"{args.max_image_side}px, rounded to multiples of {_PATCH_SIZE}.")
    else:
        preprocess = build_preprocess(args.image_size)

    results = {}

    if 'classification' in args.tasks:
        for name in args.cls_datasets:
            print(f"\n===== [cls] {name} =====")
            try:
                if name == 'imagenet_real':
                    if not (args.imagenet_real_wds or args.imagenet_real_dir):
                        print("  [SKIP] imagenet_real needs --imagenet_real_wds "
                              "(timm val shards) or --imagenet_real_dir (raw val set).")
                        continue
                    m = run_imagenet_real(model, teacher, tokenizer, preprocess,
                                          device, args.batch_size, args.cache_dir,
                                          real_wds=args.imagenet_real_wds or None,
                                          real_dir=args.imagenet_real_dir or None)
                else:
                    m = run_cls_dataset(name, teacher, tokenizer, model, preprocess,
                                        device, args.batch_size, args.num_workers,
                                        imagenet1k_wds=args.imagenet1k_wds)
                results[f'cls/{name}'] = m
                print(f"  -> {m}")
            except Exception as exc:
                print(f"  [ERROR] {name}: {exc}")
                results[f'cls/{name}'] = {'error': str(exc)}

    if 'retrieval' in args.tasks:
        for name in args.ret_datasets:
            print(f"\n===== [ret] {name} =====")
            try:
                if name == 'xm3600':
                    m = run_xm3600(model, teacher, tokenizer, preprocess, device,
                                   args.batch_size, args.xm3600_langs)
                    results[f'ret/{name}'] = m
                    print(f"  -> avg {m['average']}")
                else:
                    m = run_ret_wds(name, model, teacher, tokenizer, preprocess,
                                    device, args.batch_size)
                    results[f'ret/{name}'] = m
                    print(f"  -> {m}")
            except Exception as exc:
                print(f"  [ERROR] {name}: {exc}")
                results[f'ret/{name}'] = {'error': str(exc)}

    print("\n===== Summary =====")
    for k, v in results.items():
        print(f"{k}: {v if 'error' in v else _summ(v)}")

    if args.output:
        meta = {'ckpt': args.ckpt, 'image_size': args.image_size,
                'native_resolution': args.native_resolution}
        if args.native_resolution:
            meta['max_image_side'] = args.max_image_side
        meta['results'] = results
        with open(args.output, 'w') as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        print(f"\nSaved -> {args.output}")


def _summ(v):
    if 'top1' in v:
        return f"top1={v['top1']*100:.2f}% top5={v.get('top5', 0)*100:.2f}% n={v['n']}"
    if 'average' in v:
        a = v['average']
        return (f"t2i R@1={a['text2image']['R@1']*100:.2f}% "
                f"i2t R@1={a['image2text']['R@1']*100:.2f}% ({v['num_languages']} langs)")
    if 'text2image' in v:
        return (f"t2i R@1={v['text2image']['R@1']*100:.2f}% "
                f"i2t R@1={v['image2text']['R@1']*100:.2f}%")
    return str(v)


if __name__ == '__main__':
    main()
