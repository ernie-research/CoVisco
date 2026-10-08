"""
llava_covisco.py — CoVisco + Qwen3 模型的 lmms-eval 适配器。

支持 onevision_siglip_llm_sft 项目训练的模型，用于评测图像/视频 benchmarks。
"""
import torch

torch.backends.cuda.matmul.allow_tf32 = True

import os
import sys
import math
import warnings
import copy
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import yaml
from accelerate import Accelerator, InitProcessGroupKwargs
from packaging import version
from PIL import Image
from tqdm import tqdm

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.utils import stop_sequences_criteria

warnings.filterwarnings("ignore")

from loguru import logger as eval_logger

# ============ Model Import ============
# 本文件位于 <repo>/third_party/lmms-eval/lmms_eval/models/simple/ 下，
# 向上 6 层 (parents[5]) 即 onevision_siglip_llm_sft 仓库根目录。
# 加入 sys.path 后即可 import 仓库内的 models/ 包，不再依赖外部 PYTHONPATH。
ONEVISION_SIGLIP_ROOT = Path(__file__).resolve().parents[5]
if (ONEVISION_SIGLIP_ROOT / "models" / "config.py").exists():
    if str(ONEVISION_SIGLIP_ROOT) not in sys.path:
        sys.path.insert(0, str(ONEVISION_SIGLIP_ROOT))
else:
    eval_logger.warning(
        f"onevision_siglip_llm_sft repo root not found at {ONEVISION_SIGLIP_ROOT}; "
        f"falling back to PYTHONPATH for the models/ package"
    )

# inference implementation for attention
if version.parse(torch.__version__) >= version.parse("2.1.2"):
    best_fit_attn_implementation = "sdpa"
else:
    best_fit_attn_implementation = "eager"


# ============ dtype 字符串映射（用于 model_args 命令行传参） ============
DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "fp16": torch.float16,
    "float32": torch.float32,
    "fp32": torch.float32,
}


def _resolve_dtype(dtype: Union[str, torch.dtype]) -> torch.dtype:
    """将字符串或 torch.dtype 解析为 torch.dtype。"""
    if isinstance(dtype, torch.dtype):
        return dtype
    if dtype not in DTYPE_MAP:
        raise ValueError(f"Unsupported dtype string: {dtype!r}. Supported: {list(DTYPE_MAP.keys())}")
    return DTYPE_MAP[dtype]


def _as_bool(value: Union[bool, str, int]) -> bool:
    """解析 model_args 传入的布尔参数（可能是 "True"/"true"/"1" 等字符串）。"""
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "y")
    return bool(value)


# ============ Image Transform (与训练一致) ============
# CLIP 标准均值和方差（与 LLaVA-OneVision 对齐）
IMG_MEAN = (0.48145466, 0.4578275, 0.40821073)
IMG_STD = (0.26862954, 0.26130258, 0.27577711)


def build_image_transform(image_size: int = 224):
    """构建图像预处理 transform（与训练时的 collator.build_image_transform 一致）。

    Args:
        image_size: 目标分辨率

    Returns:
        torchvision.transforms.Compose
    """
    from torchvision import transforms

    return transforms.Compose([
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=list(IMG_MEAN), std=list(IMG_STD)),
    ])


def _fit_to_patch_grid(
    height: int,
    width: int,
    patch_size: int,
    max_patches: int,
    min_patches: int,
) -> Tuple[int, int]:
    """把 (height, width) 调整到最接近的、能被 patch_size 整除的尺寸。

    保持原始长宽比，并把 patch 数量约束在 [min_patches, max_patches] 内
    （ViT 内部是全 attention，patch 数不加约束会 OOM）。

    Returns:
        (new_height, new_width)，均为 patch_size 的整数倍
    """
    gh = max(1, round(height / patch_size))
    gw = max(1, round(width / patch_size))

    if gh * gw > max_patches:
        scale = math.sqrt(max_patches / (gh * gw))
        gh = max(1, int(gh * scale))
        gw = max(1, int(gw * scale))
    elif gh * gw < min_patches:
        scale = math.sqrt(min_patches / (gh * gw))
        gh = max(1, math.ceil(gh * scale))
        gw = max(1, math.ceil(gw * scale))

    return gh * patch_size, gw * patch_size


def build_native_resolution_transform(
    patch_size: int = 14,
    max_patches: int = 4096,
    min_patches: int = 64,
):
    """构建原生分辨率图像预处理（保持长宽比，尺寸对齐到 patch_size 的整数倍）。

    与 build_image_transform 的区别：不做「短边 Resize + CenterCrop」，
    因此不会裁掉画面内容也不会强行拉成正方形；归一化常数与训练保持一致。

    注意：不同图像输出尺寸不同，无法 stack 成 batch，调用方必须逐图前向。
    """
    from torchvision import transforms

    normalize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=list(IMG_MEAN), std=list(IMG_STD)),
    ])

    def _transform(img: Image.Image) -> torch.Tensor:
        width, height = img.size
        new_h, new_w = _fit_to_patch_grid(height, width, patch_size, max_patches, min_patches)
        if (new_h, new_w) != (height, width):
            img = img.resize((new_w, new_h), Image.BICUBIC)
        return normalize(img)

    return _transform


def _load_model_from_config(
    config_path: str,
    checkpoint_path: Optional[str] = None,
    segment_t_size: Optional[int] = None,
    dtype: Union[str, torch.dtype] = torch.bfloat16,
):
    """加载模型配置并构建模型实例。
    
    Args:
        config_path: YAML 配置文件路径
        checkpoint_path: 训练好的 checkpoint 路径（可选，用于加载 projector/llm 权重）
        segment_t_size: 覆盖 config.vit.segment_t_size (None = 使用配置文件中的值)。
            必须在构建模型之前重载，因为该值会影响 ViT 模块结构（如 rotary embedding）。
        dtype: 模型整体运行的 dtype（支持字符串如 "bfloat16"/"float16"/"float32"），
            统一转换 ViT/Projector/TokenSelector/LLM，避免各子模块 dtype 不一致导致 matmul 报错。
    
    Returns:
        model, tokenizer, config
    """
    dtype = _resolve_dtype(dtype)

    # 动态导入项目模块
    from models.config import build_model_config
    from models.llava_covisco import LlavaCoViscoModel
    from transformers import AutoTokenizer
    
    # 加载配置
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)
    config = build_model_config(config_dict)

    # 重载 segment_t_size（必须在构建模型之前）
    if segment_t_size is not None:
        eval_logger.info(
            f"Overriding segment_t_size: {config.vit.segment_t_size} -> {segment_t_size}"
        )
        config.vit.segment_t_size = segment_t_size
    
    # 构建模型
    model = LlavaCoViscoModel(config)
    
    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.llm.path,
        trust_remote_code=True,
        use_fast=False,
    )
    
    # 如果提供了 checkpoint，加载权重
    if checkpoint_path:
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"checkpoint_path 不存在: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        # train/train.py 保存的顶层键是 "model_state"（见 torch.save({... "model_state": ...})），
        # 兼容其它常见命名；都没有时把 ckpt 本身当 state_dict。
        state = None
        if isinstance(ckpt, dict):
            for key in ("model_state", "model", "state_dict"):
                if key in ckpt:
                    state = ckpt[key]
                    break
        if state is None:
            state = ckpt
        # DDP 保存的 state_dict 可能带 "module." 前缀
        if any(isinstance(k, str) and k.startswith("module.") for k in state):
            state = {k[len("module."):]: v for k, v in state.items()}

        incompatible = model.load_state_dict(state, strict=False)
        n_loaded = len(state) - len(incompatible.unexpected_keys)
        if n_loaded == 0:
            raise RuntimeError(
                f"checkpoint {checkpoint_path} 中没有任何权重被加载"
                f"（顶层键: {list(ckpt.keys())[:8] if isinstance(ckpt, dict) else type(ckpt)}）。"
                "请检查 checkpoint 格式。"
            )
        eval_logger.info(
            f"Loaded checkpoint from {checkpoint_path}: "
            f"{n_loaded}/{len(state)} tensors matched, "
            f"missing={len(incompatible.missing_keys)}, "
            f"unexpected={len(incompatible.unexpected_keys)}"
        )
        if incompatible.missing_keys:
            eval_logger.warning(f"missing keys (前 10): {incompatible.missing_keys[:10]}")
        if incompatible.unexpected_keys:
            eval_logger.warning(f"unexpected keys (前 10): {incompatible.unexpected_keys[:10]}")

    # 统一转换整个模型到目标 dtype，避免子模块间 dtype 不一致
    model = model.to(dtype)
    eval_logger.info(f"Model dtype unified to: {dtype}")
    
    return model, tokenizer, config


def _extract_video_frames(video_path: str, num_frames: int = 8) -> List[Image.Image]:
    """从视频中均匀提取指定数量的帧。
    
    Args:
        video_path: 视频文件路径
        num_frames: 要提取的帧数
    
    Returns:
        List of PIL Images
    """
    import cv2
    
    frames = []
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames <= 0:
        cap.release()
        return frames
    
    # 均匀采样帧索引
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))
        elif frames:
            # seek 后解码失败（h264 mmco 错误等）时复用上一帧。必须保证返回帧数恒为
            # num_frames，否则 T 不能被 segment_t_size 整除，ViT get_segments 的
            # assert 会失败，该样本被静默判 0 分。
            frames.append(frames[-1])

    cap.release()
    # 开头若连续解码失败，上面的 elif 兜不住，这里统一补齐
    if frames and len(frames) < num_frames:
        frames.extend([frames[-1]] * (num_frames - len(frames)))
    return frames


@register_model("llava_covisco")
class LlavaCoVisco(lmms):
    """CoVisco + Qwen3 模型的 lmms-eval 适配器。"""

    def __init__(
        self,
        config_path: str = "",
        checkpoint_path: Optional[str] = None,
        truncation: bool = True,
        device: str = "cuda",
        batch_size: int = 1,
        attn_implementation: str = best_fit_attn_implementation,
        device_map: str = "auto",
        conv_template: str = "qwen_1_5",
        use_cache: bool = True,
        max_new_tokens: int = 256,
        num_video_frames: int = 8,
        vit_ratio: float = 0.5,
        token_strategy: str = "query_and_vit",
        mmr_lambda: float = 0.0,
        segment_t_size: Optional[int] = None,
        image_size: Optional[int] = None,
        image_size_image: Optional[int] = None,
        image_size_video: Optional[int] = None,
        native_resolution: bool = False,
        native_max_patches: int = 4096,
        native_min_patches: int = 64,
        dtype: str = "bfloat16",
        codec_visidx_root: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__()

        accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
        accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])
        self.accelerator = accelerator
        self._device = torch.device(device)
        self.device_map = device_map
        self._dtype = _resolve_dtype(dtype)

        # 加载模型（segment_t_size 需在模型构建前重载，因为它影响 ViT 内部结构；
        # dtype 在模型构建后统一转换，避免子模块间 dtype 不一致导致 matmul 报错）
        eval_logger.info(f"Loading model from config: {config_path}")
        self._model, self._tokenizer, self._config = _load_model_from_config(
            config_path, checkpoint_path, segment_t_size, dtype=self._dtype
        )
        self._model = self._model.to(self._device)
        self._model.eval()
        
        self.truncation = truncation
        self.batch_size_per_gpu = batch_size
        self.conv_template = conv_template
        self.use_cache = use_cache
        self.max_new_tokens = max_new_tokens
        self.num_video_frames = num_video_frames
        self.vit_ratio = vit_ratio
        if token_strategy not in ("query_only", "vit_only", "query_and_vit"):
            raise ValueError(
                f"token_strategy 必须是 query_only / vit_only / query_and_vit，得到 {token_strategy!r}"
            )
        self.token_strategy = token_strategy
        self.mmr_lambda = mmr_lambda
        self.segment_t_size = segment_t_size

        # 原生分辨率模式：图像保持长宽比 resize 到最接近的 patch 整数倍尺寸；
        # 视频仍走固定分辨率路径。逐图前向，因此强制 batch_size=1。
        self.native_resolution = _as_bool(native_resolution)
        self.native_max_patches = int(native_max_patches)
        self.native_min_patches = int(native_min_patches)
        if self.native_resolution and self.batch_size_per_gpu != 1:
            eval_logger.warning(
                f"native_resolution=True requires batch_size=1, forcing "
                f"{self.batch_size_per_gpu} -> 1"
            )
            self.batch_size_per_gpu = 1

        # 构建图像和视频的 transform（支持通过参数覆盖分辨率）
        self._build_transforms(image_size, image_size_image, image_size_video)

        # ---- codec visidx（视频 patch 级 token 筛选）----
        # 若设置，视频评测时按 <root>/<video_stem>/visidx.npy 加载预计算的 codec 索引，
        # 传给 ViT 只计算被选中的 patch（见 models/covisco_vit.py::forward(visidx=...)）。
        # 由 scripts/precompute_codec_visidx/ 预先生成，网格参数须与本次评测一致。
        self.codec_visidx_root = codec_visidx_root or os.environ.get("CODEC_VISIDX_ROOT") or None
        self._visidx_miss_warned = False
        if self.codec_visidx_root:
            eval_logger.info(f"codec visidx enabled, root={self.codec_visidx_root}")

    def _build_transforms(
        self,
        image_size_override: Optional[int] = None,
        image_size_image_override: Optional[int] = None,
        image_size_video_override: Optional[int] = None,
    ):
        """构建图像和视频的 transform（与训练时一致）。

        分辨率优先级（从高到低）：
            显式覆盖参数 > config.vit.image_size_{image,video} > config.vit.image_size

        native_resolution=True 时图像走原生分辨率 transform（忽略 image_size_image），
        视频始终使用固定分辨率。
        """
        # 获取分辨率配置（config 中的默认值）
        base_image_size = image_size_override or self._config.vit.image_size
        image_size_image = (
            image_size_image_override
            or getattr(self._config.vit, "image_size_image", 0)
            or base_image_size
        )
        image_size_video = (
            image_size_video_override
            or getattr(self._config.vit, "image_size_video", 0)
            or base_image_size
        )
        self.image_size_image = image_size_image
        self.image_size_video = image_size_video

        # 视频始终固定分辨率
        self.video_transform = build_image_transform(image_size_video)

        if self.native_resolution:
            patch_size = self._config.vit.patch_size
            self.image_transform = build_native_resolution_transform(
                patch_size=patch_size,
                max_patches=self.native_max_patches,
                min_patches=self.native_min_patches,
            )
            eval_logger.info(
                f"Built transforms: image=native resolution "
                f"(patch_size={patch_size}, patches in "
                f"[{self.native_min_patches}, {self.native_max_patches}]), "
                f"video_size={image_size_video}"
            )
        else:
            self.image_transform = build_image_transform(image_size_image)
            eval_logger.info(
                f"Built transforms: image_size={image_size_image}, video_size={image_size_video}"
            )

    @property
    def model(self):
        return self._model

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def config(self):
        return self._config

    @property
    def device(self):
        return self._device

    @property
    def max_length(self):
        return self._model.config.llm.hidden_size * 4 if hasattr(self._model.config, "llm") else 2048

    def _load_visidx(self, video_path: str) -> Optional[torch.Tensor]:
        """按视频文件名(stem)加载预计算的 codec visidx，返回 (1, L) LongTensor 或 None。"""
        if not self.codec_visidx_root:
            return None
        stem = Path(video_path).stem
        npy_path = Path(self.codec_visidx_root) / stem / "visidx.npy"
        if not npy_path.exists():
            if not self._visidx_miss_warned:
                eval_logger.warning(
                    f"codec visidx 缺失，回退到全量 patch：{npy_path}（后续同类缺失不再提示）"
                )
                self._visidx_miss_warned = True
            return None
        arr = np.load(str(npy_path)).reshape(-1).astype(np.int64)
        return torch.from_numpy(arr).long().unsqueeze(0).to(self._device)  # (1, L)

    def generate_until(self, requests: List[Instance]) -> List[str]:
        """生成文本直到满足停止条件。

        Instance.arguments 对于 generate_until 类型为:
            (ctx, gen_kwargs, doc_to_visual, doc_id, task_name, split)
        没有 doc 字段，需要通过 doc_to_visual(task_dict[task][split][doc_id]) 获取图像/视频。
        """
        results = []

        for request in tqdm(requests, desc="Generating"):
            # 解析 arguments
            prompt, gen_kwargs, doc_to_visual, doc_id, task_name, split = request.arguments
            gen_kwargs = copy.deepcopy(gen_kwargs)
            if "until" in gen_kwargs:
                gen_kwargs.pop("until")

            # 通过 doc_to_visual 获取该样本对应的原始 doc，再取出图像/视频
            doc = self.task_dict[task_name][split][doc_id]
            visuals = doc_to_visual(doc) if doc_to_visual else []
            if visuals is None:
                visuals = []
            elif not isinstance(visuals, list):
                visuals = [visuals]

            # 区分图像 (PIL.Image) 和视频路径 (str)
            images = [v for v in visuals if isinstance(v, Image.Image)]
            video_paths = [v for v in visuals if isinstance(v, str) and os.path.exists(v)]

            video_frames = []
            for video_path in video_paths:
                frames = _extract_video_frames(video_path, self.num_video_frames)
                video_frames.extend(frames)

            # codec visidx：仅在单视频样本时按 stem 加载（多视频无法对应单一 visidx）
            video_visidx = None
            if self.codec_visidx_root and len(video_paths) == 1:
                video_visidx = self._load_visidx(video_paths[0])

            # 生成
            with torch.no_grad():
                try:
                    messages = [{"role": "user", "content": prompt}]

                    # 调用模型生成（图像与视频帧分别走各自的 transform）
                    output = self._generate(
                        messages=messages,
                        images=images,
                        video_frames=video_frames,
                        video_visidx=video_visidx,
                        max_new_tokens=gen_kwargs.get("max_new_tokens", self.max_new_tokens),
                        temperature=gen_kwargs.get("temperature", 0.0),
                        top_p=gen_kwargs.get("top_p", 1.0),
                    )
                    results.append(output)
                except Exception as e:
                    eval_logger.error(f"Generation error: {type(e).__name__}: {e}")
                    results.append("")
        
        return results

    def _validate_visidx(
        self,
        visidx: Optional[torch.Tensor],
        pixel_values: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """校验 codec visidx 是否与当前视频张量的 patch 网格匹配，不匹配则回退 None（全量 patch）。

        约束（对齐 models/_covisco_encoder_src.py::get_segments）：
          - 索引空间 [0, T*h*w)，其中 h=H//patch, w=W//patch
          - 长度须能被 segment 数整除（各段 token 数一致）
        """
        if visidx is None:
            return None
        if pixel_values.dim() != 5:
            return None
        num_frame = pixel_values.shape[2]
        patch = self._model.config.vit.patch_size
        h = pixel_values.shape[3] // patch
        w = pixel_values.shape[4] // patch
        total = num_frame * h * w
        seg_t = getattr(self._model.vit.encoder, "segment_t_size", num_frame)
        num_seg = 1 if num_frame < seg_t else num_frame // seg_t

        L = int(visidx.shape[-1])
        vmax = int(visidx.max().item()) if L > 0 else -1
        if L == 0 or L % num_seg != 0 or vmax >= total:
            eval_logger.warning(
                f"codec visidx 与当前网格不匹配（L={L}, num_seg={num_seg}, "
                f"vi_max={vmax}, total_patches={total}），本样本回退全量 patch。"
                f" 请确认预计算的 num_video_frames/image_size/patch_size/segment_t_size 与评测一致。"
            )
            return None
        return visidx.reshape(1, -1)

    def _encode_frames(
        self,
        frames: List[Image.Image],
        transform,
        modality: str,
        visidx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """把一组同尺寸的帧编码为 LLM 空间的 vision embeddings。

        Args:
            frames: PIL 图像列表（必须同尺寸，才能 stack 到 T 维）
            transform: 对应的预处理（image_transform 或 video_transform）
            modality: "image" | "video"
            visidx: 可选 (1, L) codec 预筛选的全局 patch 索引，仅视频使用

        Returns:
            vision_embeds: (1, S*(Q+K), D)
        """
        # 每帧 transform 后为 (3, H, W)，stack 后为 (N, 3, H, W)
        pixel_values = torch.stack([transform(img.convert("RGB")) for img in frames])
        # (N, 3, H, W) -> (3, N, H, W) -> (1, 3, N, H, W)，符合模型期望的 (B, C, T, H, W)
        pixel_values = pixel_values.permute(1, 0, 2, 3).unsqueeze(0)
        pixel_values = pixel_values.to(self._device, dtype=self._dtype)

        # 校验 visidx 与当前网格一致（帧数/分辨率/segment 划分），不一致则回退全量
        visidx = self._validate_visidx(visidx, pixel_values)

        query_tokens, vit_tokens, _ = self._model.vit(
            pixel_values=pixel_values.to(next(self._model.vit.parameters()).dtype),
            visidx=visidx,
            modality=modality,
        )

        # 按 token_strategy 组合视觉 token，与训练时 models/llava_covisco.py
        # ::_arrange_tokens 的三种策略保持一致
        num_seg = query_tokens.shape[1]
        if self.token_strategy == "query_only":
            b, s, q, d = query_tokens.shape
            all_vision = query_tokens.reshape(b, s * q, d)
        elif self.token_strategy == "vit_only":
            # 全量 vit patch token，不经过 token_selector（训练中 OCR 样本走此路径）
            b, s, p, d = vit_tokens.shape
            all_vision = vit_tokens.reshape(b, s * p, d)
        else:
            # query_and_vit：token selector 选 top-K（K 按 vit_ratio 动态计算），
            # 原生分辨率下 P 随图像尺寸变化
            P = vit_tokens.shape[2]
            K = max(1, round(P * self.vit_ratio))

            original_mmr_lambda = self._model.token_selector.mmr_lambda
            self._model.token_selector.mmr_lambda = self.mmr_lambda
            try:
                sel_out = self._model.token_selector(query_tokens, vit_tokens, top_k=K)
            finally:
                self._model.token_selector.mmr_lambda = original_mmr_lambda
            selected_vit = sel_out["selected_tokens"]

            # interleave: [q_seg0, v_seg0, q_seg1, v_seg1, ...]
            chunks = []
            for i in range(num_seg):
                chunks.append(query_tokens[:, i, :, :])
                chunks.append(selected_vit[:, i, :, :])
            all_vision = torch.cat(chunks, dim=1)  # (B, S*(Q+K), D)

        # Projector (对齐 projector 自身权重 dtype，而非直接假定 language_model.dtype)
        projector_dtype = next(self._model.projector.parameters()).dtype
        vision_embeds = self._model.projector(all_vision.to(projector_dtype))
        # 再转换为 language_model 期望的 dtype，用于后续 embedding 注入
        return vision_embeds.to(self._model.language_model.dtype)

    def _generate(
        self,
        messages: List[Dict[str, str]],
        images: Optional[List[Image.Image]] = None,
        video_frames: Optional[List[Image.Image]] = None,
        video_visidx: Optional[torch.Tensor] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> str:
        """内部生成方法。

        使用模型的 language_model.generate() 进行自回归生成，
        视觉 token 先经 ViT/TokenSelector/Projector 预计算，再注入 inputs_embeds。
        """
        images = images or []
        video_frames = video_frames or []

        with torch.no_grad():
            # 1. 计算 vision embeddings
            vision_chunks = []
            if images:
                if self.native_resolution:
                    # 原生分辨率下每张图尺寸不同，无法 stack，逐图编码后拼接
                    for img in images:
                        vision_chunks.append(
                            self._encode_frames([img], self.image_transform, "image")
                        )
                else:
                    vision_chunks.append(
                        self._encode_frames(images, self.image_transform, "image")
                    )
            if video_frames:
                # 视频保持当前设计：固定分辨率，所有帧一起作为 T 维送入 ViT
                vision_chunks.append(
                    self._encode_frames(video_frames, self.video_transform, "video", visidx=video_visidx)
                )

            # 2. 纯文本分支
            if not vision_chunks:
                prompt = self._tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                input_ids = self._tokenizer.encode(prompt, return_tensors="pt").to(self._device)
                output_ids = self._model.language_model.generate(
                    input_ids=input_ids,
                    attention_mask=torch.ones_like(input_ids),
                    max_new_tokens=max_new_tokens,
                    do_sample=(temperature > 0.0),
                    temperature=temperature if temperature > 0.0 else None,
                    top_p=top_p if temperature > 0.0 else None,
                    use_cache=True,
                )
                return self._tokenizer.decode(
                    output_ids[0][input_ids.shape[1]:], skip_special_tokens=True
                )

            vision_embeds = torch.cat(vision_chunks, dim=1)
            n_vision = vision_embeds.shape[1]

            # 3. 按实际 vision token 数构建 prompt
            # 训练数据的 user content 形如 "<image>\n{question}"（见 data/plugin.py 把
            # <image> 展开为 vision block），即视觉块在问题文本之前。这里必须保持一致，
            # 否则视觉 token 落在问题之后，与训练分布不匹配。
            vision_placeholder = "<|vision_start|>" + "<|image_pad|>" * n_vision + "<|vision_end|>"
            messages = copy.deepcopy(messages)
            for msg in messages:
                if msg.get("role") == "user":
                    msg["content"] = vision_placeholder + "\n" + msg["content"]
                    break
            prompt = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            input_ids = self._tokenizer.encode(prompt, return_tensors="pt").to(self._device)
            attention_mask = torch.ones_like(input_ids)

            # 4. 注入 vision embeddings
            emb_layer = self._model.language_model.get_input_embeddings()
            inputs_embeds = emb_layer(input_ids).clone()
            pad_mask = (input_ids == self._model.image_pad_token_id)
            inputs_embeds[pad_mask] = vision_embeds.reshape(-1, vision_embeds.shape[-1])

            # 5. 生成
            # 注意：使用 inputs_embeds（而非 input_ids）调用 generate 时，
            # HuggingFace transformers 返回的 output_ids 只包含新生成的 token，
            # 不包含输入部分，因此解码时不能再做 [input_ids.shape[1]:] 切片。
            output_ids = self._model.language_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=(temperature > 0.0),
                temperature=temperature if temperature > 0.0 else None,
                top_p=top_p if temperature > 0.0 else None,
                use_cache=True,
            )
            return self._tokenizer.decode(output_ids[0], skip_special_tokens=True)

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        """计算 log-likelihood（用于某些 benchmark）。"""
        results = []
        
        for request in tqdm(requests, desc="Computing log-likelihood"):
            doc = request.doc
            prompt = request.arguments[0] if request.arguments else ""
            continuation = request.arguments[1] if len(request.arguments) > 1 else ""
            
            # 简化实现：返回 0 和 False
            # 完整实现需要调用模型计算 log probabilities
            results.append((0.0, False))
        
        return results

    def generate_until_multi_round(self, requests: List[Instance]) -> List[str]:
        """多轮对话生成。"""
        return self.generate_until(requests)

    def tok_encode(self, string: str) -> List[int]:
        """Tokenize 字符串。"""
        return self._tokenizer.encode(string, add_special_tokens=False)

    def tok_decode(self, tokens: List[int]) -> str:
        """Decode tokens。"""
        return self._tokenizer.decode(tokens, skip_special_tokens=True)

    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息。"""
        return {
            "model_name": "llava_covisco",
            "config_path": getattr(self, "_config_path", ""),
            "device": str(self._device),
            "num_video_frames": self.num_video_frames,
            "token_strategy": self.token_strategy,
            "vit_ratio": self.vit_ratio,
            "mmr_lambda": self.mmr_lambda,
            "native_resolution": self.native_resolution,
            "native_max_patches": self.native_max_patches,
            "image_size": "native" if self.native_resolution else self.image_size_image,
            "video_size": self.image_size_video,
        }
