# 数据格式规范

正式训练目标是直接使用 LLaVA-OneVision-2 的数据和 dataloader；本文件描述的是当前 fallback WDS reader 支持的本地开发格式。完整对齐策略见 `docs/data_alignment.md`。

## 1. Tar 文件结构

每个 shard 是一个 `*.tar` 文件，按 `__key__` 分组多个样本。

### 1.1 图像样本

```
__key__=sample_001
├── 0.jpg              # 图像二进制
└── 0.json             # 元信息（messages + modality）
```

`0.json` 内容：
```json
{
  "messages": [
    {"role": "user", "content": "<image>\nDescribe the image."},
    {"role": "assistant", "content": "A cat is sitting on the mat."}
  ],
  "modality": "image"
}
```

### 1.2 视频样本

```
__key__=sample_002
├── 0.jpg              # 第 1 帧
├── 1.jpg              # 第 2 帧
├── 2.jpg              # 第 3 帧
├── ...
├── K.jpg              # 第 K 帧
└── 0.json             # 元信息
```

`0.json` 内容：
```json
{
  "messages": [
    {"role": "user", "content": "<video>\nDescribe the video."},
    {"role": "assistant", "content": "A person is walking in the park."}
  ],
  "modality": "video",
  "max_frames": 16
}
```

视频帧数（jpg 数）由 `frame_extraction` 工具决定，**第一版 SFT 不超过 32 帧**（与 LLaVA-OneVision-2 默认一致）。

### 1.3 纯文本样本（可选）

```
__key__=sample_003
└── 0.json
```

```json
{
  "messages": [
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "assistant", "content": "Paris."}
  ],
  "modality": "text"
}
```

## 2. 占位符

- `<image>`: 图像占位，1 个对应 1 张图
- `<video>`: 视频占位，1 个对应 1 段视频

`CoViscoPlugin.process_messages` 会把这两个占位符替换为：
```
<|vision_start|><|image_pad|>×N<|vision_end|>
```

其中 N 由 `DynamicTokenConfig.sample()` 在 collator 阶段决定（query_only / vit_only / query_and_vit 之一）。

## 3. 离线 packing

正式训练应优先复用 LLaVA-OneVision-2 已有 packing / dataloader 结果。当前 fallback reader 不实现完整 packing，只用于独立调试模型链路。
