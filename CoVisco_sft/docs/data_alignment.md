# 与 LLaVA-OneVision-2 的数据流程对齐

目标：直接使用 LLaVA-OneVision-2 的 SFT / mid-training 数据和数据加载链路，只在模型边界适配 CoVisco 的 `query_tokens` / `vit_tokens` 处理。

## 对齐原则

- 数据源、样本组织、packing、chat template、assistant-only labels 尽量保持 LLaVA-OneVision-2 原样。
- 图像/视频读取、抽帧结果、`image_grid_thw`、`video_grid_thw`、`patch_positions` 等字段尽量保持 LLaVA-OneVision-2 原样。
- 本项目不重新定义数据格式；`data/wds_dataset.py` 只是本地开发和冒烟测试用的 fallback reader。
- 允许不同的部分只放在模型侧：CoVisco ViT sequence output、query-guided ViT token selection、动态 query/vit 组合和 projector。

## LLaVA-OneVision-2 原始 batch

真实 LLaVA-OneVision-2 / Energon 数据链路通常输出：

- `tokens`：LLM input ids。
- `labels`：assistant-only target labels。
- `attn_mask`：attention mask。
- `imgs`：图像 pixel values。
- `image_grid_thw`：图像 grid 元信息。
- `pixel_values_videos`：视频 pixel values。
- `video_grid_thw`：视频 grid 元信息。
- `patch_positions`：patch 位置元信息，可选。
- `cu_lengths` / `max_lengths`：packing 相关字段。

这些字段应尽量由 LLaVA-OneVision-2 的 `Qwen2VLTaskEncoder`、`mm_plugin`、chat template 和 dataloader provider 生成。

## 当前适配边界

PyTorch debug/fallback 路径中，`data/llava_batch_adapter.py` 提供模型边界适配：

- `tokens` -> `input_ids`
- `attn_mask` -> `attention_mask`
- `imgs` / `pixel_values_videos` -> `pixel_values`
- 根据 token id 和视觉字段判断 `modality`
- 按动态策略生成 `token_plan`
- 保留 `image_grid_thw`、`video_grid_thw`、`patch_positions`，方便后续视频段数和 ViT 位置逻辑继续对齐

## 当前 fallback reader

`data/wds_dataset.py` 和 `data/collator.py` 仍然保留，用于没有 LLaVA-OneVision-2 运行环境时做本地 smoke test。正式训练建议优先接入 LLaVA-OneVision-2 的真实 dataloader。

## 后续待完全对齐

- 直接接入 LLaVA-OneVision-2 的 `get_train_dataset` / `get_train_loader`。
- 复用 `Qwen2VLTaskEncoder`、`Qwen2VLPlugin` 和 chat template，而不是本地简化 plugin/collator。
- 将动态视觉 token pad 数调整进一步下沉到 LLaVA mm_plugin 或 task encoder 的最小 override 中，让样本选择/packing 前的长度估计也与 `token_plan` 一致。
- 根据 `video_grid_thw` 或 `patch_positions` 精确确定 CoVisco 的 segment 对齐关系。
