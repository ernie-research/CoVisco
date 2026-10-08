# CoVisco SFT 整体架构文档

> 本文档覆盖 `CoVisco_sft` 项目的完整训练架构，包括数据流水线、DataLoader、模型结构与训练循环。

---

## 目录

1. [项目目录结构](#1-项目目录结构)
2. [整体数据流](#2-整体数据流)
3. [数据处理](#3-数据处理)
   - [数据集：WDSDataset](#31-数据集wdsdataset)
   - [Plugin：视觉占位符展开](#32-plugin视觉占位符展开)
   - [动态 Token 策略](#33-动态-token-策略)
   - [Collator：批次组装](#34-collator批次组装)
4. [视频采样双路径](#4-视频采样双路径)
5. [DataLoader 配置](#5-dataloader-配置)
6. [模型架构](#6-模型架构)
   - [ViT：CoVisco-L/14](#61-vitcovisco-l14)
   - [Token Selector](#62-token-selector)
   - [Projector](#63-projector)
   - [LLM：Qwen3-1.7B](#64-llmqwen3-17b)
   - [顶层模型 Forward](#65-顶层模型-forward)
7. [损失函数](#7-损失函数)
8. [训练循环](#8-训练循环)
   - [优化器与调度器](#81-优化器与调度器)
   - [冻结策略](#82-冻结策略)
   - [混合模态训练](#83-混合模态训练)
9. [关键超参数（1.7B）](#9-关键超参数17b)
10. [启动命令](#10-启动命令)

---

## 1. 项目目录结构

```
CoVisco_sft/
├── configs/
│   ├── covisco_qwen3_1.7b.yaml   # 1.7B 模型配置
│   ├── covisco_qwen3_4b.yaml
│   └── covisco_qwen3_8b.yaml
├── data/
│   ├── wds_dataset.py        # WebDataset tar shard 读取器
│   ├── collator.py           # 多模态批次 Collator
│   ├── dynamic_strategy.py   # TokenPlan / DynamicTokenConfig
│   ├── llava_batch_adapter.py
│   └── plugin.py             # mm_plugin：<image>/<video> → vision blocks
├── models/
│   ├── config.py                    # 所有 Config dataclass + build_model_config
│   ├── llava_covisco.py    # 顶层模型
│   ├── covisco_vit.py      # ViT wrapper
│   ├── _covisco_encoder_src.py    # SigLIP ViT Transformer 实现
│   ├── _token_selector_src.py       # LearnableTokenSelector
│   ├── token_selector.py            # re-export
│   ├── projector.py                 # TwoLayerMLPProjector
│   └── covisco_vit_checkpoint.py  # 预训练权重加载工具
├── train/
│   └── train.py              # DDP 训练器（run_sft_1.7b.sh 入口）
├── scripts/
│   └── run_sft_1.7b.sh
└── docs/
```

---

## 2. 整体数据流

```
  tar shards (WDS)
       │
       ▼
 CoViscoWDSDataset
  ├─ 解析 messages / modality
  ├─ 解码图片/视频帧（视频对齐到 128 帧）
  ├─ 加载 visidx.npy（稀疏候选 patch，visidx 路径用）
  ├─ plugin.process_messages() → 占位符展开
  └─ DynamicTokenConfig.sample() → TokenPlan
       │
       ▼
   DataLoader (num_workers=8)
       │
       ▼
 CoViscoCollator
  ├─ 批次级 TokenPlan 采样（覆盖 per-sample plan）
  ├─ ★ 视频 batch + query_and_vit 策略时：
  │    以 uniform_train_prob(50%) 概率决定走均匀抽帧路径
  │    → uniform_mode=True 时清空 visidx，batch 写入 uniform_sample_frames=True
  ├─ tokenizer.apply_chat_template()
  ├─ _adjust_image_pad_count() 修正 pad token 数
  ├─ tokenizer() → input_ids
  ├─ _build_labels() → 仅 assistant 位置有效 label
  └─ 图像/视频 transform → pixel_values
       │
       ▼
 LlavaCoViscoModel.forward()
  ├─ ViT: pixel_values → (query_tokens, vit_tokens)
  │    ├─ visidx 路径：128帧 + visidx → 每段 1024 sparse patch → ViT
  │    └─ 均匀路径：128帧 → 均匀采样32帧 → 每段 2048 dense patch → ViT
  ├─ Token Selector: 按 TokenPlan 选取 top-K vit token
  ├─ _arrange_tokens(): 拼接 query + selected_vit
  ├─ Projector: 1024 → 2048
  ├─ 注入 LLM embedding（替换 <|image_pad|> 位置）
  └─ Qwen3-1.7B forward → CE loss + sparsity loss
```

---

## 3. 数据处理

### 3.1 数据集：WDSDataset

**文件：** `data/wds_dataset.py` — `CoViscoWDSDataset`

- `IterableDataset`，直接读取 WebDataset `.tar` 分片，无需 `webdataset` 包依赖。
- 内部 `TarShardReader` 按文件名前缀（`__key__`）聚合同一样本的多个文件（`.jpg`、`.json`、`.mp4`、`.npy`）。

**Shard 迭代策略：**
- 每次 `__iter__` epoch 计数器递增，shard 顺序以 `seed ^ epoch ^ id(self)` 为种子打乱。
- 多 worker 场景下按 worker id 切片：`order[wid::wnum]`，避免重复。

**样本组装（`_process_sample`）：**

| 步骤 | 操作 |
|------|------|
| 1 | 读取 `json["messages"]` 和 `json["modality"]`（图像/视频/文本），不存在时自动推断 |
| 2 | 视频：从 JPEG 帧文件或 mp4 字节流（OpenCV 解码）获取帧，对齐到 `target_frames=128`（不足补末帧，超出截断） |
| 3 | 加载 `visidx.npy`（稀疏候选 patch 索引，训练 token selector 的先验） |
| 4 | `plugin.compute_video_segments(n_frames, segment_t_size=32)` 计算 `num_segments` |
| 5 | `DynamicTokenConfig.sample(num_segments)` → `TokenPlan` |
| 6 | `plugin.process_messages()` 展开 `<image>`/`<video>` 为视觉 block |

**输出字段：**
```python
{
    "messages": [...],          # 含视觉 block 的对话
    "images":   [...],          # PIL Image 列表
    "videos":   [...],          # 帧 ndarray 列表
    "visidx":   ndarray|None,   # 稀疏候选 patch 索引
    "token_plan": TokenPlan,    # 当前样本的 token 策略
    "modality": "image"|"video"|"text"
}
```

---

### 3.2 Plugin：视觉占位符展开

**文件：** `data/plugin.py` — `CoViscoPlugin`

将消息中的 `<image>` / `<video>` 占位符替换为视觉 block：

```
<|vision_start|><|image_pad|><|image_pad|>...<|vision_end|>
                 ↑───── N 个 pad token ────↑
```

`N = num_query_per_seg × num_segments`（默认 100/seg），Collator 后续会根据 `TokenPlan` 调整实际数量。

---

### 3.3 动态 Token 策略

**文件：** `data/dynamic_strategy.py` — `DynamicTokenConfig` + `TokenPlan`

每个批次从三种策略中随机采样一种（1.7B 配置）：

| 策略 | 概率 | Token 来源 | Token 数/Seg |
|------|------|-----------|------------|
| `query_only` | 20% | 仅使用 learnable query summary | 100 |
| `vit_only` | 30% | 全量 ViT patch token（无 selector） | 256（图像）/ 1024（视频） |
| `query_and_vit` | 50% | Query + Selector 选出的 top-K ViT | 100 + K，K ∈ {64, 128, 204} |

`TokenPlan` dataclass：
```python
@dataclass
class TokenPlan:
    strategy: str           # query_only | vit_only | query_and_vit
    vit_per_seg: int        # top-K 数量（query_only 时为 0）
    arrangement: str        # interleave（query + vit 交错排列）
    num_segments: int       # 图像=1，视频=N
    num_query_per_seg: int = 100
    vit_per_seg_image: Optional[int] = None
    vit_per_seg_video: Optional[int] = None
```

---

### 3.4 Collator：批次组装

**文件：** `data/collator.py` — `CoViscoCollator`

**处理步骤：**

```
1. 批次级 TokenPlan 采样
   └─ 整个 batch 使用同一 TokenPlan（保证 pad 数一致，方便 padding）

2. ★ 视频均匀抽帧模式决策（uniform_mode）
   ├─ 条件：视频 batch AND TokenPlan.strategy == "query_and_vit"
   │         （vit_only 下占位符数已按 visidx 路径的 1024/段写死，uniform 的
   │          2048/段会导致 pad 数不匹配；query_only 下 vit token 不送 LLM，
   │          uniform 路径无意义）
   ├─ 以 uniform_train_prob（默认 50%）决定本 batch 走均匀路径
   └─ uniform_mode=True 时：清空各样本 visidx（encoder 不做 sparse gather），
      batch 输出 uniform_sample_frames=True / uniform_sample_n=32 /
      uniform_segment_t_size=8

3. Chat template
   └─ tokenizer.apply_chat_template(messages, tokenize=False) → 完整对话字符串

4. Pad count 修正（_adjust_image_pad_count）
   └─ 替换所有 <|vision_start|>...<|vision_end|> 块，写入正确 pad 数量
      （两路视频均为 4 段 × plan.total_tokens_per_seg，pad 数一致）

5. Tokenization
   └─ tokenizer(text, truncation=True, max_length=8192) → input_ids

6. Label 构建（_build_labels）
   ├─ 默认全为 -100（不参与 loss）
   └─ 对每个 assistant turn：
       ├─ 重新 tokenize 前缀（add_generation_prompt=True）→ start
       ├─ 重新 tokenize 完整文本 → end
       └─ labels[start:end] = token_ids（计算 loss）

7. 图像/视频 transform
   ├─ Resize(224, BICUBIC) → CenterCrop(224) → ToTensor()
   └─ Normalize(mean=(0.481,0.458,0.408), std=(0.269,0.261,0.276))

8. 右 padding 到批次内最长序列

9. 批次校验
   └─ 所有样本 token_plan 必须一致，否则 ValueError
```

**输出 batch keys：**
`input_ids` | `attention_mask` | `labels` | `pixel_values` | `modality` | `visidx` | `token_plan` | `uniform_sample_frames` | `uniform_sample_n` | `uniform_segment_t_size`

**CollatorConfig 新增字段：**

| 字段 | 默认值 | 说明 |
|------|--------|------|
| `uniform_sample_n` | 32 | 均匀路径从 128 帧降采样到的目标帧数 |
| `uniform_segment_t_size` | 8 | 均匀路径的 segment_t_size（32帧/8=4段） |
| `uniform_train_prob` | 0.5 | 触发均匀路径的概率（0=全 visidx，1=全均匀） |

---

## 4. 视频采样双路径

视频 batch 在 `query_and_vit` 策略下以 50% 概率交替走两条路径，两路在 LLM 侧看到的 token 数完全相同。

### 路径对比

| | visidx 稀疏路径 | 均匀抽帧路径 |
|---|---|---|
| **输入帧数** | 128 帧 | 128 帧（encoder 内部降采样） |
| **ViT 输入** | visidx 预选的稀疏 patch | 均匀采样 32 帧的全量 patch |
| **每段 ViT 输入 token 数** | ~1024（由 visidx 决定） | 2048（32帧/4段 × 256 patch/帧） |
| **segment_t_size** | 32（全局配置） | 8（uniform 路径局部配置） |
| **段数** | 128 ÷ 32 = **4** | 32 ÷ 8 = **4** |
| **Token Selector 输出** | top-K per segment | top-K per segment（同上） |
| **LLM 输入 token 数** | **(100 + K) × 4** | **(100 + K) × 4** ← 相同 |
| **`<|image_pad|>` 数** | (100 + K) × 4 | (100 + K) × 4 ← 相同 |
| **ViT 显存压力** | 低（稀疏） | 高（dense，约 2× attention 计算） |

### 触发条件

```python
# collator.py
if has_video_feat and batch_plan.strategy == "query_and_vit":
    uniform_mode = random.random() < config.uniform_train_prob  # 默认 0.5
```

`vit_only` 和 `query_only` 策略下 uniform_mode 始终为 False：
- `vit_only`：占位符数按 `vit_per_seg_video=1024` 写入，uniform 路径输出 2048 token/段会造成 pad 数不匹配
- `query_only`：vit token 不送 LLM，uniform 路径的额外计算量纯属浪费

### 代码路径

```
collator.py  uniform_mode=True
    → batch["uniform_sample_frames"] = True
    → batch["uniform_sample_n"] = 32
    → batch["uniform_segment_t_size"] = 8
    → batch["visidx"] = None

train.py: model(**batch)   # 三个字段自动透传

llava_covisco.py → covisco_vit.py → _covisco_encoder_src.py
    → get_uniform_frame_segments(pixel_values, sample_frames=32, segment_t_size=8)
        ├─ torch.linspace 均匀采样 128→32 帧
        ├─ 按 segment_t_size=8 切成 4 段
        └─ 全量 patch 进 ViT（不做 sparse gather）
```

---

## 5. DataLoader 配置

```python
# 图像 DataLoader
train_loader = DataLoader(
    train_dataset,
    batch_size=24,          # --micro-batch，每 GPU 每 forward 24 张图像
    collate_fn=collator,
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
)

# 视频 DataLoader（若提供 --video-data-path）
video_loader = DataLoader(
    video_dataset,
    batch_size=4,           # --video-micro-batch
    collate_fn=collator,
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
)
```

混合模态训练逻辑：每个 optimizer step 先处理一个图像 batch，再处理一个视频 batch，共同计算梯度后统一更新参数（梯度累积 `grad_accum=8`）。

---

## 6. 模型架构

### 6.1 ViT：CoVisco-L/14

**文件：** `models/covisco_vit.py` + `models/_covisco_encoder_src.py`

- 基础架构：SigLIP-L/14，patch_size=14，image_size=224，输出 patch 数 = (224/14)² = **256/帧**
- **visidx 路径**：每 32 帧为一个 segment，通过 visidx 稀疏选取约 **1024 patch/段** 进 ViT
- **均匀路径**：128 帧均匀采样至 32 帧，每 8 帧为一个 segment，全量 **2048 patch/段** 进 ViT
- 附带 **Q=100 个 learnable query token**，通过交叉注意力聚合视觉信息

**输入/输出（视频）：**
```
输入: pixel_values (B, 3, 128, H, W)
  ├─ visidx 路径: segment_t_size=32 → 4 段，sparse gather by visidx
  └─ 均匀路径:    均匀采样→32帧，segment_t_size=8 → 4 段，全量 patch

输出（两路相同）:
  query_tokens: (B, 4, 100, 1024)   # 每 segment 100 个 query 摘要 token
  vit_tokens:   (B, 4, P,   1024)   # P=1024(visidx) 或 P=2048(uniform)
```

两路经过 Token Selector 选出相同数量的 top-K，最终 LLM 侧 token 数完全一致。

预训练权重由 `covisco_vit_checkpoint.py::load_vit_weights_direct` 加载，仅注入 `encoder` 子模块。

---

### 6.2 Token Selector

**文件：** `models/_token_selector_src.py` — `LearnableTokenSelector`

轻量级 2 层 Transformer，从 ViT patch token 中选出信息量最大的 top-K 个。

**单层架构（`TokenSelectorLayer`）：**
```
vit_tokens
    │
    ├─ Self-Attention (vit tokens 自注意力)
    │
    ├─ Cross-Attention (vit tokens 作为 Q，query_tokens 作为 KV)
    │
    └─ FFN (4× 扩展, GELU)

每个子层：pre-norm (LayerNorm) + 残差
```

**打分与选择：**
```python
# 最终打分头
score = sigmoid(LayerNorm(features) @ W_score)   # (B, S, P)

# 可微分 top-K（Straight-Through）
selected_idx = scores.topk(K).indices            # forward: 硬选择
selected_tokens = vit_tokens[selected_idx]
selected_tokens = selected_tokens * scores[selected_idx]  # backward: 连续梯度
```

**`gumbel` 模式**（可选）：训练时加 logistic 噪声 `log(u) - log(1-u)`，增强探索。

---

### 6.3 Projector

**文件：** `models/projector.py` — `TwoLayerMLPProjector`

```
Linear(1024 → 4096)  →  GELU  →  Linear(4096 → 2048)
   ↑ ViT hidden                       ↑ LLM hidden
```

将视觉 token 维度从 1024 映射到 LLM 的 2048，`ffn_mult=2`（中间层 = 2 × LLM hidden = 4096）。

---

### 6.4 LLM：Qwen3-1.7B

- 通过 `AutoModelForCausalLM.from_pretrained` 加载，`torch_dtype=bfloat16`
- `image_pad_token_id = 151655`（`<|image_pad|>`）用于定位注入位置
- SFT 阶段默认**冻结**（仅 projector 和 token_selector 可训）
- 内部计算 CE loss，仅在 `labels != -100` 的位置生效

---

### 6.5 顶层模型 Forward

**文件：** `models/llava_covisco.py` — `LlavaCoViscoModel.forward()`

```
Step 1: 纯文本捷径
    if pixel_values is None → 直接调用 LLM，跳过后续步骤

Step 2: ViT 前向（支持视频双路径）
    query_tokens, vit_tokens = self.vit(
        pixel_values, visidx,
        uniform_sample_frames, uniform_sample_n, uniform_segment_t_size
    )

Step 3: Token 策略分支
    ┌─ query_only:    selected_vit = None
    │                 （训练时 dummy selector forward × 0，保证 DDP 梯度连通）
    │
    ├─ vit_only:      selected_vit = vit_tokens（全量 patch）
    │                 （训练时 dummy selector forward × 0；不与 uniform 路径同时触发）
    │
    └─ query_and_vit: sel_out = token_selector(query_tokens, vit_tokens, top_k=K)
                      selected_vit = sel_out["selected_tokens"]   # (B, S, K, 1024)
                      → 计算 sparsity_loss（见第 7 节）
                      （uniform 路径仅在此策略下触发，vit_tokens P=2048 或 1024，
                       selector 统一选出 K 个，LLM 侧 token 数相同）

Step 4: Token 拼接（_arrange_tokens）
    query_only:     out = query_tokens.reshape(B, S×100, 1024)
    vit_only:       out = selected_vit.reshape(B, S×P, 1024)
    query_and_vit:  out = cat([query_seg_0, vit_seg_0, query_seg_1, vit_seg_1, ...])
                         → (B, S×(100+K), 1024)

Step 5: Projector
    vision_embeds = projector(all_vision)     # (B, total_tokens, 2048)

Step 6: 注入 LLM embedding
    inputs_embeds = emb_layer(input_ids)      # (B, seq_len, 2048)
    pad_mask = (input_ids == 151655)          # <|image_pad|> 位置
    inputs_embeds[pad_mask] = vision_embeds.reshape(-1, 2048)

Step 7: LLM 前向
    lm_out = language_model(inputs_embeds=inputs_embeds, labels=labels)

Step 8: 叠加稀疏性损失
    total_loss = lm_out.loss + sparsity_loss
```

---

## 7. 损失函数

| 损失项 | 公式 | 激活条件 |
|--------|------|---------|
| LM 交叉熵 | `CE(logits, labels)`，仅 `labels != -100` | 始终 |
| Sparsity anchor | `λ × MSE(top-10% scores, 1) × 0.10` | `query_and_vit` + 训练 |
| Sparsity rest | `λ × mean(bottom-90% scores) × 0.90` | `query_and_vit` + 训练 |
| **总 Loss** | `CE + λ × (anchor + rest)` | — |

`λ = sparsity_lambda = 0.2`（1.7B 配置）

**Sparsity 损失的设计意图：** 推动 token selector 的 sigmoid 分数二极化——最重要的 top-10% patch 分数趋向 1，其余 90% 趋向 0，使选择过程更确定性。

---

## 8. 训练循环

### 8.1 优化器与调度器

```python
# 优化器：AdamW，仅优化 trainable 参数
optimizer = AdamW(
    trainable_params,
    lr=3e-4,
    betas=(0.9, 0.999),
    weight_decay=0.01,
)

# 调度器：Linear warmup + constant
# warmup_steps = int(total_steps × 0.03)
# warmup 阶段：lr 从 0 线性增至 3e-4
# warmup 之后：lr 保持 3e-4 不变（constant）
scheduler = get_warmup_constant_scheduler(optimizer, warmup_steps)
```

**梯度处理：**
```python
# 梯度累积 8 步后更新
if (step + 1) % grad_accum == 0:
    torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
```

### 8.2 冻结策略

SFT 阶段（`--stage sft`）：

| 模块 | 参数量 | 状态 |
|------|--------|------|
| ViT（SigLIP-L/14） | ~300M | **冻结** |
| LLM（Qwen3-1.7B） | ~1.7B | **冻结**（默认；`--trainable-modules` 中不含 `llm`） |
| Projector | ~8M | **训练**（`--trainable-modules projector`） |
| Token Selector | ~4M | **训练**（`--trainable-modules token_selector`） |

`freeze_for_stage("sft", ["projector", "token_selector"])` 只对 `trainable-modules` 中的模块启用梯度。

### 8.3 混合模态训练

每个 optimizer step 的前向/反向顺序：

```
for accum_step in range(grad_accum):           # 8 次累积
    # 1. 图像 batch
    img_batch = next(image_loader_iter)
    img_loss = model(**img_batch).loss / (grad_accum * 2)
    img_loss.backward()

    # 2. 视频 batch（若可用）
    #    collator 已在 batch 中写入 uniform_sample_frames 等字段，model(**) 自动路由
    vid_batch = next(video_loader_iter)
    vid_loss = model(**vid_batch).loss / (grad_accum * 2)
    vid_loss.backward()

# 梯度裁剪 + 更新
clip_grad_norm_(params, 1.0)
optimizer.step()
scheduler.step()
optimizer.zero_grad()
```

---

## 9. 关键超参数（1.7B）

| 类别 | 参数 | 值 |
|------|------|----|
| **规模** | GPU 数 | 8（单节点 torchrun） |
| | 精度 | bfloat16 |
| **数据** | 图像样本数/epoch | 85M |
| | 训练 epoch | 2 |
| | 图像 micro-batch | 24/GPU |
| | 视频 micro-batch | 4/GPU |
| | 梯度累积步数 | 8 |
| | 有效 global batch | 1536 图像 + 256 视频/step |
| **优化** | 学习率 | 3e-4 |
| | Warmup ratio | 3% |
| | LR scheduler | constant（warmup 后恒定） |
| | 梯度裁剪 | max_norm=1.0 |
| | Weight decay | 0.01 |
| **Token 策略** | query_only 概率 | 30% |
| | vit_only 概率 | 30% |
| | query_and_vit 概率 | 40% |
| | top-K 选项 | {64, 128, 204} |
| **视频双路径** | uniform_sample_n | 32（均匀降采样到的帧数） |
| | uniform_segment_t_size | 8（均匀路径 segment 大小） |
| | uniform_train_prob | 0.5（仅 query_and_vit 策略时触发） |
| **正则化** | sparsity_lambda | 0.2 |
| **保存** | checkpoint 间隔 | 5000 steps |
| | log 间隔 | 10 steps |

---

## 10. 启动命令

```bash
# 单节点 8-GPU SFT（1.7B）
bash scripts/run_sft_1.7b.sh

# 关闭均匀路径（纯 visidx 模式）
UNIFORM_TRAIN_PROB=0.0 bash scripts/run_sft_1.7b.sh

# 自定义 epoch 数
NUM_EPOCHS=3 bash scripts/run_sft_1.7b.sh

# 输出目录
output/sft_1.7b/
├── checkpoint-{step}/   # 模型权重
└── runs/                # TensorBoard 日志
```
