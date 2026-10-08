# 动态 Token 策略

训练时（`eval` 模式固定为 `query_and_vit`），每个 step 随机选一种 token 组合方式：

| 模式 | 选哪些 | 视觉 token 数（每段）| 总数（4 段）|
|---|---|---|---|
| `query_only` | 仅 query | 100 | 400 |
| `vit_only` | 仅 vit（query-guided selection） | K（随机从 `[64,128,256,512]` 抽）| 4K |
| `query_and_vit` | 两者都要 | 100 + K | 4 × (100 + K) |

## 1. 配置

```yaml
token_strategy:
  enabled: true
  p_query_only: 0.2
  p_vit_only: 0.2
  p_query_and_vit: 0.6
  vit_token_counts: [64, 128, 256, 512]
  arrangement: interleave  # or "concat"
```

## 2. arrangement

- `interleave`（默认）：按段内交错，token 序列为 `[q_seg0, v_seg0, q_seg1, v_seg1, ...]`
- `concat`：先放所有 query，再放所有 vit，token 序列为 `[all_q, all_v]`

## 3. Token selector

`LearnableTokenSelector`（来自 onevision 仓库）是一个 query-guided 的 vit token selector：

- 2 层轻量 transformer，每层做 vit-self-attn + vit×query cross-attn + ffn
- 可微 top-K（DynamicViT/EViT 风格的 score-gating）
- 支持 Gumbel-Top-K 随机松弛

可调参数（`token_selector` 配置）：
- `method`: `"straight_through"` | `"gumbel"`
- `score_activation`: `"sigmoid"` | `"softmax"`
- `num_layers`: selector 层数（默认 2）
- `num_heads`: selector attention head 数
- `gumbel_temperature`: Gumbel 温度

## 4. 训练阶段策略

- **Stage-1 alignment**：只训 projector + token_selector；可适当关掉 dynamic（固定 `query_and_vit`）
- **Stage-2 mid training**：开启 dynamic，让模型学会多种 token 组合下的稳定表示
- **Stage-3 SFT**：保持 dynamic，提高对不同长度的鲁棒性
