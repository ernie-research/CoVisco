# SMS (Spatial Merge Size) 配置

## 1. 当前默认：sms=1

`TwoLayerMLPProjector` 当前默认 `sms=1`：把 vision token **1:1** 投到 LLM hidden size，**不做 spatial merge**。

```python
TwoLayerMLPProjector(vision_hidden=1024, llm_hidden=2560, sms=1, ffn_mult=2)
# 等价于：
#   fc1: Linear(1024, 5120)
#   fc2: Linear(5120, 2560)
```

## 2. 预留 sms=2/3 的扩展点

`sms=2` / `sms=3` 的路径**已经留好接口**，但 **forward 实现待补**：

```python
class TwoLayerMLPProjector(nn.Module):
    def forward(self, x, spatial_grid=None):
        if self.sms == 1:
            return self.fc2(self.act(self.fc1(x)))
        raise NotImplementedError(
            f"sms={self.sms} 的 spatial merge 路径待实现。"
            "需要把 token reshape 成 (B, h, w, vision_hidden) 后做 sms x sms 块 flatten。"
        )
```

## 3. 未来扩展

要把 sms 扩展到 2 或 3，需要：

1. **TokenSelector 输出**已知每段 vit 的空间位置（通过 `indices` 字段拼回 (h, w) 网格）
2. **Projector.forward** 接收 `spatial_grid=(h, w)`，把 `x` reshape 成 `(B, h, w, vision_hidden)`
3. **2D 块 flatten** 把 `sms × sms` 邻域拼成 `vision_hidden * sms^2` 维

具体实现留到 sms 真正需要时再做。
