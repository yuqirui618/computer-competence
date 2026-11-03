"""
MambaOut-Tiny 多标签分类模型（陈负责）。

接口（与《今日计划》第四节完全一致）：
    build_model(num_classes=10, pretrained=True, drop_path=0.0) -> nn.Module
    load_checkpoint(model, ckpt_path, strict=True) -> nn.Module

权重来源：
    timm/mambaout_tiny.in1k（ImageNet-1K / Apache-2.0 / 非 gated）
    固定 revision: d080941663a60c42d015ecf8d8fad86d12694208
    输出为 10 个 logits（不是 Softmax）；损失用 BCEWithLogitsLoss，推理时 sigmoid。
"""
from __future__ import annotations

import torch
import torch.nn as nn


def build_model(
    num_classes: int = 10,
    pretrained: bool = True,
    drop_path: float = 0.0,
) -> nn.Module:
    """构建 MambaOut-Tiny。

    num_classes=10 会把 ImageNet 的 1000 类头替换成 10 类头，骨干权重仍从
    ImageNet 预训练权重加载（timm 只跳过头，不会静默随机初始化骨干）。
    """
    import timm

    model = timm.create_model(
        "hf-hub:timm/mambaout_tiny.in1k",
        pretrained=pretrained,
        num_classes=num_classes,
        drop_path_rate=drop_path,
    )
    return model


def load_checkpoint(model: nn.Module, ckpt_path: str, strict: bool = True) -> nn.Module:
    """从 checkpoint 加载权重。

    兼容两种保存格式：
      - 直接是 state_dict
      - {"model": state_dict, ...} 或 {"state_dict": state_dict, ...}
    并自动剥掉 DDP 的 "module." 前缀。
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if isinstance(ckpt, dict):
        if "model" in ckpt:
            state = ckpt["model"]
        elif "state_dict" in ckpt:
            state = ckpt["state_dict"]
        else:
            state = ckpt
    else:
        state = ckpt

    if hasattr(state, "state_dict"):
        state = state.state_dict()

    state = {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}
    model.load_state_dict(state, strict=strict)
    return model


if __name__ == "__main__":
    # 快速自检：python model.py
    m = build_model()
    print("参数量(M):", round(sum(p.numel() for p in m.parameters()) / 1e6, 2))
    print("输出形状:", tuple(m(torch.randn(2, 3, 384, 384)).shape))