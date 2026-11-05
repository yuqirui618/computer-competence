"""
最小训练循环（陈负责）：BCE / RGB Loss + AMP。

验收：python train.py --config configs/base.yaml --subset 2000 --epochs 2
"""
from __future__ import annotations

import argparse
import os

import torch
from torch.amp import GradScaler, autocast

from model import build_model          # 自己的
from losses import build_loss            # RGB Loss / BCE
from utils import load_config, seed_everything   # 邓
from dataset import build_loaders                  # 喻


def parse_args():
    p = argparse.ArgumentParser(description="MambaOut 多标签分类最小训练循环")
    p.add_argument("--config", type=str, default="configs/base.yaml")
    p.add_argument("--subset", type=int, default=0, help="只取前 N 张训练图，0=全量")
    p.add_argument("--epochs", type=int, default=None, help="覆盖 config 里的 epochs")
    p.add_argument("--ckpt", type=str, default="checkpoints/smoke.pth")
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    seed = args.seed if args.seed is not None else cfg.get("seed", 42)
    seed_everything(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    epochs = args.epochs if args.epochs is not None else int(cfg.get("epochs", 2))
    subset = args.subset

    model = build_model(
        num_classes=int(cfg.get("num_classes", 10)),
        pretrained=bool(cfg.get("pretrained", True)),
        drop_path=float(cfg.get("drop_path", 0.0)),
    ).to(device)

    train_loader, _ = build_loaders(cfg)

    # 分组学习率：新分类头默认比骨干大 10 倍
    lr = float(cfg.get("lr", 1e-4))
    head_lr = float(cfg.get("head_lr", lr * 10))
    backbone = [p for n, p in model.named_parameters() if not n.startswith("head.")]
    head = [p for n, p in model.named_parameters() if n.startswith("head.")]
    optimizer = torch.optim.AdamW(
        [{"params": backbone, "lr": lr}, {"params": head, "lr": head_lr}],
        weight_decay=float(cfg.get("weight_decay", 0.05)),
    )

    loss_name = os.environ.get("LOSS", "BCE").upper()
    criterion = build_loss(loss_name, num_classes=int(cfg.get("num_classes", 10)))
    use_amp = bool(cfg.get("amp", True)) and device == "cuda"
    scaler = GradScaler() if use_amp else None

    ckpt_path = os.path.abspath(args.ckpt)
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

    for epoch in range(epochs):
        model.train()
        total_loss, total_n = 0.0, 0
        for batch in train_loader:
            if subset and total_n >= subset:
                break
            x = batch["image"].to(device)
            y = batch["target"].to(device)

            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                with autocast(device_type="cuda"):
                    logits = model(x)
                    loss = criterion(logits, y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * x.size(0)
            total_n += x.size(0)

        avg = total_loss / max(total_n, 1)
        print(f"[train] epoch {epoch + 1}/{epochs}  loss({loss_name})={avg:.4f}  samples={total_n}", flush=True)

    torch.save({"model": model.state_dict(), "config": cfg, "loss": loss_name}, ckpt_path)
    print(f"[train] saved checkpoint -> {ckpt_path}", flush=True)


if __name__ == "__main__":
    main()