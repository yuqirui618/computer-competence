"""
延迟测量（陈负责）：224 / 384 两组 decode / forward / total。

用法：python benchmark.py --ckpt checkpoints/smoke.pth --images DIR --sizes 224 384 --num 50
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch
from PIL import Image

from model import build_model, load_checkpoint
from dataset import build_transform, list_images   # 喻


def benchmark_one(model, transform, device, path):
    """返回 (decode_ms, forward_ms, total_ms)。"""
    # decode：读文件 + 解码为 RGB
    t0 = time.perf_counter()
    img = Image.open(path).convert("RGB")
    t1 = time.perf_counter()

    x = transform(img).unsqueeze(0).to(device)

    # forward：模型前向（前后都同步，防异步计时漏算）
    if device == "cuda":
        torch.cuda.synchronize()
    t2 = time.perf_counter()
    with torch.inference_mode():
        logits = model(x)
    if device == "cuda":
        torch.cuda.synchronize()
    t3 = time.perf_counter()

    _ = torch.sigmoid(logits)   # 概率化（计入 total）
    t4 = time.perf_counter()

    return (t1 - t0) * 1000, (t3 - t2) * 1000, (t4 - t0) * 1000


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--images", required=True)
    p.add_argument("--sizes", type=int, nargs="+", default=[224, 384])
    p.add_argument("--num", type=int, default=50)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    device = args.device

    items = list_images(args.images)
    paths = [it["path"] for it in items][: args.num + args.warmup]
    if not paths:
        print("[benchmark] 未找到图片，退出", flush=True)
        return

    for size in args.sizes:
        model = build_model(num_classes=10, pretrained=False)
        load_checkpoint(model, args.ckpt, strict=True)
        model.to(device).eval()
        transform = build_transform(is_train=False, img_size=size)

        # 预热
        for path in paths[: args.warmup]:
            benchmark_one(model, transform, device, path)

        dec, fwd, tot = [], [], []
        for path in paths[args.warmup : args.warmup + args.num]:
            d, f, t = benchmark_one(model, transform, device, path)
            dec.append(d); fwd.append(f); tot.append(t)

        print(
            f"[benchmark] img_size={size}  decode={np.mean(dec):.2f}ms  "
            f"forward={np.mean(fwd):.2f}ms  total={np.mean(tot):.2f}ms  (n={len(dec)})",
            flush=True,
        )


if __name__ == "__main__":
    main()