"""
Predictor（陈负责）。cost.py 必须复用这里的 Predictor，禁止复制第二份。
推理预处理必须 from dataset import build_transform。

CLI：python predict.py --images DIR --ckpt C --out preds.csv
preds.csv（长表）：study_id,image_path,predict_class,probability   # 每图 10 行
"""
from __future__ import annotations

import argparse
import csv

import numpy as np
import torch

from model import build_model, load_checkpoint
from dataset import build_transform, list_images   # 喻


class Predictor:
    def __init__(self, ckpt_path: str, device: str = "cuda", img_size: int = 384):
        if device == "cuda" and not torch.cuda.is_available():
            print("[predict] CUDA 不可用，退回 CPU", flush=True)
            device = "cpu"
        self.device = device
        self.img_size = img_size

        self.model = build_model(num_classes=10, pretrained=False)
        load_checkpoint(self.model, ckpt_path, strict=True)
        self.model.to(self.device).eval()

        # 推理预处理统一走 build_transform(is_train=False)
        self.transform = build_transform(is_train=False, img_size=img_size)

    @torch.inference_mode()
    def predict_paths(self, paths: list) -> np.ndarray:
        """对一批图片路径做前向，返回 (N, 10) 的概率数组。"""
        from PIL import Image

        out = []
        for p in paths:
            img = Image.open(p).convert("RGB")
            x = self.transform(img)            # -> Tensor(3, img_size, img_size)，已归一化
            x = x.unsqueeze(0).to(self.device)
            prob = torch.sigmoid(self.model(x)).squeeze(0).float().cpu().numpy()
            out.append(prob)

        if not out:
            return np.zeros((0, 10), dtype=np.float32)
        return np.stack(out, axis=0).astype(np.float32)


def parse_args():
    p = argparse.ArgumentParser(description="MambaOut 推理出概率长表")
    p.add_argument("--images", type=str, required=True, help="图片目录")
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--out", type=str, default="preds.csv")
    p.add_argument("--img-size", type=int, default=384)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    items = list_images(args.images)          # [{"study_id","subject_id","path"}]
    paths = [it["path"] for it in items]

    predictor = Predictor(args.ckpt, device=args.device, img_size=args.img_size)
    probs = predictor.predict_paths(paths)     # (N, 10)

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["study_id", "image_path", "predict_class", "probability"])
        for it, p in zip(items, probs):
            for c in range(10):                # 类别 id 1..10，不是 0..9
                w.writerow([it["study_id"], it["path"], c + 1, f"{float(p[c]):.6f}"])

    print(f"[predict] 输出 {len(items)} 张图 x 10 类 -> {args.out}", flush=True)


if __name__ == "__main__":
    main()