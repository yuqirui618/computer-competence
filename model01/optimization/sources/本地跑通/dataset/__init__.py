"""Manifest-driven CXR datasets and the shared train/inference preprocessing."""

from functools import partial
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .tools.build_cache import cache_image_path
from .tools.build_manifest import NUM_CLASSES, SPLIT_NAMES, data_root_path, image_files, image_ids, parse_label_ids, read_manifest, split_name


def list_images(root: str) -> list[dict]:
    """List images under DATA_ROOT or one split, preserving the official IDs."""
    directory = data_root_path(root)
    splits = sorted(path for path in directory.iterdir() if path.name in SPLIT_NAMES and path.is_dir())
    result = []
    for folder in splits or [directory]:
        for path in image_files(folder):
            subject_id, study_id, _ = image_ids(path)
            result.append({"study_id": study_id, "subject_id": subject_id, "path": str(path)})
    return result


def fit_rgb(image, img_size):
    return ImageOps.pad(image.convert("RGB"), (img_size, img_size), Image.Resampling.BILINEAR, color=0)


def build_transform(is_train: bool, img_size: int = 384):
    """Preserve aspect ratio, pad with black and apply ImageNet normalization."""
    if img_size < 1:
        raise ValueError("img_size must be positive")
    steps = [partial(fit_rgb, img_size=img_size)]
    if is_train:
        steps.append(transforms.RandomAffine(degrees=5, translate=(0.02, 0.02), scale=(0.95, 1.05),
                                             interpolation=transforms.InterpolationMode.BILINEAR))
    steps.extend((transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))))
    return transforms.Compose(steps)


class CXRDataset(Dataset):
    """Read one split and return image, ten binary targets and original IDs."""

    def __init__(self, manifest, data_root, split, transform=None, cache_dir=None):
        self.split = split_name(split)
        self.data_root = data_root_path(data_root)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.transform = transform if transform is not None else build_transform(self.split == "train")
        self.rows = [row for row in read_manifest(manifest) if row["split"] == self.split]
        if not self.rows:
            raise ValueError(f"No {self.split} rows in {manifest}")
        self.labels = [tuple(label - 1 for label in parse_label_ids(row["label_ids"])) for row in self.rows]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        path = self.data_root / row["image_path"]
        if self.cache_dir is not None:
            cached = cache_image_path(self.cache_dir, row["image_path"])
            if cached.is_file():
                path = cached
        with Image.open(path) as image:
            image = self.transform(image)
        target = torch.zeros(NUM_CLASSES, dtype=torch.float32)
        target[list(self.labels[index])] = 1
        return {"image": image, "target": target, "study_id": row["study_id"], "subject_id": row["subject_id"]}


def resolve_manifest(cfg, root):
    manifest = Path(cfg.get("manifest", "manifest.csv")).expanduser()
    if not manifest.is_absolute() and not manifest.is_file():
        manifest = root / manifest
    return manifest


def build_loaders(cfg: dict) -> tuple[DataLoader, DataLoader]:
    """Create the train and validation loaders from the team config."""
    root = data_root_path(cfg.get("data_root"))
    manifest = resolve_manifest(cfg, root)
    image_size = int(cfg.get("img_size", 384))
    cache = cfg.get("cache_dir")
    train = CXRDataset(manifest, root, "train", build_transform(True, image_size), cache)
    val = CXRDataset(manifest, root, "val", build_transform(False, image_size), cache)
    settings = {"batch_size": int(cfg.get("batch_size", 8)), "num_workers": int(cfg.get("num_workers", 0)),
                "pin_memory": bool(cfg.get("pin_memory", torch.cuda.is_available()))}
    return DataLoader(train, shuffle=True, **settings), DataLoader(val, shuffle=False, **settings)
