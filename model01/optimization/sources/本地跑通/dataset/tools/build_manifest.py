"""Join the official image directories and CSV files into a per-image manifest."""

import argparse
import csv
import os
import re
from pathlib import Path, PurePosixPath


SPLIT_NAMES = {"train": "train", "val": "val", "valid": "val", "validation": "val", "test": "test"}
OFFICIAL_SPLITS = {"train": "train", "val": "valid", "test": "test"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
FIELDS = ("study_id", "subject_id", "image_path", "split", "label_ids", "view_position")
NUM_CLASSES = 10


def data_root_path(value=None):
    root = value or os.environ.get("DATA_ROOT")
    if not root:
        raise ValueError("Set DATA_ROOT or supply the data_root path.")
    path = Path(root).expanduser().resolve()
    if not path.is_dir():
        raise NotADirectoryError(path)
    return path


def split_name(value):
    try:
        return SPLIT_NAMES[value.strip().lower()]
    except KeyError as error:
        raise ValueError(f"Unknown split: {value!r}") from error


def parse_label_ids(value):
    if not value.strip():
        return ()
    tokens = re.split(r"[;|]", value)
    if any(not token.strip().isdigit() for token in tokens):
        raise ValueError(f"Invalid label IDs: {value!r}")
    labels = tuple(sorted({int(token) for token in tokens}))
    if not all(1 <= label <= NUM_CLASSES for label in labels):
        raise ValueError(f"Label IDs must be in 1–{NUM_CLASSES}: {value!r}")
    return labels


def relative_image_path(value):
    path = PurePosixPath(value.replace("\\", "/"))
    if path.is_absolute() or not path.parts or ".." in path.parts or any(":" in part for part in path.parts):
        raise ValueError(f"Expected a path relative to DATA_ROOT: {value!r}")
    return path.as_posix()


def image_files(root):
    return sorted(path for path in root.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS and path.is_file())


def image_ids(path):
    subject, study = path.parent.parent.name, path.parent.name
    if not subject.startswith("subject_") or not study.startswith("study_"):
        raise ValueError(f"Expected subject_*/study_*/image layout: {path}")
    return subject, study, path.stem


def build_rows(root):
    rows = []
    for split, source in OFFICIAL_SPLITS.items():
        directory = root / source
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        indexed = {}
        for path in image_files(directory):
            key = image_ids(path)
            if key in indexed:
                raise ValueError(f"Multiple images have the same identifiers: {path}")
            indexed[key] = path
        studies = {}
        with (root / f"{source}.csv").open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"Subject_id", "Study_id", "Dicom_id", "ViewPosition", "Rows", "Columns"}
            if split != "test":
                required.add("Predict_class")
            if not required.issubset(reader.fieldnames or []):
                raise ValueError(f"Missing required columns in {source}.csv")
            for line, record in enumerate(reader, 2):
                subject, study, dicom = (record[k].strip() for k in ("Subject_id", "Study_id", "Dicom_id"))
                path = indexed.pop((subject, study, dicom), None)
                if path is None:
                    raise ValueError(f"{source}.csv:{line}: missing image or duplicate metadata row")
                labels = () if split == "test" else parse_label_ids(record["Predict_class"])
                if split != "test" and not labels:
                    raise ValueError(f"{source}.csv:{line}: missing labels")
                if studies.setdefault(study, (subject, labels)) != (subject, labels):
                    raise ValueError(f"{source}.csv:{line}: inconsistent subject or labels within a study")
                rows.append({"study_id": study, "subject_id": subject,
                             "image_path": path.relative_to(root).as_posix(), "split": split,
                             "label_ids": "|".join(map(str, labels)), "view_position": record["ViewPosition"].strip(),
                             "width": int(record["Columns"]), "height": int(record["Rows"])})
        if indexed:
            raise ValueError(f"{source}: {len(indexed)} images are missing CSV metadata")
        if not studies:
            raise ValueError(f"{source}: empty split")
    return rows


def read_manifest(path):
    rows, seen, studies = [], set(), {}
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not set(FIELDS).issubset(reader.fieldnames or []):
            raise ValueError(f"Manifest must have columns: {', '.join(FIELDS)}")
        for line, record in enumerate(reader, 2):
            row = {key: record[key].strip() for key in FIELDS}
            row["image_path"] = relative_image_path(row["image_path"])
            row["split"] = split_name(row["split"])
            labels = parse_label_ids(row["label_ids"])
            row["label_ids"] = "|".join(map(str, labels))
            if not row["study_id"] or not row["subject_id"] or bool(labels) != (row["split"] != "test"):
                raise ValueError(f"Manifest row {line}: invalid IDs or labels for {row['split']}")
            study = (row["split"], row["study_id"])
            if studies.setdefault(study, (row["subject_id"], labels)) != (row["subject_id"], labels):
                raise ValueError(f"Manifest row {line}: inconsistent subject or labels within a study")
            key = row["image_path"].casefold()
            if key in seen:
                raise ValueError(f"Manifest row {line}: duplicate image path")
            seen.add(key)
            rows.append(row)
    return rows


def write_csv(path, fields, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", help="Official dataset directory (defaults to DATA_ROOT)")
    parser.add_argument("--output", type=Path, default=Path("manifest.csv"))
    args = parser.parse_args()
    rows = build_rows(data_root_path(args.data_root))
    write_csv(args.output, FIELDS, rows)
    print(f"{args.output}: {len(rows):,} image rows")


if __name__ == "__main__":
    main()
