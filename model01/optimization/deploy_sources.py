"""Verify archived optimization sources; deploy only with an explicit --apply.

Standard-library only. Historical scripts retain frozen paths and hashes, so
this tool intentionally supports only the original E:\\比赛数据集 data root.
It never imports, executes, trains, installs, or downloads anything.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import sys
import tempfile

PACKAGE = Path(__file__).resolve().parent
FROZEN_DATA_ROOT = Path("E:/比赛数据集")
ALLOWED_SUFFIXES = {".py", ".ps1", ".yaml", ".yml", ".md", ".json"}


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def relative_parts(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, str) or not value:
        raise ValueError(f"Invalid {label}: expected a nonempty relative path")
    if "\\" in value or ":" in value or "\x00" in value:
        raise ValueError(f"Invalid {label}: use a relative path with forward slashes")
    parts = tuple(value.split("/"))
    if any(part in {"", ".", ".."} for part in parts) or PurePosixPath(value).is_absolute():
        raise ValueError(f"Invalid {label}: path traversal or absolute path")
    return parts


def check_plain_path(path: Path) -> None:
    """Reject symlinks/junctions and existing nondirectory ancestors."""
    for current in reversed((path, *path.parents)):
        try:
            information = current.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(information.st_mode) or (
            getattr(information, "st_file_attributes", 0)
            & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
        ):
            raise ValueError(f"Symlink or reparse point is not permitted: {current}")
        if current != path and not stat.S_ISDIR(information.st_mode):
            raise ValueError(f"A parent path is not a directory: {current}")


def inside(path: Path, root: Path) -> None:
    if not path.resolve(strict=False).is_relative_to(root.resolve(strict=True)):
        raise ValueError(f"Path escaped the permitted directory: {path}")


def allowed_source(path: Path) -> bool:
    return path.suffix.lower() in ALLOWED_SUFFIXES or (
        path.suffix.lower() == ".txt" and path.name.upper().startswith("LICENSE")
    )


def preflight(data_root: Path) -> list[dict]:
    if os.name != "nt":
        raise ValueError("These historical sources require Windows and E:/比赛数据集")
    check_plain_path(FROZEN_DATA_ROOT)
    check_plain_path(data_root)
    expected_root = FROZEN_DATA_ROOT.resolve(strict=True)
    actual_root = data_root.resolve(strict=True)
    if actual_root != expected_root or actual_root.drive.upper() != "E:":
        raise ValueError("--data-root must resolve exactly to E:/比赛数据集")
    if not actual_root.is_dir():
        raise ValueError("The required data root is not a directory")

    manifest_path = PACKAGE / "source_manifest.json"
    check_plain_path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("source_manifest.json must contain a nonempty files list")
    sources_root = PACKAGE / "sources"
    check_plain_path(sources_root)
    if not sources_root.is_dir():
        raise ValueError("The sources directory is missing")

    records: list[dict] = []
    destinations: set[str] = set()
    for entry in files:
        if not isinstance(entry, dict):
            raise ValueError("Invalid source manifest entry")
        repository_parts = relative_parts(entry.get("repository_path"), "repository_path")
        destination_parts = relative_parts(entry.get("data_relative_path"), "data_relative_path")
        if len(repository_parts) < 2 or repository_parts[0] != "sources":
            raise ValueError("Every repository_path must be under sources/")
        expected_sha = entry.get("sha256")
        if not isinstance(expected_sha, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", expected_sha):
            raise ValueError("Invalid SHA256 in source manifest")
        expected_sha = expected_sha.lower()
        source = PACKAGE.joinpath(*repository_parts)
        target = actual_root.joinpath(*destination_parts)
        check_plain_path(source)
        check_plain_path(target)
        inside(source, sources_root)
        inside(target, actual_root)
        if not source.is_file() or not allowed_source(source) or not allowed_source(target):
            raise ValueError(f"Only archived source/configuration/license files are permitted: {source}")
        if source.suffix.lower() != target.suffix.lower():
            raise ValueError("Source and target file types differ")
        key = str(target).casefold()
        if key in destinations:
            raise ValueError(f"Duplicate target path: {target}")
        destinations.add(key)
        if digest(source) != expected_sha:
            raise ValueError(f"Archived source SHA256 mismatch: {source}")
        action = "copy"
        if target.exists():
            if not target.is_file() or digest(target) != expected_sha:
                raise ValueError(f"Refusing to overwrite an existing different file: {target}")
            action = "skip-identical"
        records.append({"source": source, "target": target, "sha256": expected_sha,
                        "action": action, "bytes": source.stat().st_size})
    return records


def copy_new(record: dict) -> None:
    """Publish one complete verified file; Windows rename refuses replacement."""
    source, target = record["source"], record["target"]
    check_plain_path(source)
    check_plain_path(target)
    inside(target, FROZEN_DATA_ROOT)
    if target.exists():
        if target.is_file() and digest(target) == record["sha256"]:
            return
        raise ValueError(f"Target changed after preflight; refusing overwrite: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    check_plain_path(target.parent)
    temporary_path: Path | None = None
    try:
        # The temporary file is created beside the E-drive destination, never on C.
        with tempfile.NamedTemporaryFile(mode="wb", dir=target.parent,
                prefix=f".{target.name}.deploy-", suffix=".tmp", delete=False) as output:
            temporary_path = Path(output.name)
            with source.open("rb") as input_stream:
                shutil.copyfileobj(input_stream, output)
        if digest(temporary_path) != record["sha256"]:
            raise ValueError(f"Copied file failed SHA256 verification: {target}")
        # preflight enforces Windows: os.rename raises when target already exists.
        os.rename(temporary_path, target)
        temporary_path = None
        if digest(target) != record["sha256"]:
            raise ValueError(f"Final deployed file SHA256 mismatch: {target}")
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=FROZEN_DATA_ROOT,
                        help="Must resolve exactly to E:/比赛数据集")
    parser.add_argument("--apply", action="store_true",
                        help="Write missing sources after all preflight checks; default is dry-run")
    args = parser.parse_args(argv)
    try:
        records = preflight(args.data_root)
        pending = [record for record in records if record["action"] == "copy"]
        identical = len(records) - len(pending)
        print(f"SHA256 preflight passed: {len(records)} sources; "
              f"{identical} identical targets; {len(pending)} missing targets.")
        for record in records:
            print(f"{record['action']}: {record['target']}")
        if not args.apply:
            print("DRY-RUN: no files or directories were written. Use --apply to deploy missing sources.")
            return 0
        for record in pending:
            copy_new(record)
        print(f"APPLY complete: {len(pending)} missing sources deployed; "
              f"{identical} identical files preserved. No scripts were executed.")
        return 0
    except (OSError, ValueError, TypeError, KeyError) as error:
        print(f"Deployment refused: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
