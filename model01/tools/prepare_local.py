"""Prepare the verified experiment layout in a new E-drive workspace."""

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

import yaml


MODULE = Path(__file__).resolve().parents[1]
REPOSITORY = MODULE.parent


def resolve_path(value, config_directory):
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = config_directory / path
    return path.resolve()


def make_launcher(python):
    executable = str(python).replace("'", "''")
    script = r'''param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('smoke', 'baseline', 'long')]
    [string]$Phase
)
$ErrorActionPreference = 'Stop'
$runRoot = $PSScriptRoot
$cacheRoot = Join-Path $runRoot 'temp'
New-Item -ItemType Directory -Path $cacheRoot -Force | Out-Null
$env:TEMP = $cacheRoot
$env:TMP = $cacheRoot
$env:TMPDIR = $cacheRoot
$env:CUDA_CACHE_PATH = Join-Path $cacheRoot 'cuda'
$env:HF_HOME = Join-Path $cacheRoot 'huggingface'
$env:TORCH_HOME = Join-Path $cacheRoot 'torch'
$env:XDG_CACHE_HOME = $cacheRoot
$env:HF_HUB_OFFLINE = '1'
$env:HF_HUB_DISABLE_TELEMETRY = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONPATH = Join-Path $runRoot '本地跑通'
$python = '__PYTHON__'
$arguments = @()
switch ($Phase) {
    'smoke' {
        $entry = Join-Path $runRoot '本地跑通\run_local.py'
        $arguments = @('--subset', '2000', '--epochs', '2', '--workers', '0')
    }
    'baseline' { $entry = Join-Path $runRoot '全量基线\train_full.py' }
    'long' { $entry = Join-Path $runRoot 'MambaOut_100轮\train_and_optimize.py' }
}
& $python -u $entry @arguments
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
'''
    return script.replace("__PYTHON__", executable)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True, help="New directory on E: that does not exist yet")
    parser.add_argument("--weights", type=Path, required=True, help="Downloaded ImageNet model.safetensors")
    parser.add_argument("--config", type=Path, default=MODULE / "configs/data_local.example.yaml")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    args = parser.parse_args()
    if sys.platform != "win32":
        parser.error("These experiment snapshots require Windows, CUDA and an E-drive workspace")
    if not args.work_dir.is_absolute():
        parser.error("--work-dir must be an absolute E-drive path")
    work_dir = args.work_dir.resolve()
    if work_dir.drive.upper() != "E:":
        parser.error("--work-dir must resolve to the E drive")
    if work_dir.exists():
        parser.error("Choose a new workspace. Resume existing experiments with their training script")
    config_path = args.config.resolve()
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        parser.error("The configuration must be a YAML mapping")
    for key in ("data_root", "manifest", "cache_dir"):
        value = getattr(args, key) or cfg.get(key)
        cfg[key] = str(resolve_path(value, config_path.parent)) if value else None
    if not cfg["data_root"] or not Path(cfg["data_root"]).is_dir():
        parser.error("The official data_root directory must already exist")
    if not cfg["manifest"] or not Path(cfg["manifest"]).is_file():
        parser.error("The dataset manifest must already exist")
    if cfg["cache_dir"] and not Path(cfg["cache_dir"]).is_dir():
        parser.error("cache_dir must be an existing cache, or null in the configuration")
    weights = args.weights.resolve()
    metadata_path = MODULE / "licenses/weight_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    with weights.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    if digest != metadata["weight_sha256"]:
        parser.error("The ImageNet weight SHA-256 differs from the recorded source")

    work_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".model01-prepare-", dir=work_dir.parent) as temporary:
        stage = Path(temporary)
        smoke, baseline, long_run = (stage / name for name in ("本地跑通", "全量基线", "MambaOut_100轮"))
        for directory in (smoke / "pretrain_model", smoke / "outputs", smoke / "checkpoints",
                          baseline / "checkpoints", long_run):
            directory.mkdir(parents=True, exist_ok=True)
        for source_name, destination in (("run_local.py", smoke), ("train_full.py", baseline),
                                         ("train_and_optimize.py", long_run)):
            shutil.copyfile(MODULE / "src" / source_name, destination / source_name)
        for source in sorted((REPOSITORY / "dataset").rglob("*.py")):
            destination = smoke / "dataset" / source.relative_to(REPOSITORY / "dataset")
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
        shutil.copyfile(REPOSITORY / "03_datasets/class_mapping.csv", smoke / "class_mapping.csv")
        shutil.copyfile(weights, smoke / "pretrain_model/model.safetensors")
        shutil.copyfile(metadata_path, smoke / "pretrain_model/weight_metadata.json")
        shutil.copyfile(MODULE / "licenses/LICENSE.upstream.txt", smoke / "pretrain_model/LICENSE.upstream.txt")
        (smoke / "data_local.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
        (stage / "start.ps1").write_text(make_launcher(Path(sys.executable)), encoding="utf-8-sig")
        stage.rename(work_dir)
    print(f"Prepared {work_dir}. Run start.ps1 -Phase smoke, then baseline, then long.")


if __name__ == "__main__":
    main()
