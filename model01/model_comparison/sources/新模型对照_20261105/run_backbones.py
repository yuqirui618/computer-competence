"""Five-epoch full-data ImageNet-backbone pilots; single GPU, E-drive outputs.

--check-only verifies inputs using CPU only. --smoke performs one train batch
per selected model without saving weights or counting an epoch. Normal mode
runs exactly five epochs per successful model, retaining all epoch artifacts.
"""
from __future__ import annotations
import argparse
import ctypes
import gc
import hashlib
import json
import math
import msvcrt
import os
from pathlib import Path
import sys
import time
import traceback
from collections import OrderedDict
from contextlib import contextmanager
from datetime import datetime

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent
# Cache paths are set before importing torch/timm; execute this E-drive copy.
if ROOT.drive.upper() != 'E:':
    raise RuntimeError('This runner must execute from the E drive')
TEMP = ROOT / 'temp'
for key, value in {'TEMP': TEMP, 'TMP': TEMP, 'TMPDIR': TEMP,
        'XDG_CACHE_HOME': TEMP, 'CUDA_CACHE_PATH': TEMP / 'cuda',
        'HF_HOME': TEMP / 'huggingface', 'TORCH_HOME': TEMP / 'torch'}.items():
    Path(value).mkdir(parents=True, exist_ok=True)
    os.environ[key] = str(value)
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '4'

import numpy as np
import torch
import timm
import yaml
from safetensors.torch import load_file
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

sys.path.insert(0, str(DATA / '本地跑通'))
sys.path.insert(0, str(DATA / 'MambaOut_泛化优化'))
sys.path.insert(0, str(DATA / 'MambaOut_100轮'))
import train_and_optimize as ref
from experiment_metrics import paired_auc_interval
from dataset.tools.build_cache import cache_image_path

CONFIG = DATA / '本地跑通/data_local.yaml'
MANIFEST = ROOT / 'pretrained_manifest.json'
CHAMPION = DATA / 'MambaOut_ASL对照/best_model.pth'
GATES = DATA / 'MambaOut_ASL对照/gates.json'
GAMMA = DATA / 'MambaOut_二次数据增强_20261104/C_gamma_noise/best.pth'
GAMMA_GATES = DATA / 'MambaOut_二次数据增强_20261104/candidate_gates.json'
SUBMISSION = DATA / '提交结果/MambaOut_S7微调第1轮_20261104/submission.csv'
EXPECTED = {
    str(CHAMPION): '9911ef5383bb320a4770a544b85e27d58c01ab57a69bb0963392a208073a2585',
    str(GAMMA): 'b0066e479b1b0c383fa7daa2067ff0d6c954d20fbf81d1202af95481d67cc5f9',
    str(SUBMISSION): 'd49b6a7ef02489af1ed2cc38f8f49bc5f592b6519f7ee24399d4e18ab95eeae8',
}
MODEL_NAMES = ('convnext_tiny', 'efficientnetv2_s')
COMMON = dict(epochs=5, seed=42, image_size=224, batch_size=12, workers=0,
    cpu_threads=4, backbone_lr=3e-5, classifier_lr=3e-4, weight_decay=.05,
    drop_rate=.2, drop_path_rate=.1, loss='BCEWithLogits, no class weighting',
    train_images=30344, train_studies=18513, train_patients=11055,
    validation_images=1422, validation_studies=859, validation_patients=228,
    lr_factors=[.1 + .9 * (1 + math.cos(math.pi * k / 4)) / 2 for k in range(5)],
    train_source='512 JPEG cache, then pad/resize224 and original mild affine',
    validation_source='raw images, deterministic pad/resize224; no BN update',
    study_aggregation='mean image probability',
    selection='Study macro AUC, then F1 at 0.5; best of exactly five epochs',
    comparison_limits='Different backbone/initialization and short shared recipe; not a proof of architecture superiority. Official validation reused. EfficientNetV2 native pretraining sizes are larger than224.')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def save_checkpoint(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def freeze(path, value):
    path = Path(path)
    if path.is_file():
        if json.loads(path.read_text(encoding='utf-8')) != value:
            raise ValueError(f'Frozen input changed: {path}')
    else:
        save(path, value)


def status(**fields):
    save(ROOT / 'status.json', dict(pid=os.getpid(), time=datetime.now().astimezone().isoformat(), **fields))


def seed(value):
    import random
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)


def score(metrics):
    return metrics['study']['macro_auc'], metrics['study']['macro_f1']


def protected_hashes(cfg):
    paths = (CHAMPION, GATES, GAMMA, GAMMA_GATES, SUBMISSION, Path(cfg['manifest']))
    receipt = {str(p): sha(p) for p in paths}
    for p, expected in EXPECTED.items():
        if receipt[p] != expected:
            raise ValueError(f'Protected source changed: {p}')
    freeze(ROOT / 'protected_inputs.json', receipt)
    return receipt


def source_hashes():
    paths = (Path(__file__).resolve(), CONFIG,
        DATA / 'MambaOut_100轮/train_and_optimize.py',
        DATA / 'MambaOut_泛化优化/experiment_metrics.py',
        DATA / '本地跑通/dataset/__init__.py')
    return {str(p): sha(p) for p in paths}


def model_info(name):
    metadata = json.loads(MANIFEST.read_text(encoding='utf-8-sig'))['models'][name]
    architecture = metadata['architecture']
    expected_architecture = {'convnext_tiny': 'convnext_tiny.fb_in1k',
        'efficientnetv2_s': 'tf_efficientnetv2_s.in1k'}[name]
    if architecture != expected_architecture:
        raise ValueError('Unexpected model architecture or pretraining channel')
    directory = (ROOT / 'pretrained' / metadata['directory']).resolve()
    if not directory.is_relative_to((ROOT / 'pretrained').resolve()):
        raise ValueError('Pretrained path escaped its directory')
    weights = directory / 'model.safetensors'
    actual_sha = sha(weights)
    if actual_sha != metadata['sha256']:
        raise ValueError(f'Pretrained weight SHA differs: {name}')
    config_path = directory / 'config.json'
    config = json.loads(config_path.read_text(encoding='utf-8')) if config_path.is_file() else metadata['config']
    pretrained_cfg = config.get('pretrained_cfg', config)
    mean, std = pretrained_cfg['mean'], pretrained_cfg['std']
    if len(mean) != 3 or len(std) != 3 or not all(math.isfinite(float(x)) for x in mean + std) or not all(float(x) > 0 for x in std):
        raise ValueError('Invalid model normalization')
    return dict(name=name, architecture=architecture, weights=str(weights), sha256=actual_sha,
        mean=list(mean), std=list(std), pretrained_config=config,
        parameters_before_head_reset=config.get('num_classes', 1000))


def transform(is_train, info):
    result = ref.build_transform(is_train, COMMON['image_size'])
    matches = [i for i, t in enumerate(result.transforms) if isinstance(t, transforms.Normalize)]
    if len(matches) != 1:
        raise ValueError('Unexpected reference preprocessing')
    result.transforms[matches[0]] = transforms.Normalize(info['mean'], info['std'])
    return result


def datasets(cfg, info):
    train = ref.CXRDataset(cfg['manifest'], cfg['data_root'], 'train', transform(True, info), cfg['cache_dir'])
    val = ref.CXRDataset(cfg['manifest'], cfg['data_root'], 'val', transform(False, info))
    if len(train) != 30344 or len(val) != 1422:
        raise ValueError('Official image coverage differs')
    return train, val


def data_audit(cfg, info):
    train, val = datasets(cfg, info)
    subjects, studies = {}, {}
    for split in ('train', 'val', 'test'):
        ds = train if split == 'train' else val if split == 'val' else ref.CXRDataset(cfg['manifest'], cfg['data_root'], 'test', transform(False, info))
        subjects[split] = set(row['subject_id'] for row in ds.rows)
        studies[split] = set(row['study_id'] for row in ds.rows)
    if len(subjects['train']) != 11055 or len(subjects['val']) != 228 or len(studies['train']) != 18513 or len(studies['val']) != 859:
        raise ValueError('Official study/patient coverage differs')
    for a, b in (('train', 'val'), ('train', 'test'), ('val', 'test')):
        if subjects[a] & subjects[b]:
            raise ValueError('Patient leakage across official split')
    missing = [row['image_path'] for row in train.rows if not cache_image_path(Path(cfg['cache_dir']), row['image_path']).is_file()]
    if missing:
        raise ValueError(f'Missing training cache images: {len(missing)}')
    for row in val.rows:
        if not (Path(cfg['data_root']) / row['image_path']).is_file():
            raise ValueError('Missing raw validation image')
    first = train[0]
    if tuple(first['image'].shape) != (3, 224, 224) or tuple(first['target'].shape) != (10,) or not torch.isfinite(first['image']).all():
        raise ValueError('Invalid training sample shape or normalization')
    if not torch.isin(first['target'], torch.tensor([0., 1.])).all():
        raise ValueError('Invalid binary targets')
    ids = list(OrderedDict.fromkeys(row['study_id'] for row in val.rows))
    return dict(train_images=len(train), train_studies=len(studies['train']), train_patients=len(subjects['train']),
        validation_images=len(val), validation_studies=len(studies['val']), validation_patients=len(subjects['val']),
        patient_splits_disjoint=True, all_training_cache_present=True), ids


def construct(info, state_dict=None, device='cuda'):
    seed(42)
    model = timm.create_model(info['architecture'], pretrained=False,
        num_classes=1000, drop_rate=.2, drop_path_rate=.1)
    if state_dict is None:
        model.load_state_dict(load_file(info['weights'], device='cpu'), strict=True)
        model.reset_classifier(10)
    else:
        model.reset_classifier(10)
        model.load_state_dict(state_dict, strict=True)
    if device == 'cuda':
        model = model.to(device='cuda', memory_format=torch.channels_last)
    return model


def optimizer_for(model):
    classifier = list(model.get_classifier().parameters())
    identifiers = {id(p) for p in classifier}
    backbone = [p for p in model.parameters() if id(p) not in identifiers]
    if not classifier or not backbone:
        raise ValueError('Invalid classifier/backbone grouping')
    return torch.optim.AdamW([{'params': backbone, 'lr': COMMON['backbone_lr']},
        {'params': classifier, 'lr': COMMON['classifier_lr']}], weight_decay=.05)


def save_predictions(path, pred, ids):
    y, p, s = pred
    if y.shape != (859, 10) or p.shape != y.shape or len(s) != 859 or len(np.unique(s)) != 228 or len(ids) != 859:
        raise ValueError('Unexpected validation prediction coverage')
    if not np.isfinite(p).all() or not ((p >= 0) & (p <= 1)).all():
        raise ValueError('Invalid validation probabilities')
    np.savez_compressed(path, targets=y, probabilities=p, subjects=s, study_ids=np.asarray(ids))


def bn_buffers(model):
    return {name: value.detach().cpu().clone() for name, value in model.named_buffers()
        if name.endswith(('running_mean', 'running_var', 'num_batches_tracked'))}


def validate(model, val):
    before = bn_buffers(model)
    start = time.perf_counter()
    metrics, pred = ref.validate(model, val, keep_predictions=True)
    elapsed = time.perf_counter() - start
    after = bn_buffers(model)
    if before.keys() != after.keys() or any(not torch.equal(before[k], after[k]) for k in before):
        raise ValueError('Validation changed BN buffers')
    return metrics, pred, elapsed


def extras(pred):
    y, p, s = pred
    z = p >= .5
    return dict(macro_ap=float(average_precision_score(y, p, average='macro')),
        ap_by_class=average_precision_score(y, p, average=None).tolist(),
        tp=np.sum(z & (y == 1), 0).tolist(), fp=np.sum(z & (y == 0), 0).tolist(),
        fn=np.sum(~z & (y == 1), 0).tolist())


def train_epoch(model, train, optimizer, scaler, name, epoch, completed):
    seed(42 + epoch)
    factor = COMMON['lr_factors'][epoch - 1]
    for group, initial_lr in zip(optimizer.param_groups, (COMMON['backbone_lr'], COMMON['classifier_lr'])):
        group['lr'] = initial_lr * factor
    loader = DataLoader(train, batch_size=12, shuffle=True, num_workers=0, pin_memory=True,
        generator=torch.Generator().manual_seed(42 + epoch))
    model.train()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    count, loss_sum, next_progress, scale_decreases = 0, 0., 3000, 0
    for batch in loader:
        x = batch['image'].to('cuda', non_blocking=True, memory_format=torch.channels_last)
        y = batch['target'].to('cuda', non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.float16):
            logits = model(x)
        loss = F.binary_cross_entropy_with_logits(logits.float(), y)
        value = loss.item()
        if not math.isfinite(value):
            raise FloatingPointError(f'{name} epoch{epoch} images{count}: nonfinite BCE')
        previous_scale = scaler.get_scale()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scale_decreases += int(scaler.get_scale() < previous_scale)
        count += len(x)
        loss_sum += value * len(x)
        if count >= next_progress or count == 30344:
            status(state='training', stage=name, epoch=epoch, images=count, total_images=30344,
                completed_epochs=completed, train_bce=loss_sum / count)
            print(f'TRAIN {name} epoch={epoch}/5 images={count}/30344 BCE={loss_sum/count:.7f}', flush=True)
            next_progress += 3000
    torch.cuda.synchronize()
    if count != 30344:
        raise ValueError('Incomplete training epoch')
    return dict(epoch=epoch, images=count, train_loss=loss_sum/count,
        train_seconds=time.perf_counter() - started,
        learning_rates=[group['lr'] for group in optimizer.param_groups],
        grad_scaler_scale_decreases=scale_decreases,
        peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20)


def smoke(info, cfg):
    train, val = datasets(cfg, info)
    model = construct(info)
    optimizer = optimizer_for(model)
    scaler = torch.amp.GradScaler('cuda')
    batch = next(iter(DataLoader(train, batch_size=12, shuffle=False, num_workers=0)))
    model.train()
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast('cuda', dtype=torch.float16):
        logits = model(batch['image'].to('cuda', memory_format=torch.channels_last))
    loss = F.binary_cross_entropy_with_logits(logits.float(), batch['target'].to('cuda'))
    if not torch.isfinite(loss):
        raise FloatingPointError('Smoke loss not finite')
    scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
    model.eval()
    before = bn_buffers(model)
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
        p = model(batch['image'].to('cuda', memory_format=torch.channels_last)).float().sigmoid()
    if tuple(p.shape) != (12, 10) or not torch.isfinite(p).all():
        raise ValueError('Smoke output invalid')
    after = bn_buffers(model)
    if any(not torch.equal(before[k], after[k]) for k in before):
        raise ValueError('Smoke eval updated BN')
    result = dict(state='passed', model=info['name'], scope='One 12-image training batch, not a full epoch; no weights saved',
        train_loss=float(loss.item()), output_shape=list(p.shape),
        allocated_mib=torch.cuda.max_memory_allocated()/2**20)
    save(ROOT / f"smoke_{info['name']}.json", result)
    print('SMOKE ' + json.dumps(result), flush=True)
    del model, optimizer, scaler, batch, p, loss, logits, train, val
    gc.collect(); torch.cuda.empty_cache()


def run_stage(info, cfg, ids, protected):
    name = info['name']
    folder = ROOT / name
    folder.mkdir(exist_ok=True)
    plan = dict(common=COMMON, model=info, source_hashes=source_hashes(), protected_hashes=protected)
    freeze(folder / 'plan.json', plan)
    existing = folder / 'results.json'
    if existing.is_file():
        prior = json.loads(existing.read_text(encoding='utf-8'))
        if prior['state'] == 'complete':
            print(f'ALREADY_COMPLETE {name}', flush=True)
            return prior
        raise ValueError(f'Prior failed run requires inspection: {name}')
    train, val = datasets(cfg, info)
    last_path = folder / 'last_training.pth'
    resume = torch.load(last_path, map_location='cpu', weights_only=True) if last_path.is_file() else None
    if resume is not None and resume['plan'] != plan:
        raise ValueError('Resume plan differs')
    model = construct(info, None if resume is None else resume['state_dict'])
    optimizer = optimizer_for(model)
    scaler = torch.amp.GradScaler('cuda')
    history = []
    best_epoch, best_metrics = None, None
    if resume is not None:
        optimizer.load_state_dict(resume['optimizer']); scaler.load_state_dict(resume['scaler'])
        history = resume['history']; best_epoch = resume['best_epoch']; best_metrics = resume['best_metrics']
        if [row['epoch'] for row in history] != list(range(1, len(history)+1)) or len(history) > 5:
            raise ValueError('Invalid resume epoch sequence')
        expected = max(history, key=lambda row: score(row['validation']))['epoch']
        if best_epoch != expected:
            raise ValueError('Resume best epoch differs')
        for row in history:
            if not (folder / f"epoch_{row['epoch']}.pth").is_file() or not (folder / f"epoch_{row['epoch']}_predictions.npz").is_file():
                raise ValueError('Missing committed epoch artifact')
        del resume
    completed = {name: len(history)}
    for epoch in range(len(history)+1, 6):
        row = train_epoch(model, train, optimizer, scaler, name, epoch, completed)
        status(state='validation', stage=name, epoch=epoch, completed_epochs=completed, images=0, total_images=1422)
        measured, pred, seconds = validate(model, val)
        save_predictions(folder / f'epoch_{epoch}_predictions.npz', pred, ids)
        row.update(validation=measured, validation_seconds=seconds,
            calibration=ref.evaluate_gates(pred), extra=extras(pred))
        state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        checkpoint = dict(state_dict=state, training=dict(**COMMON, **info, best_epoch=epoch))
        save_checkpoint(folder / f'epoch_{epoch}.pth', checkpoint)
        if best_epoch is None or score(measured) > score(best_metrics):
            best_epoch, best_metrics = epoch, measured
            save_checkpoint(folder / 'best.pth', checkpoint)
        history.append(row)
        save_checkpoint(last_path, dict(state_dict=state, optimizer=optimizer.state_dict(), scaler=scaler.state_dict(),
            plan=plan, history=history, best_epoch=best_epoch, best_metrics=best_metrics))
        save(folder / 'history.json', history)
        completed[name] = epoch
        status(state='checkpointed', stage=name, epoch=epoch, completed_epochs=completed,
            auc=score(measured)[0], f1=score(measured)[1], oof_f1=row['calibration']['oof_macro_f1'], best_epoch=best_epoch)
        print(f"EPOCH_DONE {name} epoch={epoch}/5 AUC={score(measured)[0]:.10f} F1={score(measured)[1]:.10f} OOF_F1={row['calibration']['oof_macro_f1']:.10f} best_epoch={best_epoch}", flush=True)
        del state, checkpoint, pred
        if (ROOT / 'pause_after_epoch.flag').is_file():
            status(state='paused_after_epoch', stage=name, epoch=epoch, completed_epochs=completed)
            raise SystemExit(0)
    del model, optimizer, scaler
    gc.collect(); torch.cuda.empty_cache()
    # Independent GPU reconstruction of this model's best validation epoch.
    status(state='best_reload_verification', stage=name, epoch=best_epoch, images=0, total_images=1422)
    selected = torch.load(folder / 'best.pth', map_location='cpu', weights_only=True)
    if selected['training']['best_epoch'] != best_epoch:
        raise ValueError('Best checkpoint epoch metadata differs')
    model = construct(info, selected['state_dict']); del selected
    measured, pred, verify_seconds = validate(model, val)
    with np.load(folder / f'epoch_{best_epoch}_predictions.npz', allow_pickle=False) as saved:
        for key, array in (('targets', pred[0]), ('subjects', pred[2])):
            if not np.array_equal(saved[key], array):
                raise ValueError('Reload labels/subjects differ')
        difference = float(np.max(np.abs(saved['probabilities']-pred[1])))
        np.testing.assert_allclose(saved['probabilities'], pred[1], rtol=0, atol=1e-7)
    for level in ('image', 'study'):
        for key in ('auc', 'f1', 'macro_auc', 'macro_f1'):
            np.testing.assert_allclose(measured[level][key], best_metrics[level][key], rtol=0, atol=1e-8)
    calibrated = ref.evaluate_gates(pred)
    save(folder / 'candidate_gates.json', dict(**calibrated,
        checkpoint=str(folder/'best.pth'), checkpoint_sha256=sha(folder/'best.pth'),
        class_ids=list(range(1, 11)), decision_rule='Study mean probabilities >= class Gate',
        source='Reused official validation; fit final gates on all validation, patient-fold OOF separately', candidate_only=True))
    del model
    gc.collect(); torch.cuda.empty_cache()
    intervals = {}
    for label, prediction_path in (
        ('original_mambaout', DATA/'MambaOut_二次数据增强_20261104/baseline_predictions.npz'),
        ('latest_gamma_candidate', DATA/'MambaOut_二次数据增强_20261104/verified_candidate_predictions.npz')):
        with np.load(prediction_path, allow_pickle=False) as baseline:
            if not np.array_equal(baseline['targets'], pred[0]) or not np.array_equal(baseline['subjects'], pred[2]):
                raise ValueError('Comparison prediction ordering differs')
            intervals[label] = paired_auc_interval(pred[0], baseline['probabilities'], pred[1], pred[2], repeats=400)
    if protected_hashes(cfg) != protected:
        raise ValueError('Protected original changed after model run')
    best_row = next(row for row in history if row['epoch'] == best_epoch)
    result = dict(state='complete', name=name, architecture=info['architecture'], epochs_completed=len(history),
        best_epoch=best_epoch, best_validation=best_metrics, best_calibration=best_row['calibration'],
        best_checkpoint=str(folder/'best.pth'), best_checkpoint_sha256=sha(folder/'best.pth'),
        history=history, verification=dict(state='passed', epoch=best_epoch, max_abs_probability_difference=difference,
            metrics_reproduced=True, predictions_reproduced=True, validation_seconds=verify_seconds, bn_buffers_unchanged=True),
        paired_auc_intervals=intervals, protected_hashes_after=protected,
        no_test_inference=True, github_uploaded=False, stop_reason='Exactly five full-data epochs as requested')
    save(folder / 'results.json', result)
    print('MODEL_COMPLETE ' + json.dumps(dict(model=name, epochs=len(history), best_epoch=best_epoch, auc=score(best_metrics)[0])), flush=True)
    del train, val, pred
    gc.collect()
    return result


def check_only(names, cfg, protected):
    results = {}
    for name in names:
        info = model_info(name)
        audited, ids = data_audit(cfg, info)
        cpu_model = construct(info, device='cpu')
        parameters = sum(p.numel() for p in cpu_model.parameters())
        classifier_out = cpu_model.get_classifier().out_features
        if classifier_out != 10:
            raise ValueError('Classifier not reset to ten classes')
        del cpu_model; gc.collect()
        record = dict(state='passed', model=info, common=COMMON, data_audit=audited,
            parameters=parameters, cpu_weight_load_strict=True, source_hashes=source_hashes(),
            protected_hashes=protected, no_training_or_cuda_allocations=True)
        save(ROOT / f'preflight_{name}.json', record)
        results[name] = dict(state='passed', parameters=parameters, **audited)
    print('PREFLIGHT ' + json.dumps(results), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    select = parser.add_mutually_exclusive_group(required=True)
    select.add_argument('--model', choices=MODEL_NAMES)
    select.add_argument('--run-all', action='store_true')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--check-only', action='store_true')
    mode.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    names = MODEL_NAMES if args.run_all else (args.model,)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    ref.ARGS.batch_size = 12
    cfg = yaml.safe_load(CONFIG.read_text(encoding='utf-8'))
    protected = protected_hashes(cfg)
    if args.check_only:
        check_only(names, cfg, protected)
        return
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; refusing accidental CPU training')
    if (ROOT/'pause_after_epoch.flag').is_file():
        status(state='paused_before_start'); return
    stages = {}
    for name in names:
        info = model_info(name)
        preflight_path = ROOT / f'preflight_{name}.json'
        if not preflight_path.is_file():
            raise ValueError(f'Run --model {name} --check-only first')
        checked = json.loads(preflight_path.read_text(encoding='utf-8'))
        if checked['source_hashes'] != source_hashes() or checked['protected_hashes'] != protected or checked['model'] != info:
            raise ValueError(f'Preflight no longer matches source/input state: {name}')
        if args.smoke:
            smoke(info, cfg)
            continue
        try:
            stages[name] = run_stage(info, cfg, data_audit(cfg, info)[1], protected)
        except Exception as exc:
            failure = dict(state='failed', name=name, error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
            (ROOT/name).mkdir(exist_ok=True)
            save(ROOT/name/'failure.json', failure)
            stages[name] = failure
            print('MODEL_FAILED '+json.dumps(dict(model=name,error=failure['error']),ensure_ascii=False),flush=True)
            gc.collect(); torch.cuda.empty_cache()
            if protected_hashes(cfg) != protected:
                raise ValueError('Protected inputs changed during failed model run') from exc
    if args.smoke:
        if protected_hashes(cfg) != protected:
            raise ValueError('Protected inputs changed during smoke')
        return
    # Include any other already completed model, so single-model invocations merge.
    for name in MODEL_NAMES:
        path=ROOT/name/'results.json'
        if name not in stages and path.is_file():
            stages[name]=json.loads(path.read_text(encoding='utf-8'))
    complete = [s for s in stages.values() if s.get('state') == 'complete']
    selected = max(complete, key=lambda s: score(s['best_validation'])) if complete else None
    historical = json.loads((DATA/'MambaOut_100轮/baseline_100/history.json').read_text(encoding='utf-8'))
    history5 = [row for row in historical if row['epoch'] <= 5]
    baseline5 = max(history5,key=lambda row:score(row['validation']))
    result=dict(state='complete' if complete and all(s.get('state')=='complete' for s in stages.values()) else 'complete_with_failures',
        requested_models=list(names), stages=stages, total_new_epochs=sum(s.get('epochs_completed',0) for s in stages.values()),
        best_new_model=selected['name'] if selected else None,
        original_mambaout_auc=.8019769550500871, latest_gamma_auc=.8021774782195852,
        original_mambaout_first5_best=dict(epoch=baseline5['epoch'],validation=baseline5['validation'],
            scope='Historical first-five epochs, different optimizer recipe; descriptive comparison only'),
        protected_hashes_after=protected_hashes(cfg), no_test_inference=True, github_uploaded=False,
        common=COMMON, limitations=COMMON['comparison_limits'])
    save(ROOT/'results.json',result)
    status(state=result['state'],completed_epochs={k:v.get('epochs_completed',0) for k,v in stages.items()},
        best_new_model=result['best_new_model'], total_new_epochs=result['total_new_epochs'])
    print('COMPLETE '+json.dumps(dict(state=result['state'],models=list(stages),epochs=result['total_new_epochs'])),flush=True)


@contextmanager
def lock():
    with (ROOT/'training.lock').open('a+b') as stream:
        if stream.tell()==0:
            stream.write(b'0');stream.flush()
        stream.seek(0);msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
        try:
            yield
        finally:
            stream.seek(0);msvcrt.locking(stream.fileno(),msvcrt.LK_UNLCK,1)


if __name__=='__main__':
    with lock():
        awake=ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
        try:
            main()
        except Exception as exc:
            status(state='failed',error=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            if awake:
                ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)