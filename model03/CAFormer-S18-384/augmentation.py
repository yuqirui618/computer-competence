"""Train-only, label-preserving photometric/noise probes for native384 Study bags.

All geometric operations exactly retain FullFieldTransform's raw RGB pad and
PIL affine. Extra randomness never uses global NumPy/Python/Torch RNGs. Named
streams pair each operation across single-operation and combined arms.

Exact zero-background rule: pixels whose RGB values are all zero AFTER the
shared geometry remain exactly zero in every added augmentation. This includes
padding/affine fill and any naturally black source pixels; no lung mask or
clinical annotation is inferred. Validation/test must use the unchanged
RawStudyDataset/RawImageDataset, or policy='none' with is_train=False.
"""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision.transforms import functional as TF

from data_pipeline import FullFieldTransform, RawStudyDataset

POLICY_NAMES = ('none', 'noise', 'photo', 'photo_noise')
RECIPE = {
    'policies': list(POLICY_NAMES),
    'geometry': 'Unchanged RGB ImageOps.pad384 BICUBIC; train PIL RandomAffine5deg/translate.02/scale.95..1.05',
    'extra_operations_space': 'float32 unit RGB after shared geometry, before ImageNet normalization',
    'noise': {
        'probability': 0.5, 'sigma_uniform': [0.003, 0.01],
        'distribution': 'zero-mean Gaussian, independent per spatial pixel, same noise for all RGB channels',
    },
    'photo': {
        'operation_order': ['gamma', 'contrast', 'brightness'],
        'gamma': {'probability': 0.5, 'uniform': [0.9, 1.1]},
        'contrast': {'probability': 0.5, 'uniform': [0.9, 1.1], 'center': 0.5},
        'brightness': {'probability': 0.5, 'uniform': [0.95, 1.05]},
    },
    'combined_order': 'gamma -> contrast -> brightness -> clip[0,1] -> noise -> clip[0,1] -> restore zero pixels',
    'zero_rule': 'All-RGB-zero pixels after original geometry remain zero; includes padding/fill and natural source black pixels',
    'randomness': 'SHA256(JSON[scheme, seed, epoch, relative_path with forward slashes, named operation]); local NumPy PCG64',
    'random_stream_scheme': 'cxr-native384-augmentation-v1',
    'global_RNG_extra_consumption': False,
    'labels_changed': False,
    'validation_or_test_augmentation': False,
    'no_extra_geometric_crops_flips_or_rotations': True,
}


def recipe():
    """Return a detached, JSON-serializable description for run registration."""
    return copy.deepcopy(RECIPE)


def _valid(policy, seed, epoch):
    if policy not in POLICY_NAMES:
        raise ValueError('Unknown augmentation policy: ' + str(policy))
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError('Augmentation seed must be a nonnegative Python int')
    if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 0:
        raise ValueError('Augmentation epoch must be a nonnegative Python int')


def _rng(seed, epoch, key, operation):
    key = str(key).replace('\\', '/')
    if not key:
        raise ValueError('An image path key is required for train augmentation')
    payload = json.dumps(
        [RECIPE['random_stream_scheme'], seed, epoch, key, operation],
        ensure_ascii=False, separators=(',', ':'),
    ).encode('utf-8')
    digest = hashlib.sha256(payload).digest()
    return np.random.Generator(np.random.PCG64(int.from_bytes(digest, 'big')))


def augmentation_parameters(policy, seed, epoch, key):
    """Pure metadata; paired operation streams make ablation comparisons exact."""
    _valid(policy, seed, epoch)
    result = {'gamma': None, 'contrast': None, 'brightness': None, 'noise_sigma': None}
    if policy in ('photo', 'photo_noise'):
        for operation in ('gamma', 'contrast', 'brightness'):
            spec = RECIPE['photo'][operation]
            rng = _rng(seed, epoch, key, operation + '-gate-parameter')
            if rng.random() < spec['probability']:
                result[operation] = float(rng.uniform(*spec['uniform']))
    if policy in ('noise', 'photo_noise'):
        rng = _rng(seed, epoch, key, 'noise-gate-sigma')
        if rng.random() < RECIPE['noise']['probability']:
            result['noise_sigma'] = float(rng.uniform(*RECIPE['noise']['sigma_uniform']))
    return result


def augment_unit_rgb(unit_rgb, policy, seed, epoch, key):
    """Augment CPU float32 RGB [3,H,W] without changing any global RNG state.

    Used only inside the train transform. Exposed for small CPU numerical and
    label/zero-background audit; it does not resize, normalize, or inspect labels.
    """
    params = augmentation_parameters(policy, seed, epoch, key)
    if not isinstance(unit_rgb, torch.Tensor) or unit_rgb.device.type != 'cpu':
        raise TypeError('Augmentation expects a CPU tensor')
    if unit_rgb.dtype != torch.float32 or unit_rgb.ndim != 3 or unit_rgb.shape[0] != 3:
        raise ValueError('Expected unit float32 RGB [3,H,W]')
    if policy == 'none' or all(value is None for value in params.values()):
        return unit_rgb
    # ToTensor returns contiguous CPU float32; copy before applying any changes.
    value = unit_rgb.numpy().copy()
    if not np.isfinite(value).all() or np.any((value < 0) | (value > 1)):
        raise ValueError('Input to augmentation must be finite and in [0,1]')
    zero_pixels = np.all(value == 0, axis=0, keepdims=True)
    if params['gamma'] is not None:
        value = np.power(value, np.float32(params['gamma']))
    if params['contrast'] is not None:
        value = (value - np.float32(0.5)) * np.float32(params['contrast']) + np.float32(0.5)
    if params['brightness'] is not None:
        value = value * np.float32(params['brightness'])
    value = np.clip(value, np.float32(0), np.float32(1))
    if params['noise_sigma'] is not None:
        noise = _rng(seed, epoch, key, 'noise-pixels').standard_normal(
            (1, value.shape[1], value.shape[2]), dtype=np.float32)
        value = value + noise * np.float32(params['noise_sigma'])
    value = np.clip(value, np.float32(0), np.float32(1))
    value = np.where(zero_pixels, np.float32(0), value)
    if not np.isfinite(value).all():
        raise FloatingPointError('Nonfinite augmented image')
    return torch.from_numpy(np.ascontiguousarray(value, dtype=np.float32))


class AugmentedFullFieldTransform(FullFieldTransform):
    """Same geometry as native384, with isolated optional train-only changes."""
    def __init__(self, size, mean, std, is_train=False, policy='none', seed=42):
        _valid(policy, seed, 0)
        if not is_train and policy != 'none':
            raise ValueError('Extra augmentation is forbidden for validation/test')
        super().__init__(size, mean, std, is_train)
        self.policy = policy
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch):
        _valid(self.policy, self.seed, epoch)
        self.epoch = epoch

    def __call__(self, image, key=None):
        if self.policy == 'none':
            # Do not reimplement the control path: preserve its exact values/RNG.
            return super().__call__(image)
        if key is None:
            raise ValueError('A relative image path key is required')
        if image.mode in ('I', 'F') or image.mode.startswith('I;16'):
            raise ValueError('High-bit-depth image requires explicit normalization; refusing silent RGB truncation')
        image = ImageOps.pad(image.convert('RGB'), (self.size, self.size),
                             method=Image.Resampling.BICUBIC, color=0)
        if self.affine is not None:
            image = self.affine(image)
        unit = augment_unit_rgb(TF.to_tensor(image), self.policy, self.seed, self.epoch, key)
        # Exact normalization object/mean/std inherited from original pipeline.
        return self.tensor.transforms[1](unit)


class AugmentedStudyDataset(RawStudyDataset):
    """Original Study metadata and return contract, plus explicit epoch control."""
    def __init__(self, records, data_root, mean, std, is_train=True, policy='none', seed=42):
        super().__init__(records, data_root, mean, std, is_train)
        self.transform = AugmentedFullFieldTransform(384, mean, std, is_train, policy, seed)
        self.policy = policy
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch):
        self.transform.set_epoch(epoch)
        self.epoch = epoch

    def __getitem__(self, index):
        images = []
        for relative in self.records[index]['image_paths']:
            with Image.open(self.root / relative) as image:
                images.append(self.transform(image, key=relative))
        return torch.stack(images), torch.from_numpy(self.targets[index]), index
