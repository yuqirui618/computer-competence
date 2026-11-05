"""Fixed-threshold evaluation and patient-clustered paired AUC uncertainty."""

from collections import defaultdict

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from dataset.evaluate import evaluate_predictions


def _validated_arrays(targets, *probabilities):
    """Validate binary multilabel targets and aligned finite probabilities."""
    targets = np.asarray(targets)
    if targets.ndim != 2 or not targets.shape[0] or not targets.shape[1]:
        raise ValueError("Targets must be a nonempty two-dimensional array")
    if not np.isin(targets, (0, 1)).all():
        raise ValueError("Targets must contain only binary labels")
    arrays = []
    for values in probabilities:
        values = np.asarray(values)
        if values.shape != targets.shape:
            raise ValueError("Targets and probabilities must have matching shapes")
        if not np.isfinite(values).all() or not ((values >= 0) & (values <= 1)).all():
            raise ValueError("Probabilities must be finite and in [0, 1]")
        arrays.append(values)
    return targets, arrays


def _all_classes_defined(targets):
    return bool(((targets == 0).any(axis=0) & (targets == 1).any(axis=0)).all())


def _json_value(value):
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def evaluate_model(model, dataset, batch_size=12):
    """Evaluate every image once and return ordered study-level predictions.

    All classes must contain positive and negative examples. A study must belong
    to exactly one patient, and its image labels must agree. Metadata is returned
    to the caller but never included in progress messages or validation errors.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if not len(dataset):
        raise ValueError("Evaluation dataset must not be empty")
    device = next(model.parameters()).device
    use_cuda = device.type == "cuda"
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0,
        pin_memory=use_cuda, drop_last=False,
    )
    model.eval()
    target_batches, probability_batches = [], []
    image_subjects, image_studies = [], []
    processed = 0
    bce_sum, bce_count = 0.0, 0
    with torch.inference_mode():
        for batch_number, batch in enumerate(loader, 1):
            images = batch["image"].to(device, non_blocking=use_cuda, memory_format=torch.channels_last)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_cuda):
                logits = model(images)
            target_tensor = batch["target"].to(device, dtype=torch.float32, non_blocking=use_cuda)
            batch_bce = torch.nn.functional.binary_cross_entropy_with_logits(
                logits.float(), target_tensor, reduction="sum",
            )
            bce_sum += float(batch_bce.item())
            bce_count += target_tensor.numel()
            probabilities = logits.float().sigmoid().cpu().numpy()
            targets = batch["target"].detach().cpu().numpy()
            subjects = [str(value) for value in batch["subject_id"]]
            studies = [str(value) for value in batch["study_id"]]
            batch_count = len(images)
            if (len(targets) != batch_count or len(probabilities) != batch_count
                    or len(subjects) != batch_count or len(studies) != batch_count):
                raise ValueError("Evaluation batch arrays and metadata do not align")
            if any(not value.strip() for value in subjects + studies):
                raise ValueError("Evaluation metadata must not contain empty identifiers")
            target_batches.append(targets)
            probability_batches.append(probabilities)
            image_subjects.extend(subjects)
            image_studies.extend(studies)
            processed += batch_count
            if batch_number % 100 == 0 or batch_number == len(loader):
                print(
                    f"Evaluation: {batch_number}/{len(loader)} batches, "
                    f"{processed}/{len(dataset)} images", flush=True,
                )
    targets = np.concatenate(target_batches)
    probabilities = np.concatenate(probability_batches)
    targets, (probabilities,) = _validated_arrays(targets, probabilities)
    if len(targets) != len(dataset):
        raise ValueError("Evaluation did not cover the full dataset")
    if not _all_classes_defined(targets):
        raise ValueError("Every evaluation class must contain positive and negative examples")

    study_patient = {}
    grouped = defaultdict(list)
    for index, (subject, study) in enumerate(zip(image_subjects, image_studies)):
        previous = study_patient.setdefault(study, subject)
        if previous != subject:
            raise ValueError("A study is associated with more than one patient")
        grouped[(subject, study)].append(index)

    study_targets, study_probabilities = [], []
    study_subjects, study_ids = [], []
    for (subject, study), indices in grouped.items():
        target = targets[indices[0]]
        if not (targets[indices] == target).all():
            raise ValueError("Images within a study have inconsistent labels")
        study_targets.append(target)
        study_probabilities.append(probabilities[indices].mean(axis=0))
        study_subjects.append(subject)
        study_ids.append(study)
    study_targets = np.stack(study_targets)
    study_probabilities = np.stack(study_probabilities)
    if not _all_classes_defined(study_targets):
        raise ValueError("Every study-level class must contain positive and negative examples")

    metrics = evaluate_predictions(targets, probabilities, image_studies, threshold=0.5)
    if metrics["image"]["count"] != len(dataset) or metrics["study"]["count"] != len(grouped):
        raise ValueError("Shared evaluation counts disagree with the aggregated dataset")
    if any(metrics[level]["auc_classes"] != targets.shape[1] for level in ("image", "study")):
        raise ValueError("Shared evaluation did not compute AUC for every class")
    return {
        "metrics": _json_value(metrics),
        "bce": float(bce_sum / bce_count),
        "study_targets": study_targets,
        "study_probabilities": study_probabilities,
        "study_subjects": study_subjects,
        "study_ids": study_ids,
    }


def paired_auc_interval(targets, baseline_probs, candidate_probs, subjects,
                        seed=20261104, repeats=400):
    """Percentile interval for candidate minus baseline study macro AUC.

    Sample patients with replacement and include all of each sampled patient's
    studies, preserving pairing between models. Discard a replicate if any class
    lacks a positive or negative label. This is conditional on the fixed models
    and evaluation cohort; it does not measure training-seed uncertainty.
    """
    if not isinstance(repeats, (int, np.integer)) or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    targets, (baseline_probs, candidate_probs) = _validated_arrays(
        targets, baseline_probs, candidate_probs,
    )
    if not _all_classes_defined(targets):
        raise ValueError("Every original class must contain positive and negative examples")
    subjects = list(subjects)
    if len(subjects) != len(targets):
        raise ValueError("One patient identifier is required for each study")
    if any(subject is None or not str(subject).strip() for subject in subjects):
        raise ValueError("Patient identifiers must not be empty")
    patient_indices = defaultdict(list)
    for index, subject in enumerate(subjects):
        patient_indices[str(subject)].append(index)
    groups = [np.asarray(indices, dtype=np.intp) for indices in patient_indices.values()]
    delta = float(
        roc_auc_score(targets, candidate_probs, average="macro")
        - roc_auc_score(targets, baseline_probs, average="macro")
    )
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(repeats):
        sampled = rng.integers(0, len(groups), size=len(groups))
        indices = np.concatenate([groups[index] for index in sampled])
        sampled_targets = targets[indices]
        if not _all_classes_defined(sampled_targets):
            continue
        difference = (
            roc_auc_score(sampled_targets, candidate_probs[indices], average="macro")
            - roc_auc_score(sampled_targets, baseline_probs[indices], average="macro")
        )
        differences.append(float(difference))
    ci95 = np.quantile(differences, [0.025, 0.975]).tolist() if differences else [None, None]
    return {
        "delta": delta,
        "ci95": ci95,
        "valid_repeats": len(differences),
        "repeats": int(repeats),
    }