"""Recompute aggregate cohort counts only; no patient identifiers are emitted.

Run with the existing E-drive Python and -B. No image reads, Torch, or GPU use.
Default output is stdout; --output explicitly saves the aggregate JSON.
"""
import argparse
import collections
import csv
import hashlib
import json
import random
from pathlib import Path


def label_set(row):
    return frozenset(int(value) for value in row['label_ids'].split('|'))


def cohort_counts(rows):
    studies = {}
    patient_labels = collections.defaultdict(set)
    patient_studies = collections.defaultdict(set)
    images_per_study = collections.Counter()
    for row in rows:
        subject, study, labels = row['subject_id'], row['study_id'], label_set(row)
        record = (subject, labels)
        if study in studies and studies[study] != record:
            raise ValueError('Inconsistent patient or labels within a Study')
        studies[study] = record
        patient_labels[subject].update(labels)
        patient_studies[subject].add(study)
        images_per_study[study] += 1
    positive_studies = [sum(label in record[1] for record in studies.values()) for label in range(1, 11)]
    positive_patients = [sum(label in labels for labels in patient_labels.values()) for label in range(1, 11)]
    study_counts_desc = sorted((len(value) for value in patient_studies.values()), reverse=True)
    total_studies = len(studies)
    top_totals = {str(k): sum(study_counts_desc[:k]) for k in (1, 5, 10)}
    return {
        'images': len(rows), 'studies': total_studies, 'patients': len(patient_labels),
        'positive_studies_class1to10': positive_studies,
        'positive_patients_class1to10': positive_patients,
        'study_prevalence_pct_class1to10': [100 * value / total_studies for value in positive_studies],
        'patients_with_multiple_studies': sum(len(value) > 1 for value in patient_studies.values()),
        'studies_per_patient_mean': total_studies / len(patient_labels),
        'top_patients_study_totals': top_totals,
        'top_patients_study_percentage': {key: 100 * value / total_studies for key, value in top_totals.items()},
        'studies_with_multiple_images': sum(value > 1 for value in images_per_study.values()),
        'study_image_count_histogram': dict(sorted(collections.Counter(images_per_study.values()).items())),
        'view_position_image_counts': dict(sorted(collections.Counter(row['view_position'] for row in rows).items())),
    }


def patient_assignments(rows, seed):
    subjects = sorted({row['subject_id'] for row in rows})
    random.Random(seed).shuffle(subjects)
    count = round(0.15 * len(subjects))
    return {
        subject: ('audit' if index < count else 'early_stop' if index < 2 * count else 'train')
        for index, subject in enumerate(subjects)
    }


def assignment_hash(assignments):
    payload = ''.join(subject + '\t' + assignments[subject] + '\n' for subject in sorted(assignments))
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def run(manifest, split_seed):
    with manifest.open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    official = {name: [row for row in rows if row['split'] == name] for name in ('train', 'val', 'test')}
    if len(official['train']) != 30344 or len(official['val']) != 1422 or len(official['test']) != 2218:
        raise ValueError('Official image counts differ from the audited dataset')
    assignment = patient_assignments(official['train'], split_seed)
    previous_assignment = patient_assignments(official['train'], 20261104)
    partitions = {name: [row for row in official['train'] if assignment[row['subject_id']] == name]
                  for name in ('train', 'early_stop', 'audit')}
    subjects = {name: {row['subject_id'] for row in values} for name, values in partitions.items()}
    studies = {name: {row['study_id'] for row in values} for name, values in partitions.items()}
    official_subjects = {name: {row['subject_id'] for row in values} for name, values in official.items()}
    overlap = {}
    for left, right in (('train', 'early_stop'), ('train', 'audit'), ('early_stop', 'audit')):
        overlap[left + '_vs_' + right] = {
            'patients': len(subjects[left] & subjects[right]),
            'studies': len(studies[left] & studies[right]),
        }
        if any(overlap[left + '_vs_' + right].values()):
            raise ValueError('Patient or Study leakage between new partitions')
    external_overlap = {
        name: {official_name: len(values & official_subjects[official_name]) for official_name in ('val', 'test')}
        for name, values in subjects.items()
    }
    if any(value for group in external_overlap.values() for value in group.values()):
        raise ValueError('New partition overlaps official validation or test patients')
    cohorts = {'official_train': cohort_counts(official['train']), 'official_val': cohort_counts(official['val'])}
    cohorts.update({'new_' + name: cohort_counts(values) for name, values in partitions.items()})
    old_overlap = dict(sorted(collections.Counter(previous_assignment[subject] for subject in subjects['audit']).items()))
    return {
        'schema_version': 1,
        'source_manifest': str(manifest),
        'manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest(),
        'split_seed': split_seed,
        'previous_split_seed': 20261104,
        'assignment_sha256': assignment_hash(assignment),
        'class_order': list(range(1, 11)),
        'definitions': {
            'positive_patient': 'A patient with at least one Study positive for that class; a patient may appear in multiple classes.',
            'study_prevalence': 'Unique positive Study count divided by all unique Studies in the cohort.',
            'top_patients': 'Patients ranked by number of Studies; only total counts are included.',
        },
        'cohorts': cohorts,
        'val_minus_train_prevalence_percentage_points': [
            val - train for train, val in zip(cohorts['official_train']['study_prevalence_pct_class1to10'],
                                             cohorts['official_val']['study_prevalence_pct_class1to10'])],
        'new_partition_pairwise_overlap_counts': overlap,
        'new_partition_official_val_test_patient_overlap_counts': external_overlap,
        'new_audit_patient_overlap_with_previous_partition_counts': old_overlap,
        'limitations': [
            'New partitions are from the existing official training source, not an external validation dataset.',
            'Full-training champion saw every new holdout patient; start new models from independent ImageNet weights.',
            'Previous experiments used the same source cohort; freeze this rounds audit prospectively after design selection.',
            'Repeated Studies within a patient are correlated; bootstrap paired performance differences at patient level.',
            'Descriptive prevalence differences alone do not establish statistical significance or domain shift.',
            'No patient identifiers, image paths, image pixels, or test labels are included in this output.',
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=Path(r'E:\比赛数据集\喻数据产物\manifest.csv'))
    parser.add_argument('--split-seed', type=int, default=20261105)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = run(args.manifest, args.split_seed)
    rendered = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if args.output:
        if args.output.exists():
            raise FileExistsError('Refusing to overwrite an existing output; use a new filename.')
        args.output.write_text(rendered, encoding='utf-8')
        print(json.dumps({'output': str(args.output), 'assignment_sha256': result['assignment_sha256'],
                          'new_audit_patients': result['cohorts']['new_audit']['patients']}, ensure_ascii=False))
    else:
        print(rendered, end='')


if __name__ == '__main__':
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    main()
