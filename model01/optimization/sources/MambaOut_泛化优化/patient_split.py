"""Fixed patient-level 70/15/15 split for the official training manifest."""
import hashlib
import itertools
import random


def make_patient_split(rows, seed=20261104):
    """Return row-index lists and a reproducible split summary.

    Input rows must be the 30,344 official training rows, in dataset order.
    Index lists preserve that order. Hashes use UTF-8, no BOM, and a final LF.
    row_order_sha256 maps official_train and each partition to image-path hashes.
    No files are written and no patient identifiers are printed.
    """
    if len(rows) != 30344:
        raise ValueError(f"Expected 30344 official training images, got {len(rows)}")

    subjects, image_paths, studies = set(), set(), {}
    for row in rows:
        subject, study = row["subject_id"], row["study_id"]
        labels = tuple(sorted(int(value) for value in row["label_ids"].split("|")))
        if not labels or len(set(labels)) != len(labels):
            raise ValueError("Each Study must have nonempty, unique label IDs")
        if any(label < 1 or label > 10 for label in labels):
            raise ValueError("Label IDs must be integers from 1 through 10")
        if not subject or not study or not row["image_path"]:
            raise ValueError("Subject, Study, and image path must be nonempty")
        if row["image_path"] in image_paths:
            raise ValueError("Duplicate image_path in official training rows")
        record = (subject, labels)
        if study in studies and studies[study] != record:
            raise ValueError("A Study has inconsistent labels or multiple patients")
        studies[study] = record
        subjects.add(subject)
        image_paths.add(row["image_path"])

    if len(studies) != 18513 or len(subjects) != 11055:
        raise ValueError(
            f"Expected 18513 Studies and 11055 patients; "
            f"got {len(studies)} Studies and {len(subjects)} patients"
        )

    shuffled = sorted(subjects)
    random.Random(seed).shuffle(shuffled)
    n = round(0.15 * len(shuffled))
    assignments = {
        subject: ("audit" if i < n else "early_stop" if i < 2 * n else "train")
        for i, subject in enumerate(shuffled)
    }
    indices = {"train": [], "early_stop": [], "audit": []}
    for index, row in enumerate(rows):
        indices[assignments[row["subject_id"]]].append(index)

    patient_sets, study_sets, partitions = {}, {}, {}
    for split, row_indices in indices.items():
        patient_sets[split] = {rows[i]["subject_id"] for i in row_indices}
        study_sets[split] = {rows[i]["study_id"] for i in row_indices}
        count = len(study_sets[split])
        positive = {
            str(label): sum(label in studies[study][1] for study in study_sets[split])
            for label in range(1, 11)
        }
        negative = {label: count - value for label, value in positive.items()}
        if any(value == 0 for value in positive.values()):
            raise ValueError(f"{split} lacks a positive Study for at least one class")
        if any(value == 0 for value in negative.values()):
            raise ValueError(f"{split} lacks a negative Study for at least one class")
        partitions[split] = {
            "images": len(row_indices),
            "studies": count,
            "patients": len(patient_sets[split]),
            "study_positive_counts": positive,
            "study_negative_counts": negative,
        }

    for left, right in itertools.combinations(indices, 2):
        if patient_sets[left] & patient_sets[right]:
            raise ValueError("Patient overlap between partitions")
        if study_sets[left] & study_sets[right]:
            raise ValueError("Study overlap between partitions")
    if sum(len(value) for value in indices.values()) != len(rows):
        raise ValueError("Not all training rows were assigned exactly once")
    if set().union(*patient_sets.values()) != subjects:
        raise ValueError("Not all training patients were assigned")

    assignment_bytes = "".join(
        subject + "\t" + assignments[subject] + "\n"
        for subject in sorted(subjects)
    ).encode("utf-8")
    order_groups = {"official_train": range(len(rows)), **indices}
    order_hashes = {
        split: hashlib.sha256(
            "".join(rows[i]["image_path"] + "\n" for i in row_indices).encode("utf-8")
        ).hexdigest()
        for split, row_indices in order_groups.items()
    }
    summary = {
        "seed": seed,
        "assignment_sha256": hashlib.sha256(assignment_bytes).hexdigest(),
        "partitions": partitions,
        "assignments": assignments,
        "row_order_sha256": order_hashes,
    }
    return indices, summary
