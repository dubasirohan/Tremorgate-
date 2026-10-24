#!/usr/bin/env python3

from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

BASE = Path.home() / "tremorgate_ws"
DATA_DIR = BASE / "offline" / "data"

SOURCE = DATA_DIR / "pads_windows_expanded.npz"
OUTPUT = DATA_DIR / "pads_windows_binary_expanded.npz"
SPLIT_OUTPUT = DATA_DIR / "pads_split_subjects_expanded.npz"

SEED = 42
TEST_FRACTION = 0.15
VAL_FRACTION = 0.15

POSITIVE_CONDITIONS = {"Essential Tremor", "Parkinson's"}
NEGATIVE_CONDITIONS = {"Healthy"}
ELIGIBLE_CONDITIONS = POSITIVE_CONDITIONS | NEGATIVE_CONDITIONS


def main():
    data = np.load(SOURCE)

    X_all = data["X"]
    subject_all = data["subject_id"]
    condition_all = data["condition"]
    task_all = data["task"]
    wrist_all = data["wrist"]

    eligible = np.isin(
        condition_all,
        list(ELIGIBLE_CONDITIONS),
    )

    X = X_all[eligible]
    subject = subject_all[eligible]
    condition = condition_all[eligible]
    task = task_all[eligible]
    wrist = wrist_all[eligible]

    y = np.isin(condition, list(POSITIVE_CONDITIONS)).astype(np.int64)

    if not np.isfinite(X).all():
        raise ValueError("Dataset contains NaN or infinite values.")

    # Every subject must have one consistent binary label.
    subject_labels = {}

    for sid in np.unique(subject):
        labels = np.unique(y[subject == sid])

        if len(labels) != 1:
            raise ValueError(
                f"Subject {sid} has inconsistent binary labels."
            )

        subject_labels[sid] = int(labels[0])

    subjects = np.array(sorted(subject_labels.keys()))
    subject_y = np.array(
        [subject_labels[sid] for sid in subjects],
        dtype=np.int64,
    )

    # First reserve 15% of subjects for testing.
    train_val_subjects, test_subjects = train_test_split(
        subjects,
        test_size=TEST_FRACTION,
        random_state=SEED,
        stratify=subject_y,
    )

    train_val_labels = np.array(
        [subject_labels[sid] for sid in train_val_subjects],
        dtype=np.int64,
    )

    # Split the remaining subjects so validation is 15% overall.
    relative_val_fraction = VAL_FRACTION / (1.0 - TEST_FRACTION)

    train_subjects, val_subjects = train_test_split(
        train_val_subjects,
        test_size=relative_val_fraction,
        random_state=SEED,
        stratify=train_val_labels,
    )

    # Check that subject sets are disjoint.
    train_set = set(train_subjects)
    val_set = set(val_subjects)
    test_set = set(test_subjects)

    assert train_set.isdisjoint(val_set)
    assert train_set.isdisjoint(test_set)
    assert val_set.isdisjoint(test_set)

    # Save the binary dataset independently of the original.
    np.savez_compressed(
        OUTPUT,
        X=X,
        y=y,
        subject_id=subject,
        condition=condition,
        task=task,
        wrist=wrist,
        fs=data["fs"],
        window_size=data["window_size"],
        step=data["step"],
        label_0=np.array("Healthy"),
        label_1=np.array("Essential Tremor or Parkinson's"),
        split_method=np.array("subject-level stratified 70/15/15"),
        seed=np.array(SEED),
    )

    np.savez_compressed(
        SPLIT_OUTPUT,
        train_subjects=np.array(sorted(train_subjects)),
        val_subjects=np.array(sorted(val_subjects)),
        test_subjects=np.array(sorted(test_subjects)),
        seed=np.array(SEED),
        split_method=np.array("subject-level stratified 70/15/15"),
    )

    print("\n========== BINARY DATASET ==========")
    print("Saved:", OUTPUT)
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("Total windows:", len(y))
    print("Unique subjects:", len(subjects))
    print("Label counts:", {
        int(label): int(np.sum(y == label))
        for label in np.unique(y)
    })

    print("\nWindows by diagnosis:")

    for diagnosis in sorted(ELIGIBLE_CONDITIONS):
        mask = condition == diagnosis
        print(
            f"  {diagnosis}: "
            f"{int(mask.sum())} windows, "
            f"{len(np.unique(subject[mask]))} subjects"
        )

    print("\n========== SUBJECT SPLIT ==========")
    print("Saved:", SPLIT_OUTPUT)

    for name, split_subjects in [
        ("Train", train_subjects),
        ("Validation", val_subjects),
        ("Test", test_subjects),
    ]:
        split_mask = np.isin(subject, split_subjects)
        split_labels = y[split_mask]

        print(
            f"{name}: {len(split_subjects)} subjects, "
            f"{int(split_mask.sum())} windows, "
            f"label 0 = {int(np.sum(split_labels == 0))}, "
            f"label 1 = {int(np.sum(split_labels == 1))}"
        )

    print("\nSubject overlap checks:")
    print("Train/validation:", len(train_set & val_set))
    print("Train/test:", len(train_set & test_set))
    print("Validation/test:", len(val_set & test_set))

    print("\nOriginal dataset and split were not modified.")


if __name__ == "__main__":
    main()
