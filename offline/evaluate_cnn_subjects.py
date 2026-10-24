#!/usr/bin/env python3

from pathlib import Path
import json

import numpy as np
import torch

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from cnn_model import TremorCNN


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"

WINDOWS_FILE = DATA_DIR / "pads_windows_binary_expanded.npz"
SPLIT_FILE = DATA_DIR / "pads_split_subjects_expanded.npz"

CHECKPOINT_FILE = MODEL_DIR / "tremor_cnn_expanded_best.pt"

RESULTS_FILE = DATA_DIR / "cnn_expanded_subject_results.json"


def main():
    torch.set_num_threads(2)

    print("Loading expanded dataset...")

    with np.load(WINDOWS_FILE, allow_pickle=False) as data:
        X = data["X"].astype(np.float32)
        y = data["y"].astype(np.int64)
        subject_ids = data["subject_id"].astype(str)
        conditions = data["condition"].astype(str)

    with np.load(SPLIT_FILE, allow_pickle=False) as split:
        test_ids = set(split["test_subjects"].astype(str))

    test_mask = np.isin(subject_ids, list(test_ids))

    X_test = X[test_mask]
    y_test = y[test_mask]
    subjects_test = subject_ids[test_mask]
    conditions_test = conditions[test_mask]

    if len(X_test) == 0:
        raise ValueError("No test windows found.")

    # Confirm that every test subject has one binary ground-truth label
    # and one diagnosis.
    for sid in np.unique(subjects_test):
        mask = subjects_test == sid

        if len(np.unique(y_test[mask])) != 1:
            raise ValueError(
                f"Subject {sid} has inconsistent binary labels."
            )

        if len(np.unique(conditions_test[mask])) != 1:
            raise ValueError(
                f"Subject {sid} has inconsistent diagnoses."
            )

    print("Test windows:", len(X_test))
    print("Test subjects:", len(np.unique(subjects_test)))

    checkpoint = torch.load(
        CHECKPOINT_FILE,
        map_location="cpu",
        weights_only=False,
    )

    mean = np.asarray(
        checkpoint["normalization_mean"],
        dtype=np.float32,
    ).reshape(1, 3, 1)

    std = np.asarray(
        checkpoint["normalization_std"],
        dtype=np.float32,
    ).reshape(1, 3, 1)

    if not np.isfinite(mean).all():
        raise ValueError("Invalid normalization means.")

    if not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError("Invalid normalization standard deviations.")

    X_test = (X_test - mean) / std

    if not np.isfinite(X_test).all():
        raise ValueError("Normalized test data contains NaN or infinity.")

    model = TremorCNN(num_classes=2)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Obtain positive-class probabilities for each test window.
    scores = []

    print("\nGenerating window predictions...")

    with torch.no_grad():
        batch_size = 256

        for start in range(0, len(X_test), batch_size):
            batch = torch.from_numpy(
                np.ascontiguousarray(
                    X_test[start:start + batch_size],
                    dtype=np.float32,
                )
            )

            logits = model(batch)
            probabilities = torch.softmax(logits, dim=1)

            scores.extend(
                probabilities[:, 1].numpy().tolist()
            )

    scores = np.asarray(scores, dtype=np.float64)

    # Aggregate each subject's window probabilities by taking the mean.
    # Threshold 0.5 is fixed in advance; the test set is not used to tune it.
    subject_results = []

    for sid in sorted(np.unique(subjects_test)):
        mask = subjects_test == sid

        true_label = int(y_test[mask][0])
        diagnosis = str(conditions_test[mask][0])
        mean_score = float(scores[mask].mean())
        predicted_label = int(mean_score >= 0.5)

        subject_results.append({
            "subject_id": str(sid),
            "diagnosis": diagnosis,
            "true_label": true_label,
            "predicted_label": predicted_label,
            "mean_positive_probability": mean_score,
            "window_count": int(mask.sum()),
        })

    y_true = np.asarray(
        [row["true_label"] for row in subject_results],
        dtype=np.int64,
    )

    y_pred = np.asarray(
        [row["predicted_label"] for row in subject_results],
        dtype=np.int64,
    )

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    metrics = {
        "evaluation_level": "subject",
        "aggregation": "mean positive-class probability across subject windows",
        "decision_threshold": 0.5,
        "selected_epoch": int(checkpoint["selected_epoch"]),
        "test_subject_count": int(len(subject_results)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, y_pred)
        ),
        "precision_positive": float(
            precision_score(y_true, y_pred, zero_division=0)
        ),
        "recall_positive": float(
            recall_score(y_true, y_pred, zero_division=0)
        ),
        "f1_positive": float(
            f1_score(y_true, y_pred, zero_division=0)
        ),
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=["healthy_control", "tremor_associated"],
            output_dict=True,
            zero_division=0,
        ),
        "subjects": subject_results,
    }

    print("\n========== SUBJECT-LEVEL RESULTS ==========")
    print("Aggregation: mean positive-class probability per subject")
    print("Decision threshold: 0.5")
    print("Test subjects:", metrics["test_subject_count"])
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Balanced accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"Positive precision: {metrics['precision_positive']:.4f}")
    print(f"Positive recall: {metrics['recall_positive']:.4f}")
    print(f"Positive F1-score: {metrics['f1_positive']:.4f}")

    print("\nConfusion matrix (rows=true, columns=predicted):")
    print("                 Predicted 0   Predicted 1")
    print(f"True 0           {cm[0, 0]:8d}      {cm[0, 1]:8d}")
    print(f"True 1           {cm[1, 0]:8d}      {cm[1, 1]:8d}")

    print("\nClassification report:")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=["healthy_control", "tremor_associated"],
            zero_division=0,
        )
    )

    print("\n========== RESULTS BY DIAGNOSIS ==========")

    for diagnosis in sorted({
        row["diagnosis"] for row in subject_results
    }):
        rows = [
            row for row in subject_results
            if row["diagnosis"] == diagnosis
        ]

        true_labels = np.asarray(
            [row["true_label"] for row in rows],
            dtype=np.int64,
        )

        predicted_labels = np.asarray(
            [row["predicted_label"] for row in rows],
            dtype=np.int64,
        )

        correct = int(np.sum(true_labels == predicted_labels))

        print(
            f"{diagnosis}: "
            f"subjects={len(rows)}, "
            f"correct={correct}/{len(rows)}, "
            f"accuracy={accuracy_score(true_labels, predicted_labels):.4f}"
        )

    with RESULTS_FILE.open("w") as f:
        json.dump(metrics, f, indent=2)

    print("\nSaved subject-level results to:")
    print(RESULTS_FILE)


if __name__ == "__main__":
    main()
