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

RESULTS_FILE = DATA_DIR / "cnn_expanded_test_results.json"


def main():
    torch.set_num_threads(2)

    print("Loading expanded dataset...")

    with np.load(WINDOWS_FILE, allow_pickle=False) as data:
        X = data["X"].astype(np.float32)
        y = data["y"].astype(np.int64)
        subject_ids = data["subject_id"].astype(str)
        conditions = data["condition"].astype(str)

    with np.load(SPLIT_FILE, allow_pickle=False) as split:
        train_ids = set(split["train_subjects"].astype(str))
        val_ids = set(split["val_subjects"].astype(str))
        test_ids = set(split["test_subjects"].astype(str))

    if not (
        train_ids.isdisjoint(val_ids)
        and train_ids.isdisjoint(test_ids)
        and val_ids.isdisjoint(test_ids)
    ):
        raise ValueError("Subject splits overlap.")

    test_mask = np.isin(subject_ids, list(test_ids))

    X_test = X[test_mask]
    y_test = y[test_mask]
    test_subject_ids = subject_ids[test_mask]
    test_conditions = conditions[test_mask]

    if len(X_test) == 0:
        raise ValueError("No test windows found.")

    if not np.isin(y_test, [0, 1]).all():
        raise ValueError("Test labels must be binary.")

    print("Test windows:", len(X_test))
    print("Test subjects:", len(np.unique(test_subject_ids)))

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

    if np.any(std <= 0) or not np.isfinite(std).all():
        raise ValueError("Invalid normalization standard deviations.")

    X_test = (X_test - mean) / std

    if not np.isfinite(X_test).all():
        raise ValueError("Normalized test data contains NaN or infinity.")

    model = TremorCNN(num_classes=2)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    predictions = []
    positive_scores = []

    print("\nEvaluating held-out test set...")

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

            predictions.extend(
                logits.argmax(dim=1).numpy().tolist()
            )

            positive_scores.extend(
                probabilities[:, 1].numpy().tolist()
            )

    predictions = np.asarray(predictions, dtype=np.int64)
    positive_scores = np.asarray(positive_scores, dtype=np.float32)

    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_test, predictions)
        ),
        "precision_positive": float(
            precision_score(y_test, predictions, zero_division=0)
        ),
        "recall_positive": float(
            recall_score(y_test, predictions, zero_division=0)
        ),
        "f1_positive": float(
            f1_score(y_test, predictions, zero_division=0)
        ),
        "confusion_matrix": confusion_matrix(
            y_test,
            predictions,
            labels=[0, 1],
        ).tolist(),
        "classification_report": classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["healthy_control", "tremor_associated"],
            output_dict=True,
            zero_division=0,
        ),
        "selected_epoch": int(checkpoint["selected_epoch"]),
        "validation_balanced_accuracy": float(
            checkpoint["validation_balanced_accuracy"]
        ),
        "test_window_count": int(len(y_test)),
        "test_subject_count": int(len(np.unique(test_subject_ids))),
        "decision_rule": "argmax of two-class softmax probabilities",
        "checkpoint": str(CHECKPOINT_FILE),
    }

    print("\n========== TEST RESULTS ==========")
    print(f"Selected epoch: {metrics['selected_epoch']}")
    print(
        "Validation balanced accuracy: "
        f"{metrics['validation_balanced_accuracy']:.4f}"
    )
    print(f"Test accuracy: {metrics['accuracy']:.4f}")
    print(
        "Test balanced accuracy: "
        f"{metrics['balanced_accuracy']:.4f}"
    )
    print(f"Positive precision: {metrics['precision_positive']:.4f}")
    print(f"Positive recall: {metrics['recall_positive']:.4f}")
    print(f"Positive F1-score: {metrics['f1_positive']:.4f}")

    print("\nConfusion matrix (rows=true, columns=predicted):")
    print("                 Predicted 0   Predicted 1")
    cm = np.asarray(metrics["confusion_matrix"])
    print(f"True 0           {cm[0, 0]:8d}      {cm[0, 1]:8d}")
    print(f"True 1           {cm[1, 0]:8d}      {cm[1, 1]:8d}")

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["healthy_control", "tremor_associated"],
            zero_division=0,
        )
    )

    # Diagnosis-specific window-level metrics.
    print("\n========== RESULTS BY DIAGNOSIS ==========")

    diagnosis_results = {}

    for condition in sorted(np.unique(test_conditions)):
        mask = test_conditions == condition

        if not mask.any():
            continue

        true_labels = y_test[mask]
        predicted_labels = predictions[mask]

        result = {
            "windows": int(mask.sum()),
            "subjects": int(
                len(np.unique(test_subject_ids[mask]))
            ),
            "accuracy": float(
                accuracy_score(true_labels, predicted_labels)
            ),
            "confusion_matrix": confusion_matrix(
                true_labels,
                predicted_labels,
                labels=[0, 1],
            ).tolist(),
        }

        diagnosis_results[str(condition)] = result

        print(
            f"{condition}: "
            f"windows={result['windows']}, "
            f"subjects={result['subjects']}, "
            f"accuracy={result['accuracy']:.4f}"
        )
        print("  Confusion matrix:", result["confusion_matrix"])

    metrics["diagnosis_results"] = diagnosis_results

    with RESULTS_FILE.open("w") as f:
        json.dump(metrics, f, indent=2)

    print("\nResults saved to:", RESULTS_FILE)
    print("The checkpoint and original evaluation files were not modified.")


if __name__ == "__main__":
    main()
