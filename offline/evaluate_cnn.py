#!/usr/bin/env python3

from pathlib import Path
import csv
import json

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
PLOT_DIR = ROOT / "plots"

WINDOWS_FILE = DATA_DIR / "pads_windows.npz"
SPLIT_FILE = DATA_DIR / "pads_split_subjects.npz"
CHECKPOINT_FILE = MODEL_DIR / "tremor_cnn_best.pt"
RESULTS_FILE = DATA_DIR / "cnn_test_metrics.json"
MATRIX_FILE = PLOT_DIR / "cnn_confusion_matrix.png"


def main():
    torch.set_num_threads(2)

    with np.load(WINDOWS_FILE, allow_pickle=True) as data:
        X = data["X"].astype(np.float32)
        subjects = data["subject_id"].astype(str)
        conditions = data["condition"].astype(str)

    with np.load(SPLIT_FILE, allow_pickle=True) as split:
        test_ids = set(split["test_subjects"].astype(str))

    # Use exactly the same binary label mapping as training.
    normalized = np.array([
        " ".join(c.strip().lower().replace("_", " ").split())
        for c in conditions
    ])

    healthy_names = {"healthy"}
    positive_names = {
        "essential tremor",
        "parkinson's",
        "parkinson’s",
        "parkinsons",
    }

    healthy = np.isin(normalized, list(healthy_names))
    positive = np.isin(normalized, list(positive_names))
    keep = healthy | positive

    X = X[keep]
    subjects = subjects[keep]
    normalized = normalized[keep]

    y = np.where(
        np.isin(normalized, list(healthy_names)), 0, 1
    ).astype(np.int64)

    test_mask = np.isin(subjects, list(test_ids))
    X_test = X[test_mask]
    y_test = y[test_mask]
    test_subjects = subjects[test_mask]

    if len(X_test) == 0:
        raise RuntimeError("No test windows found.")

    counts = np.bincount(y_test, minlength=2)
    if np.any(counts == 0):
        raise RuntimeError(
            f"Test set must contain both classes; counts={counts.tolist()}"
        )

    # Explicitly use weights saved from training.
    checkpoint = torch.load(
        CHECKPOINT_FILE,
        map_location="cpu",
        weights_only=False,
    )

    mean = np.asarray(
        checkpoint["normalization_mean"], dtype=np.float32
    ).reshape(1, 3, 1)
    std = np.asarray(
        checkpoint["normalization_std"], dtype=np.float32
    ).reshape(1, 3, 1)

    X_test = (X_test - mean) / np.maximum(std, 1e-6)

    model = TremorCNN(num_classes=2)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    predictions = []
    batch_size = 64

    with torch.no_grad():
        for start in range(0, len(X_test), batch_size):
            batch = torch.from_numpy(
                np.ascontiguousarray(
                    X_test[start:start + batch_size],
                    dtype=np.float32,
                )
            )
            logits = model(batch)
            predictions.extend(logits.argmax(dim=1).numpy())

    y_pred = np.asarray(predictions, dtype=np.int64)

    metrics = {
        "test_windows": int(len(y_test)),
        "test_subjects": int(len(np.unique(test_subjects))),
        "healthy_windows": int(counts[0]),
        "positive_windows": int(counts[1]),
        "selected_epoch": int(checkpoint["selected_epoch"]),
        "validation_balanced_accuracy": float(
            checkpoint["validation_balanced_accuracy"]
        ),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_test, y_pred)
        ),
        "precision_positive": float(
            precision_score(y_test, y_pred, zero_division=0)
        ),
        "recall_positive": float(
            recall_score(y_test, y_pred, zero_division=0)
        ),
        "f1_positive": float(
            f1_score(y_test, y_pred, zero_division=0)
        ),
    }

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    print("\nCNN TEST RESULTS")
    print("=" * 50)
    print(f"Test windows:             {metrics['test_windows']}")
    print(f"Test subjects:            {metrics['test_subjects']}")
    print(f"Selected epoch:           {metrics['selected_epoch']}")
    print(
        "Validation balanced acc: "
        f"{metrics['validation_balanced_accuracy']:.4f}"
    )
    print(f"Accuracy:                 {metrics['accuracy']:.4f}")
    print(
        "Balanced accuracy:       "
        f"{metrics['balanced_accuracy']:.4f}"
    )
    print(
        "Positive precision:      "
        f"{metrics['precision_positive']:.4f}"
    )
    print(
        "Positive recall:         "
        f"{metrics['recall_positive']:.4f}"
    )
    print(f"Positive F1:              {metrics['f1_positive']:.4f}")

    print("\nConfusion matrix (rows=true, columns=predicted)")
    print("                    Predicted healthy  Predicted positive")
    print(f"True healthy         {cm[0, 0]:16d}  {cm[0, 1]:18d}")
    print(f"True positive        {cm[1, 0]:16d}  {cm[1, 1]:18d}")

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            y_pred,
            labels=[0, 1],
            target_names=["Healthy/control", "ET or Parkinson's"],
            zero_division=0,
        )
    )

    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(cm, interpolation="nearest")
    ax.set_title("CNN Test Confusion Matrix")
    fig.colorbar(image, ax=ax, label="Window count")
    ax.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["Healthy/control", "ET or Parkinson's"],
        yticklabels=["Healthy/control", "ET or Parkinson's"],
        xlabel="Predicted label",
        ylabel="True label",
    )
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")

    midpoint = cm.max() / 2 if cm.max() else 0
    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > midpoint else "black",
            )

    fig.tight_layout()
    fig.savefig(MATRIX_FILE, dpi=160)
    plt.close(fig)

    with RESULTS_FILE.open("w") as f:
        json.dump(metrics, f, indent=2)

    print("\nSaved metrics:", RESULTS_FILE)
    print("Saved confusion matrix:", MATRIX_FILE)
    print("\nFFT baseline reference balanced accuracy: 0.5837")
    print(
        "CNN test balanced accuracy: "
        f"{metrics['balanced_accuracy']:.4f}"
    )
    print(
        "This is an offline experimental classifier, not a clinically "
        "validated diagnostic system."
    )


if __name__ == "__main__":
    main()
