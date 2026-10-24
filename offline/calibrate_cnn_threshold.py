#!/usr/bin/env python3
from pathlib import Path
import sys
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

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from cnn_model import TremorCNN

DATA_FILE = ROOT / "data" / "pads_windows.npz"
SPLIT_FILE = ROOT / "data" / "pads_split_subjects.npz"
CHECKPOINT_FILE = ROOT / "models" / "tremor_cnn_best.pt"
OUTPUT_FILE = ROOT / "data" / "cnn_threshold_calibration.json"

torch.set_num_threads(2)
torch.manual_seed(42)

# Load dataset, subject splits, and the existing best checkpoint.
data = np.load(DATA_FILE)
splits = np.load(SPLIT_FILE)
checkpoint = torch.load(
    CHECKPOINT_FILE, map_location="cpu", weights_only=False
)

X_all = data["X"].astype(np.float32)
subjects_all = data["subject_id"].astype(str)
conditions_all = data["condition"].astype(str)

train_subjects = set(splits["train_subjects"].astype(str))
val_subjects = set(splits["val_subjects"].astype(str))
test_subjects = set(splits["test_subjects"].astype(str))

if train_subjects & val_subjects or train_subjects & test_subjects or val_subjects & test_subjects:
    raise RuntimeError("Subject splits overlap; refusing evaluation.")

# Healthy = 0; Essential Tremor and Parkinson's = 1.
# Other diagnostic groups are excluded.
eligible = np.isin(
    conditions_all,
    ["Healthy", "Essential Tremor", "Parkinson's", "Parkinsons"],
)
X = X_all[eligible]
subjects = subjects_all[eligible]
conditions = conditions_all[eligible]
y = np.isin(conditions, ["Essential Tremor", "Parkinson's", "Parkinsons"]).astype(np.int64)

masks = {
    "train": np.isin(subjects, list(train_subjects)),
    "val": np.isin(subjects, list(val_subjects)),
    "test": np.isin(subjects, list(test_subjects)),
}

for name, mask in masks.items():
    if not mask.any():
        raise RuntimeError(f"No eligible samples found for {name} split.")
    print(
        f"{name}: windows={int(mask.sum())}, "
        f"healthy={int((y[mask] == 0).sum())}, "
        f"positive={int((y[mask] == 1).sum())}, "
        f"subjects={len(set(subjects[mask]))}"
    )

# Reuse the exact normalization stored with the trained checkpoint.
mean = np.asarray(checkpoint["normalization_mean"], dtype=np.float32).reshape(1, 3, 1)
std = np.asarray(checkpoint["normalization_std"], dtype=np.float32).reshape(1, 3, 1)
std = np.maximum(std, 1e-8)

model = TremorCNN(num_classes=int(checkpoint["num_classes"]))
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

def positive_probabilities(X_subset):
    normalized = (X_subset - mean) / std
    results = []
    with torch.no_grad():
        for start in range(0, len(normalized), 128):
            batch = torch.from_numpy(normalized[start:start + 128])
            logits = model(batch)
            probabilities = torch.softmax(logits, dim=1)[:, 1]
            results.append(probabilities.cpu().numpy())
    return np.concatenate(results)

val_mask = masks["val"]
test_mask = masks["test"]

y_val = y[val_mask]
y_test = y[test_mask]
p_val = positive_probabilities(X[val_mask])
p_test = positive_probabilities(X[test_mask])

# Select threshold using validation balanced accuracy only.
thresholds = np.arange(0.05, 0.951, 0.01)
scores = []
for threshold in thresholds:
    predictions = (p_val >= threshold).astype(np.int64)
    scores.append(balanced_accuracy_score(y_val, predictions))

best_score = max(scores)
tied_indices = [
    i for i, score in enumerate(scores)
    if np.isclose(score, best_score)
]
best_index = min(tied_indices, key=lambda i: abs(thresholds[i] - 0.5))
best_threshold = float(thresholds[best_index])

def evaluate(y_true, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(np.int64)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "precision_positive": float(precision_score(y_true, predictions, zero_division=0)),
        "recall_positive": float(recall_score(y_true, predictions, zero_division=0)),
        "f1_positive": float(f1_score(y_true, predictions, zero_division=0)),
        "confusion_matrix_rows_true_cols_pred": confusion_matrix(
            y_true, predictions, labels=[0, 1]
        ).tolist(),
        "classification_report": classification_report(
            y_true,
            predictions,
            labels=[0, 1],
            target_names=["Healthy/control", "ET or Parkinson's"],
            output_dict=True,
            zero_division=0,
        ),
    }

# The default argmax decision for a two-class model corresponds to p >= 0.5.
default_test = evaluate(y_test, p_test, 0.5)
calibrated_test = evaluate(y_test, p_test, best_threshold)

results = {
    "experiment": "CNN validation-selected probability threshold",
    "checkpoint": str(CHECKPOINT_FILE),
    "selected_epoch": int(checkpoint["selected_epoch"]),
    "split_method": checkpoint["split_method"],
    "validation_windows": int(val_mask.sum()),
    "test_windows": int(test_mask.sum()),
    "validation_selected_threshold": best_threshold,
    "best_validation_balanced_accuracy": float(best_score),
    "default_argmax_test": default_test,
    "calibrated_threshold_test": calibrated_test,
    "test_subjects_used_for_threshold_selection": False,
}

OUTPUT_FILE.write_text(json.dumps(results, indent=2))
print("\nCNN THRESHOLD CALIBRATION")
print(f"Selected threshold (validation only): {best_threshold:.2f}")
print(f"Best validation balanced accuracy: {best_score:.4f}")
print("\nHeld-out TEST results:")
for name, metrics in [
    ("Default threshold 0.50", default_test),
    (f"Calibrated threshold {best_threshold:.2f}", calibrated_test),
]:
    print(f"\n{name}")
    print(f"  Accuracy:          {metrics['accuracy'] * 100:.2f}%")
    print(f"  Balanced accuracy: {metrics['balanced_accuracy'] * 100:.2f}%")
    print(f"  Positive precision:{metrics['precision_positive'] * 100:.2f}%")
    print(f"  Positive recall:   {metrics['recall_positive'] * 100:.2f}%")
    print(f"  Positive F1:       {metrics['f1_positive'] * 100:.2f}%")
    print("  Confusion matrix:")
    print(np.array(metrics["confusion_matrix_rows_true_cols_pred"]))
print("\nSaved:", OUTPUT_FILE)
