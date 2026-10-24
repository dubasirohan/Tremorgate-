import numpy as np
from sklearn.metrics import balanced_accuracy_score, accuracy_score, precision_score, recall_score, f1_score

FEATURE_FILE = "offline/data/pads_fft_features.npz"
SPLIT_FILE = "offline/data/pads_split_subjects.npz"

d = np.load(FEATURE_FILE)
s = np.load(SPLIT_FILE)

ratio = d["features"][:, 3]
condition = d["condition"]
subjects = d["subject_id"]

# Binary labels:
# Healthy = 0
# Essential Tremor + Parkinson's = 1
y = (condition != "Healthy").astype(np.int32)

train_mask = np.isin(subjects, s["train_subjects"])
val_mask = np.isin(subjects, s["val_subjects"])
test_mask = np.isin(subjects, s["test_subjects"])

# Search thresholds using TRAINING subjects only.
thresholds = np.linspace(0.05, 0.95, 181)

best_threshold = None
best_score = -1.0

results = []

for threshold in thresholds:
    pred = (ratio[train_mask] >= threshold).astype(np.int32)

    score = balanced_accuracy_score(y[train_mask], pred)
    acc = accuracy_score(y[train_mask], pred)

    results.append([threshold, score, acc])

    if score > best_score:
        best_score = score
        best_threshold = threshold

results = np.asarray(results)

print("========== FFT CALIBRATION ==========")
print("Best threshold:", best_threshold)
print("Training balanced accuracy:", best_score)

np.savez(
    "offline/data/fft_threshold_results.npz",
    results=results,
    best_threshold=best_threshold,
)

print("Saved: offline/data/fft_threshold_results.npz")
