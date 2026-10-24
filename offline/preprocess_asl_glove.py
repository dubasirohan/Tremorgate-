from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path.home() / "tremorgate_ws/data/asl_glove/ASL-Sensor-Dataglove-Dataset"
OUT = Path.home() / "tremorgate_ws/offline/data"
OUT.mkdir(parents=True, exist_ok=True)

WINDOW = 128
STRIDE = 64
GAP_THRESHOLD = 0.05
RANDOM_SEED = 42

CHANNELS = [
    "flex_1", "flex_2", "flex_3", "flex_4", "flex_5",
    "GYRx", "GYRy", "GYRz",
    "ACCx", "ACCy", "ACCz",
]

files = sorted(ROOT.glob("*/*.csv"))
if not files:
    raise RuntimeError(f"No CSV recordings found in {ROOT}")

# Establish a stable label index from gesture names.
gestures = sorted({p.stem for p in files})
gesture_to_id = {name: i for i, name in enumerate(gestures)}

X_list = []
y_list = []
groups_list = []
recording_list = []
segment_list = []

for i, path in enumerate(files, 1):
    df = pd.read_csv(path)

    required = ["timestamp"] + CHANNELS
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")

    timestamps = df["timestamp"].to_numpy(dtype=np.float64)
    values = df[CHANNELS].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(dtype=np.float32)

    if not np.isfinite(timestamps).all():
        raise ValueError(f"Non-finite timestamps in {path}")
    if not np.isfinite(values).all():
        raise ValueError(f"Non-finite sensor values in {path}")

    # A gap larger than 50 ms marks a segment boundary.
    dt = np.diff(timestamps)
    boundaries = np.where(dt > GAP_THRESHOLD)[0] + 1
    segments = np.split(np.arange(len(timestamps)), boundaries)

    for seg_idx, indices in enumerate(segments):
        if len(indices) < WINDOW:
            continue

        # Each window stays inside one continuous segment.
        for start in range(0, len(indices) - WINDOW + 1, STRIDE):
            idx = indices[start:start + WINDOW]
            X_list.append(values[idx])
            y_list.append(gesture_to_id[path.stem])
            groups_list.append(path.parent.name)
            recording_list.append(str(path.relative_to(ROOT)))
            segment_list.append(seg_idx)

    if i % 100 == 0:
        print(f"Processed {i}/{len(files)} recordings")

if not X_list:
    raise RuntimeError("No windows generated. Check the segment/window settings.")

X = np.stack(X_list).astype(np.float32)  # windows x time x channels
y = np.asarray(y_list, dtype=np.int64)
groups = np.asarray(groups_list)
recordings = np.asarray(recording_list)
segments = np.asarray(segment_list, dtype=np.int16)

# Split unique participants, not individual windows.
unique_subjects = np.unique(groups)
train_subjects, temp_subjects = train_test_split(
    unique_subjects,
    test_size=0.40,
    random_state=RANDOM_SEED,
)
val_subjects, test_subjects = train_test_split(
    temp_subjects,
    test_size=0.50,
    random_state=RANDOM_SEED,
)

train_mask = np.isin(groups, train_subjects)
val_mask = np.isin(groups, val_subjects)
test_mask = np.isin(groups, test_subjects)

assert not (set(train_subjects) & set(val_subjects))
assert not (set(train_subjects) & set(test_subjects))
assert not (set(val_subjects) & set(test_subjects))

# Fit channel-wise normalization on training windows only.
# Statistics are computed across all training-window time points.
train_values = X[train_mask].reshape(-1, len(CHANNELS))
mean = train_values.mean(axis=0)
std = train_values.std(axis=0)
std[std < 1e-6] = 1.0

X_norm = ((X - mean[None, None, :]) / std[None, None, :]).astype(np.float32)

# Save the normalized dataset and explicit split masks.
np.savez_compressed(
    OUT / "asl_glove_windows.npz",
    X=X_norm,
    y=y,
    groups=groups,
    recordings=recordings,
    segment_indices=segments,
    train_mask=train_mask,
    val_mask=val_mask,
    test_mask=test_mask,
    channel_names=np.asarray(CHANNELS),
    gesture_names=np.asarray(gestures),
    normalization_mean=mean.astype(np.float32),
    normalization_std=std.astype(np.float32),
)

metadata = {
    "source": "DU-ASL-DATA-GLOVE-DB",
    "window_samples": WINDOW,
    "stride_samples": STRIDE,
    "approx_sampling_rate_hz": 100,
    "window_duration_approx_seconds": WINDOW / 100,
    "gap_threshold_seconds": GAP_THRESHOLD,
    "channels": CHANNELS,
    "gesture_to_id": gesture_to_id,
    "participants": {
        "train": sorted(train_subjects.tolist()),
        "validation": sorted(val_subjects.tolist()),
        "test": sorted(test_subjects.tolist()),
    },
    "counts": {
        "recordings": len(files),
        "participants": len(unique_subjects),
        "windows_total": int(len(y)),
        "windows_train": int(train_mask.sum()),
        "windows_validation": int(val_mask.sum()),
        "windows_test": int(test_mask.sum()),
    },
    "warning": (
        "Labels are ASL gesture identities, not voluntary/involuntary "
        "or tremor labels. This dataset cannot by itself train a "
        "voluntary-versus-involuntary classifier."
    ),
}

with open(OUT / "asl_glove_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("\n=== PREPROCESSING COMPLETE ===")
print("Windows:", X_norm.shape)
print("Channels:", len(CHANNELS))
print("Gesture classes:", len(gestures))
print("Participants:", len(unique_subjects))
print("Train windows:", int(train_mask.sum()))
print("Validation windows:", int(val_mask.sum()))
print("Test windows:", int(test_mask.sum()))
print("Train participants:", sorted(train_subjects.tolist()))
print("Validation participants:", sorted(val_subjects.tolist()))
print("Test participants:", sorted(test_subjects.tolist()))
print("Saved:", OUT / "asl_glove_windows.npz")
print("Metadata:", OUT / "asl_glove_metadata.json")
