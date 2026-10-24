#!/usr/bin/env python3

from pathlib import Path
import csv
import random

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import balanced_accuracy_score

from cnn_model import TremorCNN


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"

WINDOWS_FILE = DATA_DIR / "pads_windows_binary_expanded.npz"
SPLIT_FILE = DATA_DIR / "pads_split_subjects_expanded.npz"

CHECKPOINT_FILE = MODEL_DIR / "tremor_cnn_expanded_best.pt"
HISTORY_FILE = DATA_DIR / "cnn_expanded_training_history.csv"

SEED = 42
BATCH_SIZE = 64
MAX_EPOCHS = 20
LEARNING_RATE = 1e-3


def set_seeds():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(2)


def main():
    set_seeds()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading expanded binary dataset...")

    with np.load(WINDOWS_FILE, allow_pickle=False) as data:
        X = data["X"].astype(np.float32)
        y = data["y"].astype(np.int64)
        subject_ids = data["subject_id"].astype(str)
        conditions = data["condition"].astype(str)

    with np.load(SPLIT_FILE, allow_pickle=False) as split:
        train_ids = set(split["train_subjects"].astype(str))
        val_ids = set(split["val_subjects"].astype(str))
        test_ids = set(split["test_subjects"].astype(str))

    if X.ndim != 3 or X.shape[1:] != (3, 256):
        raise ValueError(f"Unexpected input shape: {X.shape}")

    if len(X) != len(y) or len(X) != len(subject_ids):
        raise ValueError("Dataset arrays have inconsistent lengths.")

    if not np.isfinite(X).all():
        raise ValueError("Input data contains NaN or infinite values.")

    if not np.isin(y, [0, 1]).all():
        raise ValueError("Labels must be binary: 0 or 1.")

    if not (
        train_ids.isdisjoint(val_ids)
        and train_ids.isdisjoint(test_ids)
        and val_ids.isdisjoint(test_ids)
    ):
        raise ValueError("Subject splits overlap. Aborting.")

    assigned = (
        np.isin(subject_ids, list(train_ids))
        | np.isin(subject_ids, list(val_ids))
        | np.isin(subject_ids, list(test_ids))
    )

    if not assigned.all():
        missing = np.unique(subject_ids[~assigned])
        raise ValueError(
            f"Subjects missing from split: {missing.tolist()}"
        )

    # Every subject must have one consistent binary label.
    for sid in np.unique(subject_ids):
        subject_labels = np.unique(y[subject_ids == sid])

        if len(subject_labels) != 1:
            raise ValueError(
                f"Subject {sid} has inconsistent labels."
            )

    train_mask = np.isin(subject_ids, list(train_ids))
    val_mask = np.isin(subject_ids, list(val_ids))
    test_mask = np.isin(subject_ids, list(test_ids))

    X_train = X[train_mask]
    y_train = y[train_mask]

    X_val = X[val_mask]
    y_val = y[val_mask]

    X_test = X[test_mask]
    y_test = y[test_mask]

    print("\n========== DATA SUMMARY ==========")
    print("Input shape:", X.shape)
    print("Training:", X_train.shape)
    print("Validation:", X_val.shape)
    print("Test:", X_test.shape)

    print("Training subjects:", len(train_ids))
    print("Validation subjects:", len(val_ids))
    print("Test subjects:", len(test_ids))

    for name, labels in (
        ("Training", y_train),
        ("Validation", y_val),
        ("Test", y_test),
    ):
        counts = np.bincount(labels, minlength=2)
        print(
            f"{name} labels: "
            f"healthy={counts[0]}, "
            f"positive={counts[1]}"
        )

        if counts.min() == 0:
            raise ValueError(
                f"{name} does not contain both classes."
            )

    # Calculate normalization parameters using training data only.
    mean = X_train.mean(axis=(0, 2), keepdims=True)
    std = X_train.std(axis=(0, 2), keepdims=True)
    std = np.maximum(std, 1e-6)

    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std

    def make_loader(signals, labels, shuffle):
        dataset = TensorDataset(
            torch.from_numpy(
                np.ascontiguousarray(signals, dtype=np.float32)
            ),
            torch.from_numpy(
                np.ascontiguousarray(labels, dtype=np.int64)
            ),
        )

        return DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=shuffle,
            num_workers=0,
        )

    train_loader = make_loader(X_train, y_train, True)
    val_loader = make_loader(X_val, y_val, False)

    # Class weights are calculated from training labels only.
    class_counts = np.bincount(y_train, minlength=2)

    class_weights = (
        len(y_train) / (2.0 * class_counts.astype(np.float32))
    )

    model = TremorCNN(num_classes=2)

    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32)
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    print("\n========== TRAINING CONFIGURATION ==========")
    print("Device: CPU")
    print("Seed:", SEED)
    print("Batch size:", BATCH_SIZE)
    print("Maximum epochs:", MAX_EPOCHS)
    print("Learning rate:", LEARNING_RATE)
    print("Class weights:", class_weights.tolist())

    history = []

    best_balanced_accuracy = -1.0
    best_val_loss = float("inf")
    best_epoch = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()

        train_loss_sum = 0.0
        train_count = 0

        for signals, labels in train_loader:
            optimizer.zero_grad(set_to_none=True)

            logits = model(signals)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            n = labels.size(0)
            train_loss_sum += loss.item() * n
            train_count += n

        train_loss = train_loss_sum / train_count

        model.eval()

        val_loss_sum = 0.0
        val_count = 0
        val_predictions = []

        with torch.no_grad():
            for signals, labels in val_loader:
                logits = model(signals)
                loss = criterion(logits, labels)

                n = labels.size(0)
                val_loss_sum += loss.item() * n
                val_count += n

                val_predictions.extend(
                    logits.argmax(dim=1).numpy()
                )

        val_loss = val_loss_sum / val_count

        val_predictions = np.asarray(
            val_predictions,
            dtype=np.int64,
        )

        val_bacc = balanced_accuracy_score(
            y_val,
            val_predictions,
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_balanced_accuracy": val_bacc,
        })

        print(
            f"Epoch {epoch:02d}/{MAX_EPOCHS} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_balanced_accuracy={val_bacc:.4f}"
        )

        better = (
            val_bacc > best_balanced_accuracy
            or (
                np.isclose(val_bacc, best_balanced_accuracy)
                and val_loss < best_val_loss
            )
        )

        if better:
            best_balanced_accuracy = val_bacc
            best_val_loss = val_loss
            best_epoch = epoch

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_name": "TremorCNN",
                    "num_classes": 2,
                    "input_shape": [1, 3, 256],
                    "normalization_mean": (
                        mean.reshape(3).astype(np.float32)
                    ),
                    "normalization_std": (
                        std.reshape(3).astype(np.float32)
                    ),
                    "labels": {
                        "0": "healthy_control",
                        "1": "tremor_associated_ET_or_Parkinsons",
                    },
                    "selected_epoch": epoch,
                    "validation_loss": float(val_loss),
                    "validation_balanced_accuracy": float(val_bacc),
                    "sample_rate_hz": 100,
                    "window_samples": 256,
                    "window_seconds": 2.56,
                    "split_method": (
                        "subject-level train/validation/test split"
                    ),
                    "seed": SEED,
                    "training_subject_count": len(train_ids),
                    "validation_subject_count": len(val_ids),
                    "test_subject_count": len(test_ids),
                    "training_window_count": len(y_train),
                    "validation_window_count": len(y_val),
                    "test_window_count": len(y_test),
                    "training_class_counts": class_counts.tolist(),
                },
                CHECKPOINT_FILE,
            )

            print("  Saved best expanded-dataset checkpoint.")

    with HISTORY_FILE.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "train_loss",
                "val_loss",
                "val_balanced_accuracy",
            ],
        )

        writer.writeheader()
        writer.writerows(history)

    print("\n========== TRAINING COMPLETE ==========")
    print("Best epoch:", best_epoch)
    print(
        "Best validation balanced accuracy: "
        f"{best_balanced_accuracy:.4f}"
    )
    print(f"Best validation loss: {best_val_loss:.4f}")
    print("Checkpoint:", CHECKPOINT_FILE)
    print("History:", HISTORY_FILE)
    print("\nThe held-out test set has not been used for model selection.")


if __name__ == "__main__":
    main()
