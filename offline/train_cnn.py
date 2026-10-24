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

WINDOWS_FILE = DATA_DIR / "pads_windows.npz"
SPLIT_FILE = DATA_DIR / "pads_split_subjects.npz"
CHECKPOINT_FILE = MODEL_DIR / "tremor_cnn_best.pt"
HISTORY_FILE = DATA_DIR / "cnn_training_history.csv"

SEED = 42
BATCH_SIZE = 32
MAX_EPOCHS = 20
LEARNING_RATE = 1e-3

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(2)


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Load the actual keys verified in your dataset inspection.
    with np.load(WINDOWS_FILE, allow_pickle=True) as data:
        X = data["X"].astype(np.float32)
        subject_ids = data["subject_id"].astype(str)
        conditions = data["condition"].astype(str)

    with np.load(SPLIT_FILE, allow_pickle=True) as split:
        train_ids = set(split["train_subjects"].astype(str))
        val_ids = set(split["val_subjects"].astype(str))
        test_ids = set(split["test_subjects"].astype(str))

    if X.ndim != 3 or X.shape[1:] != (3, 256):
        raise ValueError(f"Unexpected window shape: {X.shape}")

    if not (train_ids.isdisjoint(val_ids)
            and train_ids.isdisjoint(test_ids)
            and val_ids.isdisjoint(test_ids)):
        raise ValueError("Subject splits overlap. Aborting to prevent leakage.")

    # Binary experiment: Healthy=0; Essential Tremor/Parkinson's=1.
    condition_normalized = np.array([
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

    healthy = np.isin(condition_normalized, list(healthy_names))
    positive = np.isin(condition_normalized, list(positive_names))
    keep = healthy | positive

    X = X[keep]
    subject_ids = subject_ids[keep]
    condition_normalized = condition_normalized[keep]

    y = np.where(
        np.isin(condition_normalized, list(healthy_names)), 0, 1
    ).astype(np.int64)

    masks = {
        "train": np.isin(subject_ids, list(train_ids)),
        "val": np.isin(subject_ids, list(val_ids)),
        "test": np.isin(subject_ids, list(test_ids)),
    }

    assigned = masks["train"] | masks["val"] | masks["test"]
    if not np.all(assigned):
        unknown = np.unique(subject_ids[~assigned])
        raise ValueError(
            f"Eligible subjects missing from split: {unknown.tolist()}"
        )

    print("Filtered dataset:", X.shape)
    print("Labels: 0=Healthy, 1=Essential Tremor or Parkinson's")

    for name, mask in masks.items():
        counts = np.bincount(y[mask], minlength=2)
        print(
            f"{name}: windows={int(mask.sum())}, "
            f"healthy={counts[0]}, positive={counts[1]}, "
            f"subjects={len(np.unique(subject_ids[mask]))}"
        )
        if counts.min() == 0:
            raise ValueError(f"{name} split does not contain both classes.")

    X_train, y_train = X[masks["train"]], y[masks["train"]]
    X_val, y_val = X[masks["val"]], y[masks["val"]]

    # Fit normalization only on training subjects.
    mean = X_train.mean(axis=(0, 2), keepdims=True)
    std = X_train.std(axis=(0, 2), keepdims=True)
    std = np.maximum(std, 1e-6)

    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std

    def make_loader(signals, labels, shuffle):
        dataset = TensorDataset(
            torch.from_numpy(np.ascontiguousarray(signals, dtype=np.float32)),
            torch.from_numpy(np.ascontiguousarray(labels, dtype=np.int64)),
        )
        return DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=shuffle,
            num_workers=0,
        )

    train_loader = make_loader(X_train, y_train, True)
    val_loader = make_loader(X_val, y_val, False)

    # Counter class imbalance using weights computed from training labels.
    class_counts = np.bincount(y_train, minlength=2)
    class_weights = len(y_train) / (2.0 * class_counts.astype(np.float32))

    model = TremorCNN(num_classes=2)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("\nDevice: CPU")
    print("Training class counts:", class_counts.tolist())
    print("Class weights:", class_weights.tolist())
    print("Epochs:", MAX_EPOCHS, "| Batch size:", BATCH_SIZE)

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
                val_predictions.extend(logits.argmax(dim=1).numpy())

        val_loss = val_loss_sum / val_count
        val_predictions = np.asarray(val_predictions, dtype=np.int64)
        val_bacc = balanced_accuracy_score(y_val, val_predictions)

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

            torch.save({
                "model_state_dict": model.state_dict(),
                "model_name": "TremorCNN",
                "num_classes": 2,
                "input_shape": [1, 3, 256],
                "normalization_mean": mean.reshape(3).astype(np.float32),
                "normalization_std": std.reshape(3).astype(np.float32),
                "labels": {
                    "0": "healthy_control",
                    "1": "tremor_associated_ET_or_Parkinsons",
                },
                "selected_epoch": epoch,
                "validation_loss": val_loss,
                "validation_balanced_accuracy": val_bacc,
                "sample_rate_hz": 100,
                "window_samples": 256,
                "window_seconds": 2.56,
                "split_method": "subject-level",
                "seed": SEED,
            }, CHECKPOINT_FILE)

            print("  Saved best validation checkpoint.")

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

    print("\nTraining finished.")
    print("Best epoch:", best_epoch)
    print(f"Best validation balanced accuracy: {best_balanced_accuracy:.4f}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print("Checkpoint:", CHECKPOINT_FILE)
    print("History:", HISTORY_FILE)
    print("Test subjects have not been evaluated.")


if __name__ == "__main__":
    main()
