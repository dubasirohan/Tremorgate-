from pathlib import Path
import copy
import json
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

ROOT = Path.home() / "tremorgate_ws"
DATA = ROOT / "data_external" / "uci_har" / "UCI HAR Dataset"
OUT = ROOT / "offline" / "models"
RESULTS = ROOT / "offline" / "data"

OUT.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cpu")
BATCH_SIZE = 64
EPOCHS = 30
PATIENCE = 6
NUM_CLASSES = 6
CHANNELS = [
    "body_acc_x", "body_acc_y", "body_acc_z",
    "body_gyro_x", "body_gyro_y", "body_gyro_z",
]
CLASS_NAMES = [
    "WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS",
    "SITTING", "STANDING", "LAYING",
]


def load_signals(split):
    folder = DATA / split / "Inertial Signals"
    channels = []
    for name in CHANNELS:
        path = folder / f"{name}_{split}.txt"
        if not path.exists():
            raise FileNotFoundError(f"Missing dataset file: {path}")
        values = np.loadtxt(path, dtype=np.float32)
        channels.append(values)

    x = np.stack(channels, axis=1)  # samples, 6 channels, 128 timesteps
    y = np.loadtxt(DATA / split / f"y_{split}.txt", dtype=np.int64) - 1
    subjects = np.loadtxt(
        DATA / split / f"subject_{split}.txt", dtype=np.int64
    )

    if len(x) != len(y) or len(y) != len(subjects):
        raise ValueError(f"Inconsistent sample counts in {split} split")
    return x, y, subjects


class ActivityCNN(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(6, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def main():
    x_all, y_all, subjects = load_signals("train")
    x_test, y_test, _ = load_signals("test")

    # Keep all windows from each subject in only one train/validation group.
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=0.20, random_state=SEED
    )
    train_idx, val_idx = next(
        splitter.split(x_all, y_all, groups=subjects)
    )

    x_train, y_train = x_all[train_idx], y_all[train_idx]
    x_val, y_val = x_all[val_idx], y_all[val_idx]

    train_subjects = set(subjects[train_idx])
    val_subjects = set(subjects[val_idx])
    assert train_subjects.isdisjoint(val_subjects)

    # Normalize each channel using training data only.
    mean = x_train.mean(axis=(0, 2), keepdims=True)
    std = x_train.std(axis=(0, 2), keepdims=True)
    std[std < 1e-6] = 1.0

    x_train = (x_train - mean) / std
    x_val = (x_val - mean) / std
    x_test = (x_test - mean) / std

    def loader(x, y, shuffle=False):
        ds = TensorDataset(
            torch.from_numpy(x.astype(np.float32)),
            torch.from_numpy(y.astype(np.int64)),
        )
        return DataLoader(
            ds, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=0
        )

    train_loader = loader(x_train, y_train, True)
    val_loader = loader(x_val, y_val)
    test_loader = loader(x_test, y_test)

    model = ActivityCNN(NUM_CLASSES).to(DEVICE)

    counts = np.bincount(y_train, minlength=NUM_CLASSES)
    weights = len(y_train) / (NUM_CLASSES * np.maximum(counts, 1))
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32, device=DEVICE)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_bacc = -1.0
    best_state = None
    best_epoch = 0
    stale_epochs = 0
    history = []

    print(f"Device: {DEVICE}")
    print(f"Windows: train={len(x_train)}, val={len(x_val)}, test={len(x_test)}")
    print(
        f"Subjects: train={len(train_subjects)}, "
        f"val={len(val_subjects)}, test={len(np.unique(load_signals('test')[2]))}"
    )
    print("Input shape: (batch, 6 channels, 128 timesteps)")
    print("Training CNN...")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_losses = []

        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_true, val_pred, val_losses = [], [], []

        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                logits = model(xb)
                val_losses.append(criterion(logits, yb).item())
                val_pred.extend(logits.argmax(dim=1).cpu().numpy())
                val_true.extend(yb.cpu().numpy())

        val_bacc = balanced_accuracy_score(val_true, val_pred)
        val_loss = float(np.mean(val_losses))
        train_loss = float(np.mean(train_losses))

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_balanced_accuracy": float(val_bacc),
        })
        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | val_BAcc={val_bacc:.4f}"
        )

        if val_bacc > best_bacc:
            best_bacc = val_bacc
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1

        if stale_epochs >= PATIENCE:
            print("Early stopping.")
            break

    model.load_state_dict(best_state)

    # Final test evaluation, performed only after model selection.
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            logits = model(xb.to(DEVICE))
            y_pred.extend(logits.argmax(dim=1).cpu().numpy())
            y_true.extend(yb.numpy())

    acc = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    report = classification_report(
        y_true, y_pred, labels=list(range(NUM_CLASSES)),
        target_names=CLASS_NAMES, output_dict=True, zero_division=0
    )

    checkpoint = {
        "model_state_dict": best_state,
        "model_name": "UCI HAR six-class activity CNN",
        "input_channels": CHANNELS,
        "class_names": CLASS_NAMES,
        "mean": mean.reshape(-1).tolist(),
        "std": std.reshape(-1).tolist(),
        "best_epoch": best_epoch,
        "validation_balanced_accuracy": float(best_bacc),
        "seed": SEED,
    }
    model_path = OUT / "uci_har_activity_cnn_best.pt"
    torch.save(checkpoint, model_path)

    pd.DataFrame(history).to_csv(
        RESULTS / "uci_har_activity_training_history.csv", index=False
    )
    result = {
        "model": str(model_path),
        "best_epoch": best_epoch,
        "validation_balanced_accuracy": float(best_bacc),
        "test_accuracy": float(acc),
        "test_balanced_accuracy": float(bacc),
        "confusion_matrix": cm.tolist(),
        "class_names": CLASS_NAMES,
        "classification_report": report,
    }
    result_path = RESULTS / "uci_har_activity_test_results.json"
    result_path.write_text(json.dumps(result, indent=2))

    print("\n=== FINAL TEST RESULTS ===")
    print(f"Best epoch: {best_epoch}")
    print(f"Validation balanced accuracy: {best_bacc:.4f}")
    print(f"Test accuracy: {acc:.4f}")
    print(f"Test balanced accuracy: {bacc:.4f}")
    print("\nConfusion matrix (rows=true, columns=predicted):")
    print(cm)
    print("\nClassification report:")
    print(classification_report(
        y_true, y_pred, target_names=CLASS_NAMES, zero_division=0
    ))
    print(f"Model saved: {model_path}")
    print(f"Results saved: {result_path}")


if __name__ == "__main__":
    main()
