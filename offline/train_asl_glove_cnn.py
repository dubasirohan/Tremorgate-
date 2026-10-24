from pathlib import Path
import json
import copy
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report

SEED = 42
EPOCHS = 60
BATCH_SIZE = 128
PATIENCE = 10
LR = 1e-3

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

BASE = Path.home() / "tremorgate_ws/offline"
DATA = BASE / "data/asl_glove_windows.npz"
MODEL_DIR = BASE / "models"
RESULT_DIR = BASE / "data"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

d = np.load(DATA, allow_pickle=False)
X = d["X"].astype(np.float32)
y = d["y"].astype(np.int64)
train_mask = d["train_mask"].astype(bool)
val_mask = d["val_mask"].astype(bool)
test_mask = d["test_mask"].astype(bool)
gesture_names = d["gesture_names"].tolist()
channel_names = d["channel_names"].tolist()

# Conv1d expects [batch, channels, time].
X = np.transpose(X, (0, 2, 1)).copy()

def make_loader(mask, shuffle=False):
    ds = TensorDataset(
        torch.from_numpy(X[mask]),
        torch.from_numpy(y[mask]),
    )
    return DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,
    )

train_loader = make_loader(train_mask, shuffle=True)
val_loader = make_loader(val_mask)
test_loader = make_loader(test_mask)

class GloveCNN(nn.Module):
    def __init__(self, n_channels, n_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(n_channels, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.35),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))

model = GloveCNN(len(channel_names), len(gesture_names)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", factor=0.5, patience=3
)

def evaluate(loader):
    model.eval()
    total_loss = 0.0
    ys, ps = [], []

    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            total_loss += loss.item() * len(yb)
            ys.extend(yb.cpu().numpy())
            ps.extend(logits.argmax(dim=1).cpu().numpy())

    return (
        total_loss / max(len(loader.dataset), 1),
        accuracy_score(ys, ps),
        balanced_accuracy_score(ys, ps),
        np.asarray(ys),
        np.asarray(ps),
    )

best_val_bacc = -1.0
best_state = None
best_epoch = 0
stale_epochs = 0
history = []

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0

    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total_loss += loss.item() * len(yb)

    train_loss = total_loss / len(train_loader.dataset)
    val_loss, val_acc, val_bacc, _, _ = evaluate(val_loader)
    scheduler.step(val_bacc)

    history.append({
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "val_accuracy": val_acc,
        "val_balanced_accuracy": val_bacc,
        "lr": optimizer.param_groups[0]["lr"],
    })

    print(
        f"Epoch {epoch:02d}/{EPOCHS} | "
        f"train_loss={train_loss:.4f} | "
        f"val_loss={val_loss:.4f} | "
        f"val_acc={val_acc:.4f} | "
        f"val_bacc={val_bacc:.4f}"
    )

    if val_bacc > best_val_bacc:
        best_val_bacc = val_bacc
        best_epoch = epoch
        best_state = copy.deepcopy(model.state_dict())
        stale_epochs = 0
    else:
        stale_epochs += 1

    if stale_epochs >= PATIENCE:
        print(f"Early stopping at epoch {epoch}")
        break

if best_state is None:
    raise RuntimeError("Training did not produce a checkpoint.")

model.load_state_dict(best_state)
test_loss, test_acc, test_bacc, y_true, y_pred = evaluate(test_loader)

print("\n=== HELD-OUT TEST RESULTS ===")
print("Best epoch:", best_epoch)
print(f"Best validation balanced accuracy: {best_val_bacc:.4f}")
print(f"Test loss: {test_loss:.4f}")
print(f"Test accuracy: {test_acc:.4f}")
print(f"Test balanced accuracy: {test_bacc:.4f}")
print("\nClassification report:")
print(classification_report(
    y_true,
    y_pred,
    labels=np.arange(len(gesture_names)),
    target_names=gesture_names,
    zero_division=0,
))

checkpoint = MODEL_DIR / "asl_glove_cnn_best.pt"
torch.save({
    "model_state_dict": best_state,
    "model_name": "GloveCNN",
    "input_channels": channel_names,
    "num_classes": len(gesture_names),
    "gesture_names": gesture_names,
    "window_samples": int(X.shape[2]),
    "normalization_mean": d["normalization_mean"],
    "normalization_std": d["normalization_std"],
    "best_epoch": best_epoch,
    "validation_balanced_accuracy": float(best_val_bacc),
    "test_accuracy": float(test_acc),
    "test_balanced_accuracy": float(test_bacc),
}, checkpoint)

with open(RESULT_DIR / "asl_glove_cnn_results.json", "w") as f:
    json.dump({
        "best_epoch": best_epoch,
        "best_validation_balanced_accuracy": float(best_val_bacc),
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
        "test_balanced_accuracy": float(test_bacc),
        "classification_report": classification_report(
            y_true, y_pred,
            labels=np.arange(len(gesture_names)),
            target_names=gesture_names,
            output_dict=True,
            zero_division=0,
        ),
        "note": (
            "40-class ASL gesture recognition on held-out participants. "
            "Not a voluntary-versus-involuntary or tremor classifier."
        ),
    }, f, indent=2)

import pandas as pd
pd.DataFrame(history).to_csv(
    RESULT_DIR / "asl_glove_cnn_training_history.csv",
    index=False,
)

print("\nSaved checkpoint:", checkpoint)
print("Saved results:", RESULT_DIR / "asl_glove_cnn_results.json")
print("Saved history:", RESULT_DIR / "asl_glove_cnn_training_history.csv")
