#!/usr/bin/env python3

from pathlib import Path
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "data" / "cnn_training_history.csv"
PLOT_DIR = ROOT / "plots"

PLOT_DIR.mkdir(parents=True, exist_ok=True)

with DATA_FILE.open(newline="") as f:
    rows = list(csv.DictReader(f))

if not rows:
    raise RuntimeError(f"No training history found in {DATA_FILE}")

epochs = [int(row["epoch"]) for row in rows]
train_loss = [float(row["train_loss"]) for row in rows]
val_loss = [float(row["val_loss"]) for row in rows]
val_bacc = [
    float(row["val_balanced_accuracy"])
    for row in rows
]

best_index = max(range(len(rows)), key=lambda i: val_bacc[i])
best_epoch = epochs[best_index]

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(epochs, train_loss, marker="o", label="Training loss")
ax.plot(epochs, val_loss, marker="o", label="Validation loss")
ax.axvline(
    best_epoch,
    linestyle="--",
    label=f"Best validation balanced accuracy: epoch {best_epoch}",
)
ax.set_xlabel("Epoch")
ax.set_ylabel("Weighted cross-entropy loss")
ax.set_title("TremorGate CNN Training and Validation Loss")
ax.grid(True, alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(PLOT_DIR / "cnn_loss_curves.png", dpi=160)
plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(epochs, val_bacc, marker="o")
ax.axhline(
    0.5,
    linestyle="--",
    label="0.50 reference",
)
ax.axvline(
    best_epoch,
    linestyle="--",
    label=f"Best epoch: {best_epoch}",
)
ax.set_xlabel("Epoch")
ax.set_ylabel("Validation balanced accuracy")
ax.set_ylim(0, 1)
ax.set_title("TremorGate CNN Validation Balanced Accuracy")
ax.grid(True, alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(PLOT_DIR / "cnn_validation_balanced_accuracy.png", dpi=160)
plt.close(fig)

print("Training history loaded:", len(rows), "epochs")
print("Best epoch:", best_epoch)
print(f"Best validation balanced accuracy: {val_bacc[best_index]:.4f}")
print("Saved:", PLOT_DIR / "cnn_loss_curves.png")
print("Saved:", PLOT_DIR / "cnn_validation_balanced_accuracy.png")
