import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

d = np.load("offline/data/pads_fft_features.npz")
s = np.load("offline/data/pads_split_subjects.npz")

ratio = d["features"][:, 3]
condition = d["condition"]
subjects = d["subject_id"]

threshold = float(np.load("offline/data/fft_threshold_results.npz")["best_threshold"])

y = (condition != "Healthy").astype(int)
test = np.isin(subjects, s["test_subjects"])

# 1. Threshold distribution
plt.figure(figsize=(9, 5))
plt.hist(ratio[test & (y == 0)], bins=50, alpha=0.7, label="Healthy")
plt.hist(ratio[test & (y == 1)], bins=50, alpha=0.7, label="Tremor-associated")
plt.axvline(threshold, linestyle="--", label=f"Threshold = {threshold:.3f}")
plt.xlabel("Maximum 4–8 Hz band-power ratio")
plt.ylabel("Number of windows")
plt.title("FFT Tremor-Band Feature Distribution")
plt.legend()
plt.tight_layout()
plt.savefig("offline/plots/fft_feature_distribution.png", dpi=150)
plt.close()

# 2. Confusion matrix
pred = (ratio[test] >= threshold).astype(int)
cm = confusion_matrix(y[test], pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Voluntary / Control", "Involuntary / Tremor-associated"]
)

disp.plot()
plt.title("FFT Baseline Confusion Matrix")
plt.tight_layout()
plt.savefig("offline/plots/fft_confusion_matrix.png", dpi=150)
plt.close()

print("Saved:")
print("offline/plots/fft_feature_distribution.png")
print("offline/plots/fft_confusion_matrix.png")
