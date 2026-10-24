import numpy as np

FS = 100.0
WINDOW_SIZE = 256
LOW_HZ = 4.0
HIGH_HZ = 8.0

def preprocess_signal(x):
    x = np.asarray(x, dtype=np.float64)
    x = x - np.mean(x)
    t = np.arange(len(x), dtype=np.float64)
    p = np.polyfit(t, x, 1)
    return x - np.polyval(p, t)

def fft_features(x, fs=FS):
    x = preprocess_signal(x)

    window = np.hanning(len(x))
    y = np.fft.rfft(x * window)
    freqs = np.fft.rfftfreq(len(x), d=1.0 / fs)

    power = np.abs(y) ** 2
    band = (freqs >= LOW_HZ) & (freqs <= HIGH_HZ)

    band_power = np.sum(power[band])
    total_power = np.sum(power[1:])

    ratio = band_power / (total_power + 1e-12)

    peak_idx = np.argmax(power[band])
    band_freqs = freqs[band]
    peak_frequency = band_freqs[peak_idx]

    return {
        "band_power": float(band_power),
        "total_power": float(total_power),
        "band_ratio": float(ratio),
        "peak_frequency": float(peak_frequency),
    }

def extract_window_features(X):
    results = []

    for window in X:
        axis_features = [fft_features(window[i]) for i in range(3)]

        ratios = [f["band_ratio"] for f in axis_features]
        powers = [f["band_power"] for f in axis_features]

        best = int(np.argmax(ratios))

        results.append([
            ratios[0], ratios[1], ratios[2],
            max(ratios),
            powers[0], powers[1], powers[2],
            axis_features[best]["peak_frequency"],
        ])

    return np.asarray(results, dtype=np.float32)

if __name__ == "__main__":
    data = np.load("offline/data/pads_windows.npz")

    keep = np.isin(
        data["condition"],
        ["Healthy", "Essential Tremor", "Parkinson's"]
    )

    X = data["X"][keep]
    subject_id = data["subject_id"][keep]
    condition = data["condition"][keep]
    task = data["task"][keep]
    wrist = data["wrist"][keep]

    features = extract_window_features(X)

    np.savez(
        "offline/data/pads_fft_features.npz",
        features=features,
        subject_id=subject_id,
        condition=condition,
        task=task,
        wrist=wrist,
        fs=FS,
        window_size=WINDOW_SIZE,
        low_hz=LOW_HZ,
        high_hz=HIGH_HZ,
    )

    print("========== FFT FEATURES ==========")
    print("Input windows:", X.shape)
    print("Feature shape:", features.shape)
    print("Features:")
    print("0-2 = X/Y/Z band ratios")
    print("3   = maximum axis band ratio")
    print("4-6 = X/Y/Z band power")
    print("7   = peak frequency in 4-8 Hz")
    print("Saved: offline/data/pads_fft_features.npz")
