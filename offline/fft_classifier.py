import numpy as np

FS = 100.0
WINDOW_SIZE = 256
LOW_HZ = 4.0
HIGH_HZ = 8.0

DEFAULT_THRESHOLD = 0.435

def preprocess_signal(x):
    x = np.asarray(x, dtype=np.float64)
    x = x - np.mean(x)

    t = np.arange(len(x), dtype=np.float64)
    p = np.polyfit(t, x, 1)

    return x - np.polyval(p, t)

def band_ratio(x, fs=FS):
    x = preprocess_signal(x)

    window = np.hanning(len(x))
    spectrum = np.fft.rfft(x * window)
    frequencies = np.fft.rfftfreq(len(x), d=1.0 / fs)

    power = np.abs(spectrum) ** 2

    band = (frequencies >= LOW_HZ) & (frequencies <= HIGH_HZ)

    band_power = np.sum(power[band])
    total_power = np.sum(power[1:])

    return float(band_power / (total_power + 1e-12))

def classify(x, threshold=DEFAULT_THRESHOLD, fs=FS):
    """
    Classify one 3-axis acceleration window.

    x shape:
        (3, 256)

    Returns:
        label:
            0 = voluntary/control
            1 = involuntary/tremor-associated

        score:
            maximum 4–8 Hz band ratio
    """

    x = np.asarray(x)

    if x.shape != (3, WINDOW_SIZE):
        raise ValueError(
            f"Expected input shape (3, {WINDOW_SIZE}), got {x.shape}"
        )

    ratios = [
        band_ratio(x[0], fs),
        band_ratio(x[1], fs),
        band_ratio(x[2], fs),
    ]

    score = max(ratios)
    label = int(score >= threshold)

    return label, score

if __name__ == "__main__":
    rng = np.random.default_rng(42)

    sample = rng.normal(
        size=(3, WINDOW_SIZE)
    )

    label, score = classify(sample)

    print("FFT classifier test")
    print("Input shape:", sample.shape)
    print("Label:", label)
    print("Score:", score)
    print("Threshold:", DEFAULT_THRESHOLD)
