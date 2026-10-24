from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path.home() / "tremorgate_ws/data/asl_glove/ASL-Sensor-Dataglove-Dataset"
OUT = Path.home() / "tremorgate_ws/offline/data"
OUT.mkdir(parents=True, exist_ok=True)

WINDOW = 256
STRIDE = 128
GAP_THRESHOLD = 0.05  # seconds; investigate gaps over 50 ms

records = []

for i, path in enumerate(sorted(ROOT.glob("*/*.csv")), 1):
    df = pd.read_csv(path, usecols=["timestamp"])
    t = df["timestamp"].to_numpy(dtype=float)
    dt = np.diff(t)
    positive = dt[dt > 0]

    if len(positive) == 0:
        records.append({
            "participant": path.parent.name,
            "gesture": path.stem,
            "file": str(path.relative_to(ROOT)),
            "rows": len(t),
            "duration_s": float(t[-1] - t[0]),
            "median_dt_s": np.nan,
            "gaps_over_50ms": 0,
            "gaps_over_500ms": 0,
            "gaps_over_2s": 0,
            "continuous_segments": 0,
            "usable_windows": 0,
        })
        continue

    # Split at timestamp gaps so windows never bridge missing time.
    boundaries = np.where(dt > GAP_THRESHOLD)[0] + 1
    segments = np.split(np.arange(len(t)), boundaries)
    usable_segments = [seg for seg in segments if len(seg) >= WINDOW]

    windows = sum(
        1 + (len(seg) - WINDOW) // STRIDE
        for seg in usable_segments
    )

    records.append({
        "participant": path.parent.name,
        "gesture": path.stem,
        "file": str(path.relative_to(ROOT)),
        "rows": len(t),
        "duration_s": float(t[-1] - t[0]),
        "median_dt_s": float(np.median(positive)),
        "gaps_over_50ms": int((dt > 0.05).sum()),
        "gaps_over_500ms": int((dt > 0.5).sum()),
        "gaps_over_2s": int((dt > 2).sum()),
        "continuous_segments": len(usable_segments),
        "usable_windows": int(windows),
    })

    if i % 100 == 0:
        print(f"Checked {i}/1000 recordings")

audit = pd.DataFrame(records)
audit.to_csv(OUT / "asl_glove_gap_audit.csv", index=False)

print("\n=== TIMESTAMP GAP AUDIT ===")
print("Recordings:", len(audit))
print("Recordings with gaps > 50 ms:",
      int((audit["gaps_over_50ms"] > 0).sum()))
print("Recordings with gaps > 500 ms:",
      int((audit["gaps_over_500ms"] > 0).sum()))
print("Recordings with gaps > 2 seconds:",
      int((audit["gaps_over_2s"] > 0).sum()))

print("\n=== GAP COUNTS PER RECORDING ===")
print(audit[["gaps_over_50ms", "gaps_over_500ms", "gaps_over_2s"]]
      .describe().round(2).to_string())

print("\n=== ESTIMATED WINDOWS ===")
print("Total 256-sample windows, 128-sample stride:",
      int(audit["usable_windows"].sum()))
print("Recordings with at least one usable window:",
      int((audit["usable_windows"] > 0).sum()))

print("\n=== RECORDINGS WITH MOST LARGE GAPS ===")
print(audit.sort_values("gaps_over_500ms", ascending=False)
      [["file", "duration_s", "gaps_over_50ms",
        "gaps_over_500ms", "gaps_over_2s", "usable_windows"]]
      .head(10).to_string(index=False))

print("\nSaved:", OUT / "asl_glove_gap_audit.csv")
