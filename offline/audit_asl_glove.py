from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path.home() / "tremorgate_ws/data/asl_glove/ASL-Sensor-Dataglove-Dataset"
OUT = Path.home() / "tremorgate_ws/offline/data"
OUT.mkdir(parents=True, exist_ok=True)

sensor_cols = [
    "flex_1", "flex_2", "flex_3", "flex_4", "flex_5",
    "GYRx", "GYRy", "GYRz",
    "ACCx", "ACCy", "ACCz",
]

files = sorted(ROOT.glob("*/*.csv"))
records = []
errors = []

for i, path in enumerate(files, 1):
    try:
        df = pd.read_csv(path)

        missing_cols = [c for c in ["timestamp", "user_id"] + sensor_cols
                        if c not in df.columns]
        if missing_cols:
            errors.append({"file": str(path), "error": f"Missing columns: {missing_cols}"})
            continue

        ts = pd.to_numeric(df["timestamp"], errors="coerce").to_numpy()
        valid_ts = ts[np.isfinite(ts)]
        dt = np.diff(valid_ts)
        dt = dt[dt > 0]

        hz = float(1.0 / np.median(dt)) if len(dt) else None
        duration = float(valid_ts[-1] - valid_ts[0]) if len(valid_ts) > 1 else 0.0

        values = df[sensor_cols].apply(pd.to_numeric, errors="coerce")
        records.append({
            "participant": path.parent.name,
            "gesture": path.stem,
            "file": str(path.relative_to(ROOT)),
            "rows": len(df),
            "duration_s": duration,
            "sampling_hz": hz,
            "missing_sensor_values": int(values.isna().sum().sum()),
            "infinite_sensor_values": int(
                np.isinf(values.to_numpy(dtype=float)).sum()
            ),
        })

    except Exception as exc:
        errors.append({"file": str(path), "error": repr(exc)})

    if i % 100 == 0:
        print(f"Audited {i}/{len(files)} recordings")

summary = pd.DataFrame(records)
summary.to_csv(OUT / "asl_glove_recording_audit.csv", index=False)

print("\n=== OVERALL DATASET ===")
print("Files discovered:", len(files))
print("Successfully audited:", len(summary))
print("Participants:", summary["participant"].nunique() if len(summary) else 0)
print("Gesture classes:", summary["gesture"].nunique() if len(summary) else 0)
print("Audit errors:", len(errors))

if len(summary):
    print("\n=== ROW COUNTS ===")
    print(summary["rows"].describe().to_string())

    print("\n=== RECORDING DURATIONS (SECONDS) ===")
    print(summary["duration_s"].describe().round(3).to_string())

    print("\n=== ESTIMATED SAMPLING RATE (Hz) ===")
    print(summary["sampling_hz"].describe().round(2).to_string())

    print("\n=== SENSOR DATA QUALITY ===")
    print("Files with missing sensor values:",
          int((summary["missing_sensor_values"] > 0).sum()))
    print("Files with infinite sensor values:",
          int((summary["infinite_sensor_values"] > 0).sum()))

    print("\n=== GESTURE RECORDINGS ===")
    print(summary.groupby("gesture").size().sort_index().to_string())

    print("\n=== RECORDINGS PER PARTICIPANT ===")
    print(summary.groupby("participant").size().describe().to_string())

    print("\n=== FIRST 10 RECORDINGS ===")
    print(summary.head(10).to_string(index=False))

if errors:
    with open(OUT / "asl_glove_audit_errors.json", "w") as f:
        json.dump(errors, f, indent=2)
    print("\nFirst audit errors:")
    print(json.dumps(errors[:5], indent=2))

print(f"\nSaved audit CSV: {OUT / 'asl_glove_recording_audit.csv'}")
