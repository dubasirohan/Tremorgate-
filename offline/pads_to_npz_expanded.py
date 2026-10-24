#!/usr/bin/env python3

import json
from pathlib import Path

import numpy as np


PADS_ROOT = Path.home() / "tremorgate_ws" / "data" / "pads" / "physionet.org" / "files" / "parkinsons-disease-smartwatch" / "1.0.0"

OBS_DIR = PADS_ROOT / "movement"
PATIENT_DIR = PADS_ROOT / "patients"
TIMESERIES_DIR = PADS_ROOT / "movement" / "timeseries"

OUTPUT = Path.home() / "tremorgate_ws" / "offline" / "data" / "pads_windows_expanded.npz"

FS = 100.0
WINDOW = 256
STEP = 128


def load_patient(patient_id):
    path = PATIENT_DIR / f"patient_{patient_id}.json"
    with open(path, "r") as f:
        return json.load(f)


def load_timeseries(path):
    data = np.loadtxt(path, delimiter=",", dtype=np.float32)

    if data.ndim != 2 or data.shape[1] != 7:
        raise ValueError(
            f"Unexpected shape in {path}: {data.shape}; expected (N, 7)"
        )

    return data


def make_windows(accel):
    windows = []

    if len(accel) < WINDOW:
        return windows

    for start in range(0, len(accel) - WINDOW + 1, STEP):
        window = accel[start:start + WINDOW]

        if not np.isfinite(window).all():
            continue

        windows.append(window.T.astype(np.float32))

    return windows


def main():
    print("PADS ROOT:", PADS_ROOT)
    print("Output:", OUTPUT)
    print()

    observation_files = sorted(OBS_DIR.glob("observation_*.json"))

    print("Observation files:", len(observation_files))

    X = []
    subject_ids = []
    conditions = []
    tasks = []
    wrists = []

    skipped = 0
    total_recordings = 0
    total_windows = 0

    for obs_path in observation_files:
        with open(obs_path, "r") as f:
            observation = json.load(f)

        subject_id = str(observation["subject_id"]).zfill(3)
        patient = load_patient(subject_id)

        condition = patient["condition"]

        sampling_rate = float(observation["sampling_rate"])

        if sampling_rate != FS:
            print(
                f"WARNING: subject {subject_id}: "
                f"metadata sampling rate = {sampling_rate}"
            )

        for session in observation.get("session", []):
            task = session["record_name"]

            for record in session.get("records", []):
                total_recordings += 1

                wrist = record["device_location"]
                relative_file = record["file_name"]

                ts_path = PADS_ROOT / "movement" / relative_file

                if not ts_path.exists():
                    print("MISSING:", ts_path)
                    skipped += 1
                    continue

                try:
                    data = load_timeseries(ts_path)
                except Exception as exc:
                    print("ERROR:", ts_path, exc)
                    skipped += 1
                    continue

                time = data[:, 0]
                accel = data[:, 1:4]

                if len(time) >= 2:
                    dt = np.diff(time)
                    estimated_fs = 1.0 / np.mean(dt)

                    if abs(estimated_fs - FS) > 1.0:
                        print(
                            f"WARNING: {subject_id} {task} {wrist}: "
                            f"estimated_fs={estimated_fs:.3f}"
                        )

                windows = make_windows(accel)

                for window in windows:
                    X.append(window)
                    subject_ids.append(subject_id)
                    conditions.append(condition)
                    tasks.append(task)
                    wrists.append(wrist)

                total_windows += len(windows)

    if not X:
        raise RuntimeError("No valid windows were generated.")

    X = np.stack(X).astype(np.float32)

    subject_ids = np.asarray(subject_ids)
    conditions = np.asarray(conditions)
    tasks = np.asarray(tasks)
    wrists = np.asarray(wrists)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        OUTPUT,
        X=X,
        subject_id=subject_ids,
        condition=conditions,
        task=tasks,
        wrist=wrists,
        fs=np.float32(FS),
        window_size=np.int32(WINDOW),
        step=np.int32(STEP),
    )

    print()
    print("========== COMPLETE ==========")
    print("Recordings processed:", total_recordings)
    print("Recordings skipped:", skipped)
    print("Windows:", len(X))
    print("X shape:", X.shape)
    print("Expected shape: (N, 3, 256)")
    print("Sampling rate:", FS)
    print("Window duration:", WINDOW / FS, "seconds")
    print("Step:", STEP, "samples =", STEP / FS, "seconds")
    print("Subjects:", len(np.unique(subject_ids)))
    print("Conditions:", sorted(np.unique(conditions).tolist()))
    print("Tasks:", sorted(np.unique(tasks).tolist()))
    print("Wrists:", sorted(np.unique(wrists).tolist()))
    print("Saved:", OUTPUT)


if __name__ == "__main__":
    main()
