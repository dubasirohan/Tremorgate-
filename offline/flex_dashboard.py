import csv
import time
import threading
import tkinter as tk
from tkinter import ttk, messagebox
from datetime import datetime
from pathlib import Path
from collections import deque

import numpy as np
import serial
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

GLOVE_PORT = "/dev/ttyUSB0"
BAUD = 115200

SENSORS = ["Thumb", "Index", "Middle", "Ring"]
FOLDER = Path.home() / "tremorgate_ws" / "offline"
CSV_PATH = FOLDER / "flex_dashboard_data_4finger.csv"

SAMPLE_RATE = 50
MIN_TREMOR_HZ = 3.0
MAX_TREMOR_HZ = 10.0
WINDOW_SECONDS = 2.0
WINDOW_SIZE = int(SAMPLE_RATE * WINDOW_SECONDS)
MIN_AMPLITUDE = 3.0

class FlexDashboard:
    def __init__(self, root):
        self.root = root
        self.root.title("TremorGate | Four-Finger Dashboard")
        self.root.geometry("1150x850")
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        self.ser = None
        self.running = False
        self.lock = threading.Lock()

        self.baseline = None
        self.latest = [0] * 4
        self.deltas = [0] * 4
        self.history = [deque(maxlen=100) for _ in SENSORS]
        self.signal_history = [
            deque(maxlen=WINDOW_SIZE) for _ in SENSORS
        ]

        self.sample_count = 0
        self.csv_file = None
        self.csv_writer = None

        self.status_text = "Disconnected"
        self.classification = "WAITING FOR DATA"
        self.frequencies = [None] * 4


        ttk.Label(
            root,
            text="TremorGate: Four-Finger Monitoring",
            font=("Arial", 18, "bold"),
        ).pack(pady=8)

        self.status = ttk.Label(
            root,
            text=self.status_text,
            font=("Arial", 12, "bold"),
        )
        self.status.pack(pady=4)

        self.classification_label = ttk.Label(
            root,
            text="Classification: WAITING FOR DATA",
            font=("Arial", 14, "bold"),
        )
        self.classification_label.pack(pady=4)

        self.frequency_label = ttk.Label(
            root,
            text="Finger frequencies: collecting samples...",
        )
        self.frequency_label.pack(pady=3)

        panel = ttk.Frame(root)
        panel.pack(fill="x", padx=10, pady=8)

        self.reading_labels = []
        self.delta_labels = []

        for i, name in enumerate(SENSORS):
            card = ttk.LabelFrame(panel, text=name, padding=8)
            card.grid(row=0, column=i, padx=4, sticky="nsew")
            panel.columnconfigure(i, weight=1)

            reading = ttk.Label(
                card, text="---", font=("Arial", 16, "bold")
            )
            reading.pack(pady=3)

            delta = ttk.Label(card, text="Delta: ---")
            delta.pack(pady=3)

            self.reading_labels.append(reading)
            self.delta_labels.append(delta)

        self.figure, self.ax = plt.subplots(figsize=(9, 4))
        self.lines = []

        for name in SENSORS:
            line, = self.ax.plot([], [], label=name)
            self.lines.append(line)

        self.ax.set_xlabel("Recent samples")
        self.ax.set_ylabel("Raw ADC reading")
        self.ax.set_title("Live Flex Sensor Readings")
        self.ax.grid(True)
        self.ax.legend(loc="upper right")
        self.figure.tight_layout()

        self.canvas = FigureCanvasTkAgg(self.figure, master=root)
        self.canvas.get_tk_widget().pack(
            fill="both", expand=True, padx=10, pady=5
        )

        controls = ttk.Frame(root)
        controls.pack(pady=8)

        self.start_button = ttk.Button(
            controls,
            text="Start Monitoring",
            command=self.start,
        )
        self.start_button.pack(side="left", padx=5)

        self.stop_button = ttk.Button(
            controls,
            text="Stop Monitoring",
            command=self.stop,
            state="disabled",
        )
        self.stop_button.pack(side="left", padx=5)

        self.count_label = ttk.Label(root, text="Samples: 0")
        self.count_label.pack(pady=4)

    def classify_signal(self, signal):
        if len(signal) < WINDOW_SIZE:
            return "UNCERTAIN", None, 0.0

        x = np.asarray(signal, dtype=float)
        x = x - np.mean(x)

        t = np.arange(len(x), dtype=float)
        slope, intercept = np.polyfit(t, x, 1)
        x = x - (slope * t + intercept)

        amplitude = float(np.sqrt(np.mean(x ** 2)))

        if amplitude < MIN_AMPLITUDE:
            return "STABLE", None, amplitude

        window = np.hanning(len(x))
        spectrum = np.abs(np.fft.rfft(x * window))
        frequencies = np.fft.rfftfreq(
            len(x), d=1.0 / SAMPLE_RATE
        )

        band = (
            (frequencies >= MIN_TREMOR_HZ)
            & (frequencies <= MAX_TREMOR_HZ)
        )

        if not np.any(band):
            return "UNCERTAIN", None, amplitude

        band_indices = np.where(band)[0]
        peak_index = band_indices[np.argmax(spectrum[band])]
        peak_frequency = float(frequencies[peak_index])

        total_energy = float(np.sum(spectrum[1:] ** 2))
        band_energy = float(np.sum(spectrum[band] ** 2))

        if total_energy <= 0:
            return "STABLE", peak_frequency, amplitude

        if band_energy / total_energy >= 0.60:
            return "POSSIBLE TREMOR", peak_frequency, amplitude

        return "UNCERTAIN", peak_frequency, amplitude

    def start(self):
        if self.running:
            return

        try:
            self.ser = serial.Serial(
                GLOVE_PORT, BAUD, timeout=1
            )
            time.sleep(2)
            self.ser.reset_input_buffer()
        except Exception as exc:
            messagebox.showerror(
                "Glove connection error",
                f"Could not open {GLOVE_PORT}.\n\n{exc}\n\n"
                "Close other programs using the glove Arduino.",
            )
            self.ser = None
            return

        FOLDER.mkdir(parents=True, exist_ok=True)
        new_file = (
            not CSV_PATH.exists()
            or CSV_PATH.stat().st_size == 0
        )

        self.csv_file = open(CSV_PATH, "a", newline="")
        self.csv_writer = csv.writer(self.csv_file)

        if new_file:
            self.csv_writer.writerow([
                "timestamp",
                "arduino_ms",
                "thumb_raw",
                "index_raw",
                "middle_raw",
                "ring_raw",
                "thumb_delta",
                "index_delta",
                "middle_delta",
                "ring_delta",
                "classification",
                "thumb_hz",
                "index_hz",
                "middle_hz",
                "ring_hz",
            ])
            self.csv_file.flush()

        self.baseline = None
        self.latest = [0] * 4
        self.deltas = [0] * 4
        self.sample_count = 0
        self.frequencies = [None] * 4
        self.classification = "COLLECTING BASELINE"
        self.status_text = "Connected — collecting baseline..."

        for history in self.history:
            history.clear()
        for history in self.signal_history:
            history.clear()

        self.running = True
        self.start_button.config(state="disabled")
        self.stop_button.config(state="normal")

        threading.Thread(
            target=self.reader, daemon=True
        ).start()

        self.root.after(200, self.refresh)

    def reader(self):
        while self.running:
            try:
                line = (
                    self.ser.readline()
                    .decode("utf-8", errors="ignore")
                    .strip()
                )

                if not line:
                    continue

                parts = line.split(",")

                if len(parts) not in (5, 6):
                    continue

                try:
                    arduino_ms = int(float(parts[0]))
                    values = [
                        int(float(value))
                        for value in parts[1:5]
                    ]
                except ValueError:
                    continue

                if self.baseline is None:
                    self.baseline = values[:]

                deltas = [
                    values[i] - self.baseline[i]
                    for i in range(4)
                ]

                with self.lock:
                    self.latest = values[:]
                    self.deltas = deltas[:]

                    for i in range(4):
                        self.history[i].append(values[i])
                        self.signal_history[i].append(
                            float(deltas[i])
                        )

                    self.sample_count += 1

                    results = [
                        self.classify_signal(signal)
                        for signal in self.signal_history
                    ]

                    self.frequencies = [
                        result[1] for result in results
                    ]

                    labels = [result[0] for result in results]

                    if "POSSIBLE TREMOR" in labels:
                        self.classification = "POSSIBLE TREMOR"
                    elif "UNCERTAIN" in labels:
                        self.classification = "UNCERTAIN"
                    else:
                        self.classification = "STABLE"

                    self.status_text = "Connected"

                    frequency_values = [
                        "" if f is None else round(f, 2)
                        for f in self.frequencies
                    ]

                    self.csv_writer.writerow([
                        datetime.now().isoformat(
                            timespec="milliseconds"
                        ),
                        arduino_ms,
                        *values,
                        *deltas,
                        self.classification,
                        *frequency_values,
                    ])

                    if self.sample_count % 10 == 0:
                        self.csv_file.flush()


            except (serial.SerialException, OSError) as exc:
                self.status_text = f"Glove serial error: {exc}"
                self.running = False
                break

            except Exception as exc:
                self.status_text = f"Processing error: {exc}"
                time.sleep(0.05)

    def refresh(self):
        with self.lock:
            for i in range(4):
                self.reading_labels[i].config(
                    text=str(self.latest[i])
                )
                self.delta_labels[i].config(
                    text=f"Delta: {self.deltas[i]:+d}"
                )

                data = list(self.history[i])
                self.lines[i].set_data(range(len(data)), data)

            max_len = max(
                (len(history) for history in self.history),
                default=0,
            )
            self.ax.set_xlim(0, max(20, max_len))

            all_values = [
                value
                for history in self.history
                for value in history
            ]

            if all_values:
                low = min(all_values)
                high = max(all_values)
                margin = max(5, (high - low) * 0.15)
                self.ax.set_ylim(low - margin, high + margin)

            self.canvas.draw_idle()
            self.status.config(text=self.status_text)
            self.classification_label.config(
                text=f"Classification: {self.classification}"
            )

            frequency_text = []
            for name, frequency in zip(SENSORS, self.frequencies):
                if frequency is None:
                    frequency_text.append(f"{name}: collecting")
                else:
                    frequency_text.append(
                        f"{name}: {frequency:.1f} Hz"
                    )

            self.frequency_label.config(
                text=" | ".join(frequency_text)
            )
            self.count_label.config(
                text=f"Samples: {self.sample_count}"
            )

        if self.running:
            self.root.after(200, self.refresh)

    def stop(self):
        self.running = False
        if self.ser is not None:
            try:
                self.ser.close()
            except Exception:
                pass
            self.ser = None

        if self.csv_file is not None:
            try:
                self.csv_file.flush()
                self.csv_file.close()
            except Exception:
                pass
            self.csv_file = None

        self.start_button.config(state="normal")
        self.stop_button.config(state="disabled")
        self.status.config(text=f"Stopped — CSV: {CSV_PATH}")

    def close(self):
        self.stop()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = FlexDashboard(root)
    root.mainloop()
