import csv
import os
import sys
import time
from datetime import datetime

import serial

PORT = "/dev/ttyUSB0"
BAUD = 115200
THRESHOLD = 8
BASELINE_SECONDS = 3

labels = ["thumb", "index", "middle", "ring", "little"]
output = os.path.expanduser(
    "~/tremorgate_ws/offline/flex_sensor_data.csv"
)

def main():
    print("\nTREMORGATE - FIVE-FINGER MONITOR")
    print("=" * 55)
    print("Connecting to Arduino...")

    try:
        ser = serial.Serial(PORT, BAUD, timeout=2)
    except Exception as exc:
        print(f"Could not open {PORT}: {exc}")
        print("Check the USB connection and close other serial monitors.")
        sys.exit(1)

    time.sleep(2)
    ser.reset_input_buffer()

    print("Keep your hand relaxed and fingers still.")
    print(f"Calibrating baseline for {BASELINE_SECONDS} seconds...\n")

    calibration = []
    start = time.time()

    try:
        while time.time() - start < BASELINE_SECONDS:
            line = ser.readline().decode("utf-8", errors="replace").strip()
            parts = line.split(",")

            if len(parts) == 6 and parts[0].isdigit():
                try:
                    calibration.append([int(x) for x in parts[1:]])
                except ValueError:
                    continue

        if len(calibration) < 3:
            print("ERROR: Not enough valid sensor readings.")
            print("Check the Arduino sketch and serial connection.")
            return

        baseline = [
            sum(row[i] for row in calibration) / len(calibration)
            for i in range(5)
        ]

        print("Baseline established:")
        for name, value in zip(labels, baseline):
            print(f"  {name:8s}: {value:.1f}")

        print("\nLive readings | Change from baseline")
        print("-" * 70)
        print("Press Ctrl+C to stop and save the CSV.\n")

        os.makedirs(os.path.dirname(output), exist_ok=True)
        new_file = not os.path.exists(output) or os.path.getsize(output) == 0

        with open(output, "a", newline="") as file:
            writer = csv.writer(file)

            if new_file:
                writer.writerow([
                    "timestamp", "arduino_ms",
                    *[f"{name}_raw" for name in labels],
                    *[f"{name}_delta" for name in labels],
                    "movement_status"
                ])

            while True:
                line = ser.readline().decode(
                    "utf-8", errors="replace"
                ).strip()
                parts = line.split(",")

                if len(parts) != 6:
                    continue

                try:
                    arduino_ms = int(parts[0])
                    values = [int(x) for x in parts[1:]]
                except ValueError:
                    continue

                deltas = [
                    round(values[i] - baseline[i], 1)
                    for i in range(5)
                ]

                moving = any(abs(d) >= THRESHOLD for d in deltas)
                status = "CHANGE DETECTED" if moving else "STABLE"

                print(
                    " | ".join(
                        f"{labels[i]} {values[i]:4d} ({deltas[i]:+5.1f})"
                        for i in range(5)
                    )
                    + f"  --> {status}"
                )

                writer.writerow([
                    datetime.now().isoformat(timespec="milliseconds"),
                    arduino_ms,
                    *values,
                    *deltas,
                    status
                ])
                file.flush()

    except KeyboardInterrupt:
        print("\nMonitoring stopped.")
    finally:
        ser.close()
        print(f"Sensor data saved to: {output}")

if __name__ == "__main__":
    main()
