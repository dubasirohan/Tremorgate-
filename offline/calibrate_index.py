import serial
import time
import statistics

PORT = "/dev/ttyUSB0"
BAUD = 115200
SAMPLE_COUNT = 100

def collect_samples(ser, label):
    input(
        f"\nPosition your index finger {label} "
        "and press ENTER when ready..."
    )

    print(f"Collecting {label} readings for approximately 2 seconds...")

    ser.reset_input_buffer()
    values = []
    deadline = time.monotonic() + 5

    while len(values) < SAMPLE_COUNT and time.monotonic() < deadline:
        line = ser.readline().decode(
            "utf-8", errors="replace"
        ).strip()

        parts = line.split(",")

        if len(parts) != 6:
            continue

        try:
            value = int(parts[2])
        except ValueError:
            continue

        if 0 <= value <= 1023:
            values.append(value)

    if len(values) < 20:
        raise RuntimeError(
            f"Only collected {len(values)} samples. Check serial data."
        )

    print(f"\n{label.capitalize()} position:")
    print(f"  Samples: {len(values)}")
    print(f"  Mean:    {statistics.mean(values):.1f}")
    print(f"  Median:  {statistics.median(values):.1f}")
    print(f"  Min:     {min(values)}")
    print(f"  Max:     {max(values)}")

    return values

print("TremorGate - Index Finger Calibration")
print("Port:", PORT)
print("No robotic-arm commands will be sent.")

try:
    with serial.Serial(PORT, BAUD, timeout=0.2) as ser:
        time.sleep(2)
        ser.reset_input_buffer()

        straight = collect_samples(ser, "straight")
        bent = collect_samples(ser, "bent")

        straight_median = statistics.median(straight)
        bent_median = statistics.median(bent)
        difference = bent_median - straight_median

        print("\n--- Calibration summary ---")
        print(f"Straight median: {straight_median:.1f}")
        print(f"Bent median:     {bent_median:.1f}")
        print(f"Difference:      {difference:+.1f}")

        if abs(difference) < 20:
            print("\nWARNING: Insufficient separation.")
            print("Repeat the test while holding each position still.")
        else:
            threshold = (straight_median + bent_median) / 2
            print(f"\nSuggested threshold: {threshold:.1f}")
            print("Direction:", "BEND when reading falls below threshold"
                  if difference < 0 else
                  "BEND when reading rises above threshold")
            print("\nCalibration appears usable for initial testing.")

except (serial.SerialException, RuntimeError) as exc:
    print("\nError:", exc)
    print("Close other programs using /dev/ttyUSB0 and retry.")
