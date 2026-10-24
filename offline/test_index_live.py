import serial
import time
from collections import deque

PORT = "/dev/ttyUSB0"
BAUD = 115200

print("=" * 55)
print("TremorGate - Live Index Flex Sensor Test")
print("=" * 55)
print("Port:", PORT)
print("Index sensor: CSV column 3 (parts[2])")
print("\nKeep your hand relaxed initially.")
print("Watch the index reading while slowly bending your index finger.")
print("Press Ctrl+C to stop.\n")

try:
    with serial.Serial(PORT, BAUD, timeout=1) as ser:
        time.sleep(2)
        ser.reset_input_buffer()

        recent_index = deque(maxlen=50)
        sample_count = 0

        while True:
            line = ser.readline().decode(
                "utf-8", errors="replace"
            ).strip()

            parts = line.split(",")

            if len(parts) != 6:
                continue

            try:
                timestamp = int(parts[0])
                thumb = int(parts[1])
                index = int(parts[2])
                middle = int(parts[3])
                ring = int(parts[4])
                little = int(parts[5])
            except ValueError:
                continue

            if not 0 <= index <= 1023:
                continue

            recent_index.append(index)
            sample_count += 1

            if sample_count % 10 == 0:
                print(
                    f"Index: {index:4d} | "
                    f"Last-second range: "
                    f"{min(recent_index):3d}-{max(recent_index):3d} | "
                    f"All fingers T/I/M/R/L: "
                    f"{thumb}/{index}/{middle}/{ring}/{little}",
                    flush=True
                )

except KeyboardInterrupt:
    print("\nLive test stopped.")

except serial.SerialException as exc:
    print("\nSerial connection error:", exc)
    print("Close any other program using /dev/ttyUSB0.")
