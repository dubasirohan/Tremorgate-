#!/usr/bin/env python3

import time
import statistics
from collections import deque

import serial

# ============================================================
# TremorGate: Ring Finger -> Gripper
# Based on the test_shoulder.py / test_elbow.py control pattern
# ============================================================

GLOVE_PORT = "/dev/ttyUSB0"
ARM_PORT = "/dev/ttyUSB1"
BAUD = 115200

RING_THRESHOLD = 236.5
HYSTERESIS = 5
FILTER_SIZE = 5
STABLE_SAMPLES = 4

GRIPPER_JOINT = 6
GRIPPER_OPEN = 60
GRIPPER_CLOSED = 110

# Glove CSV:
# time_ms,thumb,index,middle,ring,little
RING_FIELD = 4


def read_ring(glove):
    """Read one valid ring-finger ADC value."""
    while True:
        line = glove.readline().decode(
            "utf-8", errors="ignore"
        ).strip()

        if not line:
            return None

        parts = line.split(",")

        if len(parts) != 6:
            continue

        try:
            values = [float(value) for value in parts]
        except ValueError:
            continue

        ring_value = values[RING_FIELD]

        if 0 <= ring_value <= 1023:
            return ring_value

    return None


def send_command(arm, command, timeout=0.15):
    """Send one newline-terminated command and briefly read responses."""
    print(f"\\n[TX] {command}", flush=True)
    arm.write((command + "\n").encode("utf-8"))
    arm.flush()

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = arm.readline().decode(
            "utf-8", errors="ignore"
        ).strip()
        if response:
            print("[ARM]", response, flush=True)


def main():
    glove = None
    arm = None

    try:
        print("=" * 50)
        print("TremorGate: Ring Finger -> SG90 Gripper")
        print("=" * 50)

        print(f"\nConnecting to glove: {GLOVE_PORT}")
        glove = serial.Serial(
            GLOVE_PORT, BAUD, timeout=1
        )
        time.sleep(2)
        glove.reset_input_buffer()

        print(f"Connecting to arm: {ARM_PORT}")
        arm = serial.Serial(
            ARM_PORT, BAUD, timeout=1
        )
        time.sleep(2)
        arm.reset_input_buffer()

        # Attach joint 6 and set its initial position.
        send_command(
            arm, f"TEST {GRIPPER_JOINT} {GRIPPER_OPEN}"
        )

        send_command(
            arm, f"SET {GRIPPER_JOINT} {GRIPPER_OPEN}"
        )

        print("\nControl mapping:")
        print("Ring finger bent     -> Gripper CLOSE (110 degrees)")
        print("Ring finger straight -> Gripper OPEN  (60 degrees)")
        print(f"Threshold: {RING_THRESHOLD}")
        print(f"Hysteresis: +/-{HYSTERESIS}")
        print(f"Stable samples: {STABLE_SAMPLES}")
        print("\nPress CTRL+C to stop.\n")

        history = deque(maxlen=FILTER_SIZE)

        # Begin in the open state.
        is_bent = False
        candidate_state = None
        candidate_count = 0
        last_print = 0

        while True:
            raw = read_ring(glove)

            if raw is None:
                continue

            history.append(raw)

            if len(history) < FILTER_SIZE:
                continue

            filtered = statistics.median(history)

            # Match the earlier finger scripts:
            # lower ADC reading indicates a bent finger.
            if not is_bent:
                if filtered < RING_THRESHOLD - HYSTERESIS:
                    detected_state = True
                else:
                    detected_state = False
            else:
                if filtered > RING_THRESHOLD + HYSTERESIS:
                    detected_state = False
                else:
                    detected_state = True

            if detected_state == is_bent:
                candidate_state = None
                candidate_count = 0
            else:
                if candidate_state == detected_state:
                    candidate_count += 1
                else:
                    candidate_state = detected_state
                    candidate_count = 1

                if candidate_count >= STABLE_SAMPLES:
                    is_bent = detected_state
                    candidate_state = None
                    candidate_count = 0

                    if is_bent:
                        print(
                            f"\nRING BENT | ADC={filtered:.1f}"
                            " -> GRIPPER CLOSE"
                        )
                        send_command(
                            arm,
                            f"SET {GRIPPER_JOINT} "
                            f"{GRIPPER_CLOSED}"
                        )
                    else:
                        print(
                            f"\nRING STRAIGHT | ADC={filtered:.1f}"
                            " -> GRIPPER OPEN"
                        )
                        send_command(
                            arm,
                            f"SET {GRIPPER_JOINT} "
                            f"{GRIPPER_OPEN}"
                        )

            # Periodic sensor output helps diagnose the threshold.
            now = time.monotonic()

            if now - last_print >= 0.5:
                state = "BENT" if is_bent else "STRAIGHT"

                print(
                    f"Ring ADC: {filtered:.1f} | "
                    f"State: {state}",
                    flush=True
                )

                last_print = now

    except KeyboardInterrupt:
        print("\nStopping ring-finger control...")

    except (serial.SerialException, OSError) as exc:
        print(f"\nSerial error: {exc}")

    finally:
        if arm is not None and arm.is_open:
            try:
                arm.write(b"STOP\n")
                arm.flush()
                time.sleep(0.2)

                arm.write(b"DISARM\n")
                arm.flush()
                time.sleep(0.2)
            except serial.SerialException:
                pass

            arm.close()

        if glove is not None and glove.is_open:
            glove.close()

        print("Serial connections closed.")


if __name__ == "__main__":
    main()
