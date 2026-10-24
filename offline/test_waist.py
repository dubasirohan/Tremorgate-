import serial
import time
import signal
from collections import deque

GLOVE_PORT = "/dev/ttyUSB0"
ARM_PORT = "/dev/ttyUSB1"
BAUD = 115200

# Existing index-finger calibration
BEND_THRESHOLD = 236.5

STRAIGHT_ANGLE = 60
BENT_ANGLE = 110

WINDOW_SIZE = 5
STABLE_SAMPLES = 4

running = True


def stop_handler(signum, frame):
    global running
    running = False


signal.signal(signal.SIGINT, stop_handler)
signal.signal(signal.SIGTERM, stop_handler)


def read_arm_responses(arm):
    while arm.in_waiting:
        line = arm.readline().decode(
            "utf-8", errors="replace"
        ).strip()

        if line:
            print(f"[ARM] {line}")


def send_command(arm, command, delay=0.15):
    print(f"[TX ARM] {command}")

    arm.write((command + "\n").encode("ascii"))
    arm.flush()

    time.sleep(delay)
    read_arm_responses(arm)


def parse_thumb(line):
    parts = line.strip().split(",")

    if len(parts) != 6:
        return None

    try:
        values = [int(value) for value in parts]
    except ValueError:
        return None

    index = values[1]

    if 0 <= index <= 1023:
        return index

    return None


def main():
    glove = None
    arm = None

    try:
        print("=" * 55)
        print("TremorGate: THUMB FINGER -> WAIST TEST")
        print("=" * 55)
        print("Glove:       /dev/ttyUSB0")
        print("Arm:         /dev/ttyUSB1")
        print("Joint:       1")
        print("Servo pin:   D10")
        print(f"Straight:    {STRAIGHT_ANGLE} degrees")
        print(f"Bent:        {BENT_ANGLE} degrees")
        print(f"Threshold:   {BEND_THRESHOLD}")
        print()
        print("Only the shoulder servo will be attached.")
        print("External servo power must remain OFF during preparation.")

        input("\nPress ENTER to connect to both Arduinos... ")

        arm = serial.Serial(ARM_PORT, BAUD, timeout=0.2)
        time.sleep(2)
        arm.reset_input_buffer()

        glove = serial.Serial(GLOVE_PORT, BAUD, timeout=1)
        time.sleep(2)
        glove.reset_input_buffer()

        print("\nArm controller status:")
        send_command(arm, "STATUS")

        print("\nSAFETY CHECK:")
        print("- Ensure the shoulder can move freely.")
        print("- Keep hands clear of the arm.")
        print("- Support the arm if necessary.")
        print("- Other servo signals will remain detached.")

        input(
            "\nWhen ready, turn ON the external servo supply, "
            "then press ENTER to initialise the shoulder at 60 degrees. "
            "Press Ctrl+C to cancel: "
        )

        # TEST 2 attaches the shoulder while preserving the controller's
        # individual-servo testing behaviour for this isolated test.
        send_command(arm, f"TEST 1 {STRAIGHT_ANGLE}", delay=0.5)

        print("\nShoulder initialisation command sent.")
        print("Bend thumb -> waist 110 degrees.")
        print("Straighten thumb -> waist 60 degrees.")
        print("Press Ctrl+C to stop.\n")

        samples = deque(maxlen=WINDOW_SIZE)

        candidate_state = None
        stable_count = 0
        commanded_state = "STRAIGHT"

        last_print = 0.0

        while running:
            line = glove.readline().decode(
                "utf-8", errors="replace"
            ).strip()

            index = parse_thumb(line)

            if index is None:
                continue

            samples.append(index)

            if len(samples) < WINDOW_SIZE:
                continue

            ordered = sorted(samples)
            filtered = ordered[len(ordered) // 2]

            # Existing index-finger calibration and hysteresis.
            if commanded_state == "STRAIGHT":
                detected_state = (
                    "BENT"
                    if filtered < BEND_THRESHOLD
                    else "STRAIGHT"
                )
            else:
                detected_state = (
                    "STRAIGHT"
                    if filtered > BEND_THRESHOLD + 5
                    else "BENT"
                )

            if detected_state == candidate_state:
                stable_count += 1
            else:
                candidate_state = detected_state
                stable_count = 1

            if (
                stable_count >= STABLE_SAMPLES
                and detected_state != commanded_state
            ):
                commanded_state = detected_state

                if commanded_state == "BENT":
                    angle = BENT_ANGLE
                else:
                    angle = STRAIGHT_ANGLE

                send_command(arm, f"SET 1 {angle}")

                print(
                    f"THUMB {commanded_state} -> "
                    f"WAIST {angle} DEGREES"
                )

            now = time.monotonic()

            if now - last_print >= 0.5:
                print(
                    f"Thumb raw={index:3d} | "
                    f"median={filtered:3d} | "
                    f"state={commanded_state}",
                    flush=True,
                )

                last_print = now

    except KeyboardInterrupt:
        print("\nStopping shoulder test...")

    except serial.SerialException as exc:
        print(f"\nSerial error: {exc}")
        print("Check USB ports and close other serial-monitor programs.")

    finally:
        if arm is not None and arm.is_open:
            try:
                arm.write(b"STOP\n")
                arm.flush()
                time.sleep(0.2)

                arm.write(b"DISARM\n")
                arm.flush()
                time.sleep(0.2)

                read_arm_responses(arm)

            except Exception as exc:
                print(f"Shutdown command error: {exc}")

        if glove is not None and glove.is_open:
            glove.close()

        if arm is not None and arm.is_open:
            arm.close()

        print("Serial connections closed.")
        print("Switch OFF external servo power before adjusting the arm.")


if __name__ == "__main__":
    main()
