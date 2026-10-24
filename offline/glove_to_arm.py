import serial
import time
import signal
import os
from collections import deque

GLOVE_PORT = "/dev/ttyUSB0"
ARM_PORT = "/dev/ttyUSB1"
BAUD = 115200

BEND_THRESHOLDS = {
    "thumb": 236.5,
    "index": 236.5,
    "middle": 236.5,
    "ring": 236.5,
}

STRAIGHT_ANGLE = 50
BENT_ANGLE = 120
WINDOW_SIZE = 5
STABLE_SAMPLES = 4

FINGER_JOINTS = {
    "thumb":  (2, 2, "SHOULDER"),
    "index":  (1, 1, "WAIST"),
    "middle": (3, 3, "ELBOW"),
    "ring":   (6, 6, "GRIPPER"),
}

MOTOR_SEQUENCE = [
    (1, "WAIST", True),
    (2, "SHOULDER", True),
    (3, "ELBOW", True),
    (4, "WRIST ROLL", False),
    (5, "WRIST PITCH", False),
    (6, "GRIPPER", True),
]

running = True


def stop_handler(signum, frame):
    global running
    running = False


signal.signal(signal.SIGINT, stop_handler)
signal.signal(signal.SIGTERM, stop_handler)


def read_arm_responses(arm):
    while arm.in_waiting:
        arm.readline()


def send_command(arm, command, delay=0.15):
    arm.write((command + "\n").encode("ascii"))
    arm.flush()
    time.sleep(delay)
    read_arm_responses(arm)


def parse_glove(line):
    parts = line.strip().split(",")

    if len(parts) != 5:
        return None

    try:
        values = [int(value) for value in parts]
    except ValueError:
        return None

    if not all(0 <= value <= 1023 for value in values[1:]):
        return None

    return {
        "thumb": values[1],
        "index": values[2],
        "middle": values[3],
        "ring": values[4],
    }


def show_initialization(number, name, message):
    print(f"  [{number}/6] {name:<12} {message}", flush=True)


def draw_dashboard(readings, filtered_values, states, last_action, sample_count):
    os.system("clear")

    print("=" * 66)
    print("                    TREMORGATE LIVE DASHBOARD")
    print("=" * 66)
    print("Glove: CONNECTED   |   Arm: CONNECTED   |   Control: ACTIVE")
    print("-" * 66)
    print(f"{'FINGER':<12}{'RAW ADC':>12}{'FILTERED':>14}{'CLASSIFICATION':>22}")
    print("-" * 66)

    joint_names = {
        "thumb": "SHOULDER",
        "index": "WAIST",
        "middle": "ELBOW",
        "ring": "GRIPPER",
    }

    for finger in FINGER_JOINTS:
        raw = readings.get(finger, "--")
        filtered = filtered_values.get(finger, "--")
        state = states[finger]

        print(
            f"{finger.upper():<12}"
            f"{str(raw):>12}"
            f"{str(filtered):>14}"
            f"{state:>15}  "
            f""
        )
        print(f"  Motor: {joint_names[finger]}")

    print("-" * 66)
    print(f"Last motor update: {last_action}")
    print(f"Valid glove samples: {sample_count}")
    print("-" * 66)
    print("BEND a finger to command its mapped motor.")
    print("Press Ctrl+C to stop safely.")
    print("=" * 66, flush=True)


def main():
    glove = None
    arm = None

    try:
        print("=" * 66)
        print("TremorGate: Four-Finger to Four-Joint Control")
        print("=" * 66)
        print("Thumb  -> Shoulder -> Joint 2 -> D9")
        print("Index  -> Waist    -> Joint 1 -> D10")
        print("Middle -> Elbow    -> Joint 3 -> D8")
        print("Ring   -> Gripper  -> Joint 6 -> D5")
        print(f"Straight target: {STRAIGHT_ANGLE} degrees")
        print(f"Bent target:     {BENT_ANGLE} degrees")
        print("Wrist roll and wrist pitch will remain detached.")
        print("\nKeep external servo power OFF during preparation.")

        input("\nPress ENTER to connect to both Arduinos... ")

        arm = serial.Serial(ARM_PORT, BAUD, timeout=0.2)
        time.sleep(2)
        arm.reset_input_buffer()

        glove = serial.Serial(GLOVE_PORT, BAUD, timeout=1)
        time.sleep(2)
        glove.reset_input_buffer()

        send_command(arm, "STATUS")

        print("\nSAFETY CHECK:")
        print("- Ensure the arm can move freely.")
        print("- Support the arm so it cannot fall unexpectedly.")
        print("- Keep hands clear of moving joints.")
        print("- Only the selected servo is attached by each TEST command.")

        input(
            "\nWhen ready, turn ON the external servo supply, "
            "then press ENTER to initialize motors. "
            "Press Ctrl+C to cancel: "
        )

        print("\n" + "=" * 66)
        print("                    INITIALIZING MOTORS")
        print("=" * 66)

        for number, (joint, name, enabled) in enumerate(MOTOR_SEQUENCE, start=1):
            if not running:
                break

            if enabled:
                show_initialization(
                    number, name, f"Moving to {STRAIGHT_ANGLE} degrees..."
                )
                send_command(
                    arm, f"TEST {joint} {STRAIGHT_ANGLE}", delay=0.4
                )
                show_initialization(number, name, "INITIALIZED")
            else:
                show_initialization(number, name, "SKIPPED - remains detached")

        if not running:
            return

        print("\nAll six initialization steps completed.")
        print("Starting live glove readings and classifications...")
        time.sleep(2)

        samples = {
            finger: deque(maxlen=WINDOW_SIZE)
            for finger in FINGER_JOINTS
        }

        candidate_state = {
            finger: None for finger in FINGER_JOINTS
        }

        stable_count = {
            finger: 0 for finger in FINGER_JOINTS
        }

        commanded_state = {
            finger: "STRAIGHT" for finger in FINGER_JOINTS
        }

        filtered_values = {}
        last_action = "None"
        last_print = 0.0
        sample_count = 0
        readings = {}

        while running:
            line = glove.readline().decode(
                "utf-8", errors="replace"
            ).strip()

            current_readings = parse_glove(line)
            if current_readings is None:
                continue

            readings = current_readings
            sample_count += 1

            for finger, (joint, _, joint_name) in FINGER_JOINTS.items():
                raw = readings[finger]
                samples[finger].append(raw)

                if len(samples[finger]) < WINDOW_SIZE:
                    continue

                ordered = sorted(samples[finger])
                filtered = ordered[len(ordered) // 2]
                filtered_values[finger] = filtered
                threshold = BEND_THRESHOLDS[finger]

                if commanded_state[finger] == "STRAIGHT":
                    detected = "BENT" if filtered < threshold else "STRAIGHT"
                else:
                    detected = (
                        "STRAIGHT"
                        if filtered > threshold + 5
                        else "BENT"
                    )

                if detected == candidate_state[finger]:
                    stable_count[finger] += 1
                else:
                    candidate_state[finger] = detected
                    stable_count[finger] = 1

                if (
                    stable_count[finger] >= STABLE_SAMPLES
                    and detected != commanded_state[finger]
                ):
                    commanded_state[finger] = detected
                    angle = (
                        STRAIGHT_ANGLE if detected == "BENT"
                        else BENT_ANGLE
                    ) if finger == "ring" else (
                        BENT_ANGLE if detected == "BENT"
                        else STRAIGHT_ANGLE
                    )

                    send_command(arm, f"SET {joint} {angle}")
                    last_action = (
                        f"{finger.upper()} -> {joint_name}: "
                        f"{detected}, {angle} degrees"
                    )

            now = time.monotonic()
            if now - last_print >= 0.5:
                draw_dashboard(
                    readings,
                    filtered_values,
                    commanded_state,
                    last_action,
                    sample_count,
                )
                last_print = now

    except KeyboardInterrupt:
        print("\nStopping bridge...")

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

        print("\nSerial connections closed.")
        print("Switch OFF external servo power before adjusting the arm.")


if __name__ == "__main__":
    main()
