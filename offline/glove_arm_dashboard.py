import tkinter as tk
from tkinter import ttk, messagebox
import serial
import time
import threading
from collections import deque
from datetime import datetime
import csv
import os
import queue

GLOVE_PORT = "/dev/ttyUSB0"
ARM_PORT = "/dev/ttyUSB1"
BAUD = 115200

STRAIGHT_ANGLE = 50
BENT_ANGLE = 120
WINDOW_SIZE = 5
STABLE_SAMPLES = 4

FINGERS = {
    "thumb":  {"joint": 2, "motor": "Shoulder", "pin": "D9", "threshold": 236.5},
    "index":  {"joint": 1, "motor": "Waist",    "pin": "D10", "threshold": 236.5},
    "middle": {"joint": 3, "motor": "Elbow",    "pin": "D8", "threshold": 236.5},
    "ring":   {"joint": 6, "motor": "Gripper",  "pin": "D5", "threshold": 236.5},
}

class GloveArmDashboard:
    def __init__(self, root):
        self.root = root
        self.root.title("TremorGate | Glove-to-Arm Dashboard")
        self.root.geometry("1050x720")
        self.root.minsize(850, 620)

        self.glove = None
        self.arm = None
        self.connected = False
        self.initialized = False
        self.closing = False
        self.lock = threading.Lock()
        self.glove_queue = queue.Queue(maxsize=500)
        self.reader_thread = None
        self.reader_stop = threading.Event()

        self.samples = {f: deque(maxlen=WINDOW_SIZE) for f in FINGERS}
        self.candidates = {f: None for f in FINGERS}
        self.stable_counts = {f: 0 for f in FINGERS}
        self.states = {f: "UNKNOWN" for f in FINGERS}
        self.raw = {f: "--" for f in FINGERS}
        self.filtered = {f: "--" for f in FINGERS}
        self.angles = {f: "--" for f in FINGERS}
        self.last_action = "No movement yet"
        self.sample_count = 0
        self.log_file = os.path.expanduser(
            "~/tremorgate_ws/offline/glove_arm_dashboard.csv"
        )

        self.build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.after(100, self.update_ui)

    def build_ui(self):
        outer = ttk.Frame(self.root, padding=14)
        outer.pack(fill="both", expand=True)

        ttk.Label(
            outer, text="TREMORGATE", font=("TkDefaultFont", 22, "bold")
        ).pack(anchor="w")
        ttk.Label(
            outer,
            text="Live Wearable Input  →  Motor Control",
            font=("TkDefaultFont", 11)
        ).pack(anchor="w", pady=(0, 12))

        status = ttk.LabelFrame(outer, text="System Status", padding=10)
        status.pack(fill="x", pady=(0, 12))

        self.glove_status = tk.StringVar(value="Glove: Disconnected")
        self.arm_status = tk.StringVar(value="Arm: Disconnected")
        self.control_status = tk.StringVar(value="Control: Not initialized")

        ttk.Label(status, textvariable=self.glove_status).grid(
            row=0, column=0, sticky="w", padx=8
        )
        ttk.Label(status, textvariable=self.arm_status).grid(
            row=0, column=1, sticky="w", padx=8
        )
        ttk.Label(status, textvariable=self.control_status).grid(
            row=0, column=2, sticky="w", padx=8
        )

        buttons = ttk.Frame(outer)
        buttons.pack(fill="x", pady=(0, 12))

        self.connect_button = ttk.Button(
            buttons, text="Connect Arduinos", command=self.connect_devices
        )
        self.connect_button.pack(side="left", padx=(0, 8))

        self.init_button = ttk.Button(
            buttons, text="Initialize Motors", command=self.initialize_motors,
            state="disabled"
        )
        self.init_button.pack(side="left", padx=8)

        self.stop_button = ttk.Button(
            buttons, text="Stop / Disarm", command=self.stop_motors,
            state="disabled"
        )
        self.stop_button.pack(side="left", padx=8)

        panel = ttk.LabelFrame(outer, text="Finger and Motor Monitor", padding=8)
        panel.pack(fill="both", expand=True)

        headers = [
            "Finger", "Raw ADC", "Filtered ADC", "Finger State",
            "Mapped Motor", "Joint", "Pin", "Target Angle"
        ]
        for col, heading in enumerate(headers):
            ttk.Label(panel, text=heading, font=("TkDefaultFont", 9, "bold")).grid(
                row=0, column=col, padx=5, pady=8, sticky="w"
            )

        self.cells = {}
        for row, finger in enumerate(FINGERS, start=1):
            info = FINGERS[finger]
            values = [
                finger.title(), "--", "--", "UNKNOWN",
                info["motor"], str(info["joint"]), info["pin"], "--"
            ]
            self.cells[finger] = []
            for col, value in enumerate(values):
                var = tk.StringVar(value=value)
                ttk.Label(panel, textvariable=var, padding=(5, 8)).grid(
                    row=row, column=col, sticky="w", padx=5
                )
                self.cells[finger].append(var)

        mapping = ttk.LabelFrame(outer, text="Arm Configuration", padding=10)
        mapping.pack(fill="x", pady=(12, 8))
        ttk.Label(
            mapping,
            text="Thumb → Shoulder (J2)    |    Index → Waist (J1)    |    "
                 "Middle → Elbow (J3)    |    Ring → Gripper (J6)"
        ).pack(anchor="w")
        ttk.Label(
            mapping,
            text="Straight: 50°    |    Bent: 120°    |    "
                 "Gripper direction is reversed    |    "
                 "Wrist roll and wrist pitch remain detached"
        ).pack(anchor="w", pady=(5, 0))

        activity = ttk.LabelFrame(outer, text="Latest Activity", padding=10)
        activity.pack(fill="x", pady=(0, 8))
        self.action_var = tk.StringVar(value=self.last_action)
        ttk.Label(
            activity, textvariable=self.action_var, wraplength=950
        ).pack(anchor="w")

        footer = ttk.Frame(outer)
        footer.pack(fill="x")
        self.samples_var = tk.StringVar(value="Valid samples: 0")
        ttk.Label(footer, textvariable=self.samples_var).pack(side="left")
        ttk.Label(
            footer, text="CSV log: glove_arm_dashboard.csv"
        ).pack(side="right")

    def connect_devices(self):
        if self.connected:
            return

        try:
            self.glove_status.set("Glove: Connecting...")
            self.arm_status.set("Arm: Connecting...")
            self.root.update_idletasks()

            arm = serial.Serial(ARM_PORT, BAUD, timeout=0.05)
            time.sleep(2)
            arm.reset_input_buffer()

            glove = serial.Serial(GLOVE_PORT, BAUD, timeout=0.05)
            time.sleep(2)
            glove.reset_input_buffer()

            self.arm = arm
            self.glove = glove
            self.connected = True
            self.reader_stop.clear()
            self.reader_thread = threading.Thread(
                target=self.glove_reader, daemon=True
            )
            self.reader_thread.start()

            self.glove_status.set(f"Glove: Connected ({GLOVE_PORT})")
            self.arm_status.set(f"Arm: Connected ({ARM_PORT})")
            self.control_status.set("Control: Connected, not initialized")
            self.init_button.config(state="normal")
            self.stop_button.config(state="normal")
            self.connect_button.config(state="disabled")
            self.last_action = "Connected. Motors have not been initialized."

            self.arm.write(b"STATUS\n")
            self.arm.flush()

            os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
            with open(self.log_file, "w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([
                    "timestamp", "finger", "raw_adc", "filtered_adc",
                    "state", "motor", "joint", "target_angle", "action"
                ])

        except Exception as exc:
            if self.glove and self.glove.is_open:
                self.glove.close()
            if self.arm and self.arm.is_open:
                self.arm.close()
            self.glove = None
            self.arm = None
            self.connected = False
            self.glove_status.set("Glove: Disconnected")
            self.arm_status.set("Arm: Disconnected")
            messagebox.showerror(
                "Connection failed",
                f"{exc}\n\nCheck the USB ports and close other programs "
                "using either Arduino."
            )

    def arm_command(self, command, delay=0.15):
        if not self.arm or not self.arm.is_open:
            raise serial.SerialException("Arm Arduino is not connected")

        self.arm.write((command + "\n").encode("ascii"))
        self.arm.flush()
        time.sleep(delay)

        responses = []
        while self.arm.in_waiting:
            responses.append(
                self.arm.readline().decode("utf-8", errors="replace").strip()
            )
        return responses

    def initialize_motors(self):
        if not self.connected:
            return

        confirm = messagebox.askyesno(
            "Confirm motor initialization",
            "Before continuing:\n\n"
            "• Keep hands clear of the arm.\n"
            "• Ensure the arm is supported and can move freely.\n"
            "• Verify the servo power supply and wiring.\n"
            "• Only the four mapped joints will be tested.\n\n"
            "Initialize the mapped motors at 50 degrees?"
        )
        if not confirm:
            return

        self.init_button.config(state="disabled")
        self.control_status.set("Control: Initializing...")
        self.root.update_idletasks()

        try:
            sequence = [
                (1, "Waist", True),
                (2, "Shoulder", True),
                (3, "Elbow", True),
                (4, "Wrist Roll", False),
                (5, "Wrist Pitch", False),
                (6, "Gripper", True),
            ]
            for joint, name, enabled in sequence:
                if enabled:
                    self.arm_command(f"TEST {joint} {STRAIGHT_ANGLE}", 0.4)
                    self.last_action = (
                        f"Initialized {name} (Joint {joint}) at "
                        f"{STRAIGHT_ANGLE}°"
                    )
                else:
                    self.last_action = f"{name} remains detached"
                self.action_var.set(self.last_action)
                self.root.update_idletasks()

            self.initialized = True
            for finger in FINGERS:
                self.states[finger] = "STRAIGHT"
                self.angles[finger] = STRAIGHT_ANGLE

            self.control_status.set("Control: ACTIVE")
            self.last_action = "Initialization complete. Live control active."
            self.action_var.set(self.last_action)

        except Exception as exc:
            self.initialized = False
            self.control_status.set("Control: Initialization failed")
            messagebox.showerror("Initialization failed", str(exc))
        finally:
            if self.connected and not self.initialized:
                self.init_button.config(state="normal")

    def stop_motors(self):
        self.initialized = False
        self.control_status.set("Control: Stopping / disarming")

        try:
            if self.arm and self.arm.is_open:
                self.arm_command("STOP", 0.2)
                self.arm_command("DISARM", 0.2)
            self.last_action = "STOP and DISARM sent. Live motor control disabled."
            self.action_var.set(self.last_action)
            self.control_status.set("Control: DISARMED")
        except Exception as exc:
            messagebox.showerror("Disarm warning", str(exc))

        self.init_button.config(state="normal" if self.connected else "disabled")

    def parse_glove(self, line):
        parts = line.strip().split(",")
        if len(parts) != 5:
            return None
        try:
            values = [int(v) for v in parts]
        except ValueError:
            return None
        if not all(0 <= v <= 1023 for v in values[1:]):
            return None
        return dict(zip(FINGERS, values[1:]))

    def process_glove_line(self, line):
        readings = self.parse_glove(line)
        if readings is None:
            return

        self.sample_count += 1
        now = datetime.now().isoformat(timespec="milliseconds")

        for finger, raw in readings.items():
            self.raw[finger] = raw
            self.samples[finger].append(raw)

            if len(self.samples[finger]) < WINDOW_SIZE:
                continue

            filtered = sorted(self.samples[finger])[len(self.samples[finger]) // 2]
            self.filtered[finger] = filtered
            threshold = FINGERS[finger]["threshold"]

            if self.states[finger] == "BENT":
                detected = "STRAIGHT" if filtered > threshold + 5 else "BENT"
            else:
                detected = "BENT" if filtered < threshold else "STRAIGHT"

            if detected == self.candidates[finger]:
                self.stable_counts[finger] += 1
            else:
                self.candidates[finger] = detected
                self.stable_counts[finger] = 1

            if self.stable_counts[finger] >= STABLE_SAMPLES:
                previous = self.states[finger]
                if previous == "UNKNOWN":
                    self.states[finger] = detected
                    if detected == "BENT":
                        self.apply_motor_command(finger, detected)
                    else:
                        self.angles[finger] = STRAIGHT_ANGLE
                elif detected != previous:
                    self.states[finger] = detected
                    if self.initialized:
                        self.apply_motor_command(finger, detected)

            info = FINGERS[finger]
            angle = self.angles[finger]
            with open(self.log_file, "a", newline="") as file:
                csv.writer(file).writerow([
                    now, finger, raw, filtered, self.states[finger],
                    info["motor"], info["joint"], angle, self.last_action
                ])

    def apply_motor_command(self, finger, state):
        if not self.initialized:
            return

        info = FINGERS[finger]
        angle = (
            STRAIGHT_ANGLE if state == "BENT" else BENT_ANGLE
        ) if finger == "ring" else (
            BENT_ANGLE if state == "BENT" else STRAIGHT_ANGLE
        )

        try:
            self.arm_command(f"SET {info['joint']} {angle}")
            self.angles[finger] = angle
            self.last_action = (
                f"{finger.title()} → {info['motor']}: {state}, commanded {angle}°"
            )
            self.action_var.set(self.last_action)
        except Exception as exc:
            self.initialized = False
            self.control_status.set("Control: ERROR")
            self.last_action = f"Motor command failed: {exc}"
            self.action_var.set(self.last_action)

    def glove_reader(self):
        while not self.reader_stop.is_set():
            try:
                if not self.glove or not self.glove.is_open:
                    time.sleep(0.02)
                    continue
                line = self.glove.readline().decode(
                    "utf-8", errors="replace"
                ).strip()
                if line:
                    try:
                        self.glove_queue.put_nowait(line)
                    except queue.Full:
                        try:
                            self.glove_queue.get_nowait()
                            self.glove_queue.put_nowait(line)
                        except queue.Empty:
                            pass
            except Exception as exc:
                if not self.reader_stop.is_set():
                    try:
                        self.glove_queue.put_nowait("__READ_ERROR__" + str(exc))
                    except queue.Full:
                        pass
                time.sleep(0.05)

    def update_ui(self):
        if self.closing:
            return

        for _ in range(100):
            try:
                line = self.glove_queue.get_nowait()
            except queue.Empty:
                break

            if line.startswith("__READ_ERROR__"):
                self.glove_status.set("Glove: Read error")
                self.initialized = False
                self.control_status.set("Control: ERROR")
                continue

            self.process_glove_line(line)

        for finger in FINGERS:
            values = self.cells[finger]
            info = FINGERS[finger]
            values[1].set(str(self.raw[finger]))
            values[2].set(str(self.filtered[finger]))
            values[3].set(self.states[finger])
            values[4].set(info["motor"])
            values[5].set(str(info["joint"]))
            values[6].set(info["pin"])
            values[7].set(
                f"{self.angles[finger]}°"
                if self.angles[finger] != "--" else "--"
            )

        self.samples_var.set(f"Valid samples: {self.sample_count}")
        self.action_var.set(self.last_action)
        self.root.after(50, self.update_ui)

    def close(self):
        if not messagebox.askyesno(
            "Close dashboard",
            "Close the dashboard and send STOP/DISARM to the arm?"
        ):
            return

        self.closing = True
        self.initialized = False
        self.reader_stop.set()

        try:
            if self.arm and self.arm.is_open:
                self.arm.write(b"STOP\n")
                self.arm.flush()
                time.sleep(0.2)
                self.arm.write(b"DISARM\n")
                self.arm.flush()
                time.sleep(0.2)
        except Exception:
            pass

        for device in (self.glove, self.arm):
            try:
                if device and device.is_open:
                    device.close()
            except Exception:
                pass

        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = GloveArmDashboard(root)
    root.mainloop()
