import serial, time

names = {
    1: "Waist / D10",
    2: "Shoulder / D9",
    3: "Elbow / D8",
    4: "Wrist roll / D7",
    5: "Wrist pitch / D6",
    6: "Gripper / D5",
}

joint = int(input("Enter joint number (2-6): "))
if joint not in names or joint == 1:
    raise SystemExit("Choose a joint from 2 to 6.")

s = None
try:
    s = serial.Serial("/dev/ttyUSB0", 115200, timeout=1)
    time.sleep(2)
    s.reset_input_buffer()

    print(f"Testing {names[joint]}")
    input("Ensure the arm is clear. Turn ON external servo power, then press ENTER: ")

    def send(cmd, delay=1):
        print(">>>", cmd)
        s.write((cmd + "\n").encode())
        s.flush()
        time.sleep(delay)
        while s.in_waiting:
            print(s.readline().decode(errors="replace").rstrip())

    send(f"TEST {joint} 90", 2)
    input("If safe, press ENTER to move to 110 degrees; Ctrl+C cancels: ")
    send(f"SET {joint} 110", 2)
    input("Press ENTER to return to 90 degrees: ")
    send(f"SET {joint} 90", 2)
    send("DISARM")
    print("Test completed.")

except KeyboardInterrupt:
    print("\nCancelled.")

finally:
    if s is not None:
        try:
            s.write(b"STOP\n")
            s.flush()
            time.sleep(0.2)
            s.write(b"DISARM\n")
            s.flush()
            time.sleep(0.2)
            s.close()
        except Exception:
            pass
    print("Serial closed. Switch OFF external servo power.")
