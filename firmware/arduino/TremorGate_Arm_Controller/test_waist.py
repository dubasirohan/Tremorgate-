import serial
import time

PORT = "/dev/ttyUSB0"
BAUD = 115200

def send(ser, command, delay=0.5):
    print(f">>> {command}")
    ser.write((command + "\n").encode())
    ser.flush()
    time.sleep(delay)
    while ser.in_waiting:
        print(ser.readline().decode(errors="replace").rstrip())

ser = None

try:
    ser = serial.Serial(PORT, BAUD, timeout=1)
    time.sleep(2)  # Arduino resets when serial opens
    ser.reset_input_buffer()

    print("\nWAIST SERVO TEST — Joint 1, pin D10")
    print("Keep the arm clear of obstructions.")
    input("Ensure servo power is OFF while preparing. Press ENTER when ready: ")

    input("Switch ON the external servo supply, then press ENTER: ")

    send(ser, "TEST 1 90", 2)

    input("If movement was safe and the servo is stable, press ENTER to move +5 degrees. Ctrl+C cancels: ")
    send(ser, "SET 1 95", 2)

    input("Press ENTER to return to 90 degrees: ")
    send(ser, "SET 1 90", 2)

    send(ser, "DISARM")
    print("Waist test completed.")

except KeyboardInterrupt:
    print("\nTest cancelled.")

finally:
    if ser is not None:
        try:
            ser.write(b"STOP\n")
            ser.flush()
            time.sleep(0.2)
            ser.write(b"DISARM\n")
            ser.flush()
            time.sleep(0.2)
            ser.close()
        except Exception:
            pass
    print("Serial connection closed. Switch OFF servo power.")
