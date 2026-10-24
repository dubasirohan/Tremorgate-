import time
import serial
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from .common import parse_csv_line


class SerialBridge(Node):
    def __init__(self):
        super().__init__("serial_bridge_node")
        self.declare_parameter("port", "/dev/ttyUSB0")
        self.declare_parameter("baud", 115200)
        self.port = str(self.get_parameter("port").value)
        self.baud = int(self.get_parameter("baud").value)
        self.pub = self.create_publisher(Float32MultiArray, "/sensor_raw", 10)
        self.ser = None
        self.last_log = 0.0
        self.timer = self.create_timer(0.01, self.poll)

    def poll(self):
        if self.ser is None:
            try:
                self.ser = serial.Serial(self.port, self.baud, timeout=0.2)
                self.ser.reset_input_buffer()
                self.get_logger().info(f"Opened {self.port} @ {self.baud}")
            except Exception as e:
                now = time.monotonic()
                if now - self.last_log > 2:
                    self.get_logger().warning(f"Serial open failed: {e}")
                    self.last_log = now
            return

        try:
            raw = self.ser.readline()
            if not raw:
                return
            line = raw.decode("utf-8", "replace").strip()

            try:
                vals = parse_csv_line(line)
                msg = Float32MultiArray()
                msg.data = [float(x) for x in vals[1:11]]
                self.pub.publish(msg)
                self.get_logger().info(f"Published sensor row: {line}")
            except Exception as e:
                self.get_logger().warning(f"Rejected row: {line!r}; reason: {e}")
        except Exception as e:
            self.get_logger().error(f"Serial connection lost: {e}")
            try:
                self.ser.close()
            except Exception:
                pass
            self.ser = None

    def destroy_node(self):
        if self.ser:
            self.ser.close()
        super().destroy_node()


def main():
    rclpy.init()
    node = SerialBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
