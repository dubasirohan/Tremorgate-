import json
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, String, Bool
from .cnn_model import TremorCNN


class Perception(Node):
    def __init__(self):
        super().__init__("perception_node")

        self.declare_parameter(
            "model_path",
            str(Path.home() / "tremorgate_ws/offline/models/tremor_cnn_expanded_best.pt")
        )
        self.declare_parameter("fs", 100.0)
        self.declare_parameter("accel_counts_per_g", 16384.0)
        self.declare_parameter("log_path", "tremor_events.jsonl")

        torch.set_num_threads(2)
        self.fs = float(self.get_parameter("fs").value)
        self.counts_per_g = float(
            self.get_parameter("accel_counts_per_g").value
        )

        model_path = str(self.get_parameter("model_path").value)
        checkpoint = torch.load(
            model_path, map_location="cpu", weights_only=False
        )

        self.mean = np.asarray(
            checkpoint["normalization_mean"], dtype=np.float32
        ).reshape(1, 3, 1)
        self.std = np.asarray(
            checkpoint["normalization_std"], dtype=np.float32
        ).reshape(1, 3, 1)

        if np.any(self.std <= 0):
            raise ValueError("Invalid checkpoint normalization values")

        self.model = TremorCNN(num_classes=2)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        self.buf = deque(maxlen=256)
        self.samples_since_infer = 0
        self.log_path = str(self.get_parameter("log_path").value)

        self.cls_pub = self.create_publisher(String, "/classification", 10)
        self.prob_pub = self.create_publisher(
            Float32MultiArray, "/tremor_probability", 10
        )
        self.state_pub = self.create_publisher(Bool, "/tremor_state", 10)

        self.sub = self.create_subscription(
            Float32MultiArray, "/sensor_raw", self.cb, 20
        )

        self.get_logger().info(
            f"PyTorch CNN loaded. Window=256, fs={self.fs} Hz. "
            "Arm commands DISABLED."
        )

    def cb(self, msg):
        a = np.asarray(msg.data, dtype=np.float32)

        # Bridge output: flex1..4, ax, ay, az, gx, gy, gz
        if a.size != 10 or not np.isfinite(a).all():
            return

        # Convert raw accelerometer counts to g.
        accel_g = a[4:7] / self.counts_per_g
        self.buf.append(accel_g)

        if len(self.buf) < 256:
            return

        # Infer every 128 new samples, using a 256-sample window.
        self.samples_since_infer += 1
        if self.samples_since_infer < 128:
            return
        self.samples_since_infer = 0

        window = np.asarray(self.buf, dtype=np.float32).T[None, :, :]
        x = (window - self.mean) / self.std

        with torch.no_grad():
            logits = self.model(torch.from_numpy(
                np.ascontiguousarray(x)
            ))
            probs = torch.softmax(logits, dim=1)[0].numpy()

        healthy_prob = float(probs[0])
        tremor_prob = float(probs[1])
        predicted = int(np.argmax(probs))

        result = {
            "time": time.time(),
            "model": "tremor_cnn_expanded_best.pt",
            "class_id": predicted,
            "class": (
                "healthy_control" if predicted == 0
                else "tremor_associated_ET_or_Parkinsons"
            ),
            "healthy_probability": healthy_prob,
            "tremor_associated_probability": tremor_prob,
            "window_samples": 256,
            "arm_commands_enabled": False
        }

        out = String()
        out.data = json.dumps(result)
        self.cls_pub.publish(out)

        pm = Float32MultiArray()
        pm.data = [healthy_prob, tremor_prob]
        self.prob_pub.publish(pm)

        state = Bool()
        state.data = predicted == 1
        self.state_pub.publish(state)

        self.get_logger().info(
            f"CNN: {result['class']} | "
            f"tremor-associated probability={tremor_prob:.3f}"
        )

    def destroy_node(self):
        super().destroy_node()


def main():
    rclpy.init()
    node = Perception()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
