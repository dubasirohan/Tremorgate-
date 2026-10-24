import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

class ArmGate(Node):
    def __init__(self):
        super().__init__("arm_gate_node")
        self.declare_parameter("controller_topic","/forward_position_controller/commands")
        topic=self.get_parameter("controller_topic").value
        self.pub=self.create_publisher(Float64MultiArray,topic,10)
        self.sub=self.create_subscription(Float64MultiArray,"/arm_cmd",self.cb,10)
    def cb(self,msg):
        if len(msg.data)!=2:
            self.get_logger().warning("arm_cmd must contain exactly 2 joint targets")
            return
        self.pub.publish(msg)
        self.get_logger().info(f"VOLUNTARY MOVE q={list(msg.data)}")

def main():
    rclpy.init(); n=ArmGate()
    try:rclpy.spin(n)
    except KeyboardInterrupt:pass
    n.destroy_node(); rclpy.shutdown()
