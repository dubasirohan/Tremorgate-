import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

class Stub(Node):
    def __init__(self):
        super().__init__("arm_stub_node")
        self.sub=self.create_subscription(Float64MultiArray,"/arm_cmd",self.cb,10)
        self.last=None
        self.create_timer(0.5,self.hold)
    def cb(self,msg):
        self.last=list(msg.data); self.get_logger().info(f"MOVE {self.last}")
    def hold(self):
        if self.last is None:self.get_logger().info("HOLD (no voluntary command yet)")
        else:self.get_logger().debug(f"HOLD at {self.last}")
def main():
    rclpy.init(); n=Stub()
    try:rclpy.spin(n)
    except KeyboardInterrupt:pass
    n.destroy_node(); rclpy.shutdown()
