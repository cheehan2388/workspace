"""
c2_trigger.py — C2 場景：機器人前進 1.5 m 後，在它前方放一個障礙物

用法（Gazebo 開著 c2_sudden.sdf 時）：
  ros2 run nav_sim c2_trigger

做法：訂閱 /odom 看車子走了多遠，到了就呼叫 Gazebo 的 create 服務把障礙物放進世界。
障礙物不在場景檔裡，所以掃地圖時看不到，A* 也不知道它的存在。
"""

import math
import os
import subprocess

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from ament_index_python.packages import get_package_share_directory

WORLD = "c2_sudden"
TRIGGER_DISTANCE = 1.5          # 車子前進多少公尺後觸發
OBSTACLE_XY = (-0.5, 0.0)       # 障礙物位置（世界座標）：起點 x=-3，前進 1.5 m 後的前方 1.0 m


class C2Trigger(Node):
    def __init__(self):
        super().__init__("c2_trigger")
        self.done = False
        self.model_file = os.path.join(
            get_package_share_directory("nav_sim"), "models", "sudden_obstacle", "model.sdf")
        self.create_subscription(Odometry, "/odom", self.on_odom, 10)
        self.get_logger().info(f"等待車子前進 {TRIGGER_DISTANCE} m ...")

    def on_odom(self, msg):
        if self.done:
            return
        p = msg.pose.pose.position
        # /odom 從出發點算起，所以 (x, y) 離原點多遠就是走了多遠
        if math.hypot(p.x, p.y) >= TRIGGER_DISTANCE:
            self.done = True
            self.spawn()

    def spawn(self):
        x, y = OBSTACLE_XY
        req = (f'sdf_filename: "{self.model_file}", name: "sudden_obstacle", '
               f'pose: {{position: {{x: {x}, y: {y}, z: 0.0}}}}')
        cmd = ["gz", "service", "-s", f"/world/{WORLD}/create",
               "--reqtype", "gz.msgs.EntityFactory", "--reptype", "gz.msgs.Boolean",
               "--timeout", "2000", "--req", req]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if "true" in result.stdout:
            self.get_logger().info(f"障礙物出現在 {OBSTACLE_XY}")
        else:
            self.get_logger().error(f"放置失敗：{result.stdout} {result.stderr}")


def main():
    rclpy.init()
    node = C2Trigger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
