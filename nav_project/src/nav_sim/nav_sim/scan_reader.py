"""
scan_reader.py — 讀 /scan，整理成 RL 要用的觀測，並判斷有沒有撞到

用法（sim.launch.py 開著時）：
  ros2 run nav_sim scan_reader
  ros2 run nav_sim scan_reader --ros-args -p sectors:=24 -p collision_dist:=0.18

做法：
  1. +inf / nan 換成 range_max，-inf（太近）換成 range_min，再夾在 range_min~range_max
  2. 切成 N 個扇區，每區取最小值（最危險的那條）→ N 維觀測
  3. 除以 range_max 正規化到 0~1（RL 網路比較好學）
  4. 最近距離 < collision_dist 就算撞到（之後 RL 用來結束 episode）

角度：光達從 0 rad（車頭正前方）逆時針轉一圈，所以 sector 0 = 正前方附近。
"""

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan


def preprocess(msg, sectors):
    """LaserScan → (正規化後的 N 維觀測, 最近距離 m, 最近的角度 rad)"""
    r = np.asarray(msg.ranges, dtype=np.float32)
    # 沒打到東西 (+inf) 跟讀不到 (nan) 當成很遠；nan 當近的話會一直誤判撞牆
    # 比 range_min 還近 (-inf) 當成很近，不然貼著牆反而會被當成前面沒東西
    r = np.where(np.isneginf(r), msg.range_min, r)
    r = np.where(np.isfinite(r), r, msg.range_max)
    r = np.clip(r, msg.range_min, msg.range_max)

    i = int(np.argmin(r))
    min_dist = float(r[i])
    min_angle = msg.angle_min + i * msg.angle_increment

    # 360 條不一定剛好整除 N，用 array_split 平均分
    obs = np.array([s.min() for s in np.array_split(r, sectors)], dtype=np.float32)
    return obs / msg.range_max, min_dist, min_angle


class ScanReader(Node):
    def __init__(self):
        super().__init__("scan_reader")
        self.sectors = self.declare_parameter("sectors", 24).value
        self.collision_dist = self.declare_parameter("collision_dist", 0.18).value
        # 光達是 sensor 資料，用 best-effort QoS（跟 bridge 對得上，也不會卡舊資料）
        self.create_subscription(LaserScan, "/scan", self.on_scan, qos_profile_sensor_data)
        self.count = 0

    def on_scan(self, msg):
        obs, min_dist, min_angle = preprocess(msg, self.sectors)
        self.count += 1
        if min_dist < self.collision_dist:
            self.get_logger().warn(
                f"💥 撞到！最近 {min_dist:.2f} m @ {np.degrees(min_angle):.0f}°")
        elif self.count % 5 == 0:   # 5 Hz → 每秒印一次就好
            self.get_logger().info(
                f"射線 {len(msg.ranges)} 條 | 最近 {min_dist:.2f} m @ "
                f"{np.degrees(min_angle):.0f}° | obs = {np.round(obs, 2).tolist()}")


def main():
    rclpy.init()
    node = ScanReader()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
