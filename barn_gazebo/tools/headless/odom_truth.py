"""odom_truth.py — 把 Gazebo 的真實位置發成 nav_msgs/Odometry（/odom_truth），座標系跟 /odom 一樣

為什麼：TurtleBot3 Burger 在 Gazebo 轉彎時輪子會打滑，輪速里程計（/odom）的朝向會多算很多
（實測 BARN 0 跑 15 秒就差 56°、0.7 m）。2D 實驗用的是完美位置，要公平比較就給真實位置。

odom 座標系：原點 = 起點、x 軸 = 出發時車頭方向（跟 DiffDrive 的 /odom 相同），所以
run_method / barn_eval 只要 remap（-r odom:=/odom_truth）就能直接用，程式不用改。

  python3 odom_truth.py --world barn_0 --robot turtlebot3 [--start -2.25 3.0 1.5708]
"""
import argparse
import math
import threading

import rclpy
from gz.msgs10.pose_v_pb2 import Pose_V
from gz.transport13 import Node as GzNode
from nav_msgs.msg import Odometry


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", default="barn_0")
    ap.add_argument("--robot", default="turtlebot3")
    ap.add_argument("--start", type=float, nargs=3, default=[-2.25, 3.0, 1.5708])
    ap.add_argument("--rate", type=float, default=30.0, help="發布頻率（模擬時間 Hz）")
    a, _ = ap.parse_known_args()
    sx, sy, syaw = a.start
    cs, ss = math.cos(-syaw), math.sin(-syaw)

    rclpy.init()
    node = rclpy.create_node("odom_truth")
    pub = node.create_publisher(Odometry, "odom_truth", 20)
    lock = threading.Lock()
    st = {"last_t": None, "prev": None}

    def on_pose(msg):
        t = msg.header.stamp.sec + msg.header.stamp.nsec * 1e-9
        for p in msg.pose:
            if p.name != a.robot:
                continue
            with lock:
                if st["last_t"] is not None and t - st["last_t"] < 1.0 / a.rate - 1e-6:
                    return
                q = p.orientation
                yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
                dx, dy = p.position.x - sx, p.position.y - sy
                ox, oy, oyaw = cs * dx - ss * dy, ss * dx + cs * dy, wrap(yaw - syaw)
                v = w = 0.0
                if st["prev"] is not None and t > st["prev"][0]:      # 速度：跟上一筆差分（車身座標）
                    pt, px, py, pyaw = st["prev"]
                    dt = t - pt
                    v = ((ox - px) * math.cos(oyaw) + (oy - py) * math.sin(oyaw)) / dt
                    w = wrap(oyaw - pyaw) / dt
                st["prev"] = (t, ox, oy, oyaw)
                st["last_t"] = t
            m = Odometry()
            m.header.stamp.sec, m.header.stamp.nanosec = msg.header.stamp.sec, msg.header.stamp.nsec
            m.header.frame_id, m.child_frame_id = "odom", "base_footprint"
            m.pose.pose.position.x, m.pose.pose.position.y = ox, oy
            m.pose.pose.orientation.z, m.pose.pose.orientation.w = math.sin(oyaw / 2), math.cos(oyaw / 2)
            m.twist.twist.linear.x, m.twist.twist.angular.z = v, w
            pub.publish(m)
            return

    gz = GzNode()
    gz.subscribe(Pose_V, f"/world/{a.world}/dynamic_pose/info", on_pose)
    node.get_logger().info(f"發布 /odom_truth（{a.world} 裡的 {a.robot} 真實位置）")
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
