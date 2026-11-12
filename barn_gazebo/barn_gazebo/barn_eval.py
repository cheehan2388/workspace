"""
barn_eval — 在 Gazebo 的 BARN 地圖上記一回合的成績（不控制車子，只看）

從 /odom 讀車子位置（odom 的原點 = 起點、x 軸 = 出發時的車頭方向，也就是北方），
換算成世界座標後判斷：
  成功    離終點 (-2.25, 13) < goal_tol（預設 1.0 m，跟 BARN 官方相同；另外記錄第一次 < 0.25 m 的時間，對應 2D 實驗）
  碰撞    車子中心到最近圓柱的表面距離 < robot_radius（預設 0.11 m，跟 2D 實驗相同）
  逾時    從車子開始移動算起超過 timeout 秒（預設 100 s 模擬時間，跟 BARN 官方相同）
結束時印出結果，並在 out（預設 ~/barn_results.csv）加一行。

  ros2 run barn_gazebo barn_eval --ros-args -p world:=0 -p use_sim_time:=true -p method:=nav2
"""
import csv
import json
import math
import os
import time

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import String


class BarnEval(Node):
    def __init__(self):
        super().__init__("barn_eval")
        self.world = self.declare_parameter("world", 0).value
        self.goal_tol = self.declare_parameter("goal_tol", 1.0).value
        self.radius = self.declare_parameter("robot_radius", 0.11).value
        self.timeout = self.declare_parameter("timeout", 100.0).value
        self.method = self.declare_parameter("method", "").value
        self.out = os.path.expanduser(self.declare_parameter("out", "~/barn_results.csv").value)

        path = os.path.join(get_package_share_directory("barn_gazebo"), "worlds", f"barn_{self.world}.json")
        info = json.load(open(path))
        self.cyl = np.array(info["cylinders"])
        self.cyl_r = info["radius"]
        self.sx, self.sy, self.syaw = info["start"]
        self.gx, self.gy = info["goal"]

        self.t0 = None
        self.last = None
        self.path_len = 0.0
        self.min_clear = math.inf
        self.t_close = None
        self.done = False
        self.last_print = 0.0
        self.create_subscription(Odometry, "odom", self.on_odom, 20)
        self.pub_result = self.create_publisher(String, "barn_result", 10)   # run_method 收到就停車
        self.get_logger().info(f"BARN {self.world}：等車子開始移動…（終點 ({self.gx}, {self.gy})，"
                               f"在 odom 座標是 ({self.gy - self.sy:.1f}, {-(self.gx - self.sx):.1f})）")

    def to_world(self, ox, oy):
        c, s = math.cos(self.syaw), math.sin(self.syaw)
        return self.sx + c * ox - s * oy, self.sy + s * ox + c * oy

    def on_odom(self, msg):
        if self.done:
            return
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        x, y = self.to_world(msg.pose.pose.position.x, msg.pose.pose.position.y)
        if self.t0 is None:
            if math.hypot(x - self.sx, y - self.sy) < 0.02:
                return
            self.t0 = t
            self.get_logger().info("開始計時")
        if self.last is not None:
            self.path_len += math.hypot(x - self.last[0], y - self.last[1])
        self.last = (x, y)
        el = t - self.t0
        clear = float(np.min(np.hypot(self.cyl[:, 0] - x, self.cyl[:, 1] - y))) - self.cyl_r
        self.min_clear = min(self.min_clear, clear)
        d_goal = math.hypot(self.gx - x, self.gy - y)
        if self.t_close is None and d_goal < 0.25:
            self.t_close = el

        if clear < self.radius:
            self.finish("collision", el, x, y)
        elif d_goal < self.goal_tol:
            self.finish("success", el, x, y)
        elif el > self.timeout:
            self.finish("timeout", el, x, y)
        elif time.time() - self.last_print > 2.0:
            self.last_print = time.time()
            self.get_logger().info(f"t={el:5.1f}s  位置=({x:.2f}, {y:.2f})  離終點 {d_goal:.2f} m  最近圓柱 {clear:.2f} m")

    def finish(self, outcome, el, x, y):
        self.done = True
        row = {"time_stamp": time.strftime("%Y-%m-%d %H:%M:%S"), "method": self.method, "world": self.world,
               "outcome": outcome, "time": round(el, 2), "path_len": round(self.path_len, 2),
               "min_clear": round(self.min_clear, 3), "t_within_0.25m": "" if self.t_close is None else round(self.t_close, 2),
               "end_x": round(x, 2), "end_y": round(y, 2)}
        new = not os.path.exists(self.out)
        with open(self.out, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if new:
                w.writeheader()
            w.writerow(row)
        self.pub_result.publish(String(data=json.dumps(row)))
        self.get_logger().info(f"=== BARN {self.world}：{outcome}，{el:.1f} s，路徑 {self.path_len:.2f} m，"
                               f"最近圓柱 {self.min_clear:.3f} m → 已寫入 {self.out}")


def main():
    rclpy.init()
    node = BarnEval()
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.5)
        for _ in range(10):                     # 結果多發幾次，確保 run_method 收到
            node.pub_result.publish(String(data=json.dumps({"outcome": "done"})))
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
