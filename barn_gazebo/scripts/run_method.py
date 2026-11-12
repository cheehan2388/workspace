#!/usr/bin/env python3
"""
run_method.py — 把 2D 實驗的六種方法原封不動接到 Gazebo 上開車

讀 /scan、/odom → 呼叫 sim2d_levels 裡同一份 Pipeline2（A*、DWA、PPO、Hybrid 的程式完全不改）→ 發 /cmd_vel。
每 0.2 秒決策一次（5 Hz，跟 2D 一樣），用模擬時間。

先開地圖（另一個終端機）：
  ros2 launch barn_gazebo barn.launch.py world:=0 method:=astar_dwa
再開車（在裝了 PPO 套件的 venv 裡）：
  python3 run_method.py --ros-args -p method:=astar_dwa -p world:=0 -p use_sim_time:=true
2D 程式和模型都已經包在 barn_gazebo 套件裡，不用另外指定。

參數
  method      ppo / lstm / astar_ppo / dwa / astar_dwa / hybrid
  world       BARN 地圖編號（0, 6, ..., 294）
  models      v1（原獎勵，預設）或 v2（ω 平順獎勵 k=0.02，只有 ppo_waypoint 種子 0）；也可以給資料夾完整路徑
  sim2d       2D 程式資料夾（預設用套件裡附的，通常不用改）
  train_seed  用哪個訓練種子的模型（預設 0）
  replan      A* 每秒依光達重新規劃（預設 false）
  cmd_stamped 官方 TurtleBot3（Jazzy）= true（TwistStamped）；nav_sim 的 my_robot = false（Twist）
收到 barn_eval 的結果（/barn_result）、或離終點 < 0.25 m、或超過 120 s 就停車結束。
"""
import math
import os
import sys
import time

import numpy as np
import rclpy
from geometry_msgs.msg import Twist, TwistStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String

START = (-2.25, 3.0, 1.5708)
GOAL = (-2.25, 13.0)


class RunMethod(Node):
    def __init__(self):
        super().__init__("run_method")
        p = lambda n, d: self.declare_parameter(n, d).value
        self.method = p("method", "astar_dwa")
        self.world = int(p("world", 0))
        from ament_index_python.packages import get_package_share_directory
        share = get_package_share_directory("barn_gazebo")
        sim2d = os.path.expanduser(p("sim2d", os.path.join(share, "sim2d")))
        models = p("models", "v1")
        models = {"v1": os.path.join(share, "models", "v1"),
                  "v2": os.path.join(share, "models", "v2_k0.02")}.get(models, os.path.expanduser(models))
        seed = int(p("train_seed", 0))
        replan = bool(p("replan", False))
        stamped = bool(p("cmd_stamped", True))

        sys.path.insert(0, sim2d)
        from levels import barn_walls, _barn            # noqa: E402  （2D 實驗的同一份程式）
        from planners2 import METHODS2, Pipeline2       # noqa: E402

        _, idx = _barn()
        k = int(np.where(idx == self.world)[0][0])      # 第幾張（0..49）
        scene = dict(name="barn", barn_map=k, walls=barn_walls(k), obstacles=[], start=START, goal=GOAL)

        kind = METHODS2[self.method][2]
        model = None
        if kind:
            import torch
            torch.set_num_threads(1)
            path = os.path.join(models, f"{kind}_s{seed}.zip")
            if kind == "lstm_goal":
                from sb3_contrib import RecurrentPPO
                model = RecurrentPPO.load(path, device="cpu")
            else:
                from stable_baselines3 import PPO
                model = PPO.load(path, device="cpu")
            self.get_logger().info(f"模型：{path}")
        self.pipe = Pipeline2(self.method, model, replan=replan)
        # 先空跑一步讓模型熱身（第一次推論特別慢），再重設，避免開車第一步卡住
        self.pipe.reset(scene)
        self.pipe.act(START[0], START[1], START[2], 0.0, 0.0, np.full(360, np.inf))
        self.pipe.reset(scene)

        self.scan = self.odom = None
        self.t_start = None
        self.n_ppo = self.n_steps = 0
        self.finished = False
        self.stamped = stamped
        self.pub = self.create_publisher(TwistStamped if stamped else Twist, "cmd_vel", 10)
        self.create_subscription(LaserScan, "scan", self.on_scan, 10)
        self.create_subscription(Odometry, "odom", self.on_odom, 20)
        self.create_subscription(String, "barn_result", self.on_result, 10)
        self.create_timer(0.2, self.tick)               # 用 use_sim_time 時，這是模擬時間的 0.2 s
        self.get_logger().info(f"BARN {self.world}（第 {k} 張）用 {self.method}{'（重新規劃）' if replan else ''} 開車")

    # LaserScan → 2D 實驗的格式：360 條、第 0 條在正前方、每度一條逆時針、太近 -inf、太遠 inf
    def on_scan(self, msg):
        r = np.array(msg.ranges, dtype=float)
        ang = msg.angle_min + msg.angle_increment * np.arange(len(r))
        deg = np.round(np.degrees(ang)).astype(int) % 360
        out = np.full(360, np.inf)
        bad = ~np.isfinite(r) | (r > msg.range_max)
        r[bad] = np.inf
        r[np.isfinite(r) & (r < msg.range_min)] = -np.inf
        for d, v in zip(deg, r):                        # 同一度有多條時取最近
            if v < out[d]:
                out[d] = v
        self.scan = out

    def on_odom(self, msg):
        self.odom = msg

    def on_result(self, msg):
        self.get_logger().info(f"barn_eval：{msg.data}")
        self.stop()

    def pose(self):
        o = self.odom.pose.pose
        ox, oy = o.position.x, o.position.y
        q = o.orientation
        oyaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        sx, sy, syaw = START
        c, s = math.cos(syaw), math.sin(syaw)
        return sx + c * ox - s * oy, sy + s * ox + c * oy, math.atan2(math.sin(syaw + oyaw), math.cos(syaw + oyaw))

    def send(self, v, w):
        if self.stamped:
            m = TwistStamped()
            m.header.stamp = self.get_clock().now().to_msg()
            m.header.frame_id = "base_link"
            m.twist.linear.x, m.twist.angular.z = float(v), float(w)
        else:
            m = Twist()
            m.linear.x, m.angular.z = float(v), float(w)
        self.pub.publish(m)

    def stop(self):
        if not self.finished:
            self.finished = True
            for _ in range(3):
                self.send(0.0, 0.0)
            frac = self.n_ppo / max(self.n_steps, 1)
            self.get_logger().info(f"結束。{self.n_steps} 步，PPO 控制佔 {frac:.0%}")

    def tick(self):
        if self.finished or self.scan is None or self.odom is None:
            return
        now = self.get_clock().now().nanoseconds * 1e-9
        if self.t_start is None:
            self.t_start = now
        x, y, yaw = self.pose()
        if math.hypot(GOAL[0] - x, GOAL[1] - y) < 0.25 or now - self.t_start > 120.0:
            self.stop()
            return
        tw = self.odom.twist.twist
        t0 = time.time()
        v, w, mode = self.pipe.act(x, y, yaw, tw.linear.x, tw.angular.z, self.scan)
        dt = time.time() - t0
        if dt > 0.15:
            self.get_logger().warn(f"這一步算了 {dt * 1000:.0f} ms（超過 0.2 s 的決策週期會讓車反應變慢）")
        self.n_steps += 1
        self.n_ppo += mode == "ppo"
        self.send(v, w)


def main():
    rclpy.init()
    node = RunMethod()
    try:
        while rclpy.ok() and not node.finished:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        node.stop()
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
