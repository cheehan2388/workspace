"""
nav_env.py — 把 Gazebo 包成 Gymnasium 環境，給 PPO 訓練用

先開模擬（不用 SLAM、不用 RViz）：
  ros2 launch nav_sim sim.launch.py gui:=false slam:=false rviz:=false

步進方式（lockstep）：
  Gazebo 一直保持暫停，每個 step：送速度指令 → 叫 Gazebo 往前跑 0.2 秒 → 等新的 /scan
  → 算 reward。好處：結果不受電腦快慢影響，而且可以比真實時間跑得快（約 2~3 倍）。

觀測（29 維，全部是「相對機器人」的量，不用絕對座標）：
  [0:24]  光達 24 個扇區的最近距離 / 3.5（0~1）
  [24]    到目前路徑點的距離 / 3.0（超過就是 1）
  [25:27] 路徑點在車子的哪個方向：sin(角度), cos(角度)
  [27:29] 上一步的動作（讓網路知道自己現在多快）

動作（2 維，-1~1）：
  a[0] → 前進速度 0 ~ 0.22 m/s（不給倒車）
  a[1] → 轉彎速度 -1.5 ~ 1.5 rad/s

路徑點：先手寫 S 形走廊的轉角（之後可以換成在存好的地圖上跑 A* 的結果）。
位置：用輪式里程計（reset 時記下 /odom，之後算相對位移）。
"""

import math
import time

import gymnasium as gym
import numpy as np
import rclpy
from gymnasium import spaces
from nav_msgs.msg import Odometry
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan

from gz.msgs10.boolean_pb2 import Boolean
from gz.msgs10.physics_pb2 import Physics
from gz.msgs10.pose_pb2 import Pose
from gz.msgs10.twist_pb2 import Twist
from gz.msgs10.world_control_pb2 import WorldControl
from gz.transport13 import Node as GzNode

from nav_sim.scan_reader import preprocess

# 每個場景的路徑：第一個是起點，最後一個是終點（世界座標 x, y）
# c1_serpentine：下走廊往東 → 右邊缺口往上 → 中走廊往西 → 左邊缺口往上 → 上走廊往東穿過門
ROUTES = {
    "c1_serpentine": [(-1.5, 0.55), (1.45, 0.55), (1.45, 1.65),
                      (-1.45, 1.65), (-1.45, 2.75), (1.5, 2.75)],
}

MAX_V = 0.22        # m/s，TurtleBot3 Burger 上限
MAX_W = 1.5         # rad/s
STEP_TIME = 0.2     # 一個 step = 0.2 秒模擬時間（剛好一次光達掃描，5 Hz）
PHYS_DT = 0.001     # 物理步長（跟場景檔一樣 1 ms；改大會比較快但不準）


def wrap(a):
    """角度包到 -pi ~ pi"""
    return (a + math.pi) % (2 * math.pi) - math.pi


def yaw_of(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


class NavEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, world="c1_serpentine", sectors=24, max_steps=600,
                 collision_dist=0.18, goal_tol=0.3, rtf=100.0, seed=None):
        super().__init__()
        self.world = world
        self.route = ROUTES[world]
        self.sectors = sectors
        self.max_steps = max_steps
        self.collision_dist = collision_dist   # 光達量到 < 這個距離就算撞（車身半徑約 0.115 m）
        self.goal_tol = goal_tol
        self.rng = np.random.default_rng(seed)

        self.observation_space = spaces.Box(-1.0, 1.0, shape=(sectors + 5,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

        # ROS：讀 /scan、/odom（經過 bridge）
        if not rclpy.ok():
            rclpy.init()
        self.node = rclpy.create_node("nav_env")
        self.scan = None
        self.odom = None
        self.node.create_subscription(LaserScan, "/scan", self._on_scan, qos_profile_sensor_data)
        self.node.create_subscription(Odometry, "/odom", self._on_odom, 10)

        # Gazebo transport：直接送速度、控制暫停/步進、瞬移機器人（不經過 ROS，比較快）
        self.gz = GzNode()
        self.cmd_pub = self.gz.advertise("/cmd_vel", Twist)
        self.svc = f"/world/{world}"
        self._call("/control", WorldControl(pause=True), WorldControl)
        self._call("/set_physics", Physics(max_step_size=PHYS_DT, real_time_factor=rtf), Physics)

    # ---------- 跟 Gazebo / ROS 溝通 ----------
    def _on_scan(self, msg):
        self.scan = msg

    def _on_odom(self, msg):
        self.odom = msg

    @staticmethod
    def _stamp(msg):
        return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

    def _call(self, name, req, req_type):
        ok, rep = self.gz.request(self.svc + name, req, req_type, Boolean, 3000)
        if not ok:
            raise RuntimeError(f"Gazebo 服務 {self.svc + name} 沒回應，模擬有開嗎？")
        return rep

    def _send_cmd(self, v, w):
        m = Twist()
        m.linear.x = float(v)
        m.angular.z = float(w)
        self.cmd_pub.publish(m)

    def _advance(self):
        """Gazebo 往前跑 STEP_TIME 秒，等到新的 scan 和 odom 都到了才回來"""
        t0 = self._stamp(self.scan) if self.scan else -1.0
        time.sleep(0.002)   # 讓速度指令先送到
        self._call("/control", WorldControl(pause=True, multi_step=round(STEP_TIME / PHYS_DT)),
                   WorldControl)
        target = t0 + STEP_TIME - 0.01
        deadline = time.time() + 10.0
        while time.time() < deadline:
            rclpy.spin_once(self.node, timeout_sec=0.05)
            if (self.scan and self._stamp(self.scan) >= target
                    and self.odom and self._stamp(self.odom) >= target - 0.11):
                return
        self.node.get_logger().warn("等不到新的 /scan 或 /odom（bridge 有開嗎？）")

    def _teleport(self, x, y, yaw):
        p = Pose(name="my_robot")
        p.position.x, p.position.y, p.position.z = x, y, 0.0
        p.orientation.z, p.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
        self._call("/set_pose", p, Pose)

    # ---------- 位置估計：起點 + 里程計的相對位移 ----------
    def _pose(self):
        o = self.odom.pose.pose
        ox, oy, oyaw = self.odom0
        dx, dy = o.position.x - ox, o.position.y - oy
        # 把里程計的位移轉到「reset 那一刻車頭朝向」的座標系，再接到起點上
        c, s = math.cos(-oyaw), math.sin(-oyaw)
        lx, ly = c * dx - s * dy, s * dx + c * dy
        sx, sy, syaw = self.start
        c, s = math.cos(syaw), math.sin(syaw)
        return sx + c * lx - s * ly, sy + s * lx + c * ly, wrap(syaw + yaw_of(o.orientation) - oyaw)

    def _goal_rel(self):
        x, y, yaw = self._pose()
        gx, gy = self.route[self.wp]
        return math.hypot(gx - x, gy - y), wrap(math.atan2(gy - y, gx - x) - yaw)

    def _obs(self):
        scan, self.min_dist, _ = preprocess(self.scan, self.sectors)
        d, a = self._goal_rel()
        extra = [min(d / 3.0, 1.0), math.sin(a), math.cos(a), *self.last_action]
        return np.concatenate([scan, np.array(extra, dtype=np.float32)]).astype(np.float32)

    # ---------- Gymnasium 介面 ----------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        # 隨機從某一段路的開頭出發，這樣每個轉角都練得到
        leg = int(self.rng.integers(0, len(self.route) - 1)) if not (options or {}).get("from_start") else 0
        sx, sy = self.route[leg]
        nx, ny = self.route[leg + 1]
        if leg > 0:
            sx += self.rng.uniform(-0.1, 0.1)
            sy += self.rng.uniform(-0.1, 0.1)
        syaw = wrap(math.atan2(ny - sy, nx - sx) + self.rng.uniform(-0.6, 0.6))

        # 先停車 → 瞬移 → 再跑兩步讓車子站穩、感測器更新
        self._send_cmd(0, 0)
        self._advance()
        self._teleport(sx, sy, syaw)
        self._advance()
        self._advance()

        o = self.odom.pose.pose
        self.odom0 = (o.position.x, o.position.y, yaw_of(o.orientation))
        self.start = (sx, sy, syaw)
        self.wp = leg + 1
        self.steps = 0
        self.last_action = [0.0, 0.0]
        obs = self._obs()
        self.prev_d = self._goal_rel()[0]
        return obs, {"start_leg": leg}

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        v = (a[0] + 1) / 2 * MAX_V
        w = a[1] * MAX_W
        self._send_cmd(v, w)
        self._advance()
        self.steps += 1
        self.last_action = [float(a[0]), float(a[1])]
        obs = self._obs()

        d, _ = self._goal_rel()
        reward = 10.0 * (self.prev_d - d) - 0.01      # 靠近路徑點 = 加分；每一步小扣分（不要拖）
        if self.min_dist < 0.3:
            reward -= 0.1                              # 太貼牆扣一點
        self.prev_d = d
        terminated, success, collided = False, False, False

        if self.min_dist < self.collision_dist:
            reward -= 20.0
            terminated, collided = True, True
        elif d < self.goal_tol:
            if self.wp == len(self.route) - 1:
                reward += 20.0
                terminated, success = True, True
            else:
                reward += 5.0                          # 到了中途的路徑點 → 換下一個
                self.wp += 1
                obs = self._obs()
                self.prev_d = self._goal_rel()[0]
        truncated = self.steps >= self.max_steps and not terminated

        info = {"waypoint": self.wp, "min_dist": self.min_dist, "collision": collided}
        if terminated or truncated:
            info["is_success"] = success               # SB3 會自動算 success_rate
        return obs, reward, terminated, truncated, info

    def close(self):
        # 訓練結束：停車、讓模擬恢復正常速度繼續跑
        try:
            self._send_cmd(0, 0)
            self._call("/set_physics", Physics(max_step_size=PHYS_DT, real_time_factor=1.0), Physics)
            self._call("/control", WorldControl(pause=False), WorldControl)
        except RuntimeError:
            pass
        self.node.destroy_node()
