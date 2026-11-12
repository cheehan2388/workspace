"""
nav2d_env.py — 照 Gazebo 場景做的 2D 導航環境（給 PPO 訓練用）

為什麼要 2D：Gazebo 在筆電上大約 1 倍速，PPO 要練幾十萬步會等太久。
這個 2D 版本用同樣的房間、U 形死路、TurtleBot3 速度與光達規格，
但每秒能跑上千步。訓練好之後再搬回 Gazebo 驗證。

座標慣例跟 ROS 一樣：x 往前（東）、y 往左（北），角度逆時針為正，單位是弧度。
"""

import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces


# ---------------------------------------------------------------------
# 1. 場景：跟 u_trap.sdf 一模一樣
#    每面牆寫成「中心 x、中心 y、x 方向長度、y 方向長度」，就是 SDF 的 <pose> 和 <size>
# ---------------------------------------------------------------------
WALLS = [
    # 房間四面牆（8 x 8 公尺）
    (0.0, 4.0, 8.0, 0.1),    # wall_north
    (0.0, -4.0, 8.0, 0.1),   # wall_south
    (4.0, 0.0, 0.1, 8.0),    # wall_east
    (-4.0, 0.0, 0.1, 8.0),   # wall_west
    # U 形死路（開口朝南）
    (0.0, 1.5, 3.1, 0.1),    # u_back
    (-1.5, 1.0, 0.1, 1.0),   # u_left
    (1.5, 1.0, 0.1, 1.0),    # u_right
]


def walls_to_segments(walls):
    """把每面牆（長方形）拆成 4 條邊。光達射線和碰撞檢查都是對這些邊計算。"""
    segs = []
    for cx, cy, sx, sy in walls:
        x0, x1 = cx - sx / 2, cx + sx / 2
        y0, y1 = cy - sy / 2, cy + sy / 2
        corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        for i in range(4):
            segs.append((*corners[i], *corners[(i + 1) % 4]))
    return np.array(segs, dtype=np.float64)  # 每列：ax, ay, bx, by


# ---------------------------------------------------------------------
# 2. 機器人與光達規格：照 TurtleBot3 Burger 與 robot_test.sdf
# ---------------------------------------------------------------------
ROBOT_RADIUS = 0.11       # 車子外圍半徑（公尺），用來判斷有沒有撞牆
LINEAR_SPEED = 0.15       # 固定前進速度（m/s），低於 Burger 上限 0.22
ANGULAR_CHOICES = [-1.5, -0.75, 0.0, 0.75, 1.5]  # 5 種轉彎速度（rad/s），正 = 左轉
DT = 0.2                  # 每一步 0.2 秒 = 光達每秒 5 次

LIDAR_MIN = 0.12          # 光達最近量測距離（比這更近 → 真實光達回傳 -inf）
LIDAR_MAX = 3.5           # 光達最遠量測距離（比這更遠 → 真實光達回傳 inf）
LIDAR_NOISE = 0.01        # 雜訊標準差，跟 SDF 的 <stddev> 一樣
N_RAYS = 24               # 360 條射線降採樣成 24 條（每 15 度一條）

GOAL_TOLERANCE = 0.25     # 離目標 25 公分內算到達
MAX_STEPS = 500           # 一回合最多 500 步 = 100 秒


def clean_scan(ranges):
    """
    把光達讀數整理成神經網路能用的數字。
    你在 Gazebo 看過：太近是 -inf、太遠是 inf。神經網路不能吃無限大，所以：
      -inf（太近）→ 0
       inf（太遠）→ LIDAR_MAX
    之後搬到 Gazebo 時，同一個函式可以直接用在真正的 /scan 上。
    """
    r = np.array(ranges, dtype=np.float32)
    r[np.isneginf(r)] = 0.0
    r[np.isposinf(r) | np.isnan(r)] = LIDAR_MAX
    return np.clip(r, 0.0, LIDAR_MAX)


def wrap_angle(a):
    """把角度整理到 -π 到 π 之間（跟你在 turtlesim 寫的 normalize 一樣）。"""
    return (a + math.pi) % (2 * math.pi) - math.pi


class Nav2DEnv(gym.Env):
    """
    觀測（observation，共 27 個數字，都整理到大約 0~1 或 -1~1）：
      - 24 條光達距離 / 3.5
      - 到目標的距離 / 8
      - 目標在車子的哪個方向 / π（0 = 正前方，正 = 偏左）
      - 上一步的轉彎速度 / 1.5
    動作（action）：5 選 1，對應 ANGULAR_CHOICES 的 5 種轉彎速度，前進速度固定。
    """

    metadata = {"render_modes": []}

    def __init__(self, walls=WALLS, fixed_start=None, fixed_goal=None):
        super().__init__()
        self.segs = walls_to_segments(walls)
        self.fixed_start = fixed_start  # (x, y, yaw)，評估時用；None = 隨機
        self.fixed_goal = fixed_goal    # (x, y)，None = 隨機
        self.ray_angles = np.linspace(0, 2 * math.pi, N_RAYS, endpoint=False)

        self.action_space = spaces.Discrete(len(ANGULAR_CHOICES))
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(N_RAYS + 3,), dtype=np.float32
        )

    # ---------- 幾何計算 ----------
    def _raycast(self, x, y, yaw):
        """從車子位置射出 N_RAYS 條射線，回傳每條碰到牆的距離。"""
        dirs = np.stack([np.cos(yaw + self.ray_angles), np.sin(yaw + self.ray_angles)], axis=1)
        a = self.segs[:, :2]
        e = self.segs[:, 2:] - a
        ap = a - np.array([x, y])
        # 解「射線 = 牆邊」：p + t·d = a + u·e
        denom = dirs[:, None, 0] * e[None, :, 1] - dirs[:, None, 1] * e[None, :, 0]
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (ap[None, :, 0] * e[None, :, 1] - ap[None, :, 1] * e[None, :, 0]) / denom
            u = (ap[None, :, 0] * dirs[:, None, 1] - ap[None, :, 1] * dirs[:, None, 0]) / denom
        hit = (np.abs(denom) > 1e-12) & (t > 0) & (u >= 0) & (u <= 1)
        t = np.where(hit, t, np.inf)
        return t.min(axis=1)

    def _clearance(self, x, y):
        """車子中心到最近牆邊的距離。小於車身半徑 = 撞到了。"""
        a = self.segs[:, :2]
        e = self.segs[:, 2:] - a
        p = np.array([x, y])
        s = np.clip(((p - a) * e).sum(1) / (e * e).sum(1), 0, 1)
        closest = a + s[:, None] * e
        return np.sqrt(((closest - p) ** 2).sum(1)).min()

    def _sample_free_point(self, margin=0.35):
        while True:
            x, y = self.np_random.uniform(-3.6, 3.6, size=2)
            if self._clearance(x, y) > margin:
                return float(x), float(y)

    # ---------- 光達模擬：加雜訊、套用量測範圍 ----------
    def _lidar(self):
        d = self._raycast(self.x, self.y, self.yaw)
        d = d + self.np_random.normal(0, LIDAR_NOISE, size=d.shape)
        d[d < LIDAR_MIN] = -np.inf   # 太近：模仿真實光達回傳 -inf
        d[d > LIDAR_MAX] = np.inf    # 太遠：模仿真實光達回傳 inf
        return d

    def _goal_dist_angle(self):
        dx, dy = self.gx - self.x, self.gy - self.y
        dist = math.hypot(dx, dy)
        angle = wrap_angle(math.atan2(dy, dx) - self.yaw)  # 目標相對於車頭的角度
        return dist, angle

    def _obs(self):
        scan = clean_scan(self._lidar()) / LIDAR_MAX
        dist, angle = self._goal_dist_angle()
        extra = [min(dist / 8.0, 1.0), angle / math.pi, self.last_w / 1.5]
        return np.concatenate([scan, extra]).astype(np.float32)

    # ---------- Gymnasium 的兩個核心函式 ----------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.fixed_start is not None:
            self.x, self.y, self.yaw = self.fixed_start
        else:
            self.x, self.y = self._sample_free_point()
            self.yaw = float(self.np_random.uniform(-math.pi, math.pi))
        if self.fixed_goal is not None:
            self.gx, self.gy = self.fixed_goal
        else:
            while True:  # 目標至少離起點 1 公尺
                self.gx, self.gy = self._sample_free_point()
                if math.hypot(self.gx - self.x, self.gy - self.y) > 1.0:
                    break
        self.steps = 0
        self.last_w = 0.0
        self.prev_dist, _ = self._goal_dist_angle()
        self.trajectory = [(self.x, self.y)]
        return self._obs(), {}

    def step(self, action):
        w = ANGULAR_CHOICES[int(action)]
        # 差速驅動的運動公式（跟 Gazebo 的 DiffDrive 同樣的概念）
        self.yaw = wrap_angle(self.yaw + w * DT)
        self.x += LINEAR_SPEED * math.cos(self.yaw) * DT
        self.y += LINEAR_SPEED * math.sin(self.yaw) * DT
        self.last_w = w
        self.steps += 1
        self.trajectory.append((self.x, self.y))

        dist, _ = self._goal_dist_angle()
        collided = self._clearance(self.x, self.y) < ROBOT_RADIUS
        reached = dist < GOAL_TOLERANCE

        # ----- reward：照計畫的四條規則 -----
        reward = 10.0 * (self.prev_dist - dist)   # 1. 靠近目標給正分，遠離扣分
        reward -= 0.01                             # 2. 每一步小扣分，鼓勵快一點到
        if collided:
            reward -= 20.0                         # 3. 碰撞大懲罰
        if reached:
            reward += 20.0                         # 4. 到達大獎勵
        self.prev_dist = dist

        terminated = collided or reached
        truncated = self.steps >= MAX_STEPS        # 時間到（例如卡在 U 裡面）
        info = {"is_success": reached, "collided": collided}
        return self._obs(), reward, terminated, truncated, info


# U 形死路測試：起點在 U 的正南方，終點在 U 的後面（跟 robot_test.sdf 的起點、終點一樣）
U_TRAP_START = (0.0, -3.0, math.pi / 2)
U_TRAP_GOAL = (0.0, 3.0)
