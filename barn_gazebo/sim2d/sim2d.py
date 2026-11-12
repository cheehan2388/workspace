"""
sim2d.py — 所有方法共用的 2D 模擬核心

為什麼要共用：要比較 PPO、DWA、A*+PPO……最基本的公平，就是它們活在「同一個世界」：
同樣的牆、同樣的障礙物、同樣的光達、同樣的車子動力（加速度限制）。

這個世界有：
  - 牆（長方形，從 .sdf 讀進來）
  - 圓形障礙物：可以一開始就在、也可以「突然出現」、也可以「在移動」（模擬行人）
  - 光達：360 條射線，規格跟 Gazebo 一樣（0.12~3.5 m、雜訊 0.01、太近 -inf、太遠 inf）
  - 車子：差速驅動，速度上限與加速度限制照 Nav2 官方 TurtleBot3 模擬
"""

import copy
import math
import numpy as np

from nav2d_env import (walls_to_segments, wrap_angle, LIDAR_MIN, LIDAR_MAX, LIDAR_NOISE,
                       ROBOT_RADIUS, DT, GOAL_TOLERANCE)

# ---- 車子動力（跟 Gazebo 的 DiffDrive 設定一樣）----
V_MAX = 0.22      # 最大前進速度 m/s
W_MAX = 2.84      # 最大轉彎速度 rad/s
A_V = 1.0         # 前進加速度上限 m/s^2
A_W = 2.0         # 轉彎加速度上限 rad/s^2
SUBSTEPS = 4      # 每個 0.2 秒的決策，內部切 4 小步（每步 0.05 秒）來積分與檢查碰撞

N_SCAN = 360      # 光達射線數（跟 Gazebo 一樣）
MAX_STEPS = 600   # 一回合最多 600 個決策 = 120 秒


class Obstacle:
    """
    圓形障礙物。
      appear=None            一開始就在
      appear=("time", 秒)    模擬進行到第幾秒才出現
      appear=("dist", 公尺)  車子離出發點走了幾公尺才出現（C2 的「突然出現」用這個）
      vx, vy                 出現之後的移動速度（行人）
      stop_after=None        出現後走幾秒就停下（迎面行人「走來後停下」用這個）；None = 一直走
    """

    def __init__(self, x, y, r=0.2, vx=0.0, vy=0.0, appear=None, stop_after=None):
        self.x, self.y, self.r, self.vx, self.vy = x, y, r, vx, vy
        self.appear = appear
        self.active = appear is None
        self.stop_after = stop_after
        self.t_active = 0.0


class World:
    def __init__(self, walls, obstacles=()):
        self.walls = walls
        self.segs = walls_to_segments(walls)
        self.obstacles = [copy.copy(o) for o in obstacles]  # 每回合各自一份，才不會互相影響

    # ---------- 時間推進 ----------
    def advance(self, dt, t, dist_from_start):
        for o in self.obstacles:
            if not o.active:
                kind, val = o.appear
                if (kind == "time" and t >= val) or (kind == "dist" and dist_from_start >= val):
                    o.active = True
            if o.active:
                if o.stop_after is None or o.t_active < o.stop_after:
                    o.x += o.vx * dt
                    o.y += o.vy * dt
                o.t_active += dt

    def circles(self):
        c = [(o.x, o.y, o.r) for o in self.obstacles if o.active]
        return np.array(c, dtype=np.float64).reshape(-1, 3)

    # ---------- 幾何 ----------
    def raycast(self, x, y, yaw, angles):
        """回傳每條射線碰到牆或障礙物的距離（沒碰到 = inf）。"""
        a_world = yaw + angles
        dirs = np.stack([np.cos(a_world), np.sin(a_world)], axis=1)
        # 1) 牆：射線與線段求交點
        a = self.segs[:, :2]
        e = self.segs[:, 2:] - a
        ap = a - np.array([x, y])
        denom = dirs[:, None, 0] * e[None, :, 1] - dirs[:, None, 1] * e[None, :, 0]
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (ap[None, :, 0] * e[None, :, 1] - ap[None, :, 1] * e[None, :, 0]) / denom
            u = (ap[None, :, 0] * dirs[:, None, 1] - ap[None, :, 1] * dirs[:, None, 0]) / denom
        hit = (np.abs(denom) > 1e-12) & (t > 0) & (u >= 0) & (u <= 1)
        d = np.where(hit, t, np.inf).min(axis=1)
        # 2) 圓：射線與圓求交點
        circ = self.circles()
        if len(circ):
            oc = np.array([x, y]) - circ[:, :2]
            b = dirs @ oc.T
            c0 = (oc ** 2).sum(1) - circ[:, 2] ** 2
            disc = b * b - c0[None, :]
            with np.errstate(invalid="ignore"):
                tc = -b - np.sqrt(disc)
            tc = np.where((disc >= 0) & (tc > 0), tc, np.inf).min(axis=1)
            d = np.minimum(d, tc)
        return d

    def clearance(self, x, y):
        """車子中心到最近的牆面／障礙物表面的距離。小於車身半徑 = 撞到。"""
        a = self.segs[:, :2]
        e = self.segs[:, 2:] - a
        p = np.array([x, y])
        s = np.clip(((p - a) * e).sum(1) / (e * e).sum(1), 0, 1)
        closest = a + s[:, None] * e
        c = np.sqrt(((closest - p) ** 2).sum(1)).min()
        return min(c, self.obstacle_clearance(x, y))

    def obstacle_clearance(self, x, y):
        circ = self.circles()
        if len(circ) == 0:
            return np.inf
        return float((np.hypot(circ[:, 0] - x, circ[:, 1] - y) - circ[:, 2]).min())


class Sim:
    """一個回合。scene 是 scenes2d.make_scene() 產生的字典。"""

    def __init__(self, scene, seed=0, dynamics="accel"):
        assert dynamics in ("accel", "instant")
        self.dynamics = dynamics
        self.rng = np.random.default_rng(seed)
        self.world = World(scene["walls"], scene["obstacles"])
        self.x, self.y, self.yaw = scene["start"]
        self.x0, self.y0 = self.x, self.y
        self.goal = scene["goal"]
        self.v = self.w = 0.0
        self.t = 0.0
        self.steps = 0
        self.path_len = 0.0
        self.collided = False
        self.reached = False
        self.min_clear = np.inf       # 全程離「任何東西」最近的距離
        self.min_obs_clear = np.inf   # 全程離「圓形障礙物」最近的距離
        self.angles = np.linspace(0, 2 * math.pi, N_SCAN, endpoint=False)

    def scan(self):
        """光達：規格與 Gazebo 的 /scan 相同（太近 -inf、太遠 inf）。"""
        d = self.world.raycast(self.x, self.y, self.yaw, self.angles)
        d = d + self.rng.normal(0, LIDAR_NOISE, size=d.shape)
        d[d < LIDAR_MIN] = -np.inf
        d[d > LIDAR_MAX] = np.inf
        return d

    def step(self, v_cmd, w_cmd):
        v_cmd = float(np.clip(v_cmd, 0.0, V_MAX))
        w_cmd = float(np.clip(w_cmd, -W_MAX, W_MAX))
        h = DT / SUBSTEPS
        for _ in range(SUBSTEPS):
            if self.dynamics == "accel":
                self.v += float(np.clip(v_cmd - self.v, -A_V * h, A_V * h))
                self.w += float(np.clip(w_cmd - self.w, -A_W * h, A_W * h))
            else:
                self.v, self.w = v_cmd, w_cmd
            self.yaw = wrap_angle(self.yaw + self.w * h)
            dx, dy = self.v * math.cos(self.yaw) * h, self.v * math.sin(self.yaw) * h
            self.x += dx
            self.y += dy
            self.path_len += math.hypot(dx, dy)
            self.t += h
            self.world.advance(h, self.t, math.hypot(self.x - self.x0, self.y - self.y0))
            c = self.world.clearance(self.x, self.y)
            self.min_clear = min(self.min_clear, c)
            self.min_obs_clear = min(self.min_obs_clear, self.world.obstacle_clearance(self.x, self.y))
            if c < ROBOT_RADIUS:
                self.collided = True
                break
        self.steps += 1
        if math.hypot(self.goal[0] - self.x, self.goal[1] - self.y) < GOAL_TOLERANCE:
            self.reached = True

    @property
    def done(self):
        return self.collided or self.reached or self.steps >= MAX_STEPS


def hit_points(scan, x, y, yaw, angles, every=1):
    """把光達讀數轉成「障礙物打到的點」（世界座標）。-inf（太近）與 inf（沒打到）都不算。"""
    d = scan[::every]
    a = yaw + angles[::every]
    ok = np.isfinite(d)
    return np.stack([x + d[ok] * np.cos(a[ok]), y + d[ok] * np.sin(a[ok])], axis=1)


def run_episode(scene, pipeline, seed=0, dynamics="accel"):
    """跑一個回合，回傳結果與指標。pipeline 必須有 reset(scene) 和 act(...)。"""
    sim = Sim(scene, seed, dynamics)
    pipeline.reset(scene)
    traj, modes, ws = [(sim.x, sim.y)], [], []
    while not sim.done:
        scan = sim.scan()
        v, w, mode = pipeline.act(sim.x, sim.y, sim.yaw, sim.v, sim.w, scan)
        sim.step(v, w)
        traj.append((sim.x, sim.y))
        modes.append(mode)
        ws.append(sim.w)
    if sim.reached:
        outcome = "success"
    elif sim.collided:
        outcome = "collision"
    else:
        outcome = "timeout"
    ws = np.array(ws)
    return {
        "outcome": outcome,
        "time": sim.t,
        "path_len": sim.path_len,
        "mean_speed": sim.path_len / max(sim.t, 1e-9),
        "smooth": float(np.abs(np.diff(ws)).mean()) if len(ws) > 1 else 0.0,   # 平均 |Δω|，越小越平順
        "min_clear": sim.min_clear,
        "min_obs_clear": sim.min_obs_clear,
        "ppo_frac": float(np.mean([m == "ppo" for m in modes])) if modes else 0.0,
        "traj": traj,
    }
