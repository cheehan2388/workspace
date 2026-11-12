"""
planners.py — 把「全域規劃」和「局部規劃」組合成不同的方法

兩個零件：
  全域（global）：none（直接朝終點）或 A*（先規劃一條路徑，沿路取路點當臨時目標）
  局部（local） ：PPO、DWA，或 hybrid（依規則在 DWA 與 PPO 之間切換）

五種方法（METHODS）：
  ppo        = 無 + PPO          →  純 PPO（沒有 A*）
  astar_ppo  = A* + PPO          →  你的主要方法
  dwa        = 無 + DWA          →  純 DWA（沒有 A*）
  astar_dwa  = A* + DWA          →  傳統做法（Nav2 的 A* + DWB 的簡化版）
  hybrid     = A* + 切換         →  Hybrid 論文的切換規則（局部規劃器換成你自己的 PPO）

Hybrid 切換規則（Nokia Bell Labs 2024，論文有寫的部分）：
  1. 取全域路徑上的 8 個路點
  2. 車子 → 路點1 → 路點2 → … → 路點8 連成折線，檢查折線附近有沒有光達打到的障礙物
  3. 連續 3 次檢查（5 Hz，約 0.6 秒）都被擋 → 用 RL；否則用 DWA
  論文沒寫的細節（路點間隔、安全距離、DWA 權重）是我的選擇，已標在常數旁邊。
"""

from collections import deque
import math
import numpy as np

from astar import GridMap, WP_COUNT
from dwa import DWA, dist_to_polyline
from nav2d_env import LINEAR_SPEED, ANGULAR_CHOICES, ROBOT_RADIUS, N_RAYS
from policy_io import make_obs
from sim2d import hit_points, N_SCAN

LOOKAHEAD_INDEX = 1        # 臨時目標 = 第 2 個路點（沿路徑前方 1.0 m）。我的選擇
WP_BLOCK_DIST = ROBOT_RADIUS + 0.05    # 路點離障礙物小於這個距離 → 這個路點算「在障礙物上」
POLY_BLOCK_DIST = ROBOT_RADIUS + 0.01  # 折線離障礙物小於這個距離 → 算「被擋」。要小於 A* 的膨脹 0.20，否則貼牆的路會一直被判被擋
SEG_BLOCK_DIST = ROBOT_RADIUS + 0.01  # 車子到路點的直線離障礙物小於這個距離 → 直線走不過去
REPLAN_EVERY = 5          # 重新規劃的頻率：每 5 個決策 = 1 秒（Nav2 預設）
SWITCH_N = 3               # 連續幾次被擋才切到 RL（論文：3）

METHODS = {
    # 名稱: (用 A*?, 局部規劃器)
    "ppo": (False, "ppo"),
    "astar_ppo": (True, "ppo"),
    "dwa": (False, "dwa"),
    "astar_dwa": (True, "dwa"),
    "hybrid": (True, "hybrid"),
}


class PPOLocal:
    def __init__(self, model):
        self.model = model
        self.last_w = 0.0

    def act(self, x, y, yaw, scan, target):
        obs, _ = make_obs(scan, x, y, yaw, target, self.last_w)
        action, _ = self.model.predict(obs, deterministic=True)
        return LINEAR_SPEED, ANGULAR_CHOICES[int(action)]


class Pipeline:
    """一個完整的導航方法：reset(scene) 之後，每 0.2 秒呼叫一次 act(...)。"""

    def __init__(self, method, ppo_model=None, dwa_vmax=None, replan=False):
        self.method = method
        self.replan = replan and METHODS[method][0]   # 只有用 A* 的方法才有「重新規劃」
        self.use_astar, self.local = METHODS[method]
        self.ppo = PPOLocal(ppo_model) if self.local in ("ppo", "hybrid") else None
        self.dwa = DWA(**({"v_max": dwa_vmax} if dwa_vmax else {})) if self.local in ("dwa", "hybrid") else None

    def reset(self, scene):
        self.goal = scene["goal"]
        self.path = None
        if self.use_astar:
            # A* 只看得到地圖上的牆，看不到突然出現或會移動的障礙物
            self.grid = GridMap(scene["walls"])
            self.path = self.grid.plan(scene["start"][:2], self.goal)
            if self.path is None:
                raise RuntimeError(f"A* 找不到路徑：{scene['name']}")
        self.blocked_hist = deque(maxlen=SWITCH_N)
        self.k = 0
        if self.ppo:
            self.ppo.last_w = 0.0
        self.angles = np.linspace(0, 2 * math.pi, N_SCAN, endpoint=False)

    def _target(self, x, y, hits):
        """
        臨時目標（給 PPO 或 DWA 的「目標」）：
          1. 從最近的路點開始，跳過「落在障礙物上」的路點，取第一個不在障礙物上的
             （這是 Hybrid 論文的規則：first non-obstacle waypoint）
          2. 再往前看到 1.0 m 內：只要路點本身沒被擋、車子到它的直線也走得過去，就選更遠的
             （路點在牆角另一側時，朝它直走會削到牆角，所以不能選）
        """
        if self.path is None:
            return self.goal, None
        wps = self.path.waypoints(x, y)
        here = np.array([[x, y]])

        def on_obstacle(wp):
            return len(hits) > 0 and np.hypot(hits[:, 0] - wp[0], hits[:, 1] - wp[1]).min() <= WP_BLOCK_DIST

        def line_clear(wp):
            return len(hits) == 0 or dist_to_polyline(hits, np.vstack([here, wp[None]])).min() > SEG_BLOCK_DIST

        i0 = next((i for i, wp in enumerate(wps) if not on_obstacle(wp)), len(wps) - 1)
        pick = wps[i0]
        for j in range(i0 + 1, LOOKAHEAD_INDEX + 1):
            if on_obstacle(wps[j]) or not line_clear(wps[j]):
                break
            pick = wps[j]
        return tuple(pick), wps

    def act(self, x, y, yaw, v, w, scan):
        hits = hit_points(scan, x, y, yaw, self.angles)
        if self.replan and self.k % REPLAN_EVERY == 0:
            # 像 Nav2 預設那樣每秒重新規劃一次，並把光達看到的東西當成障礙物
            new = self.grid.plan((x, y), self.goal, extra=self.grid.mark_hits(hits))
            if new is not None:
                self.path = new
        self.k += 1
        target, wps = self._target(x, y, hits)

        if self.local == "ppo":
            mode = "ppo"
        elif self.local == "dwa":
            mode = "dwa"
        else:  # hybrid：看折線有沒有被擋
            blocked = False
            if len(hits):
                poly = np.vstack([[x, y], wps])
                blocked = bool(dist_to_polyline(hits, poly).min() < POLY_BLOCK_DIST)
            self.blocked_hist.append(blocked)
            all_blocked = len(self.blocked_hist) == SWITCH_N and all(self.blocked_hist)
            mode = "ppo" if all_blocked else "dwa"

        if mode == "ppo":
            v_cmd, w_cmd = self.ppo.act(x, y, yaw, scan, target)
        else:
            ref = self.path.ahead(x, y) if self.path is not None else None   # A* 真正的路徑
            v_cmd, w_cmd = self.dwa.act(x, y, yaw, v, w, hits, target, path=ref)
        if self.ppo:
            self.ppo.last_w = float(np.clip(w_cmd, -1.5, 1.5))   # PPO 的「上一步轉彎速度」，不管是誰下的指令
        return v_cmd, w_cmd, mode
