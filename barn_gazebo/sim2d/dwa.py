"""
dwa.py — 局部規劃：Dynamic Window Approach（動態窗）

每 0.2 秒做一次：
  1. 動態窗：只考慮「馬達加速度限制下，下一瞬間做得到」的 (v, ω) 組合
  2. 每個組合往前預演 1.5 秒（假設速度不變，走出一小段弧線）
  3. 預演會撞到東西的 → 淘汰；來不及煞車的 → 淘汰
  4. 剩下的用公式打分，選最高分：
        G = α·heading + β·clearance + γ·velocity
        heading   ：預演終點時，車頭對不對著目標（1 = 正對，0 = 反方向）
        clearance ：預演途中離障礙物多遠（越遠越高，超過 CAP 就滿分）
        velocity  ：跑多快（越快越高）
  α、β、γ 是人調的權重。每個數字都看得懂，所以行為可以預測、可以解釋。

障礙物資訊來自光達打到的點（跟 PPO 用同一顆光達），不偷看地圖。
"""

import math
import numpy as np

from nav2d_env import wrap_angle, ROBOT_RADIUS, DT
from sim2d import V_MAX, W_MAX, A_V, A_W

HORIZON = 1.5          # 預演幾秒
PRED_DT = 0.1          # 預演的時間間隔
N_V, N_W = 5, 17       # 動態窗內取樣幾個速度、幾個轉彎速度
ALPHA, BETA, GAMMA, DELTA = 0.5, 0.4, 0.5, 3.0   # DELTA（貼著路徑走）最大：跟 Nav2 DWB 預設「路徑相關評分項權重最大」的精神一致
CLEAR_CAP = 0.4        # 障礙物離軌跡超過這個距離就不影響分數（遠處的牆不該讓車子不敢走）
PATH_CAP = 0.6         # 離全域路徑超過這個距離，路徑分數就是 0
SAFETY = 0.01          # 硬規則：離障礙物小於「車身半徑 + 這個值」就算會撞。想離牆遠一點是 clearance 分數的工作


def dist_to_polyline(points, poly):
    """每個點到折線的最短距離。points (m,2)，poly (k,2)。"""
    a, b = poly[:-1], poly[1:]
    ab = b - a
    denom = np.maximum((ab * ab).sum(1), 1e-12)
    ap = points[:, None, :] - a[None, :, :]
    s = np.clip((ap * ab[None]).sum(2) / denom[None], 0, 1)
    closest = a[None] + s[..., None] * ab[None]
    return np.sqrt(((points[:, None, :] - closest) ** 2).sum(2)).min(axis=1)


class DWA:
    def __init__(self, v_max=V_MAX, alpha=ALPHA, beta=BETA, gamma=GAMMA, delta=DELTA):
        self.v_max, self.alpha, self.beta, self.gamma, self.delta = v_max, alpha, beta, gamma, delta
        self.t = np.arange(1, int(HORIZON / PRED_DT) + 1) * PRED_DT   # 預演的時間點

    def act(self, x, y, yaw, v, w, hits, target, path=None):
        """
        hits  ：光達打到的點 (m, 2)，世界座標
        target：要去的點（終點，或 A* 路徑上的路點）
        path  ：（選用）全域路徑折線 (k, 2)。有給就多一項「軌跡終點離路徑多遠」的分數，
                讓車子貼著路徑走，不會為了直奔目標而削到牆角（Nav2 DWB 的 PathDist 評分項）
        回傳 (v_cmd, w_cmd)
        """
        # ---- 1. 動態窗 ----
        vs = np.linspace(max(0.0, v - A_V * DT), min(self.v_max, v + A_V * DT), N_V)
        ws = np.linspace(max(-W_MAX, w - A_W * DT), min(W_MAX, w + A_W * DT), N_W)
        VV, WW = np.meshgrid(vs, ws)
        VV, WW = VV.ravel(), WW.ravel()                                  # (C,)

        # ---- 2. 預演：等速度弧線的封閉解 ----
        t = self.t[None, :]                                              # (1, S)
        th = yaw + WW[:, None] * t                                       # (C, S)
        small = np.abs(WW[:, None]) < 1e-6
        safe_w = np.where(small, 1.0, WW[:, None])
        px = np.where(small, x + VV[:, None] * t * math.cos(yaw),
                      x + VV[:, None] / safe_w * (np.sin(th) - math.sin(yaw)))
        py = np.where(small, y + VV[:, None] * t * math.sin(yaw),
                      y + VV[:, None] / safe_w * (math.cos(yaw) - np.cos(th)))

        # ---- 3. 離障礙物多遠 ----
        if len(hits):
            near = hits[np.hypot(hits[:, 0] - x, hits[:, 1] - y) < self.v_max * HORIZON + 1.0]
        else:
            near = hits
        if len(near):
            d = np.hypot(px[:, :, None] - near[None, None, :, 0], py[:, :, None] - near[None, None, :, 1])
            clear = d.min(axis=(1, 2))                                   # (C,)
        else:
            clear = np.full(len(VV), np.inf)
        margin = ROBOT_RADIUS + SAFETY
        # 預演不會撞。（經典 DWA 還會檢查「來得及煞車」，但這台車 0.22 m/s、加速度 1 m/s²，
        # 煞車距離只有 v²/2a = 2.4 cm，遠小於預演的 33 cm，所以這項檢查可以省略。）
        ok = clear > margin

        # ---- 4. 打分 ----
        ang = np.abs(wrap_angle(np.arctan2(target[1] - py[:, -1], target[0] - px[:, -1]) - th[:, -1]))
        heading = 1.0 - ang / math.pi
        clearance = np.minimum(clear, CLEAR_CAP) / CLEAR_CAP
        velocity = VV / self.v_max
        score = self.alpha * heading + self.beta * clearance + self.gamma * velocity
        if path is not None:
            d_path = dist_to_polyline(np.stack([px[:, -1], py[:, -1]], axis=1), path)
            score = score + self.delta * (1.0 - np.minimum(d_path, PATH_CAP) / PATH_CAP)
        score = np.where(ok, score, -np.inf)

        if not np.isfinite(score).any():
            # 全部都會撞：原地轉向目標（脫困）
            a = wrap_angle(math.atan2(target[1] - y, target[0] - x) - yaw)
            return 0.0, float(np.clip(2.0 * a, -1.0, 1.0))
        k = int(score.argmax())
        return float(VV[k]), float(WW[k])
