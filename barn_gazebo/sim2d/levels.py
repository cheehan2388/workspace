"""
levels.py — 「過關挑戰」的測試關卡（照 proposal 的關卡表）

每一關都是 make_level(name, rng) 產生的字典：walls（長方形牆）、obstacles（圓形障礙物／行人）、
start（x, y, 朝向）、goal（x, y）。rng 決定「這一回合的小變化」（起點、行人位置、時機各抖一點），
同一個亂數種子下，所有方法面對的是完全一樣的關卡（配對比較）。

重要：這些關卡只拿來「考試」。訓練用的是 train_env2.py 隨機產生的地圖，
不會出現這幾張固定的關卡地圖；BARN 更是完全沒看過。

關卡（由易到難）：
  u05 / u15 / u25  U 形死路，深度 0.5 / 1.5 / 2.5 m          考：沒有地圖的 PPO
  c1               蜿蜒：走廊 1.0 m、門 0.6 m、一個 180° 彎    考：平順度、Hybrid 誤切換
  c2               突現障礙：走 1.5 m 後前方 1.0 m 冒出圓柱     考：DWA、A* 不知道的東西
  c3               迎面行人：0.8 m/s 走來後停下                 考：DWA
  c4               橫越行人：0.8 m/s 在前方約 2 m 處橫越        考：Hybrid（行人進入路線前不會切換）
  barn             大魔王：BARN 資料集 50 張地圖（編號 0、6、…、294），從沒訓練過
"""

import math
import os
import numpy as np

from sim2d import Obstacle

T = 0.1   # 牆厚


def box(x0, x1, y0, y1):
    """四面牆圍成的長方形房間（內側邊界 x0..x1, y0..y1）。"""
    return [((x0 + x1) / 2, y1 + T / 2, x1 - x0 + 2 * T, T), ((x0 + x1) / 2, y0 - T / 2, x1 - x0 + 2 * T, T),
            (x1 + T / 2, (y0 + y1) / 2, T, y1 - y0), (x0 - T / 2, (y0 + y1) / 2, T, y1 - y0)]


# ---------------------------------------------------------------------------
# U 形死路：8 × 8 m 房間，U 內寬 3 m、開口朝南（朝向起點），深度 d
# ---------------------------------------------------------------------------
def u_trap_walls(depth, width=3.0, y_open=0.0):
    hw = width / 2
    return box(-4, 4, -4, 4) + [
        (0.0, y_open + depth + T / 2, width + 2 * T, T),          # 底
        (-hw - T / 2, y_open + depth / 2, T, depth),              # 左
        (hw + T / 2, y_open + depth / 2, T, depth),               # 右
    ]


# ---------------------------------------------------------------------------
# C1 蜿蜒：兩條 1.0 m 寬的走廊，一個 180° 彎；下方走廊中間有一道 0.6 m 的門
#   外框 x 0..4.2、y 0..2.1；中間隔牆 y≈1.05，從 x=0 到 3.2；門在 x=1.6
#   路徑約 3.2 + 彎 + 3.2 ≈ 7.5–8 m
# ---------------------------------------------------------------------------
C1_WALLS = box(0.0, 4.2, 0.0, 2.1) + [
    (1.6, 1.05, 3.2, T),        # 中間隔牆
    (1.6, 0.1, T, 0.2),         # 門（下半）：留出 y 0.2..0.8 = 0.6 m
    (1.6, 0.9, T, 0.2),         # 門（上半）
]

# C2 突現障礙：走廊寬 2.0 m、長 8 m
C2_WALLS = box(0.0, 8.0, -1.0, 1.0)
# C3 迎面行人：寬 4 m、長 7 m
C3_WALLS = box(0.0, 7.0, -2.0, 2.0)
# C4 橫越行人：寬 3 m、長 6 m
C4_WALLS = box(0.0, 6.0, -1.5, 1.5)

PED_R = 0.25        # 行人＝直徑 0.5 m 的圓柱
PED_SPEED = 0.8     # m/s（proposal 原版設定；可用 --ped-speed 改）

# ---------------------------------------------------------------------------
# BARN：每張地圖是 30 × 64 的格子（每格 0.15 m 的圓柱）。barn_maps.npz 由原始 .world 轉來。
# 原始設定：起點 (-2.25, 3) 朝北，終點在起點前方 10 m。
# 注意：BARN 是為 Jackal（約 0.51 × 0.43 m）設計的，本研究的車（TurtleBot3，半徑 0.11 m）小很多，
#       同一張地圖對小車比較寬鬆。結果只用來比較本研究的各方法，不和 BARN 競賽排行榜比較。
# ---------------------------------------------------------------------------
_BARN = None
BARN_CELL = 0.15


def _barn():
    global _BARN
    if _BARN is None:
        here = os.path.dirname(os.path.abspath(__file__))
        d = np.load(os.path.join(here, "barn_maps.npz"))
        _BARN = (d["grids"], d["index"])
    return _BARN


def barn_walls(k):
    """第 k 張（0..49）地圖：把每一列連在一起的格子合併成一面長方形牆，再加一個外框讓 A* 有邊界。"""
    grids, _ = _barn()
    G = grids[k]
    walls = []
    for iy in range(G.shape[0]):
        row = G[iy]
        ix = 0
        while ix < row.size:
            if row[ix]:
                j = ix
                while j + 1 < row.size and row[j + 1]:
                    j += 1
                # 格子 ix 的中心在 x = -(0.075 + 0.15 ix)；ix 越大越往西
                x_east = -(BARN_CELL * ix)
                x_west = -(BARN_CELL * (j + 1))
                walls.append(((x_east + x_west) / 2, BARN_CELL / 2 + BARN_CELL * iy, x_east - x_west, BARN_CELL))
                ix = j + 1
            else:
                ix += 1
    return walls + box(-6.5, 2.0, -0.3, 14.5)


BARN_N = 50
LEVELS = ["u05", "u15", "u25", "c1", "c2", "c3", "c4", "barn"]
LEVEL_NAMES = {
    "u05": "U 形死路 深 0.5 m", "u15": "U 形死路 深 1.5 m", "u25": "U 形死路 深 2.5 m",
    "c1": "C1 蜿蜒", "c2": "C2 突現障礙", "c3": "C3 迎面行人", "c4": "C4 橫越行人",
    "barn": "大魔王 BARN（50 張）",
}


def make_level(name, rng, episode=0, ped_speed=PED_SPEED):
    j = lambda s: float(rng.uniform(-s, s))
    swap = bool(rng.random() < 0.5)          # C1、C2：一半回合起終點對調

    if name in ("u05", "u15", "u25"):
        depth = {"u05": 0.5, "u15": 1.5, "u25": 2.5}[name]
        return dict(name=name, walls=u_trap_walls(depth), obstacles=[],
                    start=(j(0.5), -3.0 + j(0.2), math.pi / 2 + j(0.2)),
                    goal=(j(0.5), 3.4 + j(0.1)))

    if name == "c1":
        a = (0.45 + j(0.1), 0.5 + j(0.1))     # 下方走廊西端
        b = (0.45, 1.6)                        # 上方走廊西端
        s, g = (a, b) if not swap else (b, a)
        yaw = 0.0 + j(0.2)                     # 兩個起點都朝東（往彎的方向）
        return dict(name=name, walls=C1_WALLS, obstacles=[], start=(s[0], s[1], yaw), goal=g)

    if name == "c2":
        sy = j(0.2)
        oy = sy + j(0.1)          # 圓柱一定擋在車子的直線路上（偏差不到 0.1 m，不閃就會撞）
        if not swap:   # 由西往東
            start, goal, ox = (0.5, sy, j(0.1)), (7.5, 0.0), 3.0 + j(0.1)
        else:          # 由東往西
            start, goal, ox = (7.5, sy, math.pi + j(0.1)), (0.5, 0.0), 5.0 + j(0.1)
        # 車子離出發點 1.5 m 時，前方約 1.0 m 處冒出直徑 0.4 m 的圓柱（A* 不知道）
        return dict(name=name, walls=C2_WALLS, start=start, goal=goal,
                    obstacles=[Obstacle(ox, oy, 0.2, appear=("dist", 1.5))])

    if name == "c3":
        # 行人從車子前方 5 m 迎面走來，走 3 秒（約 2.4 m）後停下
        sy = j(0.2)
        py = sy + j(0.1)          # 行人正對著車子走來（偏差不到 0.1 m，不閃就會撞）
        walk = 3.0 * PED_SPEED / ped_speed if ped_speed > 0 else 0.0   # 不管速度多少，都停在差不多的位置
        return dict(name=name, walls=C3_WALLS, start=(0.5, sy, j(0.1)), goal=(6.5, 0.0),
                    obstacles=[Obstacle(5.5, py, PED_R, vx=-ped_speed, stop_after=walk)])

    if name == "c4":
        # 行人從南牆邊往北橫越 x≈3 的位置。
        # 原版寫「在機器人前方 2 m 處交會」，但這樣行人在車子到達前早就走過去，
        # 試跑時 DWA 不用反應就 20/20 過關，等於沒考到。所以改成：
        # 以車子全速 0.22 m/s 估算，行人走到車子的路線上時，車子離交會點約 0.8 m（必須停或閃）。
        xc = 3.0 + j(0.3)
        trig = (xc - 0.5) - (0.8 + 0.22 * 1.25 / ped_speed) + j(0.2)
        cross = 2.5 / ped_speed if ped_speed > 0 else 0.0     # 走到北牆邊就停
        return dict(name=name, walls=C4_WALLS, start=(0.5, j(0.2), j(0.1)), goal=(5.5, 0.0),
                    obstacles=[Obstacle(xc, -1.25, PED_R, vy=ped_speed, appear=("dist", trig), stop_after=cross)])

    if name == "barn":
        k = episode % BARN_N
        return dict(name=name, barn_map=k, walls=barn_walls(k), obstacles=[],
                    start=(-2.25 + j(0.1), 3.0 + j(0.1), math.pi / 2 + j(0.1)), goal=(-2.25, 13.0))

    raise ValueError(name)
