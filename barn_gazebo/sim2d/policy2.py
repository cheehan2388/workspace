"""
policy2.py — 第二版 PPO 的「眼睛」和「手」（不依賴 ROS，2D 與 Gazebo 共用）

跟第一版（policy_io.py、5 個動作、固定 0.15 m/s）的差別：
  1. 手：15 個動作 = 3 種前進速度（0、0.11、0.22 m/s）× 5 種轉彎速度。
     可以停下來等行人，最高速度也跟 DWA 一樣是 0.22 m/s，比較才公平。
  2. 眼睛：24 個方向，每個方向取「15 度扇形裡最近的距離」（第一版只取扇形中間那一條）。
     BARN 的圓柱只有 15 cm 粗，只看中間那條射線會漏掉。
  3. 多一個數字：目前的前進速度。共 28 個數字。
"""

import math
import numpy as np

from nav2d_env import clean_scan, wrap_angle, LIDAR_MAX, N_RAYS

V_CHOICES = (0.0, 0.11, 0.22)
W_CHOICES = (-1.5, -0.75, 0.0, 0.75, 1.5)
ACTIONS = [(v, w) for v in V_CHOICES for w in W_CHOICES]      # 15 個
OBS_DIM = N_RAYS + 4


def make_obs2(ranges, x, y, yaw, target, last_v, last_w):
    """ranges：光達 360 個讀數（第 0 條在正前方、逆時針、每度一條）。target：要去的點（終點或 A* 路點）。"""
    r = clean_scan(ranges)
    k = len(r) // N_RAYS
    # 扇形以每個方向為中心：第 i 個方向 = i*15 度，涵蓋 -7..+7 度
    rolled = np.roll(r, k // 2)
    sectors = rolled[:k * N_RAYS].reshape(N_RAYS, k).min(axis=1) / LIDAR_MAX
    dx, dy = target[0] - x, target[1] - y
    dist = math.hypot(dx, dy)
    angle = wrap_angle(math.atan2(dy, dx) - yaw)
    extra = [min(dist / 8.0, 1.0), angle / math.pi, last_v / 0.22, last_w / 1.5]
    return np.concatenate([sectors, extra]).astype(np.float32), dist
