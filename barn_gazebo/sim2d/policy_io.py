"""
policy_io.py — PPO 的「眼睛」：把光達讀數整理成 27 個數字

2D 模擬、Gazebo 都用這一個函式，保證兩邊餵給 PPO 的東西完全一樣。
（不依賴 ROS，所以在哪裡都能 import。）
"""

import math
import numpy as np

from nav2d_env import clean_scan, wrap_angle, LIDAR_MAX, N_RAYS


def make_obs(ranges, x, y, yaw, goal, last_w):
    """
    組出跟 Nav2DEnv._obs() 完全一樣的 27 個數字。
    ranges：光達 360 個讀數（第 0 條在正前方，逆時針，每 1 度一條）
    goal  ：PPO 現在要去的點（可以是終點，也可以是 A* 路徑上的路點）
    回傳 (obs, 離 goal 的距離)
    """
    step = len(ranges) // N_RAYS                      # 360 // 24 = 15
    scan = clean_scan(ranges[::step][:N_RAYS]) / LIDAR_MAX
    dx, dy = goal[0] - x, goal[1] - y
    dist = math.hypot(dx, dy)
    angle = wrap_angle(math.atan2(dy, dx) - yaw)
    extra = [min(dist / 8.0, 1.0), angle / math.pi, last_w / 1.5]
    return np.concatenate([scan, extra]).astype(np.float32), dist
