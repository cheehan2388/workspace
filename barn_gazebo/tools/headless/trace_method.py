"""trace_method.py — 跟 np2_compat.py 一樣執行 run_method.py，但把 Pipeline2.act 每一步的輸入／輸出存成 npz

  python3 trace_method.py <trace.npz> <run_method.py> --ros-args ...
"""
import atexit
import os
import runpy
import sys
import time

import numpy as np

import np2_compat  # noqa: F401  （numpy 2 模型相容）

out = sys.argv[1]
script = sys.argv[2]
sys.argv = sys.argv[2:]

from ament_index_python.packages import get_package_share_directory  # noqa: E402
sys.path.insert(0, os.path.join(get_package_share_directory("barn_gazebo"), "sim2d"))
import planners2  # noqa: E402

rec = {"t": [], "pose": [], "vw": [], "scan": [], "cmd": [], "mode": [], "target": [], "wall": [], "truth": []}

# Gazebo 真實位置（ground truth）：直接聽 gz 的 dynamic_pose/info，不經過 odom
import math  # noqa: E402
from gz.msgs10.pose_v_pb2 import Pose_V  # noqa: E402
from gz.transport13 import Node as GzNode  # noqa: E402
_truth = [np.nan, np.nan, np.nan]
_robot = os.environ.get("TRACE_ROBOT", "turtlebot3")


def _on_pose(msg):
    for p in msg.pose:
        if p.name == _robot:
            q = p.orientation
            _truth[:] = [p.position.x, p.position.y,
                         math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))]


_gz = GzNode()
_gz.subscribe(Pose_V, f"/world/{os.environ.get('TRACE_WORLD', 'barn_0')}/dynamic_pose/info", _on_pose)
_orig_act = planners2.Pipeline2.act
_orig_target = planners2.Pipeline._target


def _target(self, x, y, hits):
    res = _orig_target(self, x, y, hits)      # (目標點, 路點們)
    self._last_target = res[0]
    return res


def act(self, x, y, yaw, v, w, scan):
    v_cmd, w_cmd, mode = _orig_act(self, x, y, yaw, v, w, scan)
    rec["t"].append(len(rec["t"]) * 0.2)
    rec["wall"].append(time.time())
    rec["pose"].append((x, y, yaw))
    rec["vw"].append((v, w))
    rec["scan"].append(np.array(scan, dtype=float))
    rec["cmd"].append((v_cmd, w_cmd))
    rec["mode"].append(str(mode))
    rec["truth"].append(tuple(_truth))
    tg = getattr(self, "_last_target", None)
    rec["target"].append(tuple(tg) if tg is not None else (np.nan, np.nan))
    return v_cmd, w_cmd, mode


planners2.Pipeline2.act = act
planners2.Pipeline._target = _target


@atexit.register
def _save():
    np.savez(out, **{k: np.array(v) for k, v in rec.items()})
    print(f"[trace] {len(rec['t'])} 步 → {out}", flush=True)


runpy.run_path(script, run_name="__main__")
