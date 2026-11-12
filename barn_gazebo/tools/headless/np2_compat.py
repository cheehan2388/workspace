"""np2_compat.py — 在 numpy 1.x（ROS Jazzy 的環境）載入 numpy 2.x 存的 SB3 模型

問題：模型在 Windows（numpy 2.4.6）存檔，zip 裡的 observation_space、action_space、lr_schedule、
clip_range 是 numpy 2 的 pickle，numpy 1.26 讀不了（numpy._core 不存在、亂數產生器格式不同）。
做法：載入時用 custom_objects 直接給這幾個物件，完全跳過那些 pickle。
神經網路權重（policy.pth）是 torch 格式，不受影響，所以推論結果跟 Windows 上一樣。

用法：
  python3 np2_compat.py <要執行的腳本.py> [參數...]     # 先打補丁，再照常執行那支腳本
"""
import runpy
import sys

import numpy as np
from gymnasium import spaces

# 2D 實驗 nav2d_env.py 的定義：觀測 28 維 [-1, 1]，動作 15 種角速度
# （逐欄測過：只有這四個是 numpy 2 pickle 讀不了；lr_schedule、clip_range 等照原檔讀）
CUSTOM = {
    "observation_space": spaces.Box(-1.0, 1.0, shape=(28,), dtype=np.float32),
    "action_space": spaces.Discrete(15),
    "_last_obs": None,                 # 訓練中途的暫存狀態，推論用不到
    "_last_episode_starts": None,
}


def _patch(cls):
    orig = cls.load.__func__

    def load(klass, path, *args, custom_objects=None, **kw):
        co = dict(CUSTOM)
        co.update(custom_objects or {})
        return orig(klass, path, *args, custom_objects=co, **kw)

    cls.load = classmethod(load)


if int(np.__version__.split(".")[0]) < 2:
    from stable_baselines3 import PPO
    _patch(PPO)
    try:
        from sb3_contrib import RecurrentPPO
        _patch(RecurrentPPO)
    except ImportError:
        pass

if __name__ == "__main__" and len(sys.argv) > 1:
    script = sys.argv[1]
    sys.argv = sys.argv[1:]
    runpy.run_path(script, run_name="__main__")
