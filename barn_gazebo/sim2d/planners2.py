"""
planners2.py — 「過關挑戰」的六種方法（用第二版 PPO：15 個動作、28 個觀測值）

  ppo        無 A*  + PPO（目標 = 終點）             模型 ppo_goal
  lstm       無 A*  + PPO+LSTM（有記憶，目標 = 終點）模型 lstm_goal
  astar_ppo  A*     + PPO（目標 = A* 路點）          模型 ppo_waypoint
  dwa        無 A*  + DWA
  astar_dwa  A*     + DWA
  hybrid     A*     + 平時 DWA、路被擋連續 3 次改用 PPO（Hybrid 論文的規則）  模型 ppo_waypoint

A*、路點、DWA、切換規則都跟第一版（planners.py）完全一樣，只換掉 PPO 這個零件。
"""

import numpy as np

import planners
from planners import Pipeline
from policy2 import ACTIONS, make_obs2

METHODS2 = {
    # 名稱: (用 A*?, 局部規劃器, 需要哪一種模型)
    "ppo": (False, "ppo", "ppo_goal"),
    "lstm": (False, "ppo", "lstm_goal"),
    "astar_ppo": (True, "ppo", "ppo_waypoint"),
    "dwa": (False, "dwa", None),
    "astar_dwa": (True, "dwa", None),
    "hybrid": (True, "hybrid", "ppo_waypoint"),
}


class PPOLocal2:
    def __init__(self, model, recurrent=False):
        self.model = model
        self.recurrent = recurrent
        self.reset()

    def reset(self):
        self.last_v = self.last_w = 0.0
        self.state = None
        self.start = np.array([True])

    def act(self, x, y, yaw, scan, target):
        obs, _ = make_obs2(scan, x, y, yaw, target, self.last_v, self.last_w)
        obs = np.clip(obs, -1.0, 1.0)
        if self.recurrent:
            a, self.state = self.model.predict(obs, state=self.state, episode_start=self.start, deterministic=True)
            self.start = np.array([False])
        else:
            a, _ = self.model.predict(obs, deterministic=True)
        return ACTIONS[int(a)]


class Pipeline2(Pipeline):
    def __init__(self, method, model=None, replan=False):
        use_astar, local, kind = METHODS2[method]
        planners.METHODS.setdefault(method, (use_astar, local))   # 讓父類別認得 "lstm"
        super().__init__(method, ppo_model=None, replan=replan)   # replan：像 Nav2 每秒重新規劃（預設關，同 Hybrid 論文）
        if local in ("ppo", "hybrid"):
            self.ppo = PPOLocal2(model, recurrent=(kind == "lstm_goal"))

    def reset(self, scene):
        super().reset(scene)
        if self.ppo:
            self.ppo.reset()

    def act(self, x, y, yaw, v, w, scan):
        v_cmd, w_cmd, mode = super().act(x, y, yaw, v, w, scan)
        if self.ppo:
            self.ppo.last_v = float(v_cmd)   # 不管是 PPO 還是 DWA 下的指令，都讓 PPO 知道「上一步」
        return v_cmd, w_cmd, mode
