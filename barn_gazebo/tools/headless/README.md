# tools/headless：無頭跑模型 + 錄影 + 用真實位置評分

不開 Gazebo 視窗，一個指令跑完一回合：開地圖 → 用 `scripts/run_method.py` 載入模型開車 → `barn_eval` 記成績 → 錄成 mp4 → 自動關掉。

```bash
sudo apt install ros-jazzy-turtlebot3-gazebo        # 只做一次
source install/setup.bash                            # barn_gazebo 編譯好的 workspace
# 啟用有 torch + stable_baselines3 的 venv
./run_model.sh astar_ppo 0                           # → ~/barn_runs/astar_ppo_v1_s0_w0_truth/
```

參數說明在 `run_model.sh` 開頭。輸出資料夾內容：

| 檔案 | 內容 |
|---|---|
| `results.csv` | barn_eval 成績（同 `~/barn_results.csv` 格式） |
| `top.mp4`、`chase.mp4` | 正上方、斜後方兩個角度的錄影（H.264，10 fps = 模擬時間 1 倍速） |
| `*_first.png`、`*_last.png` | 第一張、最後一張畫面 |
| `trace.npz` | `TRACE=1` 時：每一步的 odom 位置、真實位置、光達、指令、A* 目標點 |

## 檔案

| 檔案 | 用途 |
|---|---|
| `run_model.sh` | 一鍵跑一回合 |
| `gz_recorder.py` | 無頭錄影：用 EntityFactory 動態加一台 camera sensor 進世界（不用改 .sdf），存成 mp4 |
| `odom_truth.py` | 把 Gazebo 真實位置發成 `/odom_truth`（座標系跟 `/odom` 相同），remap 後 run_method / barn_eval 不用改 |
| `np2_compat.py` | 讓 numpy 1.x 讀得了 numpy 2.x 存的 SB3 模型 |
| `trace_method.py` | 除錯：記錄 `Pipeline2.act` 每一步的輸入/輸出 + 真實位置 |

## 發現的兩個問題

### 1. 模型在 ROS 環境載不進來（`No module named 'numpy._core.numeric'`）

模型在 Windows 用 numpy 2.4.6 存檔；ROS Jazzy 的 venv 要 numpy < 2（cv_bridge 等是用 numpy 1.x 編的）。
SB3 zip 裡有四個欄位是 numpy 2 的 pickle，numpy 1 讀不了：
`observation_space`、`action_space`、`_last_obs`、`_last_episode_starts`。

`np2_compat.py` 在 `PPO.load` 時用 `custom_objects` 直接給這四個（`Box(-1, 1, (28,))`、`Discrete(15)`、`None`、`None`），
跳過那些 pickle。神經網路權重（`policy.pth`）是 torch 格式，不受影響，推論結果不變。

### 2. TurtleBot3 Burger 的 `/odom` 轉彎時會嚴重漂移

Burger 在 Gazebo 轉彎時輪子打滑，輪速里程計把轉彎角度多算很多。實測 `astar_ppo` @ BARN 0：

| 時間 | odom 認為 | Gazebo 真實 | 差距 |
|---|---|---|---|
| 8 s（第一次轉彎後） | 朝向 98.6° | 93.7° | 4.9° |
| 12 s | (-2.61, 5.53) 99.5° | (-2.34, 5.59) 84.4° | 0.28 m、15° |
| 15.6 s | (-2.77, 6.18) 115.8° | (-2.04, 6.16) 59.6° | **0.73 m、56°** |

影響：
- run_method 用錯的位置追 A* 路點 → 走歪。
- barn_eval 也用 odom 判斷碰撞 → 在錯的位置判定「碰撞」。上面這回合 barn_eval 判碰撞，
  但用真實位置算，車子離圓柱最近還有 0.53 m，**其實沒撞**。

建議：**評分一律用真實位置**（`ODOM=truth`）；控制器用真實位置或輪速 odom 當成實驗變因，兩種都報告。
2D 實驗用的是完美位置，要跟 2D 比較就用 `ODOM=truth`。

## 結果（BARN 0，TurtleBot3 Burger，v1 模型種子 0）

| 方法 | 位置 | Gazebo | 2D 實驗同一張圖 |
|---|---|---|---|
| astar_ppo | 輪速 odom | 碰撞 15.7 s（odom 漂移造成的假碰撞） | 10/10 成功 |
| astar_ppo | 真實位置 | 成功 44.3 s，路徑 9.63 m，最近圓柱 0.209 m | 10/10 成功 |
| hybrid | 真實位置 | 成功 43.4 s，最近圓柱 0.228 m | — |
| astar_dwa | 真實位置 | 成功 43.4 s，最近圓柱 0.227 m | 10/10 成功 |
| ppo | 真實位置 | 碰撞 29.9 s | 1/30 成功 |

錄影在 repo 根目錄的 `vid/`。

## 其他

- 開兩台相機時模擬速度約為真實時間的 0.4~0.5 倍；批次跑很多張圖時用 `REC=0`。
- 沒有螢幕（`DISPLAY` 沒設）時 `gz sim -s` 光達也能正常輸出；相機在沒有螢幕時沒測過。
- 打滑的根本原因（輪子/萬向輪摩擦參數）還沒查。
