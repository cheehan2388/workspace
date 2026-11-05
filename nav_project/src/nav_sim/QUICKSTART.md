# nav_sim 快速開始（下次打開看這份）

## 1. 編譯（改過檔案後才需要）
```bash
cd ~/workspace/nav_project
colcon build --symlink-install --packages-select nav_sim
source install/setup.bash
```

## 2. 一個指令開全部（Gazebo + 橋 + TF + SLAM + RViz）
```bash
source ~/workspace/nav_project/install/setup.bash
ros2 launch nav_sim sim.launch.py                 # 有 Gazebo 視窗
ros2 launch nav_sim sim.launch.py gui:=false      # 無頭模式（不開 Gazebo 視窗）
ros2 launch nav_sim sim.launch.py world:=c2_sudden.sdf
ros2 launch nav_sim sim.launch.py slam:=false rviz:=false   # 只開模擬
```
- SLAM 會自動 configure → activate，**不用再打 `ros2 lifecycle set`**
- RViz 自動載入 `rviz/slam.rviz`（Map /map、LaserScan /scan、TF），用模擬時間

## 3. 開車（另一個終端機）
```bash
ros2 run teleop_twist_keyboard teleop_twist_keyboard
```
- `k` 或空白鍵 = 停車；`,` = 倒車；`x` / `c` = 降低速度
- 慢慢開、不要撞牆（無頭模式看不到碰撞，看 RViz 裡的掃描點）

## 4. 存地圖
```bash
ros2 run nav2_map_server map_saver_cli -f ~/workspace/nav_project/maps/my_map
```

## ⚠️ 規則（踩過的坑）
| 規則 | 原因 |
|---|---|
| 關程式一律 **Ctrl+C**，不要 Ctrl+Z | Ctrl+Z 只是暫停，Gazebo server 還在跑 → 兩個模擬同時發 `/scan`、`/clock`，SLAM 爛掉 |
| 一次只開一個 launch | 同上 |
| RViz 要 `use_sim_time:=true` | 不然掃描會因為「時間太舊」被丟掉 |
| 不要卡在牆上一直開 | 輪子空轉 → 輪式里程計以為還在走 → SLAM 把同一面牆貼在很多角度 → 星星狀地圖 |
| 地圖爛了就全部重開 | slam_toolbox 沒辦法把已經貼錯的掃描拿掉 |

## 🔍 出問題時的檢查指令
```bash
# 有沒有殘留的舊程序？（應該只有一組 gz sim / parameter_bridge）
ps aux | grep -E "gz sim|parameter_bridge|slam|rviz2" | grep -v grep
# 全部清掉
pkill -9 -f "gz sim"; pkill -9 -f parameter_bridge; pkill -f async_slam_toolbox_node; pkill -f rviz2

ros2 topic hz /scan                     # 光達應該 ~5 Hz
ros2 lifecycle get /slam_toolbox        # 應該是 active
ros2 run tf2_ros tf2_echo map base_link # TF 有沒有通

# 里程計 vs Gazebo 真實位置（差很多 = 打滑 / 卡住）
ros2 topic echo /odom --once --field pose.pose.position
gz topic -e -t /world/c1_serpentine/dynamic_pose/info -n 1 | grep -A5 'name: "my_robot"'
```

## 📁 檔案在哪
| 檔案 | 用途 |
|---|---|
| `launch/sim.launch.py` | 主 launch（參數：world、gui、slam、rviz） |
| `config/slam_params.yaml` | SLAM 參數（base_frame: base_link、max_laser_range: 3.5） |
| `rviz/slam.rviz` | RViz 畫面設定 |
| `models/my_robot/model.sdf` | 機器人（差速驅動、gpu_lidar 360°、0.12–3.5 m） |
| `worlds/*.sdf` | 場景：c1_serpentine、c2_sudden、u_trap |

## 🧠 RL 訓練（PPO，c1_serpentine）
```bash
# 終端機 1：只開模擬（RL 不用 SLAM / RViz）
ros2 launch nav_sim sim.launch.py gui:=false slam:=false rviz:=false
# 終端機 2：訓練（100k 步約 2 小時；Ctrl+C 會先存檔再停）
ros2 run nav_sim train_ppo
ros2 run nav_sim train_ppo --steps 20000            # 先跑短的
ros2 run nav_sim train_ppo --resume <run>/final.zip # 接著練
# 測試：從起點跑完整條路 → 畫 play.png（真實路徑 vs 里程計，疊在地圖上）
ros2 run nav_sim train_ppo --play ~/workspace/nav_project/rl_runs/<時間>/final.zip
# 訓練到一半想看曲線 → training.png
ros2 run nav_sim plot_results ~/workspace/nav_project/rl_runs/<時間>
# 打開圖片
explorer.exe ~/workspace/nav_project/rl_runs/<時間>/training.png
```
- ⚠️ 電腦**不能同時開 Gazebo 視窗 + RViz**（會當機）→ 訓練一律 `gui:=false rviz:=false`，結果看 PNG
- 訓練結束（或 Ctrl+C）自動畫 `training.png`：分數、成功率、episode 長度
- 環境在 `nav_sim/nav_env.py`：Gazebo 會被**暫停**，每步跑 0.2 秒（約 3.8 倍速）；訓練中 Gazebo 看起來「卡住」是正常的
- 觀測 29 維：光達 24 扇區 + 路徑點距離/方向 + 上一步動作；動作：前進 0~0.22 m/s、轉彎 ±1.5 rad/s
- 路徑點手寫在 `ROUTES`（S 形走廊轉角）；撞牆 = 光達 < 0.18 m
- 位置用輪式里程計 → 長距離會漂 15~30 cm；漂太多再加 `odom:=truth`
- 改完 Python 用 `nb` 編譯（`python3 -m colcon`，`ros2 run` 才會用 venv 裡的 torch）
