#!/usr/bin/env bash
# run_barn_batch.sh — 自動跑多張 BARN 地圖 × 多種方法（不開畫面），成績都寫進 ~/barn_results.csv
#
# 用法（先 source ROS 和 workspace，再啟用裝了 PPO 套件的 venv）：
#   ./run_barn_batch.sh                                   全部 50 張 × 5 種方法（約 50×5×2 分鐘，很久）
#   ./run_barn_batch.sh "astar_dwa hybrid" "0 6 12"       指定方法和地圖
# 環境變數：MODELS（v1 = 原獎勵，預設；v2 = ω 平順獎勵；或模型資料夾完整路徑）
#           ROBOT（turtlebot3 或 my_robot，預設 turtlebot3）、REPLAN（true/false，預設 false）

METHODS=${1:-"astar_dwa hybrid astar_ppo dwa ppo"}
WORLDS=${2:-$(seq 0 6 294)}
MODELS=${MODELS:-v1}
ROBOT=${ROBOT:-turtlebot3}
REPLAN=${REPLAN:-false}
STAMPED=true; [ "$ROBOT" = "my_robot" ] && STAMPED=false
HERE=$(dirname "$(readlink -f "$0")")
TAG=""; [ "$REPLAN" = "true" ] && TAG="_replan"; [ "$MODELS" != "v1" ] && TAG="${TAG}_$(basename $MODELS)"

for w in $WORLDS; do
  for m in $METHODS; do
    echo "=== $(date +%H:%M:%S)  BARN $w  方法 $m$TAG"
    setsid ros2 launch barn_gazebo barn.launch.py world:=$w robot:=$ROBOT method:=$m$TAG gui:=false \
      > /tmp/barn_launch_${w}_${m}.log 2>&1 &
    LP=$!
    sleep 10                                   # 等 Gazebo 和車子就位
    timeout 200 python3 "$HERE/run_method.py" --ros-args -p method:=$m -p world:=$w -p use_sim_time:=true \
      -p models:=$MODELS -p replan:=$REPLAN -p cmd_stamped:=$STAMPED
    sleep 2
    kill -INT -- -$LP 2>/dev/null; sleep 3; kill -9 -- -$LP 2>/dev/null
    pkill -f "gz sim" 2>/dev/null; sleep 2     # 確保 Gazebo 關乾淨，下一回合才不會兩個世界疊在一起
  done
done
echo "全部完成，成績在 ~/barn_results.csv"
