#!/bin/bash
# run_model.sh METHOD WORLD — 無頭跑 run_method.py（2D 訓練的模型）+ 錄影 + 成績，跑完自動收掉
#
# 先準備好（只做一次）：
#   sudo apt install ros-jazzy-turtlebot3-gazebo
#   colcon build --packages-select barn_gazebo && source install/setup.bash
#   啟用有 torch + stable_baselines3 的 venv（或用 PY=... 指定 python）
#
# 用法：
#   ./run_model.sh astar_ppo 0                         # 真實位置 + 錄影
#   ODOM=wheel ./run_model.sh astar_ppo 0              # 用輪速 /odom（會漂，見 README）
#   MODELS=v2 REC=0 ./run_model.sh hybrid 150          # v2 模型、不錄影
#
#   METHOD: ppo / astar_ppo / hybrid / dwa / astar_dwa（lstm 需要 sb3_contrib）
#   環境變數：
#     MODELS=v1|v2       模型版本（預設 v1）
#     SEED=0|1|2         訓練種子（預設 0）
#     ROBOT=turtlebot3|my_robot
#     ODOM=truth|wheel   控制和評分用的位置（預設 truth = Gazebo 真實位置）
#     REC=1|0            錄影（預設 1）
#     TRACE=1|0          把每一步的輸入/輸出/真實位置存成 trace.npz（預設 0）
#     OUT=<資料夾>       輸出位置（預設 ~/barn_runs/<方法>_<模型>_s<種子>_w<地圖>_<位置來源>）
#     PY=<python>        跑 run_method 的 python（預設 python3）
METHOD=${1:-astar_ppo}
WORLD=${2:-0}
MODELS=${MODELS:-v1}; SEED=${SEED:-0}; ROBOT=${ROBOT:-turtlebot3}
ODOM=${ODOM:-truth}; REC=${REC:-1}; TRACE=${TRACE:-0}; PY=${PY:-python3}
HERE=$(cd "$(dirname "$(readlink -f "$0")")" && pwd)
OUT=${OUT:-$HOME/barn_runs/${METHOD}_${MODELS}_s${SEED}_w${WORLD}_${ODOM}}
mkdir -p "$OUT"; rm -f "$OUT"/STOP "$OUT"/results.csv

ros2 pkg prefix barn_gazebo >/dev/null 2>&1 || { echo "找不到 barn_gazebo，先 source install/setup.bash"; exit 1; }
SHARE=$(ros2 pkg prefix barn_gazebo)/share/barn_gazebo
export TURTLEBOT3_MODEL=${TURTLEBOT3_MODEL:-burger}
STAMPED=true; [ "$ROBOT" = my_robot ] && STAMPED=false
GZNAME=turtlebot3; [ "$ROBOT" = my_robot ] && GZNAME=my_robot
REMAP=""; [ "$ODOM" = truth ] && REMAP="-r odom:=/odom_truth"

# 確保沒有殘留的模擬（兩個世界疊在一起會亂掉）
for p in $(pgrep -f "^gz sim"); do kill -9 $p; done; sleep 1

echo "[run] $METHOD（$MODELS, seed $SEED）@ BARN $WORLD，車 $ROBOT，位置 $ODOM → $OUT"
: > "$OUT/launch.log"
ros2 launch barn_gazebo barn.launch.py world:=$WORLD robot:=$ROBOT gui:=false eval:=false >> "$OUT/launch.log" 2>&1 &
LAUNCH=$!
for i in $(seq 1 60); do
  grep -q "Entity creation successful" "$OUT/launch.log" && gz topic -l 2>/dev/null | grep -q "^/scan$" && break
  sleep 0.5
done
grep -q "Entity creation successful" "$OUT/launch.log" || { echo "[run] 模擬沒起來，見 $OUT/launch.log"; kill $LAUNCH; exit 1; }

if [ "$REC" = 1 ]; then
  python3 "$HERE/gz_recorder.py" --world barn_$WORLD --view top   --out "$OUT/top.mp4"   --stop-file "$OUT/STOP" > "$OUT/rec_top.log" 2>&1 &
  python3 "$HERE/gz_recorder.py" --world barn_$WORLD --view chase --out "$OUT/chase.mp4" --stop-file "$OUT/STOP" > "$OUT/rec_chase.log" 2>&1 &
fi
TRUTH=""
if [ "$ODOM" = truth ]; then
  python3 "$HERE/odom_truth.py" --world barn_$WORLD --robot $GZNAME > "$OUT/odom_truth.log" 2>&1 &
  TRUTH=$!
fi
ros2 run barn_gazebo barn_eval --ros-args -p world:=$WORLD -p use_sim_time:=true \
    -p method:=${METHOD}_${MODELS}_s${SEED}_${ODOM} -p out:="$OUT/results.csv" $REMAP > "$OUT/eval.log" 2>&1 &
sleep 2

# run_method.py 原封不動；外面包一層讓 numpy 1.x 讀得了 numpy 2.x 存的模型（np2_compat.py）
if [ "$TRACE" = 1 ]; then WRAP=("$HERE/trace_method.py" "$OUT/trace.npz"); else WRAP=("$HERE/np2_compat.py"); fi
T0=$(date +%s)
TRACE_WORLD=barn_$WORLD TRACE_ROBOT=$GZNAME timeout 300 "$PY" "${WRAP[@]}" "$SHARE/scripts/run_method.py" \
    --ros-args -p method:=$METHOD -p world:=$WORLD -p use_sim_time:=true \
    -p models:=$MODELS -p train_seed:=$SEED -p cmd_stamped:=$STAMPED $REMAP > "$OUT/method.log" 2>&1
echo "[run] run_method 結束（exit $?，真實時間 $(( $(date +%s) - T0 )) s）"

sleep 2; touch "$OUT/STOP"; sleep 3
[ -n "$TRUTH" ] && kill $TRUTH 2>/dev/null
kill -INT $LAUNCH 2>/dev/null; sleep 3
for p in $(pgrep -f "^gz sim"); do kill -9 $p; done
for p in $(pgrep -f "ros_gz_bridge/parameter_bridge|robot_state_publisher|tf2_ros/static_transform|barn_gazebo/barn_eval"); do kill $p; done
wait 2>/dev/null
grep -hE "===" "$OUT/eval.log" || echo "[run] barn_eval 沒有結果（見 $OUT/eval.log）"
grep -hE "模型|結束|Traceback" "$OUT/method.log" | head -5
[ "$REC" = 1 ] && grep -h "結束" "$OUT"/rec_*.log
