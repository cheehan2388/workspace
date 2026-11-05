#!/bin/bash
# setup_rl_env.sh — 建立 RL 用的 Python 環境（只要跑一次）
#   bash ~/workspace/nav_project/setup_rl_env.sh
set -e
cd ~/workspace/nav_project

# 1+2. 建 venv：--system-site-packages 讓 rclpy、ROS 的 numpy 都看得到
#      Ubuntu 24.04 沒有 ensurepip，所以先建沒有 pip 的 venv，再用官方 get-pip.py 補上（不需要 sudo）
python3 -m venv --system-site-packages --without-pip .venv
touch .venv/COLCON_IGNORE          # 不要讓 colcon 掃進 venv 裡
source .venv/bin/activate
curl -sSL https://bootstrap.pypa.io/get-pip.py | python

# 3. 鎖住 numpy < 2：ROS Jazzy 的 cv_bridge 等是用 numpy 1.x 編的，升到 2 會壞
echo "numpy<2" > .venv/constraints.txt
pip install -c .venv/constraints.txt torch --index-url https://download.pytorch.org/whl/cpu   # 沒 GPU → CPU 版
pip install -c .venv/constraints.txt gymnasium stable-baselines3 tensorboard tqdm rich

# 4. 重新編譯：用 venv 的 python 跑 colcon，ros2 run 的腳本才會用 venv 的 python
source /opt/ros/jazzy/setup.bash
python -m colcon build --symlink-install --packages-select nav_sim

python -c "import torch, gymnasium, stable_baselines3, numpy, rclpy; \
print('✅ torch', torch.__version__, '| gymnasium', gymnasium.__version__, \
'| sb3', stable_baselines3.__version__, '| numpy', numpy.__version__, '| rclpy OK')"
echo "完成！開一個新的終端機（或 source ~/.bashrc）就會自動載入。"
