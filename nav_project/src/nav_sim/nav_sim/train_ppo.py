"""
train_ppo.py — 用 PPO 訓練機器人走 S 形走廊

先開模擬（另一個終端機）：
  ros2 launch nav_sim sim.launch.py gui:=false slam:=false rviz:=false

訓練：
  ros2 run nav_sim train_ppo                       # 預設 100k 步（約 1.5~2.5 小時）
  ros2 run nav_sim train_ppo --steps 20000         # 先跑短的看看
  ros2 run nav_sim train_ppo --resume <model.zip>  # 從存檔繼續練

看結果（不用開 Gazebo / RViz 視窗，全部存成 PNG）：
  訓練結束（或 Ctrl+C）會自動畫 <run>/training.png：分數、成功率、episode 長度
  訓練到一半想看：ros2 run nav_sim plot_results ~/workspace/nav_project/rl_runs/<時間>
  也可以用 tensorboard --logdir ~/workspace/nav_project/rl_runs → http://localhost:6006

測試練好的模型（從起點走完整條路），跑完畫 play.png：真實路徑 vs 里程計路徑，疊在地圖上：
  ros2 run nav_sim train_ppo --play ~/workspace/nav_project/rl_runs/<時間>/final.zip
"""

import argparse
import os
import time

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from nav_sim.nav_env import NavEnv
from nav_sim.plot_results import TruthRecorder, attach_truth, plot_paths, plot_training

RUNS = os.path.expanduser("~/workspace/nav_project/rl_runs")


def train(args):
    run_dir = os.path.join(RUNS, time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    # Monitor：記錄每個 episode 的總分、長度、有沒有成功（存成 monitor.csv）
    env = Monitor(NavEnv(world=args.world), os.path.join(run_dir, "monitor"),
                  info_keywords=("is_success",))

    if args.resume:
        model = PPO.load(args.resume, env=env, tensorboard_log=run_dir)
    else:
        model = PPO(
            "MlpPolicy", env,
            n_steps=1024,          # 收集 1024 步再更新一次網路
            batch_size=128,
            learning_rate=3e-4,
            gamma=0.99,            # 多在乎未來的分數（0.99 ≈ 看 100 步 = 20 秒）
            ent_coef=0.005,        # 鼓勵一點點探索
            policy_kwargs=dict(net_arch=[128, 128]),
            tensorboard_log=run_dir,   # 每次訓練的 log、存檔都放在同一個資料夾
            verbose=1,
        )

    ckpt = CheckpointCallback(save_freq=10_000, save_path=run_dir, name_prefix="ppo")
    try:
        model.learn(total_timesteps=args.steps, callback=ckpt,
                    tb_log_name="tb", progress_bar=True,
                    reset_num_timesteps=not args.resume)
    except KeyboardInterrupt:
        print("\n中斷訓練，先存檔 ...")
    finally:
        model.save(os.path.join(run_dir, "final"))
        print(f"模型存在：{run_dir}/final.zip")
        env.close()
        plot_training(run_dir)


def play(args):
    env = NavEnv(world=args.world)
    model = PPO.load(args.play)
    truth = TruthRecorder(args.world)          # 背景錄真實位置，最後拿來畫圖
    episodes = []
    try:
        for ep in range(args.episodes):
            obs, _ = env.reset(options={"from_start": True})
            total, done, est = 0.0, False, []
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, r, term, trunc, info = env.step(action)
                x, y, _ = env._pose()
                est.append((env._stamp(env.scan), x, y))
                total += r
                done = term or trunc
            result = "✅ 到終點" if info["is_success"] else ("💥 撞牆" if info["collision"] else "⏰ 超時")
            label = "SUCCESS" if info["is_success"] else ("COLLISION" if info["collision"] else "TIMEOUT")
            episodes.append({"est": est, "label": label, "success": info["is_success"]})
            print(f"第 {ep + 1} 次：{result} | 走到路徑點 {info['waypoint']}/{len(env.route) - 1} "
                  f"| {env.steps} 步（{env.steps * 0.2:.0f} 秒）| 總分 {total:.1f}")
        env._advance()                         # 多跑一步，讓最後的位置也被錄到
    finally:
        env.close()
        attach_truth(episodes, truth.stop())
        if episodes:
            out = os.path.join(os.path.dirname(os.path.abspath(args.play)), "play.png")
            plot_paths(episodes, args.world, env.route, out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--world", default="c1_serpentine")
    p.add_argument("--steps", type=int, default=100_000)
    p.add_argument("--resume", help="從這個 .zip 繼續訓練")
    p.add_argument("--play", help="不訓練，直接跑這個 .zip 模型")
    p.add_argument("--episodes", type=int, default=5)
    args, _ = p.parse_known_args()   # 忽略 ros2 run 可能塞進來的 --ros-args
    play(args) if args.play else train(args)


if __name__ == "__main__":
    main()
