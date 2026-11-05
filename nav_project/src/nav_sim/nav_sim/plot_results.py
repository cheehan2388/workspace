"""
plot_results.py — 不用開 Gazebo / RViz 視窗也能看結果（全部畫成 PNG）

  ros2 run nav_sim plot_results <run 資料夾>     # 訓練曲線 → <run>/training.png
  （訓練結束會自動畫；train_ppo --play 跑完會自動畫路徑圖 → play.png）

打開圖片（WSL）：explorer.exe <png 路徑>
"""

import bisect
import json
import os
import subprocess
import sys
import tempfile

import matplotlib
matplotlib.use("Agg")   # 不開視窗，直接存檔
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np   # noqa: E402
import yaml   # noqa: E402

MAPS = os.path.expanduser("~/workspace/nav_project/maps")


def plot_training(run_dir):
    """monitor.csv（每個 episode 一行）→ 分數、成功率、長度三張曲線"""
    path = os.path.join(run_dir, "monitor.monitor.csv")
    rows = [line.strip().split(",") for line in open(path) if not line.startswith("#")][1:]
    if not rows:
        print("monitor.csv 還沒有資料")
        return None
    r = np.array([float(x[0]) for x in rows])
    n = np.array([int(x[1]) for x in rows])
    ok = np.array([x[3].strip() == "True" for x in rows], dtype=float)
    steps = np.cumsum(n)
    k = max(1, min(50, len(r) // 10))           # 移動平均：最近 k 個 episode
    avg = lambda a: np.convolve(a, np.ones(k) / k, mode="valid")
    x = steps[k - 1:]

    fig, ax = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    ax[0].plot(steps, r, ".", alpha=0.2, color="gray")
    ax[0].plot(x, avg(r), color="#0969da", lw=2)
    ax[0].set_ylabel("episode reward")
    ax[1].plot(x, avg(ok) * 100, color="#1a7f37", lw=2)
    ax[1].set_ylabel("success rate (%)")
    ax[1].set_ylim(-5, 105)
    ax[2].plot(x, avg(n.astype(float)), color="#bf8700", lw=2)
    ax[2].set_ylabel("episode length (steps)")
    ax[2].set_xlabel("training steps")
    for a in ax:
        a.grid(alpha=0.3)
    fig.suptitle(f"{os.path.basename(run_dir)} | {len(r)} episodes, {steps[-1]} steps | "
                 f"last {k} eps: success {avg(ok)[-1] * 100:.0f}%, reward {avg(r)[-1]:.1f}")
    fig.tight_layout()
    out = os.path.join(run_dir, "training.png")
    fig.savefig(out, dpi=90)
    plt.close(fig)
    print(f"訓練曲線：{out}")
    return out


class TruthRecorder:
    """在背景用 gz topic 錄機器人的真實位置（不能在 Python 裡直接訂閱，會跟 gz 服務呼叫卡死）"""

    def __init__(self, world):
        self.world = world
        self.file = tempfile.NamedTemporaryFile("w+", suffix=".jsonl", delete=False)
        self.proc = subprocess.Popen(
            ["gz", "topic", "-e", "-t", f"/world/{world}/pose/info", "--json-output"],
            stdout=self.file, text=True)

    def stop(self):
        """回傳依時間排好的 [(sim 時間, x, y), ...]"""
        self.proc.terminate()
        self.proc.wait()
        self.file.close()
        out = []
        for line in open(self.file.name):
            try:
                m = json.loads(line)
            except json.JSONDecodeError:
                continue
            h = m["header"]["stamp"]
            for p in m["pose"]:
                if p["name"] == "my_robot":
                    out.append((int(h.get("sec", 0)) + int(h.get("nsec", 0)) * 1e-9,
                                p["position"].get("x", 0.0), p["position"].get("y", 0.0)))
        os.unlink(self.file.name)
        return sorted(out)


def plot_paths(episodes, world, route, out):
    """episodes: [{"est": [(t, x, y)], "label": str, "success": bool, "truth": 對好時間的 (x, y)}]"""
    n = len(episodes)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows), squeeze=False)
    route = np.array(route)

    # 存好的 SLAM 地圖：原點是 odom 框架 = 機器人出生點（路徑第一個點、朝 +x）
    img, ext = None, None
    yml = os.path.join(MAPS, f"{world}.yaml")
    if os.path.exists(yml):
        mp = yaml.safe_load(open(yml))
        img = plt.imread(os.path.join(MAPS, mp["image"]))
        ox, oy = mp["origin"][0] + route[0][0], mp["origin"][1] + route[0][1]
        ext = [ox, ox + img.shape[1] * mp["resolution"], oy, oy + img.shape[0] * mp["resolution"]]

    for i, ep in enumerate(episodes):
        ax = axes[i // cols][i % cols]
        if img is not None:
            ax.imshow(img, cmap="gray", extent=ext, origin="upper", alpha=0.6)
        ax.plot(route[:, 0], route[:, 1], "o:", color="#888", ms=5, label="waypoints")
        est = np.array([(x, y) for _, x, y in ep["est"]])
        tr = ep.get("truth")
        if tr is not None and len(tr):
            ax.plot(tr[:, 0], tr[:, 1], "-", color="#1a7f37", lw=2.2, label="TRUE path")
            ax.plot(*tr[-1], "*" if ep["success"] else "X", color="black", ms=15)
        ax.plot(est[:, 0], est[:, 1], "--", color="#d1242f", lw=1.6, label="robot THINKS (odom)")
        ax.plot(*est[0], "s", color="#1a7f37", ms=9)
        # 標題用英文：matplotlib 預設字型沒有中文
        ax.set_title(f"Episode {i + 1}: {ep['label']} | {len(est)} steps ({len(est) * 0.2:.0f} s)",
                     fontsize=10)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
        if i == 0:
            ax.legend(loc="upper left", fontsize=7)
    for j in range(n, rows * cols):
        axes[j // cols][j % cols].axis("off")
    fig.tight_layout()
    fig.savefig(out, dpi=90)
    plt.close(fig)
    print(f"路徑圖：{out}")


def attach_truth(episodes, truth):
    """把錄到的真實位置，依 sim 時間對到每一步"""
    if not truth:
        return
    tt = [t for t, _, _ in truth]
    for ep in episodes:
        idx = [min(bisect.bisect_left(tt, t), len(tt) - 1) for t, _, _ in ep["est"]]
        ep["truth"] = np.array([truth[i][1:] for i in idx])


def main():
    if len(sys.argv) < 2:
        print("用法：ros2 run nav_sim plot_results <run 資料夾>")
        return
    plot_training(os.path.expanduser(sys.argv[1]))


if __name__ == "__main__":
    main()
