"""gz_recorder.py — 無頭錄影：在正在跑的世界裡生一台相機（camera sensor），把畫面存成 mp4

  python3 gz_recorder.py --world barn_0 --out run.mp4 [--stop-file STOP] [--max-sec 150]

相機是用 EntityFactory 動態加進世界的，不用改 .sdf；靠世界裡已有的 Sensors system 渲染
（無頭模式用 EGL / headless rendering 也能動）。
"""
import argparse
import os
import threading
import time

import cv2
import numpy as np
from gz.msgs10.boolean_pb2 import Boolean
from gz.msgs10.entity_factory_pb2 import EntityFactory
from gz.msgs10.image_pb2 import Image
from gz.transport13 import Node

CAM_SDF = """<?xml version="1.0"?>
<sdf version="1.8">
  <model name="{name}">
    <static>true</static>
    <pose>{x} {y} {z} {roll} {pitch} {yaw}</pose>
    <link name="link">
      <sensor name="cam" type="camera">
        <topic>{topic}</topic>
        <update_rate>{rate}</update_rate>
        <always_on>true</always_on>
        <camera>
          <horizontal_fov>{hfov}</horizontal_fov>
          <image><width>{w}</width><height>{h}</height><format>R8G8B8</format></image>
          <clip><near>0.1</near><far>60</far></clip>
        </camera>
      </sensor>
    </link>
  </model>
</sdf>"""

VIEWS = {
    # 正上方往下看，北方（前進方向）在畫面右邊
    "top": dict(x=-2.25, y=6.75, z=11.0, roll=0, pitch=1.5708, yaw=3.1416, hfov=1.2, w=1280, h=540),
    # 從起點後方斜看過去（3D 感）
    "chase": dict(x=-2.25, y=-2.5, z=4.0, roll=0, pitch=0.45, yaw=1.5708, hfov=1.3, w=960, h=540),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", default="barn_0")
    ap.add_argument("--view", default="top", choices=list(VIEWS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--rate", type=float, default=10.0, help="相機更新率（模擬時間 Hz）")
    ap.add_argument("--speedup", type=float, default=1.0, help="影片播放倍速")
    ap.add_argument("--stop-file", default="")
    ap.add_argument("--max-sec", type=float, default=180.0, help="最多錄幾秒（真實時間）")
    a = ap.parse_args()

    v = VIEWS[a.view]
    topic = f"/rec_cam_{a.view}"
    node = Node()

    frames = []
    lock = threading.Lock()

    def on_img(msg):
        if msg.pixel_format_type != 3:      # 3 = RGB_INT8
            return
        img = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, 3)
        t = msg.header.stamp.sec + msg.header.stamp.nsec * 1e-9
        with lock:
            frames.append((t, cv2.cvtColor(img, cv2.COLOR_RGB2BGR)))

    node.subscribe(Image, topic, on_img)

    req = EntityFactory(sdf=CAM_SDF.format(name=f"rec_cam_{a.view}", topic=topic, rate=a.rate, **v),
                        allow_renaming=True)
    ok, rep = node.request(f"/world/{a.world}/create", req, EntityFactory, Boolean, 5000)
    if not ok or not rep.data:
        raise SystemExit(f"生相機失敗：/world/{a.world}/create 沒回應")
    print(f"[rec] 相機 {a.view} 已加入，錄到 {a.out}", flush=True)

    fps = a.rate * a.speedup
    writer = None
    n = 0
    t_start = time.time()
    last_t = -1.0
    try:
        while time.time() - t_start < a.max_sec:
            if a.stop_file and os.path.exists(a.stop_file):
                time.sleep(1.0)              # 多錄一秒收尾
                break
            time.sleep(0.05)
            with lock:
                batch, frames[:] = frames[:], []
            for t, img in batch:
                if t <= last_t:
                    continue
                last_t = t
                if writer is None:
                    h, w = img.shape[:2]
                    for cc in ("avc1", "mp4v"):
                        writer = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*cc), fps, (w, h))
                        if writer.isOpened():
                            print(f"[rec] codec={cc} {w}x{h} @ {fps:.0f}fps", flush=True)
                            break
                cv2.putText(img, f"sim t = {t:6.1f} s", (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                            (20, 20, 20), 2, cv2.LINE_AA)
                writer.write(img)
                n += 1
                if n == 1:
                    cv2.imwrite(os.path.splitext(a.out)[0] + "_first.png", img)
        if n:
            cv2.imwrite(os.path.splitext(a.out)[0] + "_last.png", img)
    finally:
        if writer is not None:
            writer.release()
        print(f"[rec] 結束：{n} 張 → {a.out}", flush=True)


if __name__ == "__main__":
    main()
