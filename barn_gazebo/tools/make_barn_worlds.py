"""
make_barn_worlds.py — 把 BARN 原始 Gazebo Classic 世界檔（.world）轉成 Gazebo Harmonic（ROS 2 Jazzy）用的 SDF

  python3 make_barn_worlds.py <BARN 的 worlds/BARN 資料夾> <輸出 worlds 資料夾> [編號 ...]
不給編號 = 跟 2D 實驗一樣的 50 張（0, 6, 12, ..., 294）。

每張輸出兩個檔：
  barn_<編號>.sdf   世界（地板、燈、所有圓柱、起點／終點標記；不含機器人，機器人由 launch 生成）
  barn_<編號>.json  圓柱座標，給 barn_eval 判斷碰撞用
圓柱的位置、半徑（0.075 m）、高度（1 m）照原始檔，不做任何修改。
所有圓柱合併成一個靜態模型（一個 link、多個 collision），檔案小、載入快，物理上跟分開放一樣。
"""
import json
import os
import re
import sys

START = (-2.25, 3.0, 1.5708)   # BARN 官方 run.py：INIT_POSITION = [-2.25, 3, 1.57]
GOAL = (-2.25, 13.0)           # 起點 + (0, 10)

HEAD = """<?xml version="1.0"?>
<sdf version="1.8">
  <!-- BARN 第 {idx} 號地圖（原始檔 world_{idx}.world，{n} 根圓柱）。起點 ({sx}, {sy}) 朝北，終點 ({gx}, {gy}) -->
  <world name="barn_{idx}">
    <physics name="1ms" type="ignored">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
    </physics>
    <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
    <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"/>
    <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>
    <plugin filename="gz-sim-sensors-system" name="gz::sim::systems::Sensors">
      <render_engine>ogre2</render_engine>
    </plugin>
    <plugin filename="gz-sim-imu-system" name="gz::sim::systems::Imu"/>

    <light type="directional" name="sun">
      <cast_shadows>true</cast_shadows>
      <pose>0 0 10 0 0 0</pose>
      <diffuse>0.8 0.8 0.8 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
      <direction>-0.5 0.1 -0.9</direction>
    </light>

    <model name="ground_plane">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry><plane><normal>0 0 1</normal><size>40 40</size></plane></geometry>
        </collision>
        <visual name="visual">
          <geometry><plane><normal>0 0 1</normal><size>40 40</size></plane></geometry>
          <material><ambient>0.8 0.8 0.8 1</ambient><diffuse>0.8 0.8 0.8 1</diffuse></material>
        </visual>
      </link>
    </model>

    <!-- 起點（藍）與終點（綠）標記：只有外觀，沒有碰撞，光達看不到 -->
    <model name="start_marker">
      <static>true</static>
      <pose>{sx} {sy} 0.001 0 0 0</pose>
      <link name="link">
        <visual name="visual">
          <geometry><cylinder><radius>0.15</radius><length>0.002</length></cylinder></geometry>
          <material><ambient>0.1 0.3 0.9 1</ambient><diffuse>0.1 0.3 0.9 1</diffuse></material>
        </visual>
      </link>
    </model>
    <model name="goal_marker">
      <static>true</static>
      <pose>{gx} {gy} 0.001 0 0 0</pose>
      <link name="link">
        <visual name="visual">
          <geometry><cylinder><radius>1.0</radius><length>0.002</length></cylinder></geometry>
          <material><ambient>0.1 0.7 0.2 0.6</ambient><diffuse>0.1 0.7 0.2 0.6</diffuse></material>
        </visual>
      </link>
    </model>

    <!-- BARN 障礙物：半徑 0.075 m、高 1 m 的圓柱 -->
    <model name="barn_obstacles">
      <static>true</static>
      <link name="link">
"""
CYL = """        <collision name="c{i}"><pose>{x:.4f} {y:.4f} 0.5 0 0 0</pose><geometry><cylinder><radius>0.075</radius><length>1</length></cylinder></geometry></collision>
        <visual name="v{i}"><pose>{x:.4f} {y:.4f} 0.5 0 0 0</pose><geometry><cylinder><radius>0.075</radius><length>1</length></cylinder></geometry><material><ambient>0.35 0.35 0.4 1</ambient><diffuse>0.35 0.35 0.4 1</diffuse></material></visual>
"""
TAIL = """      </link>
    </model>
  </world>
</sdf>
"""

POSE = re.compile(r"<model name='unit_cylinder_\d+'>\s*<static>1</static>\s*<pose frame=''>([-\d.e ]+)</pose>")


def convert(src_dir, out_dir, idx):
    text = open(os.path.join(src_dir, f"world_{idx}.world")).read()
    pts = [tuple(float(v) for v in p.split()[:2]) for p in POSE.findall(text)]
    n_models = len(set(re.findall(r"<model name=.(unit_cylinder_\d+).>", text)))   # <state> 區段會重複列一次
    assert len(pts) == n_models, f"world_{idx}: 抓到 {len(pts)} 根，檔案裡有 {n_models} 個圓柱模型"
    sx, sy, _ = START
    gx, gy = GOAL
    sdf = HEAD.format(idx=idx, n=len(pts), sx=sx, sy=sy, gx=gx, gy=gy)
    sdf += "".join(CYL.format(i=i, x=x, y=y) for i, (x, y) in enumerate(pts))
    sdf += TAIL
    open(os.path.join(out_dir, f"barn_{idx}.sdf"), "w").write(sdf)
    json.dump({"world": idx, "radius": 0.075, "start": START, "goal": GOAL, "cylinders": pts},
              open(os.path.join(out_dir, f"barn_{idx}.json"), "w"))
    return len(pts)


if __name__ == "__main__":
    src, out = sys.argv[1], sys.argv[2]
    ids = [int(v) for v in sys.argv[3:]] or list(range(0, 300, 6))
    os.makedirs(out, exist_ok=True)
    for i in ids:
        print(f"barn_{i}: {convert(src, out, i)} 根圓柱")
