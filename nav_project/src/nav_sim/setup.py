import os
from glob import glob
from setuptools import find_packages, setup

package_name = "nav_sim"

# 要安裝到 share/nav_sim/ 的檔案：launch、場景、模型、SLAM 參數、RViz 畫面
# ros2 launch 和 Gazebo 都是從安裝後的位置找檔案，所以這裡沒列到的檔案它們就找不到
data_files = [
    ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
    ("share/" + package_name, ["package.xml"]),
    (os.path.join("share", package_name, "launch"), glob("launch/*.py")),
    (os.path.join("share", package_name, "worlds"), glob("worlds/*.sdf")),
    (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
    (os.path.join("share", package_name, "rviz"), glob("rviz/*.rviz")),
]
# models/ 底下每個模型資料夾各自安裝
for model_dir in glob("models/*"):
    data_files.append((os.path.join("share", package_name, model_dir), glob(model_dir + "/*")))

setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(exclude=["test"]),
    data_files=data_files,
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="cheehan0114",
    maintainer_email="cheehan0114@todo.todo",
    description="Navigation project: scenes, robot and launch files",
    license="TODO",
    entry_points={
        "console_scripts": [
            "c2_trigger = nav_sim.c2_trigger:main",
            "scan_reader = nav_sim.scan_reader:main",
            "train_ppo = nav_sim.train_ppo:main",
            "plot_results = nav_sim.plot_results:main",
        ],
    },
)
