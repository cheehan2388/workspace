import os
from glob import glob
from setuptools import find_packages, setup

package_name = "barn_gazebo"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.py")),
        (os.path.join("share", package_name, "worlds"), glob("worlds/*.sdf") + glob("worlds/*.json")),
        # 2D 實驗的程式（A*、DWA、PPO、Hybrid），run_method 直接用
        (os.path.join("share", package_name, "sim2d"), glob("sim2d/*.py") + glob("sim2d/*.npz")),
        # 訓練好的模型：v1 = 原獎勵（3 種 × 3 個種子），v2_k0.02 = ω 平順獎勵
        (os.path.join("share", package_name, "models", "v1"), glob("models/v1/*")),
        (os.path.join("share", package_name, "models", "v2_k0.02"), glob("models/v2_k0.02/*")),
        # 你 nav_sim 套件的 my_robot（沒裝 nav_sim 時用這份）
        (os.path.join("share", package_name, "robots", "my_robot"), glob("robots/my_robot/*")),
        (os.path.join("share", package_name, "scripts"), glob("scripts/*")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="cheehan0114",
    maintainer_email="cheehan0114@todo.todo",
    description="BARN maps for Gazebo Harmonic + TurtleBot3 spawn + evaluator",
    license="BSD-3-Clause",
    entry_points={"console_scripts": ["barn_eval = barn_gazebo.barn_eval:main"]},
)
