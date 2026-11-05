"""
sim.launch.py — 一個指令開好 Gazebo、橋、TF

用法：
  ros2 launch nav_sim sim.launch.py world:=c1_serpentine.sdf
  ros2 launch nav_sim sim.launch.py world:=c2_sudden.sdf
  ros2 launch nav_sim sim.launch.py gui:=false   # 無頭模式：只跑 server，不開視窗
  ros2 launch nav_sim sim.launch.py slam:=false rviz:=false   # 只開模擬

取代之前要開的終端機：gz sim、parameter_bridge、static_transform_publisher、
slam_toolbox（含手動 lifecycle configure/activate）、rviz2。
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription,
                            SetEnvironmentVariable)
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node


def generate_launch_description():
    pkg = get_package_share_directory("nav_sim")
    world = LaunchConfiguration("world")
    gui = LaunchConfiguration("gui")
    world_path = PathJoinSubstitution([pkg, "worlds", world])
    slam_launch = os.path.join(get_package_share_directory("slam_toolbox"),
                               "launch", "online_async_launch.py")

    # 告訴 Gazebo 去哪裡找 model://my_robot 這類模型
    models = os.path.join(pkg, "models")
    resource_path = models + os.pathsep + os.environ.get("GZ_SIM_RESOURCE_PATH", "")

    return LaunchDescription([
        DeclareLaunchArgument("world", default_value="c1_serpentine.sdf",
                              description="worlds/ 資料夾裡的場景檔名"),
        DeclareLaunchArgument("gui", default_value="true",
                              description="false = 無頭模式（不開 Gazebo 視窗）"),
        DeclareLaunchArgument("slam", default_value="true", description="是否開 slam_toolbox"),
        DeclareLaunchArgument("rviz", default_value="true", description="是否開 RViz"),
        SetEnvironmentVariable("GZ_SIM_RESOURCE_PATH", resource_path),

        # 1. Gazebo（-r：一開就開始跑，不用按 ▶）
        ExecuteProcess(
            cmd=["gz", "sim", "-r", world_path],
            output="screen",
            condition=IfCondition(gui),
        ),
        # 1'. 無頭模式：-s 只跑 server；gpu_lidar 需要渲染，所以加 --headless-rendering（EGL，不需要螢幕）
        ExecuteProcess(
            cmd=["gz", "sim", "-r", "-s", "--headless-rendering", world_path],
            output="screen",
            condition=UnlessCondition(gui),
        ),

        # 2. 橋：跟你之前手打的那串一樣
        Node(
            package="ros_gz_bridge",
            executable="parameter_bridge",
            arguments=[
                "/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan",
                "/odom@nav_msgs/msg/Odometry[gz.msgs.Odometry",
                "/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock",
                "/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist",
                "/tf@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V",
            ],
            parameters=[{"use_sim_time": True}],
            output="screen",
        ),

        # 3. 光達焊在車上的位置：base_link → lidar_link（數字來自 model.sdf 裡光達的 <pose>）
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            arguments=["--x", "-0.02", "--y", "0", "--z", "0.11",
                       "--frame-id", "base_link", "--child-frame-id", "lidar_link"],
            parameters=[{"use_sim_time": True}],
        ),

        # 4. SLAM：用官方 launch 開，它會自己 configure → activate，不用再手打 ros2 lifecycle set
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(slam_launch),
            launch_arguments={
                "use_sim_time": "true",
                "slam_params_file": os.path.join(pkg, "config", "slam_params.yaml"),
            }.items(),
            condition=IfCondition(LaunchConfiguration("slam")),
        ),

        # 5. RViz：直接載入設定好的畫面（Map /map、LaserScan /scan、TF），而且用模擬時間
        Node(
            package="rviz2",
            executable="rviz2",
            arguments=["-d", os.path.join(pkg, "rviz", "slam.rviz")],
            parameters=[{"use_sim_time": True}],
            condition=IfCondition(LaunchConfiguration("rviz")),
        ),
    ])
