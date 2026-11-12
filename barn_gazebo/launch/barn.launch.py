"""
barn.launch.py — 開一張 BARN 地圖、把車放在起點、開始記成績

  export TURTLEBOT3_MODEL=burger
  ros2 launch barn_gazebo barn.launch.py world:=0
  ros2 launch barn_gazebo barn.launch.py world:=150 robot:=my_robot      # 用你 nav_sim 套件裡的 my_robot
  ros2 launch barn_gazebo barn.launch.py world:=0 gui:=false             # 不開畫面（跑比較快）

參數
  world   地圖編號：0, 6, 12, ..., 294（跟 2D 實驗同一批 50 張）
  robot   turtlebot3（官方 turtlebot3_gazebo 的模型，預設）、my_robot（nav_sim 套件的模型）、
          custom（你自己的模型檔：model_file:=/完整路徑/model.sdf 或 .urdf / .xacro）
  gui     true / false
  eval    true / false：要不要同時開 barn_eval 記成績
起點 (-2.25, 3) 朝北（yaw = 1.5708），終點 (-2.25, 13)，跟 BARN 官方 run.py 相同。
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (AppendEnvironmentVariable, DeclareLaunchArgument, IncludeLaunchDescription,
                            OpaqueFunction)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

START_X, START_Y, START_YAW = -2.25, 3.0, 1.5708


def setup(context):
    pkg = get_package_share_directory("barn_gazebo")
    world = LaunchConfiguration("world").perform(context)
    robot = LaunchConfiguration("robot").perform(context)
    gui = LaunchConfiguration("gui").perform(context).lower() == "true"
    do_eval = LaunchConfiguration("eval").perform(context).lower() == "true"
    radius = LaunchConfiguration("robot_radius").perform(context)
    method = LaunchConfiguration("method").perform(context)

    world_file = os.path.join(pkg, "worlds", f"barn_{world}.sdf")
    if not os.path.exists(world_file):
        raise RuntimeError(f"找不到 {world_file}。可用的地圖編號是 0, 6, 12, ..., 294")

    actions = []
    sim = {"use_sim_time": True}

    # ---- 機器人模型 ----
    if robot == "turtlebot3":
        if "TURTLEBOT3_MODEL" not in os.environ:
            raise RuntimeError("請先 export TURTLEBOT3_MODEL=burger 再啟動")
        tb3 = get_package_share_directory("turtlebot3_gazebo")
        model = os.environ["TURTLEBOT3_MODEL"]
        model_file = os.path.join(tb3, "models", f"turtlebot3_{model}", "model.sdf")
        actions.append(AppendEnvironmentVariable("GZ_SIM_RESOURCE_PATH", os.path.join(tb3, "models")))
        # 跟官方 spawn_turtlebot3.launch.py 用同一份橋設定（注意：Jazzy 版 cmd_vel 是 TwistStamped）
        actions.append(Node(package="ros_gz_bridge", executable="parameter_bridge", output="screen",
                            arguments=["--ros-args", "-p",
                                       f"config_file:={os.path.join(tb3, 'params', f'turtlebot3_{model}_bridge.yaml')}"],
                            parameters=[sim]))
        actions.append(IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(tb3, "launch", "robot_state_publisher.launch.py")),
            launch_arguments={"use_sim_time": "true"}.items()))
        name = "turtlebot3"
    elif robot == "my_robot":
        try:                                    # 有裝 nav_sim 就用它的；沒有就用套件裡附的同一份
            robots = os.path.join(get_package_share_directory("nav_sim"), "models")
        except Exception:
            robots = os.path.join(pkg, "robots")
        model_file = os.path.join(robots, "my_robot", "model.sdf")
        actions.append(AppendEnvironmentVariable("GZ_SIM_RESOURCE_PATH", robots))
        # 跟 nav_sim 的 sim.launch.py 同一組橋（cmd_vel 是 Twist）
        actions.append(Node(package="ros_gz_bridge", executable="parameter_bridge", output="screen",
                            arguments=["/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan",
                                       "/odom@nav_msgs/msg/Odometry[gz.msgs.Odometry",
                                       "/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock",
                                       "/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist",
                                       "/tf@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V"],
                            parameters=[sim]))
        actions.append(Node(package="tf2_ros", executable="static_transform_publisher", parameters=[sim],
                            arguments=["--x", "-0.02", "--y", "0", "--z", "0.11",
                                       "--frame-id", "base_link", "--child-frame-id", "lidar_link"]))
        name = "my_robot"
    elif robot == "custom":
        # 你自己的車：給模型檔（.sdf 或 .urdf）的完整路徑；橋可以給 yaml，不給就用預設的 5 條
        model_file = os.path.expanduser(LaunchConfiguration("model_file").perform(context))
        if not os.path.exists(model_file):
            raise RuntimeError(f"robot:=custom 要給 model_file:=<模型檔完整路徑>，找不到：{model_file}")
        actions.append(AppendEnvironmentVariable("GZ_SIM_RESOURCE_PATH", os.path.dirname(os.path.dirname(model_file))))
        bridge = os.path.expanduser(LaunchConfiguration("bridge_config").perform(context))
        if bridge:
            args = ["--ros-args", "-p", f"config_file:={bridge}"]
        else:
            cmd = "TwistStamped" if LaunchConfiguration("cmd_stamped").perform(context).lower() == "true" else "Twist"
            args = ["/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan",
                    "/odom@nav_msgs/msg/Odometry[gz.msgs.Odometry",
                    "/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock",
                    f"/cmd_vel@geometry_msgs/msg/{cmd}]gz.msgs.Twist",
                    "/tf@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V"]
        actions.append(Node(package="ros_gz_bridge", executable="parameter_bridge", output="screen",
                            arguments=args, parameters=[sim]))
        if model_file.endswith((".urdf", ".xacro")):
            import xacro
            desc = xacro.process_file(model_file).toxml() if model_file.endswith(".xacro") else open(model_file).read()
            actions.append(Node(package="robot_state_publisher", executable="robot_state_publisher",
                                parameters=[sim, {"robot_description": desc}]))
        name = "my_car"
    else:
        raise RuntimeError("robot 只能是 turtlebot3、my_robot 或 custom")

    # ---- Gazebo ----
    gz_args = f"-r -v2 {world_file}" if gui else f"-r -s -v2 {world_file}"
    actions.insert(0, IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(get_package_share_directory("ros_gz_sim"), "launch", "gz_sim.launch.py")),
        launch_arguments={"gz_args": gz_args, "on_exit_shutdown": "true"}.items()))

    # ---- 把車放在起點、朝北 ----
    actions.append(Node(package="ros_gz_sim", executable="create", output="screen",
                        arguments=["-name", name, "-file", model_file,
                                   "-x", str(START_X), "-y", str(START_Y), "-z", LaunchConfiguration("spawn_z").perform(context),
                                   "-Y", str(START_YAW)]))

    # ---- 記成績 ----
    if do_eval:
        actions.append(Node(package="barn_gazebo", executable="barn_eval", output="screen",
                            parameters=[{"use_sim_time": True, "world": int(world), "robot_radius": float(radius),
                                         "method": method}]))
    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("world", default_value="0", description="BARN 地圖編號（0, 6, ..., 294）"),
        DeclareLaunchArgument("robot", default_value="turtlebot3", description="turtlebot3 或 my_robot"),
        DeclareLaunchArgument("gui", default_value="true"),
        DeclareLaunchArgument("eval", default_value="true"),
        DeclareLaunchArgument("robot_radius", default_value="0.11", description="碰撞判斷用的車身半徑（m）"),
        DeclareLaunchArgument("method", default_value="", description="寫進成績檔的方法名稱"),
        DeclareLaunchArgument("model_file", default_value="", description="robot:=custom 時，你的車的模型檔（.sdf / .urdf / .xacro）"),
        DeclareLaunchArgument("bridge_config", default_value="", description="robot:=custom 時的橋設定 yaml（可省略）"),
        DeclareLaunchArgument("cmd_stamped", default_value="false", description="robot:=custom 且沒給 yaml 時，cmd_vel 用 TwistStamped 嗎"),
        DeclareLaunchArgument("spawn_z", default_value="0.01", description="生成高度（車子底盤比較高時調大）"),
        OpaqueFunction(function=setup),
    ])
