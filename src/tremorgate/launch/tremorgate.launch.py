from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    serial=LaunchConfiguration("serial_port")
    model=LaunchConfiguration("model_path")
    norm=LaunchConfiguration("norm_path")
    mode=LaunchConfiguration("mode")
    use_gazebo=LaunchConfiguration("use_gazebo")
    return LaunchDescription([
        DeclareLaunchArgument("serial_port",default_value="/tmp/ttyV1"),
        DeclareLaunchArgument("model_path",default_value=""),
        DeclareLaunchArgument("norm_path",default_value=""),
        DeclareLaunchArgument("mode",default_value="cnn",description="cnn or fft"),
        DeclareLaunchArgument("use_gazebo",default_value="true"),
        Node(package="tremorgate",executable="serial_bridge_node",name="serial_bridge_node",
             parameters=[{"port":serial,"baud":115200}]),
        Node(package="tremorgate",executable="perception_node",name="perception_node",
             parameters=[{"model_path":model,"norm_path":norm,"mode":mode}]),
        Node(package="tremorgate",executable="arm_gate_node",name="arm_gate_node"),
        Node(package="tremorgate",executable="arm_stub_node",name="arm_stub_node",
             condition=UnlessCondition(use_gazebo)),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([FindPackageShare("ros2_control_demo_example_9"),
                                      "bringup","launch","rrbot_gazebo_classic.launch.py"])),
            condition=IfCondition(use_gazebo),
            launch_arguments={"gui":"true"}.items()),
    ])
