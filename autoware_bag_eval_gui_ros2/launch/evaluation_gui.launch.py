from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import FindExecutable, LaunchConfiguration


def generate_launch_description():
    bag = LaunchConfiguration("bag")
    storage_id = LaunchConfiguration("storage_id")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "bag",
                description="Path to the ROS 2 bag directory to evaluate.",
            ),
            DeclareLaunchArgument(
                "storage_id",
                default_value="sqlite3",
                description="ROS bag storage ID, usually sqlite3 or mcap.",
            ),
            ExecuteProcess(
                cmd=[
                    FindExecutable(name="autoware_bag_eval_gui"),
                    "--bag",
                    bag,
                    "--storage-id",
                    storage_id,
                ],
                output="screen",
            ),
        ]
    )
