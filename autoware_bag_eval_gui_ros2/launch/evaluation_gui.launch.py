from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch.substitutions import FindExecutable, LaunchConfiguration


def launch_gui(context, *args, **kwargs):
    bag = LaunchConfiguration("bag").perform(context)
    storage_id = LaunchConfiguration("storage_id").perform(context)
    command = [
        FindExecutable(name="autoware_bag_eval_gui"),
        "--storage-id",
        storage_id,
    ]

    if bag:
        command.extend(["--bag", bag])

    return [
        ExecuteProcess(
            cmd=command,
            output="screen",
        )
    ]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "bag",
                default_value="",
                description="Optional path to the ROS 2 bag directory to evaluate. Leave empty and select a bag in the GUI.",
            ),
            DeclareLaunchArgument(
                "storage_id",
                default_value="sqlite3",
                description="ROS bag storage ID, usually sqlite3 or mcap.",
            ),
            OpaqueFunction(function=launch_gui),
        ]
    )
