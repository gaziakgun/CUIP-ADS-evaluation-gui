# Autoware Evaluation Workspace

This workspace is ready to build the GUI ROS 2 package from this project.

## Build

```bash
cd /home/gazi/Desktop/Evaluation/autoware_eval_ws
source /opt/ros/$ROS_DISTRO/setup.bash
source ~/autoware/install/setup.bash
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install --packages-select autoware_bag_eval_gui_ros2
source install/setup.bash
```

If your Autoware install path is different, source that install workspace instead of `~/autoware/install/setup.bash`.

## Run Test Route 03

```bash
ros2 run autoware_bag_eval_gui_ros2 autoware_bag_eval_gui \
  --bag /home/gazi/Desktop/Evaluation/test_route_03 \
  --storage-id sqlite3 \
  --origin-lat 35.0422327201 \
  --origin-lon -85.2983169612 \
  --origin-yaw-deg 0 \
  --metric speed
```

