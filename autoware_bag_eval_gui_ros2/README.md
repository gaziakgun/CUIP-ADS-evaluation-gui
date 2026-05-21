# Autoware Bag Evaluation GUI ROS 2 Package

This package wraps the Autoware bag evaluation GUI as an `ament_python` ROS 2 package.

## Build

From a ROS 2 workspace:

```bash
mkdir -p ~/autoware_eval_ws/src
cp -r /home/gazi/Desktop/Evaluation/autoware_bag_eval_gui_ros2 ~/autoware_eval_ws/src/
cd ~/autoware_eval_ws
rosdep install --from-paths src --ignore-src -r -y
colcon build --packages-select autoware_bag_eval_gui_ros2
source install/setup.bash
```

## Run

```bash
ros2 run autoware_bag_eval_gui_ros2 autoware_bag_eval_gui --bag /path/to/test_route_03 --storage-id sqlite3
```

With heatmap defaults:

```bash
ros2 run autoware_bag_eval_gui_ros2 autoware_bag_eval_gui \
  --bag /path/to/test_route_03 \
  --storage-id sqlite3 \
  --origin-lat 35.0422327201 \
  --origin-lon -85.2983169612 \
  --origin-yaw-deg 0 \
  --metric speed
```

You can also launch the basic GUI command:

```bash
ros2 launch autoware_bag_eval_gui_ros2 evaluation_gui.launch.py bag:=/path/to/test_route_03 storage_id:=sqlite3
```
