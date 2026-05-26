# Autoware Bag Evaluation GUI ROS 2 Package

This package wraps the Autoware bag evaluation GUI as an `ament_python` ROS 2 package.

## Build

From a ROS 2 workspace:

```bash
mkdir -p ~/autoware_eval_ws/src
cp -r /path/to/CUIP-ADS-evaluation-gui/src/autoware_bag_eval_gui_ros2 ~/autoware_eval_ws/src/
cd ~/autoware_eval_ws
rosdep install --from-paths src --ignore-src -r -y
colcon build --packages-select autoware_bag_eval_gui_ros2
source install/setup.bash
```

## Run

```bash
ros2 run autoware_bag_eval_gui_ros2 autoware_bag_eval_gui --storage-id sqlite3
```

Use **Select Bag** in the dashboard header to choose the ROS 2 bag folder after the GUI starts.
Use the **Units** selector in the same header to switch between metric units and US customary units (miles, feet, mph, ft/s²).

The heatmap tab uses these map-frame defaults:

```text
origin-lat: 35.0422327201
origin-lon: -85.2983169612
origin-yaw-deg: 0
```

The heatmap metric selector also includes `object_density` and `object_speed`.
These object heatmaps use classified vehicles and pedestrians from
`/perception/object_recognition/detection/objects`; unknown/other objects are excluded.
For object speed, detected objects are matched to nearby tracked objects when the
detection message does not include a usable speed.
The Summary page also reports parked-car density, traffic density, and mean traffic
speed from the same detected vehicle objects.

You can still load a bag immediately:

```bash
ros2 run autoware_bag_eval_gui_ros2 autoware_bag_eval_gui --bag /path/to/test_route_03 --storage-id sqlite3
```

You can override the heatmap defaults if needed:

```bash
ros2 run autoware_bag_eval_gui_ros2 autoware_bag_eval_gui \
  --bag /path/to/test_route_03 \
  --storage-id sqlite3 \
  --origin-lat 35.0422327201 \
  --origin-lon -85.2983169612 \
  --origin-yaw-deg 0 \
  --metric speed
```

You can also launch the GUI and select the bag after it starts:

```bash
ros2 launch autoware_bag_eval_gui_ros2 evaluation_gui.launch.py
```

Or pass the bag path through launch:

```bash
ros2 launch autoware_bag_eval_gui_ros2 evaluation_gui.launch.py bag:=/path/to/test_route_03 storage_id:=sqlite3
```
