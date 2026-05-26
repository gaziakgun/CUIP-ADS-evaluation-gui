# Autoware Evaluation Workspace

This workspace is ready to build the GUI ROS 2 package from this project.

## Build

```bash
cd CUIP-ADS-evaluation-gui
source /opt/ros/$ROS_DISTRO/setup.bash
source ~/autoware/install/setup.bash
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install --packages-select autoware_bag_eval_gui_ros2
source install/setup.bash
```

If your Autoware install path is different, source that install workspace instead of `~/autoware/install/setup.bash`.

## Run And Select Bag In The GUI

```bash
ros2 run autoware_bag_eval_gui_ros2 autoware_bag_eval_gui \
  --storage-id sqlite3
```

Or with the launch file:

```bash
ros2 launch autoware_bag_eval_gui_ros2 evaluation_gui.launch.py
```

Use **Select Bag** in the dashboard header to choose the ROS 2 bag folder after the GUI starts.
Use the **Units** selector in the same header to switch between metric units and US customary units (miles, feet, mph, ft/s²).

## Record Bag From GPS Start/Finish Gate

The recorder waits for the vehicle to pass this GNSS point, starts `ros2 bag record`, then stops after the vehicle leaves the gate and later returns through it:

```text
35.0423036111, -85.2988136944
```

Run:

```bash
./record_autoware_eval_bag.sh
```

Optional arguments are:

```bash
./record_autoware_eval_bag.sh BAG_NAME TRIGGER_LAT TRIGGER_LON RADIUS_METERS
```

The default radius is `5` meters, with a `10` second minimum recording time before finish detection can stop the bag. Use `GPS_TOPIC=/your/navsatfix/topic` if your GNSS fix topic is not `/sensing/novatel/oem7/fix`, and `MIN_RECORD_SECONDS=0` if you want to disable the minimum time guard.

The heatmap tab uses these map-frame defaults:

```text
origin-lat: 35.0422327201
origin-lon: -85.2983169612
origin-yaw-deg: 0
```

The OSM heatmap metric selector also includes `object_density` and `object_speed`.
Those use classified vehicles and pedestrians from
`/perception/object_recognition/detection/objects`; unknown/other objects are excluded.
For object speed, detected objects are matched to nearby tracked objects when the
detection message does not include a usable speed.
The Summary page also reports parked-car density, traffic density, and mean traffic
speed from the same detected vehicle objects.

## Run Test Route 03 Directly

```bash
ros2 run autoware_bag_eval_gui_ros2 autoware_bag_eval_gui \
  --bag /path/to/test_route_03 \
  --storage-id sqlite3 \
  --metric speed
```

Or use the helper script:

```bash
./run_test_route_03.sh /path/to/test_route_03
```
