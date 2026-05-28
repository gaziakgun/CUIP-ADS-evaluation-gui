# Autoware Bag Evaluation GUI

A ROS 2 package that provides a Qt-based GUI for evaluating Autoware ROS 2 bags.
It supports OSM heatmaps, vehicle and object metrics, plots, and bag selection from the GUI or CLI.
This workspace contains the `autoware_bag_eval_gui_ros2` package and the helper bag recorder script.

## Features

- Load a ROS 2 bag directory from the GUI or via `--bag` CLI option.
- Support for `sqlite3` and `mcap` bag storage backends.
- OSM heatmap metrics: `speed`, `lateral_error`, `brake`, `density`, `operation_mode`, `object_density`, `object_speed`, and `sdsm_object_density`.
- SDSM object evaluation from `/v2i/sdsm/objects`, including SDSM object counts, density, and nearest-object comparison metrics against onboard perception detections.
- Configurable heatmap origin, zoom, bin count, alpha, and colormap.
- CSV/PDF export support for evaluation results.
- GPS-triggered bag recorder script for Autoware evaluation runs.

## Requirements

- ROS 2 installed and sourced: `/opt/ros/$ROS_DISTRO/setup.bash`
- `rosbag2_py` available in the ROS environment
- Python dependencies: `PyQt5`, `matplotlib`, `pandas`, `numpy`, `requests`, `Pillow`, `pyproj`

Example packages on Debian/Ubuntu:

```bash
sudo apt install python3-pyqt5 python3-matplotlib python3-pandas python3-numpy python3-requests python3-pillow python3-pyproj
```

If you are using Autoware, also source your Autoware install workspace before building:

```bash
source ~/autoware/install/setup.bash
```

## Build

From a ROS 2 workspace:

```bash
mkdir -p ~/autoware_eval_ws/src
cp -r /path/to/eval_ws/src/autoware_bag_eval_gui_ros2 ~/autoware_eval_ws/src/
cd ~/autoware_eval_ws
source /opt/ros/$ROS_DISTRO/setup.bash
source ~/autoware/install/setup.bash  # if using Autoware
rosdep install --from-paths src --ignore-src -r -y
colcon build --packages-select autoware_bag_eval_gui_ros2
source install/setup.bash
```

## Run

Start the GUI and choose a bag after launch:

```bash
ros2 run autoware_bag_eval_gui_ros2 autoware_bag_eval_gui --storage-id sqlite3
```

Load a bag immediately:

```bash
ros2 run autoware_bag_eval_gui_ros2 autoware_bag_eval_gui \
  --bag /path/to/bag --storage-id sqlite3
```

Use the available command-line options:

- `--storage-id` : `sqlite3` or `mcap`
- `--origin-lat` : OSM heatmap origin latitude
- `--origin-lon` : OSM heatmap origin longitude
- `--origin-yaw-deg` : OSM heatmap origin yaw
- `--metric` : default heatmap metric (`speed`, `lateral_error`, `brake`, `density`, `operation_mode`, `object_density`, `object_speed`, `sdsm_object_density`)
- `--output` : default heatmap PNG save path
- `--heatmap-zoom` : default map tile zoom level
- `--heatmap-bins` : default number of heatmap bins
- `--heatmap-alpha` : default heatmap overlay alpha
- `--heatmap-cmap` : default heatmap colormap

## Launch File

Start the GUI with ROS 2 launch:

```bash
ros2 launch autoware_bag_eval_gui_ros2 evaluation_gui.launch.py
```

Or pass bag and storage arguments through launch:

```bash
ros2 launch autoware_bag_eval_gui_ros2 evaluation_gui.launch.py bag:=/path/to/bag storage_id:=sqlite3
```

## Record Bag From GPS Start/Finish Gate

The recorder script captures evaluation topics around a GPS gate and stops when the vehicle returns.

```bash
chmod +x record_autoware_eval_bag.sh
./record_autoware_eval_bag.sh
```

Optional arguments:

```bash
./record_autoware_eval_bag.sh BAG_NAME TRIGGER_LAT TRIGGER_LON RADIUS_METERS
```

Defaults:

- `TRIGGER_LAT`: `35.0423036111`
- `TRIGGER_LON`: `-85.2988136944`
- `RADIUS_METERS`: `5`
- `MIN_RECORD_SECONDS`: `10`
- `GPS_TOPIC`: `/sensing/novatel/oem7/fix`

Override the GNSS fix topic:

```bash
GPS_TOPIC=/your/navsatfix/topic ./record_autoware_eval_bag.sh
```

Disable the minimum recording time guard:

```bash
MIN_RECORD_SECONDS=0 ./record_autoware_eval_bag.sh
```

The recorder captures available Autoware evaluation topics and reports missing topics before starting.
When present, it also records `/v2i/sdsm/objects` and `/tf_static` so SDSM object heatmaps and onboard-vs-SDSM comparison metrics can be generated.
