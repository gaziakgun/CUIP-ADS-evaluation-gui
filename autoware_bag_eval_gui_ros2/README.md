# Autoware Bag Evaluation GUI ROS 2 Package

This package provides an `ament_python` ROS 2 wrapper for the Autoware bag evaluation GUI.
It launches a Qt-based application that can load ROS 2 bag folders, render OSM-based heatmaps,
plot vehicle and object metrics, and export evaluation results.

## Features

- GUI-based bag selection and immediate bag loading.
- Support for `sqlite3` and `mcap` ROS 2 bag storage backends.
- Heatmap metrics: `speed`, `lateral_error`, `brake`, `density`, `operation_mode`, `object_density`, `object_speed`, and `sdsm_object_density`.
- SDSM object counts, density, and nearest-object comparison metrics against onboard perception from `/v2i/sdsm/objects`.
- Configurable map origin and heatmap rendering options.
- Launch file support via `evaluation_gui.launch.py`.

## Requirements

- ROS 2 installed and sourced.
- `rosbag2_py` available in the ROS environment.
- Python dependencies: `PyQt5`, `matplotlib`, `pandas`, `numpy`, `requests`, `Pillow`, `pyproj`.

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

Start the GUI and select a bag from the dashboard:

```bash
ros2 run autoware_bag_eval_gui_ros2 autoware_bag_eval_gui --storage-id sqlite3
```

Load a bag immediately:

```bash
ros2 run autoware_bag_eval_gui_ros2 autoware_bag_eval_gui \
  --bag /path/to/bag --storage-id sqlite3
```

### Optional CLI Arguments

- `--storage-id`: bag storage backend (`sqlite3` or `mcap`)
- `--origin-lat`: OSM heatmap origin latitude
- `--origin-lon`: OSM heatmap origin longitude
- `--origin-yaw-deg`: OSM heatmap origin yaw
- `--metric`: default heatmap metric, including `sdsm_object_density`
- `--output`: default heatmap PNG save path
- `--heatmap-zoom`: default OSM tile zoom
- `--heatmap-bins`: default heatmap bin count
- `--heatmap-alpha`: default heatmap overlay alpha
- `--heatmap-cmap`: default heatmap colormap

## Launch File

```bash
ros2 launch autoware_bag_eval_gui_ros2 evaluation_gui.launch.py
```

Pass a bag path and storage ID through launch:

```bash
ros2 launch autoware_bag_eval_gui_ros2 evaluation_gui.launch.py bag:=/path/to/bag storage_id:=sqlite3
```

## Map Defaults

The package uses these default OSM heatmap origin values:

```text
origin-lat: 35.0422327201
origin-lon: -85.2983169612
origin-yaw-deg: 0
```

## Notes

The heatmap object metrics use classified vehicles and pedestrians from
`/perception/object_recognition/detection/objects` and `/v2i/sdsm/objects`.
Unknown/other object labels are excluded. `object_speed` uses tracked object matches
when a detection message does not supply usable speed.
