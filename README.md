# Autoware ADS Evaluation GUI

This repository contains lightweight tools for evaluating recorded Autoware ROS 2 bags. It can run as a standalone Python script or as a ROS 2 `ament_python` package.

## What It Does

- Reads selected Autoware topics from a ROS 2 bag.
- Computes route distance, autonomous distance/time, takeovers, tracking error, velocity error, and braking events.
- Displays a PyQt5 dashboard with summary, OSM heatmap, tracking, velocity, braking, and events tabs.
- Supports zoom/pan on figures, robust Y-axis scaling for outliers, figure export, CSV export, and PDF report export.
- Draws OSM route heatmaps for speed, lateral error, braking intensity, density, and operation mode.

## Repository Layout

```text
.
├── autoware_bag_eval_gui.py              # Standalone GUI script
├── autoware_osm_heatmap_png.py           # Standalone OSM heatmap PNG utility
├── record_autoware_eval_bag.sh           # Helper for recording evaluation bags
├── autoware_bag_eval_gui_ros2/           # ROS 2 ament_python package
└── autoware_eval_ws/                     # Local convenience workspace and test scripts
```

Generated bags, reports, OSM tile cache, and colcon build folders are ignored by Git.

## Prerequisites

- Linux environment with Bash
- ROS 2 Humble or compatible ROS 2 environment
- Autoware message packages available in your sourced workspace
- Python packages: `PyQt5`, `matplotlib`, `numpy`, `pandas`, `Pillow`, `pyproj`, `requests`
- ROS Python packages: `rosbag2_py`, `rosidl_runtime_py`

Typical Ubuntu dependencies:

```bash
sudo apt install \
  python3-pyqt5 python3-matplotlib python3-numpy python3-pandas \
  python3-pil python3-pyproj python3-requests
```

## Quick Start

### Standalone GUI

```bash
source /opt/ros/humble/setup.bash
source ~/autoware/install/setup.bash

python3 autoware_bag_eval_gui.py \
  --bag /path/to/test_route_03 \
  --storage-id sqlite3 \
  --origin-lat 35.0422327201 \
  --origin-lon -85.2983169612 \
  --origin-yaw-deg 0 \
  --metric speed
```

### ROS 2 Package

```bash
cd /path/to/this/repo/autoware_eval_ws
./build_eval_gui_package.sh
./run_test_route_03.sh
```

Manual ROS 2 package flow:

```bash
cd /path/to/this/repo/autoware_eval_ws
source /opt/ros/humble/setup.bash
source ~/autoware/install/setup.bash
colcon build --symlink-install --packages-select autoware_bag_eval_gui_ros2
source install/setup.bash

ros2 run autoware_bag_eval_gui_ros2 autoware_bag_eval_gui \
  --bag /path/to/test_route_03 \
  --storage-id sqlite3
```

### OSM Heatmap Utility

```bash
python3 autoware_osm_heatmap_png.py \
  --bag /path/to/test_route_03 \
  --storage-id sqlite3 \
  --output speed_heatmap.png \
  --origin-lat 35.0422327201 \
  --origin-lon -85.2983169612 \
  --origin-yaw-deg 0
```

## GitHub Preparation Notes

Do not commit ROS bags, generated reports, OSM tile cache, or colcon build outputs. The `.gitignore` file already excludes those paths and file patterns.

Before pushing:

```bash
git status --short
git add .gitignore README.md autoware_bag_eval_gui.py autoware_osm_heatmap_png.py \
  record_autoware_eval_bag.sh autoware_bag_eval_gui_ros2 autoware_eval_ws
git status --short
```

## Contributing

1. Create a feature branch.
2. Make focused changes with clear commit messages.
3. Open a pull request describing motivation and validation steps.

## License

License is not selected yet. Add a `LICENSE` file before public release.
