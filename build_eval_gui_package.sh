#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

source_setup() {
  set +u
  source "$1"
  set -u
}

source_setup /opt/ros/"${ROS_DISTRO:?ROS_DISTRO is not set. Source ROS 2 first, or set ROS_DISTRO.}"/setup.bash

if [ -f "$HOME/autoware/install/setup.bash" ]; then
  source_setup "$HOME/autoware/install/setup.bash"
else
  echo "Warning: $HOME/autoware/install/setup.bash not found."
  echo "If Autoware is installed somewhere else, source it before running this script."
fi

rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install --packages-select autoware_bag_eval_gui_ros2

echo
echo "Build complete. Load this workspace with:"
echo "source $(pwd)/install/setup.bash"
