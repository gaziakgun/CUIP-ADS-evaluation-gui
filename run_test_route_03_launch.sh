#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
BAG_PATH=${1:-${BAG_PATH:-test_route_03}}

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

if [ -f install/setup.bash ]; then
  source_setup install/setup.bash
else
  echo "Workspace is not built yet. Run ./build_eval_gui_package.sh first."
  exit 1
fi

ros2 launch autoware_bag_eval_gui_ros2 evaluation_gui.launch.py \
  bag:="$BAG_PATH" \
  storage_id:=sqlite3
