#!/usr/bin/env bash

# ============================================================
# Autoware Autonomous Driving Evaluation Bag Recorder
# ============================================================
# Usage:
#   chmod +x record_autoware_eval_bag.sh
#   ./record_autoware_eval_bag.sh
#
# Optional:
#   ./record_autoware_eval_bag.sh my_test_bag
# ============================================================

set -e

BAG_NAME=${1:-autoware_eval_$(date +"%Y%m%d_%H%M%S")}

echo "=============================================="
echo " Autoware Evaluation Bag Recorder"
echo " Bag name: $BAG_NAME"
echo "=============================================="

# ------------------------------------------------------------
# Main Autoware topics for evaluation
# ------------------------------------------------------------

TOPICS=(

  # Localization / vehicle pose
  "/localization/kinematic_state"

  # Planned trajectory
  "/planning/scenario_planning/trajectory"

  # Control command
  "/control/command/control_cmd"

  # Vehicle velocity
  "/vehicle/status/velocity_status"
 

  # Autoware operation mode
  "/vehicle/status/control_mode"

  # Diagnostics
  "/diagnostics"

  # Perception
  "/perception/object_recognition/detection/objects"
  "/perception/object_recognition/tracking/objects"

  # Planning stop reasons
  "/planning/scenario_planning/status/stop_reasons"

  # NovAtel / GNSS / INS topics
  "/sensing/novatel/oem7/fix"
  "/sensing/novatel/oem7/inspva"
  "/sensing/novatel/oem7/bestvel"
  "/sensing/novatel/oem7/odom"

  # Raptor / New Eagle DBW topics
  "/raptor_dbw_interface/driver_input_report"
  "/raptor_dbw_interface/brake_cmd"
  "/raptor_dbw_interface/accelerator_pedal_cmd"
  "/raptor_dbw_interface/steering_cmd"
  "/raptor_dbw_interface/dbw_enabled"
  "/raptor_dbw_interface/brake_report"
  "/raptor_dbw_interface/brake_2_report"
)

# ------------------------------------------------------------
# Check available topics
# ------------------------------------------------------------

echo ""
echo "Checking available ROS 2 topics..."
AVAILABLE_TOPICS=$(ros2 topic list)

RECORD_TOPICS=()

for topic in "${TOPICS[@]}"; do
  if echo "$AVAILABLE_TOPICS" | grep -qx "$topic"; then
    echo "[FOUND]   $topic"
    RECORD_TOPICS+=("$topic")
  else
    echo "[MISSING] $topic"
  fi
done

if [ ${#RECORD_TOPICS[@]} -eq 0 ]; then
  echo ""
  echo "No matching topics found. Please check your Autoware topic names."
  exit 1
fi

echo ""
echo "=============================================="
echo "Starting ros2 bag record..."
echo "Press Ctrl+C to stop recording."
echo "=============================================="

ros2 bag record -o "$BAG_NAME" "${RECORD_TOPICS[@]}"
