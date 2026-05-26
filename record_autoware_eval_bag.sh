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
#   ./record_autoware_eval_bag.sh my_test_bag 35.0423036111 -85.2988136944 5
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BAG_NAME=${1:-autoware_eval_$(date +"%Y%m%d_%H%M%S")}
DEFAULT_TRIGGER_LAT="35.0423036111"
DEFAULT_TRIGGER_LON="-85.2988136944"
TRIGGER_LAT=${2:-$DEFAULT_TRIGGER_LAT}
TRIGGER_LON=${3:-$DEFAULT_TRIGGER_LON}
TRIGGER_RADIUS_METERS=${4:-${TRIGGER_RADIUS_METERS:-5}}
MIN_RECORD_SECONDS=${MIN_RECORD_SECONDS:-10}
GPS_TOPIC=${GPS_TOPIC:-"/sensing/novatel/oem7/fix"}

echo "=============================================="
echo " Autoware Evaluation Bag Recorder"
echo " Bag name: $BAG_NAME"
echo " GPS gate: lat=$TRIGGER_LAT lon=$TRIGGER_LON radius=${TRIGGER_RADIUS_METERS}m"
echo " Minimum record time before finish: ${MIN_RECORD_SECONDS}s"
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

if ! echo "$AVAILABLE_TOPICS" | grep -qx "$GPS_TOPIC"; then
  echo ""
  echo "Required GPS trigger topic is missing: $GPS_TOPIC"
  echo "Set GPS_TOPIC=/your/navsatfix/topic if your GNSS fix topic is different."
  exit 1
fi

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
echo "Waiting for GPS start/finish gate..."
echo "Recording starts on first pass and stops on return pass."
echo "Press Ctrl+C to cancel."
echo "=============================================="

python3 "$SCRIPT_DIR/gps_triggered_bag_recorder.py" \
  --bag-name "$BAG_NAME" \
  --gps-topic "$GPS_TOPIC" \
  --trigger-lat "$TRIGGER_LAT" \
  --trigger-lon "$TRIGGER_LON" \
  --trigger-radius-m "$TRIGGER_RADIUS_METERS" \
  --min-record-seconds "$MIN_RECORD_SECONDS" \
  "${RECORD_TOPICS[@]}"
