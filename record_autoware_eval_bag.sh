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

  # Autonomy / engagement / operation mode
  "/autoware/engage"
  "/autoware/state"
  "/vehicle/engage"
  "/vehicle/status/control_mode"
  "/system/operation_mode/state"
  "/system/operation_mode/availability"
  "/system/component_state_monitor/component/autonomous/control"
  "/system/component_state_monitor/component/autonomous/localization"
  "/system/component_state_monitor/component/autonomous/perception"
  "/system/component_state_monitor/component/autonomous/planning"
  "/system/component_state_monitor/component/autonomous/sensing"
  "/system/component_state_monitor/component/autonomous/system"
  "/system/component_state_monitor/component/autonomous/vehicle"

  # Localization / vehicle pose
  "/localization/initialization_state"
  "/localization/pose_with_covariance"
  "/localization/kinematic_state"
  "/localization/twist_estimator/twist_with_covariance"
  "/localization/pose_estimator/pose"
  "/tf"
  "/tf_static"

  # Planned trajectory / route / mission planning
  "/planning/scenario_planning/trajectory"
  "/planning/trajectory"
  "/planning/route"
  "/planning/route_state"
  "/planning/mission_planning/state"
  "/planning/mission_planning/route"
  "/planning/turn_indicators_cmd"
  "/planning/remaining_distance_time_calculator/debug/processing_time_detail_ms"
  "/planning/remaining_distance_time"

  # Control command and actuation
  "/control/command/control_cmd"
  "/control/command/gear_cmd"
  "/control/command/hazard_lights_cmd"
  "/control/command/turn_indicators_cmd"
  "/control/trajectory_follower/control_cmd"

  # Perception / objects / tracking
  "/perception/object_recognition/detection/objects"
  "/perception/object_recognition/objects"
  "/perception/object_recognition/tracking/objects"
  "/perception/object_recognition/prediction/map_based_prediction/objects"
  "/perception/obstacle_segmentation/pointcloud"
  "/sensing/lidar/concatenated/pointcloud"
  "/sensing/points"

  # Safety / emergency / fail-safe
  "/api/autoware/get/emergency"
  "/system/emergency/control_cmd"
  "/system/emergency/gear_cmd"
  "/system/emergency/hazard_lights_cmd"
  "/system/emergency/hazard_status"
  "/system/fail_safe/mrm_state"
  "/system/mrm/comfortable_stop/status"
  "/system/mrm/emergency_stop/status"
  "/system/mrm/pull_over_manager/status"

  # Diagnostics / system health
  "/diagnostics"
  "/diagnostics_graph/status"
  "/diagnostics_graph/struct"
  "/system/system_monitor/cpu_monitor/cpu_usage"
  "/system/system_monitor/mem_monitor/memory_status"
  "/system/system_monitor/hdd_monitor/hdd_status"
  "/system/system_monitor/net_monitor/network_status"
  "/system/pipeline_latency_monitor/debug/planning_latency_ms"
  "/system/pipeline_latency_monitor/output/total_latency_ms"
  "/control/control_validator/validation_status"
  "/planning/planning_validator/validation_status"

  # Vehicle status and metrics
  "/vehicle/status/velocity_status"
  "/vehicle/status/steering_status"
  "/vehicle/status/hazard_lights_status"
  "/vehicle/status/gear_status"
  "/vehicle/status/battery_charge"
  "/vehicle/doors/status"

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
