#!/usr/bin/env python3

# ============================================================
# Autoware Autonomous Driving Evaluation GUI
# ============================================================
#
# Usage:
#   chmod +x autoware_bag_eval_gui.py
#   ./autoware_bag_eval_gui.py --bag test_route_01
#
# If your bag is MCAP:
#   ./autoware_bag_eval_gui.py --bag test_route_01 --storage-id mcap
#
# If your bag is SQLite:
#   ./autoware_bag_eval_gui.py --bag test_route_01 --storage-id sqlite3
#
# Required packages:
#   sudo apt install ros-$ROS_DISTRO-rosbag2-py
#   sudo apt install python3-pyqt5 python3-matplotlib python3-pandas
#
# ============================================================

import argparse
import math
import os
import sys
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QLabel,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QPushButton,
    QFileDialog,
    QTextEdit,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
)
from PyQt5.QtCore import Qt

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


# ============================================================
# Topic configuration
# ============================================================

TOPIC_LOCALIZATION = "/localization/kinematic_state"
TOPIC_TRAJECTORY = "/planning/scenario_planning/trajectory"
TOPIC_CONTROL_CMD = "/control/command/control_cmd"
TOPIC_VELOCITY = "/vehicle/status/velocity_status"
TOPIC_OPERATION_MODE =  "/vehicle/status/control_mode"
TOPIC_STOP_REASONS = "/planning/scenario_planning/status/stop_reasons"

TOPIC_DRIVER_INPUT = "/raptor_dbw_interface/driver_input_report"
TOPIC_BRAKE_CMD = "/raptor_dbw_interface/brake_cmd"
TOPIC_DBW_ENABLED = "/raptor_dbw_interface/dbw_enabled"
TOPIC_NOVATEL_ODOM = "/sensing/novatel/oem7/odom"


# ============================================================
# Data containers
# ============================================================

@dataclass
class PoseSample:
    t: float
    x: float
    y: float
    yaw: float


@dataclass
class VelocitySample:
    t: float
    v: float


@dataclass
class ControlSample:
    t: float
    target_v: float = math.nan
    target_accel: float = math.nan
    steering: float = math.nan


@dataclass
class ModeSample:
    t: float
    mode: str
    autoware_control_enabled: bool = False


@dataclass
class TrajectorySample:
    t: float
    xs: np.ndarray
    ys: np.ndarray
    yaws: np.ndarray
    velocities: np.ndarray


@dataclass
class BrakeEvent:
    start_t: float
    end_t: float
    min_accel: float
    max_jerk: float
    event_type: str


@dataclass
class EvalResults:
    poses: list = field(default_factory=list)
    velocities: list = field(default_factory=list)
    controls: list = field(default_factory=list)
    modes: list = field(default_factory=list)
    trajectories: list = field(default_factory=list)

    lateral_errors: list = field(default_factory=list)
    heading_errors: list = field(default_factory=list)
    velocity_errors: list = field(default_factory=list)

    distance_total: float = 0.0
    autonomous_distance: float = 0.0
    autonomous_time: float = 0.0
    total_time: float = 0.0

    takeover_count: int = 0
    mode_change_count: int = 0
    brake_events: list = field(default_factory=list)
    harsh_brake_count: int = 0

    stop_reason_count: int = 0

    localization_vs_gnss_errors: list = field(default_factory=list)


# ============================================================
# Utility functions
# ============================================================

def stamp_to_sec(stamp):
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def bag_time_to_sec(t_nanosec):
    return float(t_nanosec) * 1e-9


def quaternion_to_yaw(q):
    x = q.x
    y = q.y
    z = q.z
    w = q.w

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)

    return math.atan2(siny_cosp, cosy_cosp)


def angle_wrap(a):
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def safe_getattr(obj, name, default=None):
    return getattr(obj, name, default)


def mode_to_string(mode_value):
    """
    Autoware AD API OperationModeState common mapping:
      0 UNKNOWN
      1 STOP
      2 AUTONOMOUS
      3 LOCAL
      4 REMOTE
    """
    mapping = {
        0: "NO_COMMAND",
        1: "AUTONOMOUS",
        2: "AUTONOMOUS_STEER_ONLY",
        3: "AUTONOMOUS_VELOCITY_ONLY",
        4: "MANUAL",
        5: "DISENGAGED",
        6: "NOT_READY"
    }
    return mapping.get(int(mode_value), f"MODE_{mode_value}")


def get_pose_from_msg(msg):
    """
    Supports nav_msgs/Odometry and geometry_msgs/PoseStamped-like messages.
    """
    if hasattr(msg, "pose") and hasattr(msg.pose, "pose"):
        pose = msg.pose.pose
    elif hasattr(msg, "pose"):
        pose = msg.pose
    else:
        return None

    x = pose.position.x
    y = pose.position.y
    yaw = quaternion_to_yaw(pose.orientation)

    if hasattr(msg, "header"):
        t = stamp_to_sec(msg.header.stamp)
    else:
        t = math.nan

    return PoseSample(t=t, x=x, y=y, yaw=yaw)


def get_velocity_from_msg(msg, fallback_time):
    """
    Supports common Autoware velocity report fields.
    """
    t = fallback_time
    if hasattr(msg, "header"):
        t = stamp_to_sec(msg.header.stamp)

    if hasattr(msg, "longitudinal_velocity"):
        v = msg.longitudinal_velocity
    elif hasattr(msg, "twist") and hasattr(msg.twist, "twist"):
        v = msg.twist.twist.linear.x
    elif hasattr(msg, "twist"):
        v = msg.twist.linear.x
    else:
        return None

    return VelocitySample(t=t, v=float(v))


def get_control_from_msg(msg, fallback_time):
    """
    Supports autoware_control_msgs/msg/Control style:
      msg.longitudinal.velocity
      msg.longitudinal.acceleration
      msg.lateral.steering_tire_angle
    """
    t = fallback_time
    if hasattr(msg, "stamp"):
        t = stamp_to_sec(msg.stamp)
    elif hasattr(msg, "header"):
        t = stamp_to_sec(msg.header.stamp)

    target_v = math.nan
    target_accel = math.nan
    steering = math.nan

    if hasattr(msg, "longitudinal"):
        target_v = safe_getattr(msg.longitudinal, "velocity", math.nan)
        target_accel = safe_getattr(msg.longitudinal, "acceleration", math.nan)

    if hasattr(msg, "lateral"):
        steering = safe_getattr(msg.lateral, "steering_tire_angle", math.nan)

    return ControlSample(
        t=t,
        target_v=float(target_v) if target_v is not None else math.nan,
        target_accel=float(target_accel) if target_accel is not None else math.nan,
        steering=float(steering) if steering is not None else math.nan,
    )


def get_mode_from_msg(msg, fallback_time):
    t = fallback_time
    if hasattr(msg, "stamp"):
        t = stamp_to_sec(msg.stamp)
    elif hasattr(msg, "header"):
        t = stamp_to_sec(msg.header.stamp)

    mode_value = safe_getattr(msg, "mode", -1)
    mode_str = mode_to_string(mode_value)

    autoware_control_enabled = bool(
        safe_getattr(msg, "is_autoware_control_enabled", False)
    )

    return ModeSample(
        t=t,
        mode=mode_str,
        autoware_control_enabled=autoware_control_enabled,
    )


def get_trajectory_from_msg(msg, fallback_time):
    """
    Supports autoware_planning_msgs/msg/Trajectory.
    """
    t = fallback_time
    if hasattr(msg, "header"):
        t = stamp_to_sec(msg.header.stamp)

    if not hasattr(msg, "points"):
        return None

    xs = []
    ys = []
    yaws = []
    velocities = []

    for p in msg.points:
        if not hasattr(p, "pose"):
            continue

        xs.append(p.pose.position.x)
        ys.append(p.pose.position.y)
        yaws.append(quaternion_to_yaw(p.pose.orientation))

        if hasattr(p, "longitudinal_velocity_mps"):
            velocities.append(p.longitudinal_velocity_mps)
        else:
            velocities.append(math.nan)

    if len(xs) < 2:
        return None

    return TrajectorySample(
        t=t,
        xs=np.array(xs, dtype=float),
        ys=np.array(ys, dtype=float),
        yaws=np.array(yaws, dtype=float),
        velocities=np.array(velocities, dtype=float),
    )


def find_nearest_trajectory(trajectories, t, max_dt=1.0):
    if not trajectories:
        return None

    times = np.array([tr.t for tr in trajectories])
    idx = int(np.argmin(np.abs(times - t)))

    if abs(times[idx] - t) > max_dt:
        return None

    return trajectories[idx]


def compute_nearest_path_error(pose, traj):
    dx = traj.xs - pose.x
    dy = traj.ys - pose.y
    dist = np.sqrt(dx * dx + dy * dy)

    idx = int(np.argmin(dist))

    lateral_error = float(dist[idx])
    heading_error = angle_wrap(pose.yaw - traj.yaws[idx])

    target_v = math.nan
    if idx < len(traj.velocities):
        target_v = float(traj.velocities[idx])

    return lateral_error, heading_error, target_v


def compute_distance(poses):
    if len(poses) < 2:
        return 0.0

    distance = 0.0
    for i in range(1, len(poses)):
        dx = poses[i].x - poses[i - 1].x
        dy = poses[i].y - poses[i - 1].y
        step = math.sqrt(dx * dx + dy * dy)

        if step < 5.0:
            distance += step

    return distance


def get_mode_at_time(modes, t):
    if not modes:
        return "UNKNOWN"

    times = np.array([m.t for m in modes])
    idx = np.searchsorted(times, t) - 1

    if idx < 0:
        idx = 0

    return modes[idx].mode


def compute_autonomous_distance(poses, modes):
    if len(poses) < 2 or not modes:
        return 0.0

    distance = 0.0

    for i in range(1, len(poses)):
        mode = get_mode_at_time(modes, poses[i].t)

        dx = poses[i].x - poses[i - 1].x
        dy = poses[i].y - poses[i - 1].y
        step = math.sqrt(dx * dx + dy * dy)

        if step > 5.0:
            continue

        if mode == "AUTONOMOUS":
            distance += step

    return distance


def compute_mode_stats(modes):
    if len(modes) < 2:
        return 0, 0, 0.0, 0.0

    mode_change_count = 0
    takeover_count = 0
    autonomous_time = 0.0
    total_time = modes[-1].t - modes[0].t

    previous_mode = modes[0].mode

    for i in range(1, len(modes)):
        current_mode = modes[i].mode
        dt = modes[i].t - modes[i - 1].t

        if previous_mode == "AUTONOMOUS":
            autonomous_time += max(0.0, dt)

        if current_mode != previous_mode:
            mode_change_count += 1

            if previous_mode == "AUTONOMOUS" and current_mode != "AUTONOMOUS":
                takeover_count += 1

        previous_mode = current_mode

    return mode_change_count, takeover_count, autonomous_time, total_time


def compute_brake_events(velocities, accel_threshold=-0.5, harsh_threshold=-3.0):
    """
    Brake event detection from velocity derivative.
    """
    if len(velocities) < 5:
        return []

    ts = np.array([v.t for v in velocities])
    vs = np.array([v.v for v in velocities])

    order = np.argsort(ts)
    ts = ts[order]
    vs = vs[order]

    dt = np.diff(ts)
    dv = np.diff(vs)

    valid = dt > 1e-3
    accel = np.zeros_like(dv)
    accel[valid] = dv[valid] / dt[valid]

    jerk = np.zeros_like(accel)
    if len(accel) > 2:
        jerk[1:] = np.diff(accel) / np.maximum(dt[1:], 1e-3)

    events = []
    in_event = False
    start_idx = 0

    for i, a in enumerate(accel):
        if a < accel_threshold and not in_event:
            in_event = True
            start_idx = i

        if in_event and (a >= -0.2 or i == len(accel) - 1):
            end_idx = i
            event_accel = accel[start_idx:end_idx + 1]
            event_jerk = jerk[start_idx:end_idx + 1]

            if len(event_accel) > 0:
                min_accel = float(np.min(event_accel))
                max_jerk = float(np.max(np.abs(event_jerk)))

                event_type = "HARSH" if min_accel < harsh_threshold else "NORMAL"

                events.append(
                    BrakeEvent(
                        start_t=float(ts[start_idx]),
                        end_t=float(ts[end_idx]),
                        min_accel=min_accel,
                        max_jerk=max_jerk,
                        event_type=event_type,
                    )
                )

            in_event = False

    return events


# ============================================================
# Bag reader and evaluator
# ============================================================

class AutowareBagEvaluator:

    def __init__(self, bag_path, storage_id="sqlite3"):
        self.bag_path = bag_path
        self.storage_id = storage_id
        self.results = EvalResults()
        self.topic_types = {}

    def read_bag(self):
        if not os.path.exists(self.bag_path):
            raise FileNotFoundError(f"Bag path does not exist: {self.bag_path}")

        storage_options = rosbag2_py.StorageOptions(
            uri=self.bag_path,
            storage_id=self.storage_id,
        )

        converter_options = rosbag2_py.ConverterOptions(
            input_serialization_format="cdr",
            output_serialization_format="cdr",
        )

        reader = rosbag2_py.SequentialReader()
        reader.open(storage_options, converter_options)

        topic_types = reader.get_all_topics_and_types()
        self.topic_types = {t.name: t.type for t in topic_types}

        print("Available topics in bag:")
        for topic, typ in self.topic_types.items():
            print(f"  {topic}: {typ}")

        msg_types = {}
        for topic, type_name in self.topic_types.items():
            try:
                msg_types[topic] = get_message(type_name)
            except Exception as e:
                print(f"Could not load message type for {topic}: {type_name}. Error: {e}")

        while reader.has_next():
            topic, data, t_nanosec = reader.read_next()
            fallback_time = bag_time_to_sec(t_nanosec)

            if topic not in msg_types:
                continue

            try:
                msg = deserialize_message(data, msg_types[topic])
            except Exception:
                continue

            if topic == TOPIC_LOCALIZATION:
                pose = get_pose_from_msg(msg)
                if pose is not None:
                    if math.isnan(pose.t):
                        pose.t = fallback_time
                    self.results.poses.append(pose)

            elif topic == TOPIC_VELOCITY:
                vel = get_velocity_from_msg(msg, fallback_time)
                if vel is not None:
                    self.results.velocities.append(vel)

            elif topic == TOPIC_CONTROL_CMD:
                ctrl = get_control_from_msg(msg, fallback_time)
                self.results.controls.append(ctrl)

            elif topic == TOPIC_OPERATION_MODE:
                mode = get_mode_from_msg(msg, fallback_time)
                self.results.modes.append(mode)

            elif topic == TOPIC_TRAJECTORY:
                traj = get_trajectory_from_msg(msg, fallback_time)
                if traj is not None:
                    self.results.trajectories.append(traj)

            elif topic == TOPIC_STOP_REASONS:
                self.results.stop_reason_count += 1

        self.post_process()

    def post_process(self):
        r = self.results

        r.poses = sorted(r.poses, key=lambda p: p.t)
        r.velocities = sorted(r.velocities, key=lambda v: v.t)
        r.controls = sorted(r.controls, key=lambda c: c.t)
        r.modes = sorted(r.modes, key=lambda m: m.t)
        r.trajectories = sorted(r.trajectories, key=lambda tr: tr.t)

        r.distance_total = compute_distance(r.poses)
        r.autonomous_distance = compute_autonomous_distance(r.poses, r.modes)

        (
            r.mode_change_count,
            r.takeover_count,
            r.autonomous_time,
            r.total_time,
        ) = compute_mode_stats(r.modes)

        r.brake_events = compute_brake_events(r.velocities)
        r.harsh_brake_count = sum(1 for e in r.brake_events if e.event_type == "HARSH")

        self.compute_tracking_errors()

    def compute_tracking_errors(self):
        r = self.results

        if not r.poses or not r.trajectories:
            return

        velocity_times = np.array([v.t for v in r.velocities]) if r.velocities else np.array([])
        velocity_values = np.array([v.v for v in r.velocities]) if r.velocities else np.array([])

        for pose in r.poses:
            traj = find_nearest_trajectory(r.trajectories, pose.t, max_dt=1.0)

            if traj is None:
                continue

            lateral_error, heading_error, target_v = compute_nearest_path_error(pose, traj)

            r.lateral_errors.append((pose.t, lateral_error))
            r.heading_errors.append((pose.t, heading_error))

            if len(velocity_times) > 0 and not math.isnan(target_v):
                idx = int(np.argmin(np.abs(velocity_times - pose.t)))
                actual_v = velocity_values[idx]
                velocity_error = target_v - actual_v
                r.velocity_errors.append((pose.t, velocity_error))


# ============================================================
# GUI widgets
# ============================================================

class PlotCanvas(FigureCanvas):
    def __init__(self, title="Plot", width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.ax.set_title(title)
        self.fig.tight_layout()

    def plot_xy(self, x, y, title, xlabel, ylabel):
        self.ax.clear()
        self.ax.plot(x, y, linewidth=2)
        self.ax.set_title(title)
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        self.ax.grid(True)
        self.fig.tight_layout()
        self.draw()

    def plot_route(self, poses, modes):
        self.ax.clear()

        if len(poses) < 2:
            self.ax.set_title("Route")
            self.ax.text(0.5, 0.5, "No pose data", ha="center")
            self.draw()
            return

        xs_auto = []
        ys_auto = []
        xs_other = []
        ys_other = []

        for p in poses:
            mode = get_mode_at_time(modes, p.t)
            if mode == "AUTONOMOUS":
                xs_auto.append(p.x)
                ys_auto.append(p.y)
            else:
                xs_other.append(p.x)
                ys_other.append(p.y)

        if xs_other:
            self.ax.scatter(xs_other, ys_other, s=8, label="Other / Manual / Stop")

        if xs_auto:
            self.ax.scatter(xs_auto, ys_auto, s=8, label="Autonomous")

        self.ax.set_aspect("equal", adjustable="box")
        self.ax.set_title("Route colored by operation mode")
        self.ax.set_xlabel("X [m]")
        self.ax.set_ylabel("Y [m]")
        self.ax.legend()
        self.ax.grid(True)
        self.fig.tight_layout()
        self.draw()


class EvaluationGUI(QMainWindow):

    def __init__(self, results, bag_path):
        super().__init__()
        self.results = results
        self.bag_path = bag_path

        self.setWindowTitle("Autoware Autonomous Driving Evaluation Dashboard")
        self.resize(1500, 900)

        self.setup_ui()

    def setup_ui(self):
        main_widget = QWidget()
        main_layout = QVBoxLayout()

        title = QLabel("Autoware Autonomous Driving Evaluation Dashboard")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet(
            """
            QLabel {
                font-size: 26px;
                font-weight: bold;
                padding: 14px;
                color: white;
                background-color: #263238;
                border-radius: 8px;
            }
            """
        )

        subtitle = QLabel(f"Bag: {self.bag_path}")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet("font-size: 14px; padding: 6px; color: #455A64;")

        main_layout.addWidget(title)
        main_layout.addWidget(subtitle)

        tabs = QTabWidget()

        tabs.addTab(self.create_summary_tab(), "Summary")
        tabs.addTab(self.create_route_tab(), "Route")
        tabs.addTab(self.create_tracking_tab(), "Tracking")
        tabs.addTab(self.create_velocity_tab(), "Velocity")
        tabs.addTab(self.create_braking_tab(), "Braking")
        tabs.addTab(self.create_events_tab(), "Events")

        main_layout.addWidget(tabs)

        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

    def create_metric_card(self, name, value, unit="", color="#ECEFF1"):
        widget = QWidget()
        layout = QVBoxLayout()

        label_name = QLabel(name)
        label_name.setAlignment(Qt.AlignCenter)
        label_name.setStyleSheet("font-size: 14px; color: #37474F;")

        label_value = QLabel(f"{value} {unit}")
        label_value.setAlignment(Qt.AlignCenter)
        label_value.setStyleSheet("font-size: 24px; font-weight: bold; color: #102027;")

        layout.addWidget(label_name)
        layout.addWidget(label_value)

        widget.setLayout(layout)
        widget.setStyleSheet(
            f"""
            QWidget {{
                background-color: {color};
                border-radius: 12px;
                padding: 10px;
            }}
            """
        )

        return widget

    def create_summary_tab(self):
        r = self.results
        widget = QWidget()
        layout = QVBoxLayout()

        auto_dist_pct = 0.0
        if r.distance_total > 1e-6:
            auto_dist_pct = 100.0 * r.autonomous_distance / r.distance_total

        auto_time_pct = 0.0
        if r.total_time > 1e-6:
            auto_time_pct = 100.0 * r.autonomous_time / r.total_time

        mean_lat = self.mean_value(r.lateral_errors)
        max_lat = self.max_value(r.lateral_errors)
        rms_lat = self.rms_value(r.lateral_errors)

        mean_vel_err = self.mean_abs_value(r.velocity_errors)
        max_vel_err = self.max_abs_value(r.velocity_errors)

        grid = QGridLayout()

        grid.addWidget(self.create_metric_card("Total Distance", f"{r.distance_total:.2f}", "m", "#E3F2FD"), 0, 0)
        grid.addWidget(self.create_metric_card("Autonomous Distance", f"{r.autonomous_distance:.2f}", "m", "#E8F5E9"), 0, 1)
        grid.addWidget(self.create_metric_card("Autonomous Distance", f"{auto_dist_pct:.1f}", "%", "#C8E6C9"), 0, 2)
        grid.addWidget(self.create_metric_card("Autonomous Time", f"{auto_time_pct:.1f}", "%", "#DCEDC8"), 0, 3)

        grid.addWidget(self.create_metric_card("Takeovers", f"{r.takeover_count}", "", "#FFCDD2"), 1, 0)
        grid.addWidget(self.create_metric_card("Mode Changes", f"{r.mode_change_count}", "", "#FFE0B2"), 1, 1)
        grid.addWidget(self.create_metric_card("Brake Events", f"{len(r.brake_events)}", "", "#FFF9C4"), 1, 2)
        grid.addWidget(self.create_metric_card("Harsh Brakes", f"{r.harsh_brake_count}", "", "#FFAB91"), 1, 3)

        grid.addWidget(self.create_metric_card("Mean Lateral Error", f"{mean_lat:.3f}", "m", "#E1F5FE"), 2, 0)
        grid.addWidget(self.create_metric_card("RMS Lateral Error", f"{rms_lat:.3f}", "m", "#B3E5FC"), 2, 1)
        grid.addWidget(self.create_metric_card("Max Lateral Error", f"{max_lat:.3f}", "m", "#81D4FA"), 2, 2)
        grid.addWidget(self.create_metric_card("Mean |Velocity Error|", f"{mean_vel_err:.3f}", "m/s", "#D1C4E9"), 2, 3)

        layout.addLayout(grid)

        report = QTextEdit()
        report.setReadOnly(True)
        report.setStyleSheet("font-size: 14px; background-color: #FAFAFA; padding: 10px;")
        report.setText(self.generate_text_report())
        layout.addWidget(report)

        button_layout = QHBoxLayout()
        export_button = QPushButton("Export CSV Report")
        export_button.clicked.connect(self.export_csv)
        export_button.setStyleSheet("font-size: 14px; padding: 8px;")
        button_layout.addWidget(export_button)
        layout.addLayout(button_layout)

        widget.setLayout(layout)
        return widget

    def create_route_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()

        canvas = PlotCanvas(width=10, height=7)
        canvas.plot_route(self.results.poses, self.results.modes)

        layout.addWidget(canvas)
        widget.setLayout(layout)
        return widget

    def create_tracking_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()

        lateral = self.results.lateral_errors
        heading = self.results.heading_errors

        if lateral:
            t0 = lateral[0][0]
            ts = [p[0] - t0 for p in lateral]
            vals = [p[1] for p in lateral]
        else:
            ts = []
            vals = []

        canvas1 = PlotCanvas(width=10, height=4)
        canvas1.plot_xy(ts, vals, "Lateral Tracking Error", "Time [s]", "Lateral error [m]")

        if heading:
            t0 = heading[0][0]
            ts_h = [p[0] - t0 for p in heading]
            vals_h = [math.degrees(p[1]) for p in heading]
        else:
            ts_h = []
            vals_h = []

        canvas2 = PlotCanvas(width=10, height=4)
        canvas2.plot_xy(ts_h, vals_h, "Heading Error", "Time [s]", "Heading error [deg]")

        layout.addWidget(canvas1)
        layout.addWidget(canvas2)
        widget.setLayout(layout)
        return widget

    def create_velocity_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()

        r = self.results

        if r.velocities:
            t0 = r.velocities[0].t
            ts = [v.t - t0 for v in r.velocities]
            vs = [v.v for v in r.velocities]
        else:
            ts = []
            vs = []

        canvas1 = PlotCanvas(width=10, height=4)
        canvas1.plot_xy(ts, vs, "Actual Vehicle Velocity", "Time [s]", "Velocity [m/s]")

        if r.velocity_errors:
            t0 = r.velocity_errors[0][0]
            ts_e = [p[0] - t0 for p in r.velocity_errors]
            vals_e = [p[1] for p in r.velocity_errors]
        else:
            ts_e = []
            vals_e = []

        canvas2 = PlotCanvas(width=10, height=4)
        canvas2.plot_xy(ts_e, vals_e, "Velocity Tracking Error", "Time [s]", "Target - Actual [m/s]")

        layout.addWidget(canvas1)
        layout.addWidget(canvas2)
        widget.setLayout(layout)
        return widget

    def create_braking_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()

        r = self.results

        if len(r.velocities) >= 3:
            ts = np.array([v.t for v in r.velocities])
            vs = np.array([v.v for v in r.velocities])

            order = np.argsort(ts)
            ts = ts[order]
            vs = vs[order]

            dt = np.diff(ts)
            dv = np.diff(vs)
            accel = np.zeros_like(dv)
            valid = dt > 1e-3
            accel[valid] = dv[valid] / dt[valid]

            t0 = ts[0]
            t_acc = ts[1:] - t0
        else:
            t_acc = []
            accel = []

        canvas = PlotCanvas(width=10, height=5)
        canvas.plot_xy(t_acc, accel, "Estimated Longitudinal Acceleration", "Time [s]", "Acceleration [m/s²]")

        table = QTableWidget()
        table.setColumnCount(5)
        table.setHorizontalHeaderLabels(["Start [s]", "End [s]", "Duration [s]", "Min Accel [m/s²]", "Type"])
        table.setRowCount(len(r.brake_events))

        if r.brake_events:
            t0 = r.brake_events[0].start_t
        else:
            t0 = 0.0

        for row, e in enumerate(r.brake_events):
            table.setItem(row, 0, QTableWidgetItem(f"{e.start_t - t0:.2f}"))
            table.setItem(row, 1, QTableWidgetItem(f"{e.end_t - t0:.2f}"))
            table.setItem(row, 2, QTableWidgetItem(f"{e.end_t - e.start_t:.2f}"))
            table.setItem(row, 3, QTableWidgetItem(f"{e.min_accel:.2f}"))
            table.setItem(row, 4, QTableWidgetItem(e.event_type))

        layout.addWidget(canvas)
        layout.addWidget(table)

        widget.setLayout(layout)
        return widget

    def create_events_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()

        table = QTableWidget()
        table.setColumnCount(3)
        table.setHorizontalHeaderLabels(["Time [s]", "Mode", "Autoware Control Enabled"])
        table.setRowCount(len(self.results.modes))

        if self.results.modes:
            t0 = self.results.modes[0].t
        else:
            t0 = 0.0

        for row, m in enumerate(self.results.modes):
            table.setItem(row, 0, QTableWidgetItem(f"{m.t - t0:.2f}"))
            table.setItem(row, 1, QTableWidgetItem(m.mode))
            table.setItem(row, 2, QTableWidgetItem(str(m.autoware_control_enabled)))

        layout.addWidget(table)
        widget.setLayout(layout)
        return widget

    def mean_value(self, data):
        if not data:
            return 0.0
        return float(np.mean([p[1] for p in data]))

    def mean_abs_value(self, data):
        if not data:
            return 0.0
        return float(np.mean(np.abs([p[1] for p in data])))

    def max_value(self, data):
        if not data:
            return 0.0
        return float(np.max([p[1] for p in data]))

    def max_abs_value(self, data):
        if not data:
            return 0.0
        return float(np.max(np.abs([p[1] for p in data])))

    def rms_value(self, data):
        if not data:
            return 0.0
        arr = np.array([p[1] for p in data], dtype=float)
        return float(np.sqrt(np.mean(arr * arr)))

    def generate_text_report(self):
        r = self.results

        auto_dist_pct = 0.0
        if r.distance_total > 1e-6:
            auto_dist_pct = 100.0 * r.autonomous_distance / r.distance_total

        auto_time_pct = 0.0
        if r.total_time > 1e-6:
            auto_time_pct = 100.0 * r.autonomous_time / r.total_time

        mean_lat = self.mean_value(r.lateral_errors)
        rms_lat = self.rms_value(r.lateral_errors)
        max_lat = self.max_value(r.lateral_errors)

        mean_vel_err = self.mean_abs_value(r.velocity_errors)
        max_vel_err = self.max_abs_value(r.velocity_errors)

        takeover_per_km = 0.0
        if r.distance_total > 1e-6:
            takeover_per_km = r.takeover_count / (r.distance_total / 1000.0)

        harsh_per_km = 0.0
        if r.distance_total > 1e-6:
            harsh_per_km = r.harsh_brake_count / (r.distance_total / 1000.0)

        text = f"""
AUTOWARE AUTONOMOUS DRIVING EVALUATION REPORT

MISSION / ROUTE PERFORMANCE
---------------------------
Total driven distance:              {r.distance_total:.2f} m
Autonomous driven distance:         {r.autonomous_distance:.2f} m
Autonomous distance percentage:     {auto_dist_pct:.2f} %
Total evaluated time:               {r.total_time:.2f} s
Autonomous time:                    {r.autonomous_time:.2f} s
Autonomous time percentage:         {auto_time_pct:.2f} %

INTERVENTION / HANDOVER PERFORMANCE
-----------------------------------
Mode change count:                  {r.mode_change_count}
Takeover count:                     {r.takeover_count}
Takeover rate:                      {takeover_per_km:.2f} takeovers/km

TRAJECTORY TRACKING PERFORMANCE
-------------------------------
Mean lateral error:                 {mean_lat:.3f} m
RMS lateral error:                  {rms_lat:.3f} m
Maximum lateral error:              {max_lat:.3f} m
Mean absolute velocity error:       {mean_vel_err:.3f} m/s
Maximum absolute velocity error:    {max_vel_err:.3f} m/s

BRAKING / COMFORT PERFORMANCE
-----------------------------
Brake event count:                  {len(r.brake_events)}
Harsh brake count:                  {r.harsh_brake_count}
Harsh brake rate:                   {harsh_per_km:.2f} harsh brakes/km

PLANNING EVENTS
---------------
Stop reason message count:          {r.stop_reason_count}

INTERPRETATION
--------------
This report estimates route-level autonomy, trajectory tracking quality,
braking behavior, and intervention frequency from the recorded Autoware bag.

If lateral error is high, check localization, map alignment, trajectory generation,
and control tuning. If takeover rate is high, inspect operation mode transitions,
driver input reports, diagnostics, and stop reasons around each event.
"""
        return text

    def export_csv(self):
        output_dir = QFileDialog.getExistingDirectory(self, "Select output directory")
        if not output_dir:
            return

        r = self.results

        summary = {
            "total_distance_m": [r.distance_total],
            "autonomous_distance_m": [r.autonomous_distance],
            "autonomous_distance_percent": [
                100.0 * r.autonomous_distance / r.distance_total if r.distance_total > 1e-6 else 0.0
            ],
            "total_time_s": [r.total_time],
            "autonomous_time_s": [r.autonomous_time],
            "autonomous_time_percent": [
                100.0 * r.autonomous_time / r.total_time if r.total_time > 1e-6 else 0.0
            ],
            "takeover_count": [r.takeover_count],
            "mode_change_count": [r.mode_change_count],
            "brake_event_count": [len(r.brake_events)],
            "harsh_brake_count": [r.harsh_brake_count],
        }

        pd.DataFrame(summary).to_csv(os.path.join(output_dir, "summary_report.csv"), index=False)

        if r.lateral_errors:
            pd.DataFrame(r.lateral_errors, columns=["time_s", "lateral_error_m"]).to_csv(
                os.path.join(output_dir, "lateral_errors.csv"),
                index=False,
            )

        if r.heading_errors:
            pd.DataFrame(r.heading_errors, columns=["time_s", "heading_error_rad"]).to_csv(
                os.path.join(output_dir, "heading_errors.csv"),
                index=False,
            )

        if r.velocity_errors:
            pd.DataFrame(r.velocity_errors, columns=["time_s", "velocity_error_mps"]).to_csv(
                os.path.join(output_dir, "velocity_errors.csv"),
                index=False,
            )

        if r.brake_events:
            brake_rows = []
            for e in r.brake_events:
                brake_rows.append(
                    {
                        "start_time_s": e.start_t,
                        "end_time_s": e.end_t,
                        "duration_s": e.end_t - e.start_t,
                        "min_accel_mps2": e.min_accel,
                        "max_jerk_mps3": e.max_jerk,
                        "type": e.event_type,
                    }
                )

            pd.DataFrame(brake_rows).to_csv(
                os.path.join(output_dir, "brake_events.csv"),
                index=False,
            )

        print(f"CSV report exported to: {output_dir}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", required=True, help="Path to ROS 2 bag folder")
    parser.add_argument(
        "--storage-id",
        default="sqlite3",
        help="Bag storage type: sqlite3 or mcap",
    )

    args = parser.parse_args()

    print("==============================================")
    print(" Autoware Bag Evaluation GUI")
    print("==============================================")
    print(f"Bag path:   {args.bag}")
    print(f"Storage ID: {args.storage_id}")
    print("Reading bag...")

    evaluator = AutowareBagEvaluator(args.bag, args.storage_id)
    evaluator.read_bag()

    print("Bag analysis completed.")
    print("Opening GUI...")

    app = QApplication(sys.argv)
    gui = EvaluationGUI(evaluator.results, args.bag)
    gui.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
