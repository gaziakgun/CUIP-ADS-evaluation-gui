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
import textwrap
import time
from dataclasses import dataclass, field
from io import BytesIO

import numpy as np
import pandas as pd
import requests
from PIL import Image

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
    QLineEdit,
    QComboBox,
    QSpinBox,
    QDoubleSpinBox,
    QSlider,
    QMessageBox,
    QSizePolicy,
    QTextEdit,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QAbstractItemView,
)
from PyQt5.QtCore import Qt

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.collections import LineCollection
from matplotlib.colors import BoundaryNorm, ListedColormap, LogNorm, Normalize
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.widgets import SpanSelector

import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from pyproj import Transformer


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

EVALUATION_TOPICS = {
    TOPIC_LOCALIZATION,
    TOPIC_TRAJECTORY,
    TOPIC_CONTROL_CMD,
    TOPIC_VELOCITY,
    TOPIC_OPERATION_MODE,
    TOPIC_STOP_REASONS,
}

OUTLIER_MAD_THRESHOLD = 5.0
OUTLIER_MIN_SAMPLES = 10


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


@dataclass
class CompareBag:
    label: str
    bag_path: str
    storage_id: str
    results: EvalResults


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


def sample_times(samples):
    return np.array([s.t for s in samples], dtype=float)


def nearest_time_index(times, t, max_dt=None):
    if len(times) == 0:
        return None

    right = int(np.searchsorted(times, t, side="left"))
    candidates = []

    if right < len(times):
        candidates.append(right)
    if right > 0:
        candidates.append(right - 1)

    if not candidates:
        return None

    idx = min(candidates, key=lambda i: abs(times[i] - t))

    if max_dt is not None and abs(times[idx] - t) > max_dt:
        return None

    return idx


_transformer_to_3857 = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)


def latlon_to_webmercator(lat, lon):
    x, y = _transformer_to_3857.transform(lon, lat)
    return x, y


def rotate_local_xy_to_enu(x, y, origin_yaw_deg=0.0):
    yaw = math.radians(origin_yaw_deg)
    east = math.cos(yaw) * x - math.sin(yaw) * y
    north = math.sin(yaw) * x + math.cos(yaw) * y
    return east, north


def make_local_to_latlon_transformer(origin_lat, origin_lon):
    local_crs = (
        f"+proj=aeqd +lat_0={origin_lat} +lon_0={origin_lon} "
        "+datum=WGS84 +units=m +no_defs"
    )
    return Transformer.from_crs(local_crs, "EPSG:4326", always_xy=True)


def local_xy_to_latlon(x, y, origin_lat, origin_lon, origin_yaw_deg, transformer):
    east, north = rotate_local_xy_to_enu(x, y, origin_yaw_deg)
    lon, lat = transformer.transform(east, north)
    return lat, lon


def latlon_to_tile(lat, lon, zoom):
    lat = max(min(lat, 85.05112878), -85.05112878)
    lat_rad = math.radians(lat)
    n = 2.0 ** zoom

    xtile = int((lon + 180.0) / 360.0 * n)
    ytile = int(
        (1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi)
        / 2.0
        * n
    )

    max_tile = int(n - 1)
    return max(0, min(xtile, max_tile)), max(0, min(ytile, max_tile))


def tile_bounds_webmercator(x, y, zoom):
    n = 2.0 ** zoom

    lon_left = x / n * 360.0 - 180.0
    lon_right = (x + 1) / n * 360.0 - 180.0

    lat_top = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    lat_bottom = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (y + 1) / n))))

    x0, y_bottom = latlon_to_webmercator(lat_bottom, lon_left)
    x1, y_top = latlon_to_webmercator(lat_top, lon_right)

    return x0, x1, y_bottom, y_top


def download_osm_tile(x, y, zoom, cache_dir="osm_tile_cache"):
    os.makedirs(cache_dir, exist_ok=True)

    tile_path = os.path.join(cache_dir, f"{zoom}_{x}_{y}.png")
    if os.path.exists(tile_path):
        return Image.open(tile_path).convert("RGB")

    url = f"https://tile.openstreetmap.org/{zoom}/{x}/{y}.png"
    headers = {"User-Agent": "AutowareEvaluationDashboard/1.0"}
    response = requests.get(url, headers=headers, timeout=10)

    if response.status_code != 200:
        raise RuntimeError(f"Could not download OSM tile: {url}")

    image = Image.open(BytesIO(response.content)).convert("RGB")
    image.save(tile_path)
    time.sleep(0.1)
    return image


def create_osm_background(lat_list, lon_list, zoom=19, padding_tiles=1):
    tiles = [latlon_to_tile(lat, lon, zoom) for lat, lon in zip(lat_list, lon_list)]
    if not tiles:
        raise RuntimeError("No global pose samples available for OSM map.")

    xs = [tile[0] for tile in tiles]
    ys = [tile[1] for tile in tiles]

    min_x = min(xs) - padding_tiles
    max_x = max(xs) + padding_tiles
    min_y = min(ys) - padding_tiles
    max_y = max(ys) + padding_tiles

    tile_size = 256
    width = (max_x - min_x + 1) * tile_size
    height = (max_y - min_y + 1) * tile_size
    mosaic = Image.new("RGB", (width, height))

    for tx in range(min_x, max_x + 1):
        for ty in range(min_y, max_y + 1):
            tile = download_osm_tile(tx, ty, zoom)
            px = (tx - min_x) * tile_size
            py = (ty - min_y) * tile_size
            mosaic.paste(tile, (px, py))

    left, _, _, top = tile_bounds_webmercator(min_x, min_y, zoom)
    _, right, bottom, _ = tile_bounds_webmercator(max_x, max_y, zoom)

    return np.array(mosaic), [left, right, bottom, top]


def enable_scroll_zoom(canvas, base_scale=1.2):
    pan_state = {
        "active": False,
        "ax": None,
        "x": None,
        "y": None,
        "xlim": None,
        "ylim": None,
    }

    def on_scroll(event):
        if event.inaxes is None or event.xdata is None or event.ydata is None:
            return

        if event.button == "up":
            scale_factor = 1.0 / base_scale
        elif event.button == "down":
            scale_factor = base_scale
        else:
            return

        ax = event.inaxes
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        x_range = (x_max - x_min) * scale_factor
        y_range = (y_max - y_min) * scale_factor

        x_frac = (event.xdata - x_min) / (x_max - x_min)
        y_frac = (event.ydata - y_min) / (y_max - y_min)

        ax.set_xlim(event.xdata - x_range * x_frac, event.xdata + x_range * (1.0 - x_frac))
        ax.set_ylim(event.ydata - y_range * y_frac, event.ydata + y_range * (1.0 - y_frac))
        canvas.draw_idle()

    def on_press(event):
        if event.inaxes is None or event.xdata is None or event.ydata is None:
            return

        if event.button not in (2, 3):
            return

        pan_state["active"] = True
        pan_state["ax"] = event.inaxes
        pan_state["x"] = event.xdata
        pan_state["y"] = event.ydata
        pan_state["xlim"] = event.inaxes.get_xlim()
        pan_state["ylim"] = event.inaxes.get_ylim()

    def on_motion(event):
        if not pan_state["active"] or pan_state["ax"] is None:
            return
        if event.inaxes != pan_state["ax"] or event.xdata is None or event.ydata is None:
            return

        ax = pan_state["ax"]
        dx = event.xdata - pan_state["x"]
        dy = event.ydata - pan_state["y"]
        xlim = pan_state["xlim"]
        ylim = pan_state["ylim"]

        ax.set_xlim(xlim[0] - dx, xlim[1] - dx)
        ax.set_ylim(ylim[0] - dy, ylim[1] - dy)
        canvas.draw_idle()

    def on_release(event):
        pan_state["active"] = False
        pan_state["ax"] = None

    canvas._scroll_zoom_cid = canvas.mpl_connect("scroll_event", on_scroll)
    canvas._pan_press_cid = canvas.mpl_connect("button_press_event", on_press)
    canvas._pan_motion_cid = canvas.mpl_connect("motion_notify_event", on_motion)
    canvas._pan_release_cid = canvas.mpl_connect("button_release_event", on_release)


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


def find_nearest_trajectory(trajectories, t, max_dt=1.0, trajectory_times=None):
    if not trajectories:
        return None

    if trajectory_times is None:
        trajectory_times = sample_times(trajectories)

    idx = nearest_time_index(trajectory_times, t, max_dt=max_dt)
    if idx is None:
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


def cumulative_distance_samples(poses, max_step=5.0):
    if not poses:
        return np.array([]), np.array([])

    ordered = sorted(poses, key=lambda pose: pose.t)
    times = np.array([pose.t for pose in ordered], dtype=float)
    distances = np.zeros(len(ordered), dtype=float)

    for i in range(1, len(ordered)):
        dx = ordered[i].x - ordered[i - 1].x
        dy = ordered[i].y - ordered[i - 1].y
        step = math.sqrt(dx * dx + dy * dy)

        if step < max_step:
            distances[i] = distances[i - 1] + step
        else:
            distances[i] = distances[i - 1]

    return times, distances


class RouteLocationMapper:
    def __init__(self, poses):
        self.times, self.distances = cumulative_distance_samples(poses)
        self.available = len(self.times) >= 2 and len(self.distances) >= 2 and self.distances[-1] > 1e-6

        if self.available:
            unique_distances, unique_indices = np.unique(self.distances, return_index=True)
            self.unique_distances = unique_distances
            self.unique_times = self.times[unique_indices]
            self.available = len(self.unique_distances) >= 2
        else:
            self.unique_distances = np.array([])
            self.unique_times = np.array([])

    def time_to_distance(self, t):
        values = np.asarray(t, dtype=float)
        if not self.available:
            return np.zeros_like(values)

        return np.interp(
            values,
            self.times,
            self.distances,
            left=self.distances[0],
            right=self.distances[-1],
        )

    def distance_to_time(self, distance):
        values = np.asarray(distance, dtype=float)
        if not self.available:
            return np.zeros_like(values)

        return np.interp(
            values,
            self.unique_distances,
            self.unique_times,
            left=self.unique_times[0],
            right=self.unique_times[-1],
        )


def get_mode_at_time(modes, t, mode_times=None):
    if not modes:
        return "UNKNOWN"

    if mode_times is None:
        mode_times = sample_times(modes)

    idx = np.searchsorted(mode_times, t, side="right") - 1

    if idx < 0:
        idx = 0

    return modes[idx].mode


def compute_autonomous_distance(poses, modes):
    if len(poses) < 2 or not modes:
        return 0.0

    distance = 0.0
    mode_times = sample_times(modes)

    for i in range(1, len(poses)):
        mode = get_mode_at_time(modes, poses[i].t, mode_times)

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


def compute_mode_stats_for_interval(modes, start_t, end_t):
    if start_t is None or end_t is None or end_t <= start_t:
        return 0, 0, 0.0, 0.0
    if not modes:
        return 0, 0, 0.0, end_t - start_t

    mode_times = sample_times(modes)
    previous_mode = get_mode_at_time(modes, start_t, mode_times)
    previous_t = start_t
    mode_change_count = 0
    takeover_count = 0
    autonomous_time = 0.0

    for mode in modes:
        if mode.t <= start_t:
            continue
        if mode.t > end_t:
            break

        dt = mode.t - previous_t
        if previous_mode == "AUTONOMOUS":
            autonomous_time += max(0.0, dt)

        if mode.mode != previous_mode:
            mode_change_count += 1
            if previous_mode == "AUTONOMOUS" and mode.mode != "AUTONOMOUS":
                takeover_count += 1

        previous_mode = mode.mode
        previous_t = mode.t

    if previous_mode == "AUTONOMOUS":
        autonomous_time += max(0.0, end_t - previous_t)

    return mode_change_count, takeover_count, autonomous_time, end_t - start_t


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


def compute_acceleration_arrays(velocities):
    if len(velocities) < 3:
        return np.array([]), np.array([])

    ts = sample_times(velocities)
    vs = np.array([v.v for v in velocities], dtype=float)

    order = np.argsort(ts)
    ts = ts[order]
    vs = vs[order]

    dt = np.diff(ts)
    dv = np.diff(vs)

    accel = np.zeros_like(dv)
    valid = dt > 1e-3
    accel[valid] = dv[valid] / dt[valid]

    return ts[1:], accel


def gaussian_kernel1d(sigma):
    if sigma <= 0:
        return np.array([1.0])

    radius = max(1, int(math.ceil(3.0 * sigma)))
    xs = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-(xs * xs) / (2.0 * sigma * sigma))
    return kernel / np.sum(kernel)


def smooth_grid(grid, sigma):
    kernel = gaussian_kernel1d(sigma)
    if len(kernel) == 1:
        return grid

    smoothed = np.apply_along_axis(
        lambda row: np.convolve(row, kernel, mode="same"),
        axis=0,
        arr=grid,
    )
    smoothed = np.apply_along_axis(
        lambda col: np.convolve(col, kernel, mode="same"),
        axis=1,
        arr=smoothed,
    )
    return smoothed


def heatmap_sigma(metric, bins):
    base = max(1.0, bins / 250.0)
    if metric == "density":
        return 2.5 * base
    if metric in ("lateral_error", "brake"):
        return 1.8 * base
    return 1.2 * base


def robust_norm(data, metric):
    if metric == "operation_mode":
        return BoundaryNorm([-0.5, 0.5, 1.5], 2), 0.0, 1.0

    finite = np.asarray(data[np.isfinite(data)], dtype=float)

    if metric in ("density", "brake"):
        finite = finite[finite > 1e-6]

    if len(finite) == 0:
        return Normalize(vmin=0.0, vmax=1.0), 0.0, 1.0

    if metric == "density":
        vmin = float(np.percentile(finite, 5))
        vmax = float(np.percentile(finite, 98))
        vmin = max(vmin, float(np.min(finite)), 1e-6)
        vmax = max(vmax, vmin * 1.01)

        if vmax / vmin > 3.0:
            return LogNorm(vmin=vmin, vmax=vmax), vmin, vmax

        return Normalize(vmin=vmin, vmax=vmax), vmin, vmax

    if metric == "brake":
        vmin = 0.0
        vmax = float(np.percentile(finite, 98))
    else:
        vmin = float(np.percentile(finite, 2))
        vmax = float(np.percentile(finite, 98))

    if vmax <= vmin:
        center = float(np.mean(finite))
        pad = max(abs(center) * 0.05, 1e-3)
        vmin = center - pad
        vmax = center + pad

        if metric in ("lateral_error", "brake"):
            vmin = max(0.0, vmin)

    return Normalize(vmin=vmin, vmax=vmax), vmin, vmax


def robust_center_scale(values):
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)

    finite_values = values[finite]
    if len(finite_values) == 0:
        return 0.0, 0.0

    median = float(np.median(finite_values))
    deviations = np.abs(finite_values - median)
    mad = float(np.median(deviations))

    if mad > 1e-9:
        return median, 1.4826 * mad

    q1, q3 = np.percentile(finite_values, [25.0, 75.0])
    iqr = float(q3 - q1)
    if iqr > 1e-9:
        return median, iqr / 1.349

    return median, 0.0


def robust_outlier_mask(values, threshold=OUTLIER_MAD_THRESHOLD, min_samples=OUTLIER_MIN_SAMPLES):
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    mask = finite.copy()

    finite_count = int(np.count_nonzero(finite))
    if finite_count < min_samples:
        return mask

    median, robust_sigma = robust_center_scale(values)
    if robust_sigma <= 1e-9:
        return mask

    mask[finite] = np.abs(values[finite] - median) <= threshold * robust_sigma
    return mask


def isolated_spike_mask(values, threshold=OUTLIER_MAD_THRESHOLD, min_samples=OUTLIER_MIN_SAMPLES):
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    keep = finite.copy()

    if int(np.count_nonzero(finite)) < min_samples:
        return keep

    median, robust_sigma = robust_center_scale(values)
    if robust_sigma <= 1e-9:
        return keep

    extreme = finite & (np.abs(values - median) > threshold * robust_sigma)
    finite_indices = np.flatnonzero(finite)
    jump_threshold = max(2.0 * robust_sigma, 1e-9)

    for position, idx in enumerate(finite_indices):
        if not extreme[idx]:
            continue
        if position == 0 or position == len(finite_indices) - 1:
            continue

        prev_idx = finite_indices[position - 1]
        next_idx = finite_indices[position + 1]

        if extreme[prev_idx] or extreme[next_idx]:
            continue

        prev_value = values[prev_idx]
        value = values[idx]
        next_value = values[next_idx]

        jumps_away_from_neighbors = (
            abs(value - prev_value) > jump_threshold
            and abs(value - next_value) > jump_threshold
        )
        neighbors_are_consistent = abs(prev_value - next_value) <= jump_threshold

        if jumps_away_from_neighbors and neighbors_are_consistent:
            keep[idx] = False

    return keep


def filter_outliers(values):
    values = np.asarray(values, dtype=float)
    return values[isolated_spike_mask(values)]


def filter_sample_pairs(data):
    if not data:
        return []

    values = np.array([p[1] for p in data], dtype=float)
    mask = isolated_spike_mask(values)
    return [point for point, keep in zip(data, mask) if keep]


def outlier_count(data):
    if not data:
        return 0

    values = np.array([p[1] for p in data], dtype=float)
    mask = isolated_spike_mask(values)
    return int(len(values) - np.count_nonzero(mask))


def filtered_values_from_pairs(data, transform=None):
    if not data:
        return np.array([])

    values = np.array([p[1] for p in data], dtype=float)
    if transform is not None:
        values = transform(values)

    filtered = filter_outliers(values)
    if len(filtered) == 0:
        return values[np.isfinite(values)]

    return filtered


def mean_or_zero(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return 0.0
    return float(np.mean(values))


def max_or_zero(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return 0.0
    return float(np.max(values))


def rms_or_zero(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return 0.0
    return float(np.sqrt(np.mean(values * values)))


def results_time_range(results):
    times = []

    if results.poses:
        times.extend([results.poses[0].t, results.poses[-1].t])

    for samples in (
        results.velocities,
        results.controls,
        results.modes,
        results.trajectories,
    ):
        if samples:
            times.extend([samples[0].t, samples[-1].t])

    for pairs in (
        results.lateral_errors,
        results.heading_errors,
        results.velocity_errors,
    ):
        if pairs:
            times.extend([pairs[0][0], pairs[-1][0]])

    finite_times = [t for t in times if np.isfinite(t)]
    if not finite_times:
        return None, None

    return min(finite_times), max(finite_times)


def results_time_origin(results):
    start_t, _ = results_time_range(results)
    return start_t if start_t is not None else 0.0


def summarize_results_for_compare(results):
    lateral = filtered_values_from_pairs(results.lateral_errors)
    heading_deg = filtered_values_from_pairs(results.heading_errors, transform=np.degrees)
    velocity_error = filtered_values_from_pairs(results.velocity_errors)
    accel_times, accel_values = compute_acceleration_arrays(results.velocities)
    speed_values = np.array([v.v for v in results.velocities], dtype=float) if results.velocities else np.array([])
    start_t, end_t = results_time_range(results)

    distance_total = results.distance_total
    autonomous_distance = results.autonomous_distance
    auto_dist_pct = 0.0
    if distance_total > 1e-6:
        auto_dist_pct = 100.0 * autonomous_distance / distance_total

    total_time = results.total_time
    if total_time <= 1e-6 and start_t is not None and end_t is not None:
        total_time = max(0.0, end_t - start_t)

    auto_time_pct = 0.0
    if total_time > 1e-6:
        auto_time_pct = 100.0 * results.autonomous_time / total_time

    harsh_brake_count = sum(1 for event in results.brake_events if event.event_type == "HARSH")

    return {
        "distance_total_m": distance_total,
        "autonomous_distance_m": autonomous_distance,
        "autonomous_distance_percent": auto_dist_pct,
        "total_time_s": total_time,
        "autonomous_time_percent": auto_time_pct,
        "takeover_count": results.takeover_count,
        "mode_change_count": results.mode_change_count,
        "brake_event_count": len(results.brake_events),
        "harsh_brake_count": harsh_brake_count,
        "mean_lateral_error_m": mean_or_zero(lateral),
        "rms_lateral_error_m": rms_or_zero(lateral),
        "max_lateral_error_m": max_or_zero(lateral),
        "mean_abs_heading_error_deg": mean_or_zero(np.abs(heading_deg)),
        "max_abs_heading_error_deg": max_or_zero(np.abs(heading_deg)),
        "mean_abs_velocity_error_mps": mean_or_zero(np.abs(velocity_error)),
        "max_abs_velocity_error_mps": max_or_zero(np.abs(velocity_error)),
        "mean_speed_mps": mean_or_zero(speed_values),
        "max_speed_mps": max_or_zero(speed_values),
        "mean_abs_accel_mps2": mean_or_zero(np.abs(accel_values)),
        "max_abs_accel_mps2": max_or_zero(np.abs(accel_values)),
        "lateral_sample_count": len(results.lateral_errors),
        "velocity_sample_count": len(results.velocity_errors),
        "accel_sample_count": len(accel_times),
    }


def compare_metric_series(results, metric):
    origin = results_time_origin(results)

    if metric == "lateral_error":
        data = filter_sample_pairs(results.lateral_errors)
        xs = np.array([t - origin for t, _ in data], dtype=float)
        ys = np.array([v for _, v in data], dtype=float)
        return xs, ys, "Lateral error [m]", "Lateral Tracking Error"

    if metric == "heading_error":
        data = filter_sample_pairs(results.heading_errors)
        xs = np.array([t - origin for t, _ in data], dtype=float)
        ys = np.array([math.degrees(v) for _, v in data], dtype=float)
        return xs, ys, "Heading error [deg]", "Heading Error"

    if metric == "velocity_error":
        data = filter_sample_pairs(results.velocity_errors)
        xs = np.array([t - origin for t, _ in data], dtype=float)
        ys = np.array([v for _, v in data], dtype=float)
        return xs, ys, "Target - actual [m/s]", "Velocity Tracking Error"

    if metric == "vehicle_speed":
        xs = np.array([v.t - origin for v in results.velocities], dtype=float)
        ys = np.array([v.v for v in results.velocities], dtype=float)
        return xs, ys, "Velocity [m/s]", "Actual Vehicle Speed"

    if metric == "acceleration":
        ts, accel = compute_acceleration_arrays(results.velocities)
        xs = ts - origin if len(ts) > 0 else np.array([])
        return xs, accel, "Acceleration [m/s^2]", "Estimated Longitudinal Acceleration"

    raise ValueError(f"Unknown compare metric: {metric}")


def sample_grid_values(xs, ys, heat, extent):
    xmin, xmax, ymin, ymax = extent
    bins_x, bins_y = heat.shape

    ix = np.floor((xs - xmin) / max(xmax - xmin, 1e-9) * bins_x).astype(int)
    iy = np.floor((ys - ymin) / max(ymax - ymin, 1e-9) * bins_y).astype(int)

    ix = np.clip(ix, 0, bins_x - 1)
    iy = np.clip(iy, 0, bins_y - 1)

    return heat[ix, iy]


def safe_filename(text):
    cleaned = []
    for char in text.lower():
        if char.isalnum():
            cleaned.append(char)
        elif cleaned and cleaned[-1] != "_":
            cleaned.append("_")

    name = "".join(cleaned).strip("_")
    return name or "figure"


def build_global_pose_samples(poses, origin_lat, origin_lon, origin_yaw_deg):
    transformer = make_local_to_latlon_transformer(origin_lat, origin_lon)
    samples = []

    for pose in poses:
        lat, lon = local_xy_to_latlon(
            pose.x,
            pose.y,
            origin_lat,
            origin_lon,
            origin_yaw_deg,
            transformer,
        )
        mx, my = latlon_to_webmercator(lat, lon)

        samples.append(
            {
                "pose": pose,
                "lat": lat,
                "lon": lon,
                "mx": mx,
                "my": my,
            }
        )

    return samples


def create_heatmap_values(global_samples, velocities, trajectories, metric, modes=None):
    velocity_times = sample_times(velocities) if velocities else np.array([])
    velocity_values = np.array([v.v for v in velocities], dtype=float) if velocities else np.array([])
    trajectory_times = sample_times(trajectories) if trajectories else np.array([])
    accel_times, accel_values = compute_acceleration_arrays(velocities)
    mode_times = sample_times(modes) if modes else np.array([])

    xs = []
    ys = []
    values = []
    times = []

    for sample in global_samples:
        pose = sample["pose"]

        if metric == "speed":
            idx = nearest_time_index(velocity_times, pose.t)
            value = float(velocity_values[idx]) if idx is not None else math.nan

        elif metric == "lateral_error":
            traj = find_nearest_trajectory(
                trajectories,
                pose.t,
                max_dt=1.0,
                trajectory_times=trajectory_times,
            )
            if traj is None:
                value = math.nan
            else:
                value, _, _ = compute_nearest_path_error(pose, traj)

        elif metric == "brake":
            idx = nearest_time_index(accel_times, pose.t)
            if idx is None:
                value = math.nan
            else:
                value = max(0.0, -float(accel_values[idx]))

        elif metric == "density":
            value = 1.0

        elif metric == "operation_mode":
            if not modes:
                value = math.nan
            else:
                mode = get_mode_at_time(modes, pose.t, mode_times)
                value = 1.0 if mode == "AUTONOMOUS" else 0.0

        else:
            raise ValueError(f"Unknown heatmap metric: {metric}")

        if math.isnan(value):
            continue

        xs.append(sample["mx"])
        ys.append(sample["my"])
        values.append(value)
        times.append(pose.t)

    xs = np.array(xs)
    ys = np.array(ys)
    values = np.array(values)
    times = np.array(times)
    removed_outliers = 0

    if metric in ("speed", "lateral_error", "brake") and len(values) > 0:
        mask = isolated_spike_mask(values)
        removed_outliers = int(len(values) - np.count_nonzero(mask))
        xs = xs[mask]
        ys = ys[mask]
        values = values[mask]

    return xs, ys, values, removed_outliers


def draw_osm_heatmap(
    fig,
    global_samples,
    velocities,
    trajectories,
    modes,
    background,
    extent,
    metric,
    bins=250,
    alpha=0.65,
    cmap="jet",
):
    xs, ys, values, removed_outliers = create_heatmap_values(
        global_samples,
        velocities,
        trajectories,
        metric,
        modes=modes,
    )
    route_xs = np.array([sample["mx"] for sample in global_samples], dtype=float)
    route_ys = np.array([sample["my"] for sample in global_samples], dtype=float)

    if len(xs) < 3:
        raise RuntimeError("Not enough valid samples for heatmap.")

    xmin, xmax, ymin, ymax = extent
    if metric == "operation_mode":
        heat_cmap = ListedColormap(["#E53935", "#2E7D32"])
    else:
        heat_cmap = matplotlib.cm.get_cmap(cmap).copy()
    heat_cmap.set_bad(alpha=0.0)

    display_values = values

    if metric == "density":
        sigma = heatmap_sigma(metric, bins)
        counts, _, _ = np.histogram2d(
            xs,
            ys,
            bins=bins,
            range=[[xmin, xmax], [ymin, ymax]],
        )
        heat = smooth_grid(counts, sigma)
        display_values = sample_grid_values(xs, ys, heat, extent)

    norm, vmin, vmax = robust_norm(display_values, metric)

    fig.clear()
    ax = fig.add_subplot(111)
    ax.imshow(background, extent=extent, origin="upper", zorder=0)

    ax.plot(
        route_xs,
        route_ys,
        color="black",
        linewidth=7.0,
        alpha=0.45,
        label="Driven route",
        zorder=5,
    )

    if len(xs) > 1:
        points = np.column_stack([xs, ys]).reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        if metric == "operation_mode":
            segment_values = display_values[:-1]
        else:
            segment_values = 0.5 * (display_values[:-1] + display_values[1:])

        valid_segments = np.isfinite(segment_values)
        if metric == "brake":
            valid_segments &= segment_values > 0.05
        elif metric == "density":
            valid_segments &= segment_values > 0.0

        segments = segments[valid_segments]
        segment_values = segment_values[valid_segments]

        if len(segments) > 0:
            colored_route = LineCollection(
                segments,
                cmap=heat_cmap,
                norm=norm,
                linewidth=4.8,
                alpha=min(1.0, alpha + 0.2),
                zorder=8,
            )
            colored_route.set_array(segment_values)
            ax.add_collection(colored_route)

    ax.scatter(
        route_xs[0],
        route_ys[0],
        s=80,
        marker="o",
        color="lime",
        edgecolor="black",
        label="Start",
        zorder=10,
    )
    ax.scatter(
        route_xs[-1],
        route_ys[-1],
        s=80,
        marker="X",
        color="red",
        edgecolor="black",
        label="End",
        zorder=10,
    )

    metric_titles = {
        "speed": "Vehicle Speed Heatmap",
        "lateral_error": "Trajectory Tracking Error Heatmap",
        "brake": "Braking Intensity Heatmap",
        "density": "Driving Density Heatmap",
        "operation_mode": "Route by Operation Mode",
    }
    metric_units = {
        "speed": "m/s",
        "lateral_error": "m",
        "brake": "m/s^2",
        "density": "sample count",
        "operation_mode": "",
    }

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(metric_titles.get(metric, "Autoware Heatmap"), fontsize=14, fontweight="bold")
    ax.set_xlabel("WebMercator X [m]")
    ax.set_ylabel("WebMercator Y [m]")
    ax.grid(True, alpha=0.2)

    if metric == "operation_mode":
        legend_handles = [
            Line2D([0], [0], color="#E53935", linewidth=4.8, label="Other / Manual / Stop"),
            Line2D([0], [0], color="#2E7D32", linewidth=4.8, label="Autonomous"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="lime", markeredgecolor="black", markersize=8, label="Start"),
            Line2D([0], [0], marker="X", color="w", markerfacecolor="red", markeredgecolor="black", markersize=8, label="End"),
        ]
        ax.legend(handles=legend_handles, loc="upper right")
    else:
        ax.legend(loc="upper right")
        mappable = matplotlib.cm.ScalarMappable(norm=norm, cmap=heat_cmap)
        mappable.set_array([])
        colorbar = fig.colorbar(mappable, ax=ax, shrink=0.75)
        colorbar.set_label(metric_units.get(metric, ""), fontsize=10)

    fig.tight_layout()
    return {
        "sample_count": len(xs),
        "outlier_count": removed_outliers,
        "vmin": vmin,
        "vmax": vmax,
    }


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

        selected_topics = sorted(set(self.topic_types).intersection(EVALUATION_TOPICS))
        if selected_topics:
            print("Reading evaluation topics only:")
            for topic in selected_topics:
                print(f"  {topic}")

            try:
                reader.set_filter(rosbag2_py.StorageFilter(topics=selected_topics))
            except Exception as e:
                print(f"Could not apply rosbag topic filter; falling back to full scan. Error: {e}")

        msg_types = {}
        for topic in selected_topics:
            type_name = self.topic_types[topic]
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

        trajectory_times = sample_times(r.trajectories)
        velocity_times = sample_times(r.velocities) if r.velocities else np.array([])
        velocity_values = np.array([v.v for v in r.velocities]) if r.velocities else np.array([])

        for pose in r.poses:
            traj = find_nearest_trajectory(
                r.trajectories,
                pose.t,
                max_dt=1.0,
                trajectory_times=trajectory_times,
            )

            if traj is None:
                continue

            lateral_error, heading_error, target_v = compute_nearest_path_error(pose, traj)

            r.lateral_errors.append((pose.t, lateral_error))
            r.heading_errors.append((pose.t, heading_error))

            if len(velocity_times) > 0 and not math.isnan(target_v):
                idx = nearest_time_index(velocity_times, pose.t)
                if idx is not None:
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
        self.has_y_scale_controls = False
        self.has_time_cursor = False
        self.time_origin = None
        self.time_cursor_min = None
        self.time_cursor_max = None
        self.time_cursor_line = None
        self.interval_patch = None
        self.span_selector = None
        self.location_axis = None
        super().__init__(self.fig)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.updateGeometry()
        enable_scroll_zoom(self)
        self.ax.set_title(title)
        self.fig.tight_layout()

    def plot_xy(self, x, y, title, xlabel, ylabel, location_mapper=None, time_origin=None):
        self.clear_location_axis()
        self.ax.clear()
        self.has_time_cursor = False
        self.time_cursor_line = None
        self.interval_patch = None
        self.ax.plot(x, y, linewidth=2)
        self.ax.set_title(title)
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        self.ax.grid(True)
        self.add_location_axis(location_mapper, time_origin)
        self.has_y_scale_controls = True
        self.apply_robust_y_scale()
        self.configure_time_cursor(x, time_origin)
        self.fig.tight_layout()
        self.draw()

    def plot_compare_series(self, series, title, xlabel, ylabel):
        self.clear_location_axis()
        self.ax.clear()
        self.has_time_cursor = False
        self.time_cursor_line = None
        self.interval_patch = None
        plotted = False

        for label, x, y in series:
            x = np.asarray(x, dtype=float)
            y = np.asarray(y, dtype=float)
            finite = np.isfinite(x) & np.isfinite(y)
            if not np.any(finite):
                continue

            self.ax.plot(x[finite], y[finite], linewidth=1.8, label=label)
            plotted = True

        self.ax.set_title(title)
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        self.ax.grid(True, alpha=0.35)

        if plotted:
            self.ax.legend(loc="best", fontsize=8)
            self.has_y_scale_controls = True
            self.apply_robust_y_scale()
        else:
            self.ax.text(0.5, 0.5, "No data available", ha="center", va="center", transform=self.ax.transAxes)
            self.has_y_scale_controls = False

        self.fig.tight_layout()
        self.draw()

    def plot_compare_bars(self, labels, values, title, ylabel):
        self.clear_location_axis()
        self.ax.clear()
        self.has_time_cursor = False
        self.time_cursor_line = None
        self.interval_patch = None
        self.has_y_scale_controls = False

        values = np.asarray(values, dtype=float)
        if not labels or len(values) == 0:
            self.ax.set_title(title)
            self.ax.text(0.5, 0.5, "No data available", ha="center", va="center", transform=self.ax.transAxes)
            self.fig.tight_layout()
            self.draw()
            return

        positions = np.arange(len(labels))
        bars = self.ax.bar(positions, values, color="#1976D2", alpha=0.82)
        self.ax.set_title(title)
        self.ax.set_ylabel(ylabel)
        self.ax.set_xticks(positions)
        self.ax.set_xticklabels(labels, rotation=20, ha="right")
        self.ax.grid(True, axis="y", alpha=0.3)

        finite_values = values[np.isfinite(values)]
        if len(finite_values) > 0:
            min_value = float(np.min(finite_values))
            max_value = float(np.max(finite_values))
            pad = max((max_value - min_value) * 0.12, abs(max_value) * 0.08, 1e-3)
            bottom = min(0.0, min_value - pad)
            top = max_value + pad
            if top <= bottom:
                top = bottom + 1.0
            self.ax.set_ylim(bottom, top)

        for bar, value in zip(bars, values):
            if not np.isfinite(value):
                continue
            self.ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height(),
                f"{value:.3g}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        self.fig.tight_layout()
        self.draw()

    def clear_location_axis(self):
        if self.location_axis is None:
            return

        try:
            self.location_axis.remove()
        except (KeyError, ValueError):
            pass

        self.location_axis = None

    def add_location_axis(self, location_mapper, time_origin):
        if location_mapper is None or time_origin is None:
            return
        if not getattr(location_mapper, "available", False):
            return

        def time_to_distance(relative_time):
            return location_mapper.time_to_distance(np.asarray(relative_time, dtype=float) + time_origin)

        def distance_to_time(distance):
            return location_mapper.distance_to_time(np.asarray(distance, dtype=float)) - time_origin

        self.location_axis = self.ax.secondary_xaxis(
            "top",
            functions=(time_to_distance, distance_to_time),
        )
        self.location_axis.set_xlabel("Route distance [m]")

    def visible_y_values(self):
        xlim = self.ax.get_xlim()
        ys = []

        for line in self.ax.get_lines():
            if getattr(line, "_is_time_cursor", False):
                continue

            x_data = np.asarray(line.get_xdata(), dtype=float)
            y_data = np.asarray(line.get_ydata(), dtype=float)

            if len(x_data) != len(y_data) or len(y_data) == 0:
                continue

            finite = np.isfinite(x_data) & np.isfinite(y_data)
            visible = finite & (x_data >= min(xlim)) & (x_data <= max(xlim))

            if not np.any(visible):
                visible = finite

            if np.any(visible):
                ys.append(y_data[visible])

        if not ys:
            return np.array([])

        return np.concatenate(ys)

    def set_y_limits_from_values(self, values, robust=False):
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]

        if len(values) == 0:
            return

        if robust and len(values) >= 10:
            low, high = np.percentile(values, [2.0, 98.0])
        else:
            low = float(np.min(values))
            high = float(np.max(values))

        if high <= low:
            center = float(np.mean(values))
            pad = max(abs(center) * 0.05, 1e-3)
            low = center - pad
            high = center + pad
        else:
            pad = 0.08 * (high - low)
            low -= pad
            high += pad

        if np.nanmin(values) >= 0.0:
            low = min(0.0, low)

        self.ax.set_ylim(low, high)
        self.fig.tight_layout()
        self.draw_idle()

    def apply_robust_y_scale(self):
        self.set_y_limits_from_values(self.visible_y_values(), robust=True)

    def apply_full_y_scale(self):
        self.set_y_limits_from_values(self.visible_y_values(), robust=False)

    def configure_time_cursor(self, x_values, time_origin):
        if time_origin is None:
            return

        x_values = np.asarray(x_values, dtype=float)
        finite = x_values[np.isfinite(x_values)]
        if len(finite) == 0:
            return

        self.time_origin = float(time_origin)
        self.time_cursor_min = self.time_origin + float(np.min(finite))
        self.time_cursor_max = self.time_origin + float(np.max(finite))
        self.has_time_cursor = self.time_cursor_max >= self.time_cursor_min

        self.time_cursor_line = self.ax.axvline(
            finite[0],
            color="#FFC107",
            linewidth=2.0,
            linestyle="--",
            alpha=0.95,
            zorder=20,
        )
        self.time_cursor_line._is_time_cursor = True
        self.time_cursor_line.set_visible(False)

    def set_time_cursor(self, selected_time, keep_visible=False):
        if not self.has_time_cursor or self.time_cursor_line is None:
            return

        if selected_time is None or not np.isfinite(selected_time):
            self.time_cursor_line.set_visible(False)
            self.draw_idle()
            return

        selected_time = float(selected_time)
        in_range = self.time_cursor_min <= selected_time <= self.time_cursor_max
        self.time_cursor_line.set_visible(in_range)

        if in_range:
            cursor_x = selected_time - self.time_origin
            self.time_cursor_line.set_xdata([cursor_x, cursor_x])
            if keep_visible:
                self.keep_time_cursor_visible(cursor_x)

        self.draw_idle()

    def keep_time_cursor_visible(self, cursor_x, margin_fraction=0.15):
        x_min, x_max = self.ax.get_xlim()
        if x_max <= x_min:
            return

        span = x_max - x_min
        data_min = self.time_cursor_min - self.time_origin
        data_max = self.time_cursor_max - self.time_origin
        if data_max <= data_min:
            return

        margin = span * margin_fraction
        shift = 0.0
        if cursor_x < x_min + margin:
            shift = cursor_x - (x_min + margin)
        elif cursor_x > x_max - margin:
            shift = cursor_x - (x_max - margin)

        if abs(shift) <= 1e-9:
            return

        new_min = x_min + shift
        new_max = x_max + shift

        if span < data_max - data_min:
            if new_min < data_min:
                new_min = data_min
                new_max = data_min + span
            if new_max > data_max:
                new_max = data_max
                new_min = data_max - span

        self.ax.set_xlim(new_min, new_max)

    def set_analysis_interval(self, interval, mode="include"):
        if self.interval_patch is not None:
            try:
                self.interval_patch.remove()
            except (KeyError, ValueError):
                pass
            self.interval_patch = None

        if interval is None or self.time_origin is None:
            self.draw_idle()
            return

        start_t, end_t = interval
        start_x = start_t - self.time_origin
        end_x = end_t - self.time_origin
        color = "#C62828" if mode == "exclude" else "#1976D2"
        alpha = 0.22 if mode == "exclude" else 0.16
        self.interval_patch = self.ax.axvspan(
            start_x,
            end_x,
            color=color,
            alpha=alpha,
            zorder=1,
        )
        self.draw_idle()

    def plot_route(self, poses, modes):
        self.ax.clear()
        self.has_y_scale_controls = False
        self.has_time_cursor = False
        self.time_cursor_line = None
        self.interval_patch = None

        if len(poses) < 2:
            self.ax.set_title("Route")
            self.ax.text(0.5, 0.5, "No pose data", ha="center")
            self.draw()
            return

        xs_auto = []
        ys_auto = []
        xs_other = []
        ys_other = []
        mode_times = sample_times(modes) if modes else None

        for p in poses:
            mode = get_mode_at_time(modes, p.t, mode_times)
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

    def __init__(self, results, bag_path, storage_id="sqlite3", heatmap_defaults=None):
        super().__init__()
        self.results = results
        self.bag_path = bag_path
        self.storage_id = storage_id
        self.heatmap_defaults = heatmap_defaults or {}
        self.route_location_mapper = RouteLocationMapper(self.results.poses)
        self.metric_cards = []
        self.metric_card_values = {}
        self.compare_entries = [
            CompareBag(
                label=self.compare_label_for_path(bag_path),
                bag_path=bag_path,
                storage_id=storage_id,
                results=results,
            )
        ]
        self.compare_table = None
        self.compare_metric_combo = None
        self.compare_bar_metric_combo = None
        self.compare_storage_combo = None
        self.compare_status = None
        self.compare_series_canvas = None
        self.compare_bar_canvas = None
        self.analysis_interval = None
        self.analysis_interval_mode = "include"
        self.analysis_interval_status = None
        self.plot_canvases = []
        self.plot_entries = []
        self.time_scroll_controls = []
        self.selected_time = None
        self._syncing_time_scrolls = False
        self.heatmap_global_samples = []
        self.heatmap_pose_times = np.array([])
        self.heatmap_fig = None
        self.heatmap_canvas = None
        self.heatmap_marker = None
        self.heatmap_marker_label = None
        self.heatmap_time_scroll = None
        self.heatmap_time_label = None
        self.heatmap_save_button = None
        self.heatmap_status = None
        self.time_cursor_status = None
        self._last_style_scale = None

        self.setWindowTitle("Autoware Autonomous Driving Evaluation Dashboard")
        self.resize(1500, 900)

        self.setup_ui()
        self.apply_adaptive_styles()

    def setup_ui(self):
        main_widget = QWidget()
        self.main_widget = main_widget
        main_layout = QVBoxLayout()
        self.main_layout = main_layout

        self.title_label = QLabel("Autoware Autonomous Driving Evaluation Dashboard")
        self.title_label.setAlignment(Qt.AlignCenter)

        self.subtitle_label = QLabel(f"Bag: {self.bag_path}")
        self.subtitle_label.setAlignment(Qt.AlignCenter)

        main_layout.addWidget(self.title_label)
        main_layout.addWidget(self.subtitle_label)

        tabs = QTabWidget()
        self.tabs = tabs

        tabs.addTab(self.create_summary_tab(), "Summary")
        tabs.addTab(self.create_compare_tab(), "Compare")
        tabs.addTab(self.create_heatmap_tab(), "OSM Heatmap")
        tabs.addTab(self.create_tracking_tab(), "Tracking")
        tabs.addTab(self.create_velocity_tab(), "Velocity")
        tabs.addTab(self.create_braking_tab(), "Braking")
        tabs.addTab(self.create_events_tab(), "Events")

        main_layout.addWidget(tabs)

        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

    def compare_label_for_path(self, bag_path):
        label = os.path.basename(os.path.normpath(bag_path))
        return label or bag_path

    def unique_compare_label(self, bag_path):
        base_label = self.compare_label_for_path(bag_path)
        existing_labels = {entry.label for entry in self.compare_entries}
        if base_label not in existing_labels:
            return base_label

        suffix = 2
        while f"{base_label} ({suffix})" in existing_labels:
            suffix += 1
        return f"{base_label} ({suffix})"

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.apply_adaptive_styles()

    def responsive_scale(self):
        area_ratio = (max(self.width(), 1) * max(self.height(), 1)) / (1500.0 * 900.0)
        return max(0.95, min(1.75, math.sqrt(area_ratio)))

    def apply_adaptive_styles(self, force=False):
        if not hasattr(self, "title_label"):
            return

        scale = self.responsive_scale()
        if not force and self._last_style_scale is not None and abs(scale - self._last_style_scale) < 0.03:
            return
        self._last_style_scale = scale

        title_px = int(26 * scale)
        subtitle_px = int(13 * scale)
        tab_px = int(16 * scale)
        card_name_px = int(14 * scale)
        card_value_px = int(24 * scale)
        report_px = int(13 * scale)
        control_px = int(13 * scale)

        outer_margin = int(14 * scale)
        main_spacing = int(8 * scale)
        card_radius = int(8 * scale)
        card_pad = int(10 * scale)
        tab_pad_x = int(18 * scale)
        tab_pad_y = int(8 * scale)

        self.main_layout.setContentsMargins(outer_margin, outer_margin, outer_margin, outer_margin)
        self.main_layout.setSpacing(main_spacing)

        self.title_label.setStyleSheet(
            f"""
            QLabel {{
                font-size: {title_px}px;
                font-weight: bold;
                padding: {int(12 * scale)}px;
                color: white;
                background-color: #263238;
                border-radius: {card_radius}px;
            }}
            """
        )
        self.subtitle_label.setStyleSheet(
            f"font-size: {subtitle_px}px; padding: {int(5 * scale)}px; color: #455A64;"
        )

        self.tabs.setStyleSheet(
            f"""
            QTabBar::tab {{
                font-size: {tab_px}px;
                padding: {tab_pad_y}px {tab_pad_x}px;
                min-height: {int(22 * scale)}px;
            }}
            QTabWidget::pane {{
                border: 1px solid #BDBDBD;
            }}
            QPushButton {{
                font-size: {control_px}px;
                padding: {int(6 * scale)}px {int(10 * scale)}px;
            }}
            QTabWidget QLabel {{
                font-size: {control_px}px;
            }}
            QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
                font-size: {control_px}px;
                min-height: {int(24 * scale)}px;
            }}
            QTableWidget {{
                font-size: {control_px}px;
            }}
            """
        )

        for card, label_name, label_value, card_layout, color in self.metric_cards:
            card_layout.setContentsMargins(card_pad, card_pad, card_pad, card_pad)
            card_layout.setSpacing(int(8 * scale))
            card.setMinimumHeight(int(108 * scale))
            card.setStyleSheet(
                f"""
                QWidget {{
                    background-color: {color};
                    border-radius: {card_radius}px;
                }}
                """
            )
            label_name.setStyleSheet(f"font-size: {card_name_px}px; color: #37474F;")
            label_value.setStyleSheet(
                f"font-size: {card_value_px}px; font-weight: bold; color: #102027;"
            )

        if hasattr(self, "summary_grid"):
            self.summary_grid.setHorizontalSpacing(int(12 * scale))
            self.summary_grid.setVerticalSpacing(int(12 * scale))

        if hasattr(self, "report"):
            self.report.setStyleSheet(
                f"font-size: {report_px}px; background-color: #FAFAFA; padding: {int(10 * scale)}px;"
            )

        if hasattr(self, "summary_layout"):
            self.summary_layout.setContentsMargins(int(12 * scale), int(12 * scale), int(12 * scale), int(12 * scale))
            self.summary_layout.setSpacing(int(10 * scale))

        if hasattr(self, "heatmap_status") and self.heatmap_status is not None:
            self.heatmap_status.setMaximumHeight(int(24 * scale))
            self.heatmap_status.setStyleSheet(
                f"font-size: {control_px}px; color: #455A64; padding: {int(2 * scale)}px;"
            )

        if hasattr(self, "time_cursor_status") and self.time_cursor_status is not None:
            self.time_cursor_status.setMaximumHeight(int(36 * scale))
            self.time_cursor_status.setStyleSheet(
                f"font-size: {control_px}px; color: #263238; padding: {int(2 * scale)}px;"
            )

        if hasattr(self, "heatmap_control_panel"):
            self.heatmap_control_panel.setMaximumHeight(int(255 * scale))

        plot_title_px = int(13 * scale)
        plot_label_px = int(10 * scale)
        plot_tick_px = int(9 * scale)
        for canvas in self.plot_canvases:
            figure = getattr(canvas, "fig", None)
            if figure is None:
                figure = getattr(canvas, "figure", None)
            if figure is None:
                continue

            for ax in figure.axes:
                ax.title.set_fontsize(plot_title_px)
                ax.xaxis.label.set_fontsize(plot_label_px)
                ax.yaxis.label.set_fontsize(plot_label_px)
                ax.tick_params(labelsize=plot_tick_px)
                legend = ax.get_legend()
                if legend is not None:
                    for text in legend.get_texts():
                        text.set_fontsize(plot_tick_px)

            location_axis = getattr(canvas, "location_axis", None)
            if location_axis is not None:
                location_axis.xaxis.label.set_fontsize(plot_label_px)
                location_axis.tick_params(labelsize=plot_tick_px)

            figure.tight_layout()
            canvas.draw_idle()

    def create_metric_card(self, name, value, unit="", color="#ECEFF1", key=None):
        widget = QWidget()
        layout = QVBoxLayout()

        label_name = QLabel(name)
        label_name.setAlignment(Qt.AlignCenter)

        label_value = QLabel(f"{value} {unit}")
        label_value.setAlignment(Qt.AlignCenter)

        layout.addWidget(label_name)
        layout.addWidget(label_value)

        widget.setLayout(layout)
        self.metric_cards.append((widget, label_name, label_value, layout, color))
        if key is not None:
            self.metric_card_values[key] = (label_value, unit)

        return widget

    def set_metric_card_value(self, key, value):
        entry = self.metric_card_values.get(key)
        if entry is None:
            return

        label_value, unit = entry
        suffix = f" {unit}" if unit else " "
        label_value.setText(f"{value}{suffix}")

    def add_plot_with_toolbar(self, layout, canvas, title=None, stretch=1):
        toolbar = NavigationToolbar(canvas, self)
        toolbar.setMaximumHeight(34)
        toolbar.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        if title is None:
            figure = self.figure_for_canvas(canvas)
            title = "Figure"
            if figure is not None and figure.axes:
                title = figure.axes[0].get_title() or title

        if canvas not in self.plot_canvases:
            self.plot_canvases.append(canvas)

        if all(entry["canvas"] is not canvas for entry in self.plot_entries):
            self.plot_entries.append({"title": title, "canvas": canvas})

        toolbar_layout = QHBoxLayout()
        toolbar_layout.addWidget(toolbar)

        if getattr(canvas, "has_y_scale_controls", False):
            robust_button = QPushButton("Robust Y")
            robust_button.setToolTip(
                "Scale the Y axis using the 2nd to 98th percentile of the visible time range."
            )
            robust_button.clicked.connect(
                lambda checked=False, c=canvas: c.apply_robust_y_scale()
            )

            full_button = QPushButton("Full Y")
            full_button.setToolTip(
                "Scale the Y axis using the full min/max of the visible time range."
            )
            full_button.clicked.connect(
                lambda checked=False, c=canvas: c.apply_full_y_scale()
            )

            toolbar_layout.addWidget(robust_button)
            toolbar_layout.addWidget(full_button)

        if getattr(canvas, "has_time_cursor", False):
            time_label = QLabel("Time --")
            time_label.setMinimumWidth(95)
            time_label.setToolTip("Selected time on this plot.")

            time_scroll = QSlider(Qt.Horizontal)
            time_scroll.setRange(0, 10000)
            time_scroll.setSingleStep(10)
            time_scroll.setPageStep(250)
            time_scroll.setMinimumWidth(180)
            time_scroll.setToolTip(
                "Scroll through this graph's X axis. The yellow graph cursor and heatmap marker follow this time."
            )
            time_scroll.valueChanged.connect(
                lambda value, c=canvas: self.on_time_scroll_changed(c, value)
            )

            toolbar_layout.addWidget(QLabel("Time Scroll"))
            toolbar_layout.addWidget(time_scroll, 1)
            toolbar_layout.addWidget(time_label)
            self.register_time_scroll(canvas, time_scroll, time_label)
            self.register_interval_selector(canvas)

            clear_interval_button = QPushButton("Clear Interval")
            clear_interval_button.setToolTip("Clear the selected analysis time interval.")
            clear_interval_button.clicked.connect(self.clear_analysis_interval)
            toolbar_layout.addWidget(clear_interval_button)

            exclude_interval_button = QPushButton("Exclude Interval")
            exclude_interval_button.setToolTip(
                "Exclude the selected interval from calculations and use the unselected data."
            )
            exclude_interval_button.clicked.connect(self.exclude_analysis_interval)
            toolbar_layout.addWidget(exclude_interval_button)

        save_button = QPushButton("Save Figure")
        save_button.clicked.connect(
            lambda checked=False, c=canvas, t=title: self.save_figure_dialog(c, t)
        )
        toolbar_layout.addWidget(save_button)

        layout.addLayout(toolbar_layout)
        layout.addWidget(canvas, stretch)

    def register_interval_selector(self, canvas):
        if not getattr(canvas, "has_time_cursor", False):
            return
        if getattr(canvas, "span_selector", None) is not None:
            return

        callback = lambda start_x, end_x, c=canvas: self.on_plot_interval_selected(c, start_x, end_x)
        selector_kwargs = {
            "useblit": True,
            "button": 1,
            "minspan": 0.05,
            "interactive": True,
        }
        fallback_selector_kwargs = {
            "useblit": True,
            "button": 1,
            "minspan": 0.05,
        }

        try:
            canvas.span_selector = SpanSelector(
                canvas.ax,
                callback,
                "horizontal",
                props={"facecolor": "#1976D2", "alpha": 0.18},
                **selector_kwargs,
            )
        except TypeError:
            canvas.span_selector = SpanSelector(
                canvas.ax,
                callback,
                "horizontal",
                rectprops={"facecolor": "#1976D2", "alpha": 0.18},
                **fallback_selector_kwargs,
            )

    def on_plot_interval_selected(self, canvas, start_x, end_x):
        if canvas.time_origin is None:
            return

        start_t = canvas.time_origin + min(start_x, end_x)
        end_t = canvas.time_origin + max(start_x, end_x)
        self.set_analysis_interval(start_t, end_t)

    def set_analysis_interval(self, start_t, end_t):
        if start_t is None or end_t is None:
            return

        start_t, end_t = sorted((float(start_t), float(end_t)))
        route_start, route_end = self.route_time_range()
        if route_start is not None and route_end is not None:
            start_t = max(start_t, route_start)
            end_t = min(end_t, route_end)

        if end_t <= start_t:
            return

        self.analysis_interval = (start_t, end_t)
        self.analysis_interval_mode = "include"
        self.update_analysis_interval_visuals()
        self.update_summary_display()

    def exclude_analysis_interval(self, checked=False):
        if self.analysis_interval is None:
            return

        self.analysis_interval_mode = "exclude"
        self.update_analysis_interval_visuals()
        self.update_summary_display()

    def clear_analysis_interval(self, checked=False):
        self.analysis_interval = None
        self.analysis_interval_mode = "include"
        self.update_analysis_interval_visuals()
        self.update_summary_display()

    def update_analysis_interval_visuals(self):
        for canvas in self.plot_canvases:
            if hasattr(canvas, "set_analysis_interval"):
                canvas.set_analysis_interval(self.analysis_interval, self.analysis_interval_mode)

        if self.analysis_interval_status is not None:
            self.analysis_interval_status.setText(self.analysis_interval_text())

    def analysis_interval_text(self):
        if self.analysis_interval is None:
            return "Analysis interval: full route. Drag across any time-series graph to calculate only that time window."

        start_t, end_t = self.analysis_interval
        origin = self.route_time_origin()
        if self.analysis_interval_mode == "exclude":
            prefix = "Excluded interval"
            suffix = "calculating outside the red interval"
        else:
            prefix = "Analysis interval"
            suffix = "calculating inside the blue interval"

        return (
            f"{prefix}: "
            f"{start_t - origin:.2f}s to {end_t - origin:.2f}s "
            f"({end_t - start_t:.2f}s duration); {suffix}."
        )

    def figure_for_canvas(self, canvas):
        figure = getattr(canvas, "fig", None)
        if figure is None:
            figure = getattr(canvas, "figure", None)
        return figure

    def register_time_scroll(self, canvas, time_scroll, time_label):
        self.time_scroll_controls.append(
            {
                "canvas": canvas,
                "scroll": time_scroll,
                "label": time_label,
            }
        )
        self.sync_time_scroll_controls()

    def on_time_scroll_changed(self, canvas, value):
        if self._syncing_time_scrolls:
            return

        selected_time = self.time_from_scroll_value(canvas, value)
        if selected_time is None:
            return

        self.set_selected_time(selected_time, keep_plots_visible=True)

    def time_from_scroll_value(self, canvas, value):
        if not getattr(canvas, "has_time_cursor", False):
            return None

        t_min = canvas.time_cursor_min
        t_max = canvas.time_cursor_max
        if t_min is None or t_max is None or t_max < t_min:
            return None

        fraction = float(value) / 10000.0
        return t_min + (t_max - t_min) * fraction

    def scroll_value_from_time(self, canvas, selected_time):
        if selected_time is None or not getattr(canvas, "has_time_cursor", False):
            return 0

        t_min = canvas.time_cursor_min
        t_max = canvas.time_cursor_max
        if t_min is None or t_max is None or t_max <= t_min:
            return 0

        fraction = (float(selected_time) - t_min) / (t_max - t_min)
        fraction = max(0.0, min(1.0, fraction))
        return int(round(fraction * 10000.0))

    def set_selected_time(self, selected_time, keep_plots_visible=False):
        if selected_time is None or not np.isfinite(selected_time):
            return

        self.selected_time = float(selected_time)

        for canvas in self.plot_canvases:
            if hasattr(canvas, "set_time_cursor"):
                canvas.set_time_cursor(self.selected_time, keep_visible=keep_plots_visible)

        self.sync_time_scroll_controls()
        self.update_heatmap_time_marker()

    def sync_time_scroll_controls(self):
        if not hasattr(self, "time_scroll_controls"):
            return

        self._syncing_time_scrolls = True
        try:
            if self.heatmap_time_scroll is not None and self.heatmap_time_label is not None:
                if self.selected_time is None:
                    self.heatmap_time_label.setText("Time --")
                else:
                    self.heatmap_time_scroll.setValue(
                        self.route_scroll_value_from_time(self.selected_time)
                    )
                    route_t = self.selected_time - self.route_time_origin()
                    self.heatmap_time_label.setText(f"{route_t:.2f} s")

            for entry in self.time_scroll_controls:
                canvas = entry["canvas"]
                time_scroll = entry["scroll"]
                time_label = entry["label"]

                if self.selected_time is None:
                    time_label.setText("Time --")
                    continue

                time_scroll.setValue(self.scroll_value_from_time(canvas, self.selected_time))
                rel_time = self.selected_time - canvas.time_origin
                in_range = canvas.time_cursor_min <= self.selected_time <= canvas.time_cursor_max
                if in_range:
                    time_label.setText(f"{rel_time:.2f} s")
                else:
                    time_label.setText("outside")
        finally:
            self._syncing_time_scrolls = False

    def route_time_range(self):
        if self.results.poses:
            return self.results.poses[0].t, self.results.poses[-1].t

        times = []
        for samples in (
            self.results.velocities,
            self.results.controls,
            self.results.modes,
            self.results.trajectories,
        ):
            times.extend([sample.t for sample in samples])

        if not times:
            return None, None

        return min(times), max(times)

    def route_time_origin(self):
        if self.results.poses:
            return self.results.poses[0].t

        candidate_times = []
        for samples in (
            self.results.velocities,
            self.results.controls,
            self.results.modes,
            self.results.trajectories,
        ):
            if samples:
                candidate_times.append(samples[0].t)

        return min(candidate_times) if candidate_times else 0.0

    def time_from_route_scroll_value(self, value):
        t_min, t_max = self.route_time_range()
        if t_min is None or t_max is None:
            return None
        if t_max <= t_min:
            return t_min

        fraction = float(value) / 10000.0
        return t_min + (t_max - t_min) * fraction

    def route_scroll_value_from_time(self, selected_time):
        t_min, t_max = self.route_time_range()
        if selected_time is None or t_min is None or t_max is None or t_max <= t_min:
            return 0

        fraction = (float(selected_time) - t_min) / (t_max - t_min)
        fraction = max(0.0, min(1.0, fraction))
        return int(round(fraction * 10000.0))

    def on_route_time_scroll_changed(self, value):
        if self._syncing_time_scrolls:
            return

        selected_time = self.time_from_route_scroll_value(value)
        if selected_time is None:
            return

        self.set_selected_time(selected_time, keep_plots_visible=True)

    def set_time_cursor_status(self, sample=None):
        if self.time_cursor_status is None:
            return

        if self.selected_time is None:
            self.time_cursor_status.setText(
                "Time cursor: use a graph's Time Scroll to inspect the matching route point."
            )
            return

        route_t = self.selected_time - self.route_time_origin()
        if sample is None:
            self.time_cursor_status.setText(
                f"Time cursor: {route_t:.2f} s from route start. Generate the heatmap to show the map marker."
            )
            return

        pose = sample["pose"]
        pose_route_t = pose.t - self.route_time_origin()
        distance_text = ""
        if self.route_location_mapper.available:
            distance = float(self.route_location_mapper.time_to_distance(np.array([pose.t]))[0])
            distance_text = f", distance {distance:.1f} m"

        self.time_cursor_status.setText(
            f"Time cursor: {route_t:.2f} s from route start; "
            f"nearest pose {pose_route_t:.2f} s{distance_text}, "
            f"lat {sample['lat']:.7f}, lon {sample['lon']:.7f}."
        )

    def update_heatmap_time_marker(self):
        if (
            self.selected_time is None
            or self.heatmap_canvas is None
            or not self.heatmap_global_samples
            or len(self.heatmap_pose_times) == 0
        ):
            self.set_time_cursor_status(sample=None)
            return

        idx = nearest_time_index(self.heatmap_pose_times, self.selected_time)
        if idx is None:
            self.set_time_cursor_status(sample=None)
            return

        sample = self.heatmap_global_samples[idx]
        ax = self.heatmap_fig.axes[0] if self.heatmap_fig is not None and self.heatmap_fig.axes else None
        if ax is None:
            self.set_time_cursor_status(sample=None)
            return

        if self.heatmap_marker is None or self.heatmap_marker.axes is None:
            self.heatmap_marker = ax.scatter(
                sample["mx"],
                sample["my"],
                s=190,
                marker="o",
                facecolors="none",
                edgecolors="#FFC107",
                linewidths=3.0,
                label="Selected time",
                zorder=30,
            )
        else:
            self.heatmap_marker.set_offsets([[sample["mx"], sample["my"]]])
            self.heatmap_marker.set_visible(True)

        label_text = f"{sample['pose'].t - self.route_time_origin():.2f}s"
        if self.heatmap_marker_label is None or self.heatmap_marker_label.axes is None:
            self.heatmap_marker_label = ax.text(
                sample["mx"],
                sample["my"],
                label_text,
                color="black",
                fontsize=9,
                fontweight="bold",
                ha="left",
                va="bottom",
                bbox={"boxstyle": "round,pad=0.2", "facecolor": "#FFC107", "edgecolor": "black", "alpha": 0.9},
                zorder=31,
            )
        else:
            self.heatmap_marker_label.set_position((sample["mx"], sample["my"]))
            self.heatmap_marker_label.set_text(label_text)
            self.heatmap_marker_label.set_visible(True)

        self.set_time_cursor_status(sample=sample)
        self.heatmap_canvas.draw_idle()

    def save_figure_dialog(self, canvas, title="figure", default_path=None):
        figure = self.figure_for_canvas(canvas)
        if figure is None:
            QMessageBox.warning(self, "Save Figure", "No figure is available to save.")
            return

        if default_path is None:
            default_path = f"{safe_filename(title)}.png"

        output_path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Save figure",
            default_path,
            "PNG files (*.png);;PDF files (*.pdf)",
        )
        if not output_path:
            return

        _, ext = os.path.splitext(output_path)
        if not ext:
            ext = ".pdf" if "PDF" in selected_filter else ".png"
            output_path += ext

        figure.savefig(output_path, dpi=200, bbox_inches="tight")

    def time_pairs_in_interval(self, data):
        if not data or self.analysis_interval is None:
            return data

        start_t, end_t = self.analysis_interval
        if self.analysis_interval_mode == "exclude":
            return [point for point in data if point[0] < start_t or point[0] > end_t]

        return [point for point in data if start_t <= point[0] <= end_t]

    def samples_in_interval(self, samples):
        if not samples or self.analysis_interval is None:
            return samples

        start_t, end_t = self.analysis_interval
        if self.analysis_interval_mode == "exclude":
            return [sample for sample in samples if sample.t < start_t or sample.t > end_t]

        return [sample for sample in samples if start_t <= sample.t <= end_t]

    def brake_events_in_interval(self):
        if self.analysis_interval is None:
            return self.results.brake_events

        start_t, end_t = self.analysis_interval
        if self.analysis_interval_mode == "exclude":
            return [
                event
                for event in self.results.brake_events
                if event.end_t < start_t or event.start_t > end_t
            ]

        return [
            event
            for event in self.results.brake_events
            if event.end_t >= start_t and event.start_t <= end_t
        ]

    def summary_stats(self):
        r = self.results

        if self.analysis_interval is None:
            distance_total = r.distance_total
            autonomous_distance = r.autonomous_distance
            mode_change_count = r.mode_change_count
            takeover_count = r.takeover_count
            autonomous_time = r.autonomous_time
            total_time = r.total_time
        elif self.analysis_interval_mode == "exclude":
            start_t, end_t = self.analysis_interval
            poses = self.samples_in_interval(r.poses)
            interval_duration = end_t - start_t
            (
                interval_mode_change_count,
                interval_takeover_count,
                interval_autonomous_time,
                _,
            ) = compute_mode_stats_for_interval(r.modes, start_t, end_t)

            distance_total = compute_distance(poses)
            autonomous_distance = compute_autonomous_distance(poses, r.modes)
            mode_change_count = max(0, r.mode_change_count - interval_mode_change_count)
            takeover_count = max(0, r.takeover_count - interval_takeover_count)
            autonomous_time = max(0.0, r.autonomous_time - interval_autonomous_time)
            total_time = max(0.0, r.total_time - interval_duration)
        else:
            start_t, end_t = self.analysis_interval
            poses = self.samples_in_interval(r.poses)
            distance_total = compute_distance(poses)
            autonomous_distance = compute_autonomous_distance(poses, r.modes)
            (
                mode_change_count,
                takeover_count,
                autonomous_time,
                total_time,
            ) = compute_mode_stats_for_interval(r.modes, start_t, end_t)

        brake_events = self.brake_events_in_interval()
        harsh_brake_count = sum(1 for event in brake_events if event.event_type == "HARSH")

        lateral_errors = self.time_pairs_in_interval(r.lateral_errors)
        velocity_errors = self.time_pairs_in_interval(r.velocity_errors)

        auto_dist_pct = 0.0
        if distance_total > 1e-6:
            auto_dist_pct = 100.0 * autonomous_distance / distance_total

        auto_time_pct = 0.0
        if total_time > 1e-6:
            auto_time_pct = 100.0 * autonomous_time / total_time

        takeover_per_km = 0.0
        if distance_total > 1e-6:
            takeover_per_km = takeover_count / (distance_total / 1000.0)

        harsh_per_km = 0.0
        if distance_total > 1e-6:
            harsh_per_km = harsh_brake_count / (distance_total / 1000.0)

        return {
            "distance_total": distance_total,
            "autonomous_distance": autonomous_distance,
            "auto_dist_pct": auto_dist_pct,
            "total_time": total_time,
            "autonomous_time": autonomous_time,
            "auto_time_pct": auto_time_pct,
            "mode_change_count": mode_change_count,
            "takeover_count": takeover_count,
            "takeover_per_km": takeover_per_km,
            "brake_events": brake_events,
            "brake_event_count": len(brake_events),
            "harsh_brake_count": harsh_brake_count,
            "harsh_per_km": harsh_per_km,
            "mean_lat": self.mean_value(lateral_errors),
            "rms_lat": self.rms_value(lateral_errors),
            "max_lat": self.max_value(lateral_errors),
            "mean_vel_err": self.mean_abs_value(velocity_errors),
            "max_vel_err": self.max_abs_value(velocity_errors),
            "lateral_outliers": outlier_count(lateral_errors),
            "velocity_outliers": outlier_count(velocity_errors),
            "lateral_sample_count": len(lateral_errors),
            "velocity_sample_count": len(velocity_errors),
        }

    def update_summary_display(self):
        if not self.metric_card_values:
            return

        stats = self.summary_stats()
        self.set_metric_card_value("distance_total", f"{stats['distance_total']:.2f}")
        self.set_metric_card_value("autonomous_distance", f"{stats['autonomous_distance']:.2f}")
        self.set_metric_card_value("auto_dist_pct", f"{stats['auto_dist_pct']:.1f}")
        self.set_metric_card_value("auto_time_pct", f"{stats['auto_time_pct']:.1f}")
        self.set_metric_card_value("takeover_count", f"{stats['takeover_count']}")
        self.set_metric_card_value("mode_change_count", f"{stats['mode_change_count']}")
        self.set_metric_card_value("brake_event_count", f"{stats['brake_event_count']}")
        self.set_metric_card_value("harsh_brake_count", f"{stats['harsh_brake_count']}")
        self.set_metric_card_value("mean_lat", f"{stats['mean_lat']:.3f}")
        self.set_metric_card_value("rms_lat", f"{stats['rms_lat']:.3f}")
        self.set_metric_card_value("max_lat", f"{stats['max_lat']:.3f}")
        self.set_metric_card_value("mean_vel_err", f"{stats['mean_vel_err']:.3f}")

        if hasattr(self, "report") and self.report is not None:
            self.report.setText(self.generate_text_report())

        if self.analysis_interval_status is not None:
            self.analysis_interval_status.setText(self.analysis_interval_text())

    def create_summary_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()
        self.summary_layout = layout
        stats = self.summary_stats()

        grid = QGridLayout()
        self.summary_grid = grid

        grid.addWidget(self.create_metric_card("Total Distance", f"{stats['distance_total']:.2f}", "m", "#E3F2FD", key="distance_total"), 0, 0)
        grid.addWidget(self.create_metric_card("Autonomous Distance", f"{stats['autonomous_distance']:.2f}", "m", "#E8F5E9", key="autonomous_distance"), 0, 1)
        grid.addWidget(self.create_metric_card("Autonomous Distance", f"{stats['auto_dist_pct']:.1f}", "%", "#C8E6C9", key="auto_dist_pct"), 0, 2)
        grid.addWidget(self.create_metric_card("Autonomous Time", f"{stats['auto_time_pct']:.1f}", "%", "#DCEDC8", key="auto_time_pct"), 0, 3)

        grid.addWidget(self.create_metric_card("Takeovers", f"{stats['takeover_count']}", "", "#FFCDD2", key="takeover_count"), 1, 0)
        grid.addWidget(self.create_metric_card("Mode Changes", f"{stats['mode_change_count']}", "", "#FFE0B2", key="mode_change_count"), 1, 1)
        grid.addWidget(self.create_metric_card("Brake Events", f"{stats['brake_event_count']}", "", "#FFF9C4", key="brake_event_count"), 1, 2)
        grid.addWidget(self.create_metric_card("Harsh Brakes", f"{stats['harsh_brake_count']}", "", "#FFAB91", key="harsh_brake_count"), 1, 3)

        grid.addWidget(self.create_metric_card("Mean Lateral Error", f"{stats['mean_lat']:.3f}", "m", "#E1F5FE", key="mean_lat"), 2, 0)
        grid.addWidget(self.create_metric_card("RMS Lateral Error", f"{stats['rms_lat']:.3f}", "m", "#B3E5FC", key="rms_lat"), 2, 1)
        grid.addWidget(self.create_metric_card("Max Lateral Error", f"{stats['max_lat']:.3f}", "m", "#81D4FA", key="max_lat"), 2, 2)
        grid.addWidget(self.create_metric_card("Mean |Velocity Error|", f"{stats['mean_vel_err']:.3f}", "m/s", "#D1C4E9", key="mean_vel_err"), 2, 3)

        layout.addLayout(grid)

        interval_layout = QHBoxLayout()
        self.analysis_interval_status = QLabel(self.analysis_interval_text())
        self.analysis_interval_status.setWordWrap(True)
        exclude_interval_button = QPushButton("Exclude Analysis Interval")
        exclude_interval_button.clicked.connect(self.exclude_analysis_interval)
        clear_interval_button = QPushButton("Clear Analysis Interval")
        clear_interval_button.clicked.connect(self.clear_analysis_interval)
        interval_layout.addWidget(self.analysis_interval_status, 1)
        interval_layout.addWidget(exclude_interval_button)
        interval_layout.addWidget(clear_interval_button)
        layout.addLayout(interval_layout)

        report = QTextEdit()
        self.report = report
        report.setReadOnly(True)
        report.setText(self.generate_text_report())
        layout.addWidget(report)

        button_layout = QHBoxLayout()
        export_button = QPushButton("Export CSV Report")
        self.export_button = export_button
        export_button.clicked.connect(self.export_csv)

        pdf_button = QPushButton("Create PDF Report")
        self.pdf_button = pdf_button
        pdf_button.clicked.connect(self.export_pdf_report)

        button_layout.addWidget(export_button)
        button_layout.addWidget(pdf_button)
        layout.addLayout(button_layout)

        widget.setLayout(layout)
        return widget

    def compare_table_columns(self):
        return [
            ("Bag", "label", None),
            ("Storage", "storage_id", None),
            ("Distance [m]", "distance_total_m", "{:.2f}"),
            ("Auto Dist [%]", "autonomous_distance_percent", "{:.1f}"),
            ("Auto Time [%]", "autonomous_time_percent", "{:.1f}"),
            ("Takeovers", "takeover_count", "{:.0f}"),
            ("Mode Changes", "mode_change_count", "{:.0f}"),
            ("Brake Events", "brake_event_count", "{:.0f}"),
            ("Harsh Brakes", "harsh_brake_count", "{:.0f}"),
            ("Mean Lat [m]", "mean_lateral_error_m", "{:.3f}"),
            ("RMS Lat [m]", "rms_lateral_error_m", "{:.3f}"),
            ("Max Lat [m]", "max_lateral_error_m", "{:.3f}"),
            ("Mean |Vel Err| [m/s]", "mean_abs_velocity_error_mps", "{:.3f}"),
            ("Max |Vel Err| [m/s]", "max_abs_velocity_error_mps", "{:.3f}"),
            ("Mean |Heading| [deg]", "mean_abs_heading_error_deg", "{:.3f}"),
            ("Max |Heading| [deg]", "max_abs_heading_error_deg", "{:.3f}"),
            ("Mean Speed [m/s]", "mean_speed_mps", "{:.3f}"),
            ("Max Speed [m/s]", "max_speed_mps", "{:.3f}"),
            ("Mean |Accel| [m/s^2]", "mean_abs_accel_mps2", "{:.3f}"),
            ("Max |Accel| [m/s^2]", "max_abs_accel_mps2", "{:.3f}"),
            ("Lat Samples", "lateral_sample_count", "{:.0f}"),
            ("Vel Err Samples", "velocity_sample_count", "{:.0f}"),
            ("Path", "bag_path", None),
        ]

    def compare_bar_metrics(self):
        return [
            ("RMS lateral error", "rms_lateral_error_m", "RMS lateral error [m]"),
            ("Mean lateral error", "mean_lateral_error_m", "Mean lateral error [m]"),
            ("Max lateral error", "max_lateral_error_m", "Max lateral error [m]"),
            ("Mean abs velocity error", "mean_abs_velocity_error_mps", "Mean |velocity error| [m/s]"),
            ("Max abs velocity error", "max_abs_velocity_error_mps", "Max |velocity error| [m/s]"),
            ("Mean abs heading error", "mean_abs_heading_error_deg", "Mean |heading error| [deg]"),
            ("Distance", "distance_total_m", "Distance [m]"),
            ("Autonomous distance", "autonomous_distance_percent", "Autonomous distance [%]"),
            ("Autonomous time", "autonomous_time_percent", "Autonomous time [%]"),
            ("Takeovers", "takeover_count", "Takeovers"),
            ("Brake events", "brake_event_count", "Brake events"),
            ("Harsh brakes", "harsh_brake_count", "Harsh brakes"),
            ("Mean speed", "mean_speed_mps", "Mean speed [m/s]"),
            ("Max speed", "max_speed_mps", "Max speed [m/s]"),
            ("Mean abs acceleration", "mean_abs_accel_mps2", "Mean |acceleration| [m/s^2]"),
        ]

    def create_compare_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        controls = QGridLayout()
        controls.setHorizontalSpacing(10)
        controls.setVerticalSpacing(6)

        self.compare_storage_combo = QComboBox()
        self.compare_storage_combo.addItems(["sqlite3", "mcap"])
        storage_index = self.compare_storage_combo.findText(self.storage_id)
        if storage_index >= 0:
            self.compare_storage_combo.setCurrentIndex(storage_index)

        add_button = QPushButton("Add Bag")
        add_button.setToolTip("Add another ROS 2 bag folder to the comparison.")
        add_button.clicked.connect(self.add_compare_bag_dialog)

        remove_button = QPushButton("Remove Selected")
        remove_button.setToolTip("Remove selected comparison rows. The opened bag remains as the baseline.")
        remove_button.clicked.connect(self.remove_selected_compare_bags)

        clear_button = QPushButton("Clear Added Bags")
        clear_button.setToolTip("Keep only the opened baseline bag.")
        clear_button.clicked.connect(self.clear_added_compare_bags)

        self.compare_metric_combo = QComboBox()
        self.compare_metric_combo.addItem("Lateral Error", "lateral_error")
        self.compare_metric_combo.addItem("Heading Error", "heading_error")
        self.compare_metric_combo.addItem("Velocity Error", "velocity_error")
        self.compare_metric_combo.addItem("Vehicle Speed", "vehicle_speed")
        self.compare_metric_combo.addItem("Longitudinal Acceleration", "acceleration")
        self.compare_metric_combo.currentIndexChanged.connect(self.update_compare_plots)

        self.compare_bar_metric_combo = QComboBox()
        for label, key, _ in self.compare_bar_metrics():
            self.compare_bar_metric_combo.addItem(label, key)
        self.compare_bar_metric_combo.currentIndexChanged.connect(self.update_compare_bar_plot)

        controls.addWidget(QLabel("Storage ID for added bags"), 0, 0)
        controls.addWidget(self.compare_storage_combo, 0, 1)
        controls.addWidget(add_button, 0, 2)
        controls.addWidget(remove_button, 0, 3)
        controls.addWidget(clear_button, 0, 4)
        controls.addWidget(QLabel("Time-series metric"), 1, 0)
        controls.addWidget(self.compare_metric_combo, 1, 1)
        controls.addWidget(QLabel("Summary graph metric"), 1, 2)
        controls.addWidget(self.compare_bar_metric_combo, 1, 3)
        controls.setColumnStretch(5, 1)

        self.compare_status = QLabel(
            "Compare: opened bag is the baseline. Add one or more bag folders to compare metrics."
        )
        self.compare_status.setWordWrap(True)

        self.compare_table = QTableWidget()
        self.compare_table.setColumnCount(len(self.compare_table_columns()))
        self.compare_table.setHorizontalHeaderLabels([column[0] for column in self.compare_table_columns()])
        self.compare_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.compare_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.compare_table.setAlternatingRowColors(True)
        self.compare_table.verticalHeader().setVisible(False)
        self.compare_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.compare_table.horizontalHeader().setStretchLastSection(True)
        self.compare_table.setMinimumHeight(190)
        self.compare_table.setMaximumHeight(280)

        self.compare_series_canvas = PlotCanvas(width=10, height=4)
        self.compare_bar_canvas = PlotCanvas(width=10, height=3)
        self.update_compare_display()

        layout.addLayout(controls)
        layout.addWidget(self.compare_status)
        layout.addWidget(self.compare_table)
        self.add_plot_with_toolbar(layout, self.compare_series_canvas, "Compare Time Series")
        self.add_plot_with_toolbar(layout, self.compare_bar_canvas, "Compare Summary Metric")

        widget.setLayout(layout)
        return widget

    def update_compare_display(self):
        self.update_compare_table()
        self.update_compare_plots()
        self.update_compare_bar_plot()

        if self.compare_status is not None:
            count = len(self.compare_entries)
            if count == 1:
                self.compare_status.setText(
                    "Compare: opened bag is the baseline. Add one or more bag folders to compare metrics."
                )
            else:
                self.compare_status.setText(f"Compare: {count} bags loaded.")

    def update_compare_table(self):
        if self.compare_table is None:
            return

        columns = self.compare_table_columns()
        self.compare_table.setRowCount(len(self.compare_entries))

        for row, entry in enumerate(self.compare_entries):
            stats = summarize_results_for_compare(entry.results)

            for col, (_, key, fmt) in enumerate(columns):
                if key == "label":
                    text = entry.label
                elif key == "bag_path":
                    text = entry.bag_path
                elif key == "storage_id":
                    text = entry.storage_id
                else:
                    value = stats.get(key, 0.0)
                    text = fmt.format(value) if fmt is not None else str(value)

                item = QTableWidgetItem(text)
                if key not in ("label", "bag_path", "storage_id"):
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.compare_table.setItem(row, col, item)

        self.compare_table.resizeColumnsToContents()

    def update_compare_plots(self, *args):
        if self.compare_series_canvas is None or self.compare_metric_combo is None:
            return

        metric = self.compare_metric_combo.currentData() or "lateral_error"
        series = []
        title = "Comparison"
        ylabel = ""

        for entry in self.compare_entries:
            xs, ys, ylabel, title = compare_metric_series(entry.results, metric)
            series.append((entry.label, xs, ys))

        self.compare_series_canvas.plot_compare_series(
            series,
            title=f"{title} Comparison",
            xlabel="Time from bag start [s]",
            ylabel=ylabel,
        )

    def update_compare_bar_plot(self, *args):
        if self.compare_bar_canvas is None or self.compare_bar_metric_combo is None:
            return

        metric = self.compare_bar_metric_combo.currentData() or "rms_lateral_error_m"
        metric_defs = {key: (label, ylabel) for label, key, ylabel in self.compare_bar_metrics()}
        title, ylabel = metric_defs.get(metric, ("Summary Metric", metric))

        labels = []
        values = []
        for entry in self.compare_entries:
            stats = summarize_results_for_compare(entry.results)
            labels.append(entry.label)
            values.append(stats.get(metric, 0.0))

        self.compare_bar_canvas.plot_compare_bars(
            labels,
            values,
            title=f"{title} by Bag",
            ylabel=ylabel,
        )

    def add_compare_bag_dialog(self):
        bag_path = QFileDialog.getExistingDirectory(self, "Select ROS 2 bag folder")
        if not bag_path:
            return

        storage_id = self.compare_storage_combo.currentText() if self.compare_storage_combo is not None else self.storage_id
        self.add_compare_bag(bag_path, storage_id)

    def add_compare_bag(self, bag_path, storage_id):
        normalized_path = os.path.abspath(bag_path)
        for entry in self.compare_entries:
            if os.path.abspath(entry.bag_path) == normalized_path and entry.storage_id == storage_id:
                QMessageBox.information(self, "Compare Bag", "That bag is already loaded for comparison.")
                return

        if self.compare_status is not None:
            self.compare_status.setText(f"Reading comparison bag: {normalized_path}")
        QApplication.setOverrideCursor(Qt.WaitCursor)
        QApplication.processEvents()

        try:
            evaluator = AutowareBagEvaluator(normalized_path, storage_id)
            evaluator.read_bag()
            self.compare_entries.append(
                CompareBag(
                    label=self.unique_compare_label(normalized_path),
                    bag_path=normalized_path,
                    storage_id=storage_id,
                    results=evaluator.results,
                )
            )
            self.update_compare_display()

        except Exception as exc:
            if self.compare_status is not None:
                self.compare_status.setText(f"Compare failed: {exc}")
            QMessageBox.warning(self, "Compare Bag Error", str(exc))

        finally:
            QApplication.restoreOverrideCursor()

    def remove_selected_compare_bags(self):
        if self.compare_table is None:
            return

        rows = sorted({index.row() for index in self.compare_table.selectedIndexes()}, reverse=True)
        if not rows:
            current_row = self.compare_table.currentRow()
            rows = [current_row] if current_row >= 0 else []

        removable_rows = [row for row in rows if row > 0 and row < len(self.compare_entries)]
        if not removable_rows:
            if self.compare_status is not None:
                self.compare_status.setText("The opened baseline bag stays pinned; select added bags to remove.")
            return

        for row in removable_rows:
            del self.compare_entries[row]

        self.update_compare_display()

    def clear_added_compare_bags(self):
        if len(self.compare_entries) <= 1:
            return

        self.compare_entries = self.compare_entries[:1]
        self.update_compare_display()

    def create_route_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()

        canvas = PlotCanvas(width=10, height=7)
        canvas.plot_route(self.results.poses, self.results.modes)

        self.add_plot_with_toolbar(layout, canvas, "Route by Operation Mode")
        widget.setLayout(layout)
        return widget

    def create_heatmap_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        control_panel = QWidget()
        self.heatmap_control_panel = control_panel
        control_panel.setMaximumHeight(255)
        control_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        control_layout = QVBoxLayout()
        control_layout.setContentsMargins(0, 0, 0, 0)
        control_layout.setSpacing(6)

        controls = QGridLayout()
        controls.setContentsMargins(0, 0, 0, 0)
        controls.setHorizontalSpacing(10)
        controls.setVerticalSpacing(6)

        self.heatmap_origin_lat_input = QLineEdit()
        self.heatmap_origin_lat_input.setPlaceholderText("Map origin latitude")
        if self.heatmap_defaults.get("origin_lat") is not None:
            self.heatmap_origin_lat_input.setText(str(self.heatmap_defaults["origin_lat"]))

        self.heatmap_origin_lon_input = QLineEdit()
        self.heatmap_origin_lon_input.setPlaceholderText("Map origin longitude")
        if self.heatmap_defaults.get("origin_lon") is not None:
            self.heatmap_origin_lon_input.setText(str(self.heatmap_defaults["origin_lon"]))

        self.heatmap_origin_yaw_input = QDoubleSpinBox()
        self.heatmap_origin_yaw_input.setRange(-360.0, 360.0)
        self.heatmap_origin_yaw_input.setDecimals(3)
        self.heatmap_origin_yaw_input.setValue(float(self.heatmap_defaults.get("origin_yaw_deg", 0.0)))

        self.heatmap_metric_combo = QComboBox()
        self.heatmap_metric_combo.addItems(["speed", "lateral_error", "brake", "density", "operation_mode"])
        metric_default = self.heatmap_defaults.get("metric", "speed")
        metric_index = self.heatmap_metric_combo.findText(metric_default)
        if metric_index >= 0:
            self.heatmap_metric_combo.setCurrentIndex(metric_index)

        self.heatmap_zoom_spin = QSpinBox()
        self.heatmap_zoom_spin.setRange(1, 20)
        self.heatmap_zoom_spin.setValue(int(self.heatmap_defaults.get("zoom", 19)))

        self.heatmap_bins_spin = QSpinBox()
        self.heatmap_bins_spin.setRange(25, 1000)
        self.heatmap_bins_spin.setSingleStep(25)
        self.heatmap_bins_spin.setValue(int(self.heatmap_defaults.get("bins", 250)))

        self.heatmap_alpha_spin = QDoubleSpinBox()
        self.heatmap_alpha_spin.setRange(0.05, 1.0)
        self.heatmap_alpha_spin.setSingleStep(0.05)
        self.heatmap_alpha_spin.setDecimals(2)
        self.heatmap_alpha_spin.setValue(float(self.heatmap_defaults.get("alpha", 0.65)))

        self.heatmap_cmap_combo = QComboBox()
        self.heatmap_cmap_combo.addItems(["jet", "turbo", "viridis", "plasma", "inferno", "magma"])
        cmap_default = self.heatmap_defaults.get("cmap", "jet")
        cmap_index = self.heatmap_cmap_combo.findText(cmap_default)
        if cmap_index >= 0:
            self.heatmap_cmap_combo.setCurrentIndex(cmap_index)

        controls.addWidget(QLabel("Origin lat"), 0, 0)
        controls.addWidget(self.heatmap_origin_lat_input, 0, 1)
        controls.addWidget(QLabel("Origin lon"), 0, 2)
        controls.addWidget(self.heatmap_origin_lon_input, 0, 3)
        controls.addWidget(QLabel("Origin yaw deg"), 0, 4)
        controls.addWidget(self.heatmap_origin_yaw_input, 0, 5)

        controls.addWidget(QLabel("Metric"), 1, 0)
        controls.addWidget(self.heatmap_metric_combo, 1, 1)
        controls.addWidget(QLabel("Zoom"), 1, 2)
        controls.addWidget(self.heatmap_zoom_spin, 1, 3)
        controls.addWidget(QLabel("Bins"), 1, 4)
        controls.addWidget(self.heatmap_bins_spin, 1, 5)
        controls.addWidget(QLabel("Alpha"), 2, 0)
        controls.addWidget(self.heatmap_alpha_spin, 2, 1)
        controls.addWidget(QLabel("Colormap"), 2, 2)
        controls.addWidget(self.heatmap_cmap_combo, 2, 3)

        button_layout = QHBoxLayout()
        generate_button = QPushButton("Generate Heatmap")
        self.heatmap_generate_button = generate_button
        generate_button.clicked.connect(self.generate_heatmap)

        self.heatmap_save_button = QPushButton("Save Heatmap")
        self.heatmap_save_button.clicked.connect(self.save_heatmap_figure)
        self.heatmap_save_button.setEnabled(False)

        button_layout.addWidget(generate_button)
        button_layout.addWidget(self.heatmap_save_button)
        button_layout.addStretch()

        time_scroll_layout = QHBoxLayout()
        self.heatmap_time_scroll = QSlider(Qt.Horizontal)
        self.heatmap_time_scroll.setRange(0, 10000)
        self.heatmap_time_scroll.setSingleStep(10)
        self.heatmap_time_scroll.setPageStep(250)
        self.heatmap_time_scroll.setEnabled(self.route_time_range()[0] is not None)
        self.heatmap_time_scroll.setToolTip(
            "Scroll through route time. The heatmap marker and graph cursors follow this time."
        )
        self.heatmap_time_scroll.valueChanged.connect(self.on_route_time_scroll_changed)

        self.heatmap_time_label = QLabel("Time --")
        self.heatmap_time_label.setMinimumWidth(95)
        self.heatmap_time_label.setToolTip("Selected time from route start.")

        time_scroll_layout.addWidget(QLabel("Route Time Scroll"))
        time_scroll_layout.addWidget(self.heatmap_time_scroll, 1)
        time_scroll_layout.addWidget(self.heatmap_time_label)

        self.heatmap_status = QLabel(
            "Enter the Autoware map origin, then generate the OSM heatmap."
        )
        self.heatmap_status.setMaximumHeight(24)
        self.heatmap_status.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.heatmap_status.setStyleSheet("font-size: 13px; color: #455A64; padding: 2px;")

        self.time_cursor_status = QLabel(
            "Time cursor: use a graph's Time Scroll to inspect the matching route point."
        )
        self.time_cursor_status.setMaximumHeight(36)
        self.time_cursor_status.setWordWrap(True)
        self.time_cursor_status.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.time_cursor_status.setStyleSheet("font-size: 13px; color: #263238; padding: 2px;")

        self.heatmap_fig = Figure(figsize=(10, 7), dpi=100)
        self.heatmap_canvas = FigureCanvas(self.heatmap_fig)
        self.heatmap_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.heatmap_canvas.updateGeometry()
        enable_scroll_zoom(self.heatmap_canvas)
        initial_ax = self.heatmap_fig.add_subplot(111)
        initial_ax.text(
            0.5,
            0.5,
            "Heatmap will appear here",
            ha="center",
            va="center",
            fontsize=14,
        )
        initial_ax.set_axis_off()
        self.heatmap_fig.tight_layout()

        control_layout.addLayout(controls)
        control_layout.addLayout(button_layout)
        control_layout.addLayout(time_scroll_layout)
        control_layout.addWidget(self.heatmap_status)
        control_layout.addWidget(self.time_cursor_status)
        control_panel.setLayout(control_layout)

        layout.addWidget(control_panel)
        self.add_plot_with_toolbar(layout, self.heatmap_canvas, "OSM Heatmap")

        widget.setLayout(layout)
        return widget

    def read_heatmap_origin(self):
        try:
            origin_lat = float(self.heatmap_origin_lat_input.text().strip())
            origin_lon = float(self.heatmap_origin_lon_input.text().strip())
        except ValueError as exc:
            raise ValueError("Origin latitude and longitude must be numeric.") from exc

        origin_yaw_deg = float(self.heatmap_origin_yaw_input.value())
        return origin_lat, origin_lon, origin_yaw_deg

    def generate_heatmap(self):
        if len(self.results.poses) < 2:
            QMessageBox.warning(self, "Heatmap", "Not enough pose samples to draw a heatmap.")
            return

        try:
            origin_lat, origin_lon, origin_yaw_deg = self.read_heatmap_origin()
        except ValueError as exc:
            QMessageBox.warning(self, "Heatmap Origin", str(exc))
            return

        metric = self.heatmap_metric_combo.currentText()
        zoom = int(self.heatmap_zoom_spin.value())
        bins = int(self.heatmap_bins_spin.value())
        alpha = float(self.heatmap_alpha_spin.value())
        cmap = self.heatmap_cmap_combo.currentText()

        QApplication.setOverrideCursor(Qt.WaitCursor)
        self.heatmap_save_button.setEnabled(False)

        try:
            self.heatmap_status.setText("Projecting route into latitude/longitude...")
            QApplication.processEvents()

            global_samples = build_global_pose_samples(
                self.results.poses,
                origin_lat,
                origin_lon,
                origin_yaw_deg,
            )
            lat_list = [sample["lat"] for sample in global_samples]
            lon_list = [sample["lon"] for sample in global_samples]

            self.heatmap_status.setText("Downloading/stitching OSM tiles...")
            QApplication.processEvents()

            background, extent = create_osm_background(
                lat_list,
                lon_list,
                zoom=zoom,
                padding_tiles=1,
            )

            self.heatmap_status.setText("Rendering heatmap...")
            QApplication.processEvents()

            self.heatmap_marker = None
            self.heatmap_marker_label = None
            heatmap_stats = draw_osm_heatmap(
                self.heatmap_fig,
                global_samples,
                self.results.velocities,
                self.results.trajectories,
                self.results.modes,
                background,
                extent,
                metric,
                bins=bins,
                alpha=alpha,
                cmap=cmap,
            )
            self.heatmap_global_samples = global_samples
            self.heatmap_pose_times = np.array(
                [sample["pose"].t for sample in global_samples],
                dtype=float,
            )

            if self.selected_time is None and self.heatmap_global_samples:
                self.set_selected_time(self.heatmap_global_samples[0]["pose"].t)
            else:
                self.update_heatmap_time_marker()

            self.apply_adaptive_styles(force=True)
            self.heatmap_canvas.draw()
            self.heatmap_save_button.setEnabled(True)
            if metric == "operation_mode":
                self.heatmap_status.setText(
                    f"Rendered operation mode route with {heatmap_stats['sample_count']} samples."
                )
            else:
                outlier_text = ""
                if heatmap_stats.get("outlier_count", 0) > 0:
                    outlier_text = f", excluded {heatmap_stats['outlier_count']} outliers"

                self.heatmap_status.setText(
                    f"Rendered {metric}: {heatmap_stats['sample_count']} samples, "
                    f"color range {heatmap_stats['vmin']:.3g} to {heatmap_stats['vmax']:.3g}"
                    f"{outlier_text}."
                )

        except Exception as exc:
            self.heatmap_status.setText(f"Heatmap failed: {exc}")
            QMessageBox.warning(self, "Heatmap Error", str(exc))

        finally:
            QApplication.restoreOverrideCursor()

    def save_heatmap_figure(self):
        if self.heatmap_fig is None:
            return

        self.save_figure_dialog(
            self.heatmap_canvas,
            "OSM Heatmap",
            default_path=self.heatmap_defaults.get("output", "autoware_heatmap.png"),
        )

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
            t0 = None
            ts = []
            vals = []

        canvas1 = PlotCanvas(width=10, height=4)
        canvas1.plot_xy(
            ts,
            vals,
            "Lateral Tracking Error",
            "Time [s]",
            "Lateral error [m]",
            location_mapper=self.route_location_mapper,
            time_origin=t0,
        )

        if heading:
            t0 = heading[0][0]
            ts_h = [p[0] - t0 for p in heading]
            vals_h = [math.degrees(p[1]) for p in heading]
        else:
            t0 = None
            ts_h = []
            vals_h = []

        canvas2 = PlotCanvas(width=10, height=4)
        canvas2.plot_xy(
            ts_h,
            vals_h,
            "Heading Error",
            "Time [s]",
            "Heading error [deg]",
            location_mapper=self.route_location_mapper,
            time_origin=t0,
        )

        self.add_plot_with_toolbar(layout, canvas1, "Lateral Tracking Error")
        self.add_plot_with_toolbar(layout, canvas2, "Heading Error")
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
            t0 = None
            ts = []
            vs = []

        canvas1 = PlotCanvas(width=10, height=4)
        canvas1.plot_xy(
            ts,
            vs,
            "Actual Vehicle Velocity",
            "Time [s]",
            "Velocity [m/s]",
            location_mapper=self.route_location_mapper,
            time_origin=t0,
        )

        if r.velocity_errors:
            t0 = r.velocity_errors[0][0]
            ts_e = [p[0] - t0 for p in r.velocity_errors]
            vals_e = [p[1] for p in r.velocity_errors]
        else:
            t0 = None
            ts_e = []
            vals_e = []

        canvas2 = PlotCanvas(width=10, height=4)
        canvas2.plot_xy(
            ts_e,
            vals_e,
            "Velocity Tracking Error",
            "Time [s]",
            "Target - Actual [m/s]",
            location_mapper=self.route_location_mapper,
            time_origin=t0,
        )

        self.add_plot_with_toolbar(layout, canvas1, "Actual Vehicle Velocity")
        self.add_plot_with_toolbar(layout, canvas2, "Velocity Tracking Error")
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
            t0 = None
            t_acc = []
            accel = []

        canvas = PlotCanvas(width=10, height=5)
        canvas.plot_xy(
            t_acc,
            accel,
            "Estimated Longitudinal Acceleration",
            "Time [s]",
            "Acceleration [m/s²]",
            location_mapper=self.route_location_mapper,
            time_origin=t0,
        )

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

        self.add_plot_with_toolbar(layout, canvas, "Estimated Longitudinal Acceleration")
        layout.addWidget(table)

        widget.setLayout(layout)
        return widget

    def create_events_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()

        mode_events = []
        previous_mode = None
        previous_enabled = None

        for m in self.results.modes:
            enabled = bool(m.autoware_control_enabled)
            if m.mode != previous_mode or enabled != previous_enabled:
                mode_events.append(m)
                previous_mode = m.mode
                previous_enabled = enabled

        table = QTableWidget()
        table.setColumnCount(3)
        table.setHorizontalHeaderLabels(["Time [s]", "Mode / Control Change", "Autoware Control Enabled"])
        table.setRowCount(len(mode_events))

        if self.results.modes:
            t0 = self.results.modes[0].t
        else:
            t0 = 0.0

        for row, m in enumerate(mode_events):
            table.setItem(row, 0, QTableWidgetItem(f"{m.t - t0:.2f}"))
            table.setItem(row, 1, QTableWidgetItem(m.mode))
            table.setItem(row, 2, QTableWidgetItem(str(m.autoware_control_enabled)))

        layout.addWidget(table)
        widget.setLayout(layout)
        return widget

    def mean_value(self, data):
        values = self.filtered_metric_values(data)
        if len(values) == 0:
            return 0.0
        return float(np.mean(values))

    def mean_abs_value(self, data):
        values = self.filtered_metric_values(data)
        if len(values) == 0:
            return 0.0
        return float(np.mean(np.abs(values)))

    def max_value(self, data):
        values = self.filtered_metric_values(data)
        if len(values) == 0:
            return 0.0
        return float(np.max(values))

    def max_abs_value(self, data):
        values = self.filtered_metric_values(data)
        if len(values) == 0:
            return 0.0
        return float(np.max(np.abs(values)))

    def rms_value(self, data):
        values = self.filtered_metric_values(data)
        if len(values) == 0:
            return 0.0
        return float(np.sqrt(np.mean(values * values)))

    def filtered_metric_values(self, data):
        if not data:
            return np.array([])

        values = np.array([p[1] for p in data], dtype=float)
        filtered = filter_outliers(values)
        if len(filtered) == 0:
            return values[np.isfinite(values)]

        return filtered

    def generate_text_report(self):
        r = self.results
        stats = self.summary_stats()
        interval_text = "Full route"
        if self.analysis_interval is not None:
            start_t, end_t = self.analysis_interval
            origin = self.route_time_origin()
            if self.analysis_interval_mode == "exclude":
                interval_text = f"Excluding {start_t - origin:.2f}s to {end_t - origin:.2f}s"
            else:
                interval_text = f"Using {start_t - origin:.2f}s to {end_t - origin:.2f}s"

        text = f"""
AUTOWARE AUTONOMOUS DRIVING EVALUATION REPORT

ANALYSIS WINDOW
---------------
Selected interval:                  {interval_text}

MISSION / ROUTE PERFORMANCE
---------------------------
Total driven distance:              {stats['distance_total']:.2f} m
Autonomous driven distance:         {stats['autonomous_distance']:.2f} m
Autonomous distance percentage:     {stats['auto_dist_pct']:.2f} %
Total evaluated time:               {stats['total_time']:.2f} s
Autonomous time:                    {stats['autonomous_time']:.2f} s
Autonomous time percentage:         {stats['auto_time_pct']:.2f} %

INTERVENTION / HANDOVER PERFORMANCE
-----------------------------------
Mode change count:                  {stats['mode_change_count']}
Takeover count:                     {stats['takeover_count']}
Takeover rate:                      {stats['takeover_per_km']:.2f} takeovers/km

TRAJECTORY TRACKING PERFORMANCE
-------------------------------
Mean lateral error:                 {stats['mean_lat']:.3f} m
RMS lateral error:                  {stats['rms_lat']:.3f} m
Maximum lateral error:              {stats['max_lat']:.3f} m
Mean absolute velocity error:       {stats['mean_vel_err']:.3f} m/s
Maximum absolute velocity error:    {stats['max_vel_err']:.3f} m/s

BRAKING / COMFORT PERFORMANCE
-----------------------------
Brake event count:                  {stats['brake_event_count']}
Harsh brake count:                  {stats['harsh_brake_count']}
Harsh brake rate:                   {stats['harsh_per_km']:.2f} harsh brakes/km

OUTLIER FILTERING
-----------------
Continuous summary metrics exclude isolated one-sample spikes only.
Spike threshold:                     {OUTLIER_MAD_THRESHOLD:.1f} robust sigma from median
Excluded lateral error samples:      {stats['lateral_outliers']} of {stats['lateral_sample_count']}
Excluded velocity error samples:     {stats['velocity_outliers']} of {stats['velocity_sample_count']}

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

    def wrapped_report_text(self, width=112):
        wrapped_lines = []
        for line in self.generate_text_report().strip().splitlines():
            if len(line) <= width or not line.strip():
                wrapped_lines.append(line)
            else:
                wrapped_lines.append(textwrap.fill(line, width=width))
        return "\n".join(wrapped_lines)

    def add_text_pdf_page(self, pdf, title, text):
        fig = Figure(figsize=(11, 8.5))
        ax = fig.add_subplot(111)
        ax.axis("off")
        ax.text(
            0.03,
            0.97,
            title,
            fontsize=17,
            fontweight="bold",
            va="top",
            transform=ax.transAxes,
        )
        ax.text(
            0.03,
            0.91,
            text,
            fontsize=8.5,
            family="monospace",
            va="top",
            transform=ax.transAxes,
        )
        pdf.savefig(fig, bbox_inches="tight")

    def add_table_pdf_page(self, pdf, title, headers, rows, max_rows=42):
        fig = Figure(figsize=(11, 8.5))
        ax = fig.add_subplot(111)
        ax.axis("off")
        ax.text(
            0.03,
            0.97,
            title,
            fontsize=16,
            fontweight="bold",
            va="top",
            transform=ax.transAxes,
        )

        if rows:
            visible_rows = rows[:max_rows]
            table = ax.table(
                cellText=visible_rows,
                colLabels=headers,
                loc="center",
                cellLoc="left",
            )
            table.auto_set_font_size(False)
            table.set_fontsize(7.5)
            table.scale(1.0, 1.25)

            if len(rows) > max_rows:
                ax.text(
                    0.03,
                    0.04,
                    f"Showing first {max_rows} of {len(rows)} rows.",
                    fontsize=8,
                    transform=ax.transAxes,
                )
        else:
            ax.text(
                0.03,
                0.85,
                "No rows available.",
                fontsize=11,
                transform=ax.transAxes,
            )

        pdf.savefig(fig, bbox_inches="tight")

    def mode_event_rows(self):
        rows = []
        previous_mode = None
        previous_enabled = None
        t0 = self.results.modes[0].t if self.results.modes else 0.0
        start_t, end_t = self.analysis_interval if self.analysis_interval is not None else (None, None)

        for mode in self.results.modes:
            if start_t is not None:
                in_interval = start_t <= mode.t <= end_t
                if self.analysis_interval_mode == "exclude" and in_interval:
                    continue
                if self.analysis_interval_mode != "exclude" and not in_interval:
                    continue

            enabled = bool(mode.autoware_control_enabled)
            if mode.mode != previous_mode or enabled != previous_enabled:
                rows.append([
                    f"{mode.t - t0:.2f}",
                    mode.mode,
                    str(enabled),
                ])
                previous_mode = mode.mode
                previous_enabled = enabled

        return rows

    def brake_event_rows(self):
        brake_events = self.brake_events_in_interval()
        if not brake_events:
            return []

        t0 = brake_events[0].start_t
        rows = []

        for event in brake_events:
            rows.append([
                f"{event.start_t - t0:.2f}",
                f"{event.end_t - t0:.2f}",
                f"{event.end_t - event.start_t:.2f}",
                f"{event.min_accel:.2f}",
                f"{event.max_jerk:.2f}",
                event.event_type,
            ])

        return rows

    def export_pdf_report(self):
        default_path = f"{safe_filename(os.path.basename(self.bag_path))}_evaluation_report.pdf"
        output_path, _ = QFileDialog.getSaveFileName(
            self,
            "Create PDF report",
            default_path,
            "PDF files (*.pdf)",
        )
        if not output_path:
            return

        if not output_path.lower().endswith(".pdf"):
            output_path += ".pdf"

        try:
            with PdfPages(output_path) as pdf:
                self.add_text_pdf_page(
                    pdf,
                    "Autoware Autonomous Driving Evaluation Report",
                    self.wrapped_report_text(),
                )

                for entry in self.plot_entries:
                    figure = self.figure_for_canvas(entry["canvas"])
                    if figure is None:
                        continue

                    figure.tight_layout()
                    pdf.savefig(figure, bbox_inches="tight")

                self.add_table_pdf_page(
                    pdf,
                    "Brake Events",
                    ["Start [s]", "End [s]", "Duration [s]", "Min Accel [m/s^2]", "Max Jerk [m/s^3]", "Type"],
                    self.brake_event_rows(),
                )
                self.add_table_pdf_page(
                    pdf,
                    "Mode / Control Events",
                    ["Time [s]", "Mode", "Autoware Control Enabled"],
                    self.mode_event_rows(),
                )

            QMessageBox.information(self, "PDF Report", f"PDF report saved:\n{output_path}")

        except Exception as exc:
            QMessageBox.warning(self, "PDF Report Error", str(exc))

    def export_csv(self):
        output_dir = QFileDialog.getExistingDirectory(self, "Select output directory")
        if not output_dir:
            return

        r = self.results
        stats = self.summary_stats()
        interval_start = ""
        interval_end = ""
        interval_mode = "full_route"
        if self.analysis_interval is not None:
            interval_start, interval_end = self.analysis_interval
            interval_mode = self.analysis_interval_mode

        summary = {
            "analysis_interval_mode": [interval_mode],
            "analysis_interval_start_s": [interval_start],
            "analysis_interval_end_s": [interval_end],
            "total_distance_m": [stats["distance_total"]],
            "autonomous_distance_m": [stats["autonomous_distance"]],
            "autonomous_distance_percent": [stats["auto_dist_pct"]],
            "total_time_s": [stats["total_time"]],
            "autonomous_time_s": [stats["autonomous_time"]],
            "autonomous_time_percent": [stats["auto_time_pct"]],
            "takeover_count": [stats["takeover_count"]],
            "mode_change_count": [stats["mode_change_count"]],
            "brake_event_count": [stats["brake_event_count"]],
            "harsh_brake_count": [stats["harsh_brake_count"]],
            "mean_lateral_error_m_filtered": [stats["mean_lat"]],
            "rms_lateral_error_m_filtered": [stats["rms_lat"]],
            "max_lateral_error_m_filtered": [stats["max_lat"]],
            "mean_abs_velocity_error_mps_filtered": [stats["mean_vel_err"]],
            "max_abs_velocity_error_mps_filtered": [stats["max_vel_err"]],
            "outlier_filter": [f"isolated spike > {OUTLIER_MAD_THRESHOLD:.1f} robust sigma from median"],
            "lateral_error_outliers_excluded": [stats["lateral_outliers"]],
            "velocity_error_outliers_excluded": [stats["velocity_outliers"]],
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
    parser.add_argument("--origin-lat", type=float, default=None, help="Optional OSM heatmap map origin latitude")
    parser.add_argument("--origin-lon", type=float, default=None, help="Optional OSM heatmap map origin longitude")
    parser.add_argument("--origin-yaw-deg", type=float, default=0.0, help="Optional OSM heatmap map origin yaw")
    parser.add_argument(
        "--metric",
        default="speed",
        choices=["speed", "lateral_error", "brake", "density", "operation_mode"],
        help="Default OSM heatmap metric",
    )
    parser.add_argument("--output", default="autoware_heatmap.png", help="Default OSM heatmap PNG save path")
    parser.add_argument("--heatmap-zoom", type=int, default=19, help="Default OSM heatmap tile zoom")
    parser.add_argument("--heatmap-bins", type=int, default=250, help="Default OSM heatmap bin count")
    parser.add_argument("--heatmap-alpha", type=float, default=0.65, help="Default OSM heatmap overlay alpha")
    parser.add_argument("--heatmap-cmap", default="jet", help="Default OSM heatmap matplotlib colormap")

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
    heatmap_defaults = {
        "origin_lat": args.origin_lat,
        "origin_lon": args.origin_lon,
        "origin_yaw_deg": args.origin_yaw_deg,
        "metric": args.metric,
        "output": args.output,
        "zoom": args.heatmap_zoom,
        "bins": args.heatmap_bins,
        "alpha": args.heatmap_alpha,
        "cmap": args.heatmap_cmap,
    }
    gui = EvaluationGUI(
        evaluator.results,
        args.bag,
        storage_id=args.storage_id,
        heatmap_defaults=heatmap_defaults,
    )
    gui.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
