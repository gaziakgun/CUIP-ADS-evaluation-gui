#!/usr/bin/env python3

import argparse
import math
import os
import time
from io import BytesIO

import numpy as np
import requests
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from pyproj import Transformer


# ============================================================
# Default topics
# ============================================================

TOPIC_POSE = "/localization/kinematic_state"
TOPIC_TRAJECTORY = "/planning/scenario_planning/trajectory"
TOPIC_VELOCITY = "/vehicle/status/velocity_status"


# ============================================================
# Coordinate helpers
# ============================================================

_transformer_to_3857 = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
_transformer_to_4326 = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)


def latlon_to_webmercator(lat, lon):
    x, y = _transformer_to_3857.transform(lon, lat)
    return x, y


def webmercator_to_latlon(x, y):
    lon, lat = _transformer_to_4326.transform(x, y)
    return lat, lon


def local_xy_to_latlon(x, y, origin_lat, origin_lon, origin_yaw_deg=0.0):
    origin_x, origin_y = latlon_to_webmercator(origin_lat, origin_lon)

    yaw = math.radians(origin_yaw_deg)

    east = math.cos(yaw) * x - math.sin(yaw) * y
    north = math.sin(yaw) * x + math.cos(yaw) * y

    mx = origin_x + east
    my = origin_y + north

    return webmercator_to_latlon(mx, my)


def latlon_to_tile(lat, lon, zoom):
    lat_rad = math.radians(lat)
    n = 2.0 ** zoom

    xtile = int((lon + 180.0) / 360.0 * n)
    ytile = int(
        (1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi)
        / 2.0
        * n
    )

    return xtile, ytile


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

    headers = {
        "User-Agent": "AutowareEvaluationHeatmap/1.0"
    }

    r = requests.get(url, headers=headers, timeout=10)

    if r.status_code != 200:
        raise RuntimeError(f"Could not download OSM tile: {url}")

    img = Image.open(BytesIO(r.content)).convert("RGB")
    img.save(tile_path)

    time.sleep(0.1)
    return img


def create_osm_background(lat_list, lon_list, zoom=19, padding_tiles=1):
    tiles = []

    for lat, lon in zip(lat_list, lon_list):
        tx, ty = latlon_to_tile(lat, lon, zoom)
        tiles.append((tx, ty))

    xs = [t[0] for t in tiles]
    ys = [t[1] for t in tiles]

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

    extent = [left, right, bottom, top]

    return np.array(mosaic), extent


# ============================================================
# ROS helpers
# ============================================================

def bag_time_to_sec(t_nanosec):
    return float(t_nanosec) * 1e-9


def stamp_to_sec(stamp):
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def quaternion_to_yaw(q):
    x = q.x
    y = q.y
    z = q.z
    w = q.w

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)

    return math.atan2(siny_cosp, cosy_cosp)


def extract_pose(msg, fallback_time):
    if hasattr(msg, "header"):
        t = stamp_to_sec(msg.header.stamp)
    else:
        t = fallback_time

    if hasattr(msg, "pose") and hasattr(msg.pose, "pose"):
        pose = msg.pose.pose
    elif hasattr(msg, "pose"):
        pose = msg.pose
    else:
        return None

    return {
        "t": t,
        "x": float(pose.position.x),
        "y": float(pose.position.y),
        "yaw": quaternion_to_yaw(pose.orientation),
    }


def extract_velocity(msg, fallback_time):
    if hasattr(msg, "header"):
        t = stamp_to_sec(msg.header.stamp)
    else:
        t = fallback_time

    if hasattr(msg, "longitudinal_velocity"):
        v = float(msg.longitudinal_velocity)
    elif hasattr(msg, "twist") and hasattr(msg.twist, "twist"):
        v = float(msg.twist.twist.linear.x)
    elif hasattr(msg, "twist"):
        v = float(msg.twist.linear.x)
    else:
        return None

    return {
        "t": t,
        "v": v,
    }


def extract_trajectory(msg, fallback_time):
    if hasattr(msg, "header"):
        t = stamp_to_sec(msg.header.stamp)
    else:
        t = fallback_time

    if not hasattr(msg, "points"):
        return None

    xs = []
    ys = []

    for p in msg.points:
        if not hasattr(p, "pose"):
            continue

        xs.append(float(p.pose.position.x))
        ys.append(float(p.pose.position.y))

    if len(xs) < 2:
        return None

    return {
        "t": t,
        "xs": np.array(xs),
        "ys": np.array(ys),
    }


def read_bag(bag_path, storage_id):
    storage_options = rosbag2_py.StorageOptions(
        uri=bag_path,
        storage_id=storage_id,
    )

    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr",
    )

    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)

    topic_types = reader.get_all_topics_and_types()
    topic_type_dict = {t.name: t.type for t in topic_types}

    msg_types = {}

    for topic, type_name in topic_type_dict.items():
        try:
            msg_types[topic] = get_message(type_name)
        except Exception:
            pass

    poses = []
    velocities = []
    trajectories = []

    while reader.has_next():
        topic, data, t_nanosec = reader.read_next()

        if topic not in msg_types:
            continue

        fallback_time = bag_time_to_sec(t_nanosec)

        try:
            msg = deserialize_message(data, msg_types[topic])
        except Exception:
            continue

        if topic == TOPIC_POSE:
            item = extract_pose(msg, fallback_time)
            if item:
                poses.append(item)

        elif topic == TOPIC_VELOCITY:
            item = extract_velocity(msg, fallback_time)
            if item:
                velocities.append(item)

        elif topic == TOPIC_TRAJECTORY:
            item = extract_trajectory(msg, fallback_time)
            if item:
                trajectories.append(item)

    poses = sorted(poses, key=lambda p: p["t"])
    velocities = sorted(velocities, key=lambda v: v["t"])
    trajectories = sorted(trajectories, key=lambda tr: tr["t"])

    return poses, velocities, trajectories


# ============================================================
# Metric helpers
# ============================================================

def get_velocity_at_time(velocities, t):
    if not velocities:
        return math.nan

    ts = np.array([v["t"] for v in velocities])
    vs = np.array([v["v"] for v in velocities])

    idx = int(np.argmin(np.abs(ts - t)))
    return float(vs[idx])


def get_trajectory_at_time(trajectories, t, max_dt=1.0):
    if not trajectories:
        return None

    ts = np.array([tr["t"] for tr in trajectories])
    idx = int(np.argmin(np.abs(ts - t)))

    if abs(ts[idx] - t) > max_dt:
        return None

    return trajectories[idx]


def compute_lateral_error(pose, trajectory):
    if trajectory is None:
        return math.nan

    dx = trajectory["xs"] - pose["x"]
    dy = trajectory["ys"] - pose["y"]

    dist = np.sqrt(dx * dx + dy * dy)

    if len(dist) == 0:
        return math.nan

    return float(np.min(dist))


def compute_acceleration_series(velocities):
    if len(velocities) < 3:
        return []

    ts = np.array([v["t"] for v in velocities])
    vs = np.array([v["v"] for v in velocities])

    order = np.argsort(ts)
    ts = ts[order]
    vs = vs[order]

    dt = np.diff(ts)
    dv = np.diff(vs)

    acc = np.zeros_like(dv)
    valid = dt > 1e-3
    acc[valid] = dv[valid] / dt[valid]

    return [{"t": float(ts[i + 1]), "a": float(acc[i])} for i in range(len(acc))]


def get_accel_at_time(accels, t):
    if not accels:
        return math.nan

    ts = np.array([a["t"] for a in accels])
    vals = np.array([a["a"] for a in accels])

    idx = int(np.argmin(np.abs(ts - t)))
    return float(vals[idx])


def convert_poses_to_global(poses, origin_lat, origin_lon, origin_yaw_deg):
    for p in poses:
        lat, lon = local_xy_to_latlon(
            p["x"],
            p["y"],
            origin_lat,
            origin_lon,
            origin_yaw_deg,
        )

        mx, my = latlon_to_webmercator(lat, lon)

        p["lat"] = lat
        p["lon"] = lon
        p["mx"] = mx
        p["my"] = my

    return poses


# ============================================================
# Heatmap generation
# ============================================================

def create_heatmap_png(
    poses,
    velocities,
    trajectories,
    background,
    extent,
    output_png,
    metric="speed",
    bins=250,
    alpha=0.65,
    cmap="jet",
    origin_lat=None,
    origin_lon=None,
    origin_yaw_deg=0.0,
):
    if len(poses) < 2:
        raise RuntimeError("Not enough pose samples.")

    accels = compute_acceleration_series(velocities)

    xs = []
    ys = []
    values = []

    for p in poses:
        x = p["mx"]
        y = p["my"]
        t = p["t"]

        if metric == "speed":
            value = get_velocity_at_time(velocities, t)

        elif metric == "lateral_error":
            traj = get_trajectory_at_time(trajectories, t)
            value = compute_lateral_error(p, traj)

        elif metric == "brake":
            a = get_accel_at_time(accels, t)

            if math.isnan(a):
                value = math.nan
            else:
                # only braking intensity, positive value
                value = max(0.0, -a)

        elif metric == "density":
            value = 1.0

        else:
            raise ValueError(f"Unknown metric: {metric}")

        if math.isnan(value):
            continue

        xs.append(x)
        ys.append(y)
        values.append(value)

    xs = np.array(xs)
    ys = np.array(ys)
    values = np.array(values)

    if len(xs) < 3:
        raise RuntimeError("Not enough valid samples for heatmap.")

    xmin, xmax, ymin, ymax = extent

    if metric == "density":
        heat, xedges, yedges = np.histogram2d(
            xs,
            ys,
            bins=bins,
            range=[[xmin, xmax], [ymin, ymax]],
        )
    else:
        weighted_sum, xedges, yedges = np.histogram2d(
            xs,
            ys,
            bins=bins,
            range=[[xmin, xmax], [ymin, ymax]],
            weights=values,
        )

        counts, _, _ = np.histogram2d(
            xs,
            ys,
            bins=bins,
            range=[[xmin, xmax], [ymin, ymax]],
        )

        heat = weighted_sum / np.maximum(counts, 1.0)
        heat[counts == 0] = np.nan

    fig, ax = plt.subplots(figsize=(14, 14), dpi=200)

    ax.imshow(background, extent=extent, origin="upper")

    # driven route line
    ax.plot(
        xs,
        ys,
        color="black",
        linewidth=1.5,
        alpha=0.55,
        label="Driven route",
        zorder=5,
    )

    # heatmap overlay
    im = ax.imshow(
        heat.T,
        extent=extent,
        origin="lower",
        cmap=cmap,
        alpha=alpha,
        interpolation="gaussian",
        zorder=6,
    )

    # start/end points
    ax.scatter(xs[0], ys[0], s=80, marker="o", color="lime", edgecolor="black", label="Start", zorder=10)
    ax.scatter(xs[-1], ys[-1], s=80, marker="X", color="red", edgecolor="black", label="End", zorder=10)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")

    metric_titles = {
        "speed": "Vehicle Speed Heatmap",
        "lateral_error": "Trajectory Tracking Error Heatmap",
        "brake": "Braking Intensity Heatmap",
        "density": "Driving Density Heatmap",
    }

    metric_units = {
        "speed": "m/s",
        "lateral_error": "m",
        "brake": "m/s²",
        "density": "sample count",
    }

    title = metric_titles.get(metric, "Autoware Heatmap")
    unit = metric_units.get(metric, "")

    ax.set_title(title, fontsize=18, fontweight="bold")
    ax.set_xlabel("WebMercator X [m]")
    ax.set_ylabel("WebMercator Y [m]")

    cbar = plt.colorbar(im, ax=ax, shrink=0.75)
    cbar.set_label(unit, fontsize=12)

    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(output_png)
    plt.close(fig)

    print(f"Saved heatmap PNG: {output_png}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Create one PNG heatmap over OSM map from Autoware ROS 2 bag."
    )

    parser.add_argument("--bag", required=True, help="ROS 2 bag folder")
    parser.add_argument("--storage-id", default="sqlite3", help="sqlite3 or mcap")
    parser.add_argument("--output", default="autoware_heatmap.png", help="Output PNG file")

    parser.add_argument("--origin-lat", type=float, required=True)
    parser.add_argument("--origin-lon", type=float, required=True)
    parser.add_argument("--origin-yaw-deg", type=float, default=0.0)

    parser.add_argument(
        "--metric",
        default="speed",
        choices=["speed", "lateral_error", "brake", "density"],
        help="Heatmap metric",
    )

    parser.add_argument("--zoom", type=int, default=19)
    parser.add_argument("--bins", type=int, default=250)
    parser.add_argument("--alpha", type=float, default=0.65)
    parser.add_argument("--cmap", default="jet")

    args = parser.parse_args()

    print("==============================================")
    print(" Autoware OSM Heatmap PNG Generator")
    print("==============================================")
    print(f"Bag:        {args.bag}")
    print(f"Storage:    {args.storage_id}")
    print(f"Metric:     {args.metric}")
    print(f"Output:     {args.output}")
    print("==============================================")

    poses, velocities, trajectories = read_bag(args.bag, args.storage_id)

    print(f"Pose samples:       {len(poses)}")
    print(f"Velocity samples:   {len(velocities)}")
    print(f"Trajectory samples: {len(trajectories)}")

    if len(poses) < 2:
        raise RuntimeError("No enough pose data. Check /localization/kinematic_state in the bag.")

    poses = convert_poses_to_global(
        poses,
        args.origin_lat,
        args.origin_lon,
        args.origin_yaw_deg,
    )

    lat_list = [p["lat"] for p in poses]
    lon_list = [p["lon"] for p in poses]

    print("Downloading/stitching OSM map...")
    background, extent = create_osm_background(
        lat_list,
        lon_list,
        zoom=args.zoom,
        padding_tiles=1,
    )

    create_heatmap_png(
        poses=poses,
        velocities=velocities,
        trajectories=trajectories,
        background=background,
        extent=extent,
        output_png=args.output,
        metric=args.metric,
        bins=args.bins,
        alpha=args.alpha,
        cmap=args.cmap,
        origin_lat=args.origin_lat,
        origin_lon=args.origin_lon,
        origin_yaw_deg=args.origin_yaw_deg,
    )


if __name__ == "__main__":
    main()
