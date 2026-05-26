#!/usr/bin/env python3
"""Start and stop a ROS 2 bag recording from a GNSS start/finish gate."""

import argparse
import math
import os
import shlex
import signal
import subprocess
import sys
import time
from typing import Optional

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import NavSatFix
except ImportError as exc:
    print(
        "Failed to import ROS 2 Python modules. Source your ROS 2 and "
        "Autoware setup files before running this recorder.",
        file=sys.stderr,
    )
    raise SystemExit(1) from exc


EARTH_RADIUS_METERS = 6371008.8


def distance_meters(lat_a: float, lon_a: float, lat_b: float, lon_b: float) -> float:
    """Return the haversine distance between two WGS84 points."""
    lat_1 = math.radians(lat_a)
    lat_2 = math.radians(lat_b)
    d_lat = math.radians(lat_b - lat_a)
    d_lon = math.radians(lon_b - lon_a)

    hav = (
        math.sin(d_lat / 2.0) ** 2
        + math.cos(lat_1) * math.cos(lat_2) * math.sin(d_lon / 2.0) ** 2
    )
    return 2.0 * EARTH_RADIUS_METERS * math.asin(math.sqrt(hav))


class GpsTriggeredBagRecorder(Node):
    def __init__(
        self,
        *,
        bag_name: str,
        gps_topic: str,
        trigger_lat: float,
        trigger_lon: float,
        trigger_radius_m: float,
        exit_radius_m: float,
        min_record_seconds: float,
        record_topics: list[str],
    ) -> None:
        super().__init__("gps_triggered_bag_recorder")
        self.bag_name = bag_name
        self.gps_topic = gps_topic
        self.trigger_lat = trigger_lat
        self.trigger_lon = trigger_lon
        self.trigger_radius_m = trigger_radius_m
        self.exit_radius_m = exit_radius_m
        self.min_record_seconds = min_record_seconds
        self.record_topics = record_topics

        self.record_process: Optional[subprocess.Popen] = None
        self.record_started_at: Optional[float] = None
        self.started = False
        self.left_gate_after_start = False
        self.done = False
        self.last_status_time = self.get_clock().now()

        self.create_subscription(NavSatFix, gps_topic, self.on_fix, qos_profile_sensor_data)

        self.get_logger().info(
            "Waiting for GNSS gate: "
            f"lat={trigger_lat:.10f}, lon={trigger_lon:.10f}, "
            f"start/finish radius={trigger_radius_m:.1f} m"
        )
        self.get_logger().info(
            "Recording will start on the first pass through the gate, "
            "then stop after the vehicle leaves the gate and returns."
        )
        self.get_logger().info(
            f"Finish detection is armed after leaving {exit_radius_m:.1f} m "
            f"from the gate and recording for at least {min_record_seconds:.1f} s."
        )

    def on_fix(self, msg: NavSatFix) -> None:
        if self.done:
            return

        if not math.isfinite(msg.latitude) or not math.isfinite(msg.longitude):
            self.get_logger().warn("Ignoring GNSS fix with invalid latitude/longitude.")
            return

        distance = distance_meters(
            msg.latitude,
            msg.longitude,
            self.trigger_lat,
            self.trigger_lon,
        )
        inside_gate = distance <= self.trigger_radius_m

        self.log_status(distance)

        if not self.started:
            if inside_gate:
                self.start_recording(distance)
            return

        if not self.left_gate_after_start:
            if distance > self.exit_radius_m:
                self.left_gate_after_start = True
                self.get_logger().info(
                    f"Vehicle left start/finish gate ({distance:.1f} m away). "
                    "Waiting for return pass."
                )
            return

        if inside_gate and self.minimum_record_time_elapsed():
            self.get_logger().info(
                f"Return pass detected ({distance:.1f} m from gate). Stopping bag."
            )
            self.stop_recording()
            self.done = True

    def log_status(self, distance: float) -> None:
        now = self.get_clock().now()
        if (now - self.last_status_time).nanoseconds < 2_000_000_000:
            return

        self.last_status_time = now
        if not self.started:
            state = "waiting for start pass"
        elif not self.left_gate_after_start:
            state = "recording; waiting to leave gate"
        else:
            state = "recording; waiting for return pass"

        self.get_logger().info(f"{state}; distance to gate: {distance:.1f} m")

    def start_recording(self, distance: float) -> None:
        command = ["ros2", "bag", "record", "-o", self.bag_name, *self.record_topics]
        self.get_logger().info(
            f"Start pass detected ({distance:.1f} m from gate). Starting bag."
        )
        self.get_logger().info("Command: " + " ".join(shlex.quote(part) for part in command))

        self.record_process = subprocess.Popen(command, start_new_session=True)
        self.record_started_at = time.monotonic()
        self.started = True

    def minimum_record_time_elapsed(self) -> bool:
        if self.record_started_at is None:
            return False
        return (time.monotonic() - self.record_started_at) >= self.min_record_seconds

    def stop_recording(self) -> None:
        if self.record_process is None:
            return

        if self.record_process.poll() is not None:
            self.get_logger().warn(
                f"ros2 bag record exited early with code {self.record_process.returncode}."
            )
            return

        os.killpg(self.record_process.pid, signal.SIGINT)
        try:
            self.record_process.wait(timeout=20.0)
        except subprocess.TimeoutExpired:
            self.get_logger().warn("ros2 bag record did not stop after SIGINT; terminating.")
            os.killpg(self.record_process.pid, signal.SIGTERM)
            self.record_process.wait(timeout=10.0)

    def destroy_node(self) -> bool:
        if self.record_process is not None and self.record_process.poll() is None:
            self.get_logger().info("Stopping active bag recorder.")
            self.stop_recording()
        return super().destroy_node()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record a ROS 2 bag between two passes through a GNSS gate."
    )
    parser.add_argument("--bag-name", required=True)
    parser.add_argument("--gps-topic", default="/sensing/novatel/oem7/fix")
    parser.add_argument("--trigger-lat", type=float, required=True)
    parser.add_argument("--trigger-lon", type=float, required=True)
    parser.add_argument("--trigger-radius-m", type=float, default=5.0)
    parser.add_argument("--exit-radius-m", type=float)
    parser.add_argument("--min-record-seconds", type=float, default=10.0)
    parser.add_argument("record_topics", nargs="+")
    args = parser.parse_args()

    if args.trigger_radius_m <= 0:
        parser.error("--trigger-radius-m must be greater than zero")

    if args.exit_radius_m is None:
        args.exit_radius_m = max(args.trigger_radius_m * 1.5, args.trigger_radius_m + 2.0)
    elif args.exit_radius_m <= args.trigger_radius_m:
        parser.error("--exit-radius-m must be larger than --trigger-radius-m")

    if args.min_record_seconds < 0:
        parser.error("--min-record-seconds must be zero or greater")

    return args


def main() -> int:
    args = parse_args()
    rclpy.init()
    node = GpsTriggeredBagRecorder(
        bag_name=args.bag_name,
        gps_topic=args.gps_topic,
        trigger_lat=args.trigger_lat,
        trigger_lon=args.trigger_lon,
        trigger_radius_m=args.trigger_radius_m,
        exit_radius_m=args.exit_radius_m,
        min_record_seconds=args.min_record_seconds,
        record_topics=args.record_topics,
    )

    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.2)
    except KeyboardInterrupt:
        node.get_logger().info("Interrupted by user.")
    finally:
        node.destroy_node()
        rclpy.shutdown()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
