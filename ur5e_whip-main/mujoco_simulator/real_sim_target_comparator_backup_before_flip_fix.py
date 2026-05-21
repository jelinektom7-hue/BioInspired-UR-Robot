#!/usr/bin/env python3

import math
import time

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Point
from std_msgs.msg import Bool, Float64, String


# ============================================================
# USER CONFIGURATION
# ============================================================

SIM_TARGET_TOPIC = "/target_position_world"
REAL_TARGET_TOPIC = "/real/target_position_world"

MATCH_OK_TOPIC = "/whip_vision/target_match_ok"
MATCH_ERROR_TOPIC = "/whip_vision/target_match_error"
MATCH_STATUS_TOPIC = "/whip_vision/target_match_status"

# Maximum allowed 3D target-position difference.
# Start loose, then tighten later.
MAX_ALLOWED_ERROR_M = 0.10

# If either target position is too old, report false.
MAX_TARGET_AGE_S = 1.0

PRINT_PERIOD_S = 0.5


class RealSimTargetComparator(Node):
    def __init__(self):
        super().__init__("real_sim_target_comparator")

        self.sim_target = None
        self.real_target = None

        self.sim_target_time = None
        self.real_target_time = None

        self.last_print_time = 0.0

        self.create_subscription(
            Point,
            SIM_TARGET_TOPIC,
            self.sim_callback,
            10,
        )

        self.create_subscription(
            Point,
            REAL_TARGET_TOPIC,
            self.real_callback,
            10,
        )

        self.ok_pub = self.create_publisher(
            Bool,
            MATCH_OK_TOPIC,
            10,
        )

        self.error_pub = self.create_publisher(
            Float64,
            MATCH_ERROR_TOPIC,
            10,
        )

        self.status_pub = self.create_publisher(
            String,
            MATCH_STATUS_TOPIC,
            10,
        )

        self.timer = self.create_timer(0.05, self.timer_callback)

        self.get_logger().info("Real/sim target comparator started.")
        self.get_logger().info(f"Sim target topic:  {SIM_TARGET_TOPIC}")
        self.get_logger().info(f"Real target topic: {REAL_TARGET_TOPIC}")
        self.get_logger().info(f"Match OK topic:    {MATCH_OK_TOPIC}")
        self.get_logger().info(f"Max allowed error: {MAX_ALLOWED_ERROR_M:.3f} m")

    def sim_callback(self, msg):
        self.sim_target = msg
        self.sim_target_time = time.time()

    def real_callback(self, msg):
        self.real_target = msg
        self.real_target_time = time.time()

    def publish_result(self, ok: bool, error_m: float, status: str):
        ok_msg = Bool()
        ok_msg.data = bool(ok)
        self.ok_pub.publish(ok_msg)

        error_msg = Float64()
        error_msg.data = float(error_m)
        self.error_pub.publish(error_msg)

        status_msg = String()
        status_msg.data = status
        self.status_pub.publish(status_msg)

    def timer_callback(self):
        now = time.time()

        if self.sim_target is None:
            self.publish_result(
                ok=False,
                error_m=float("inf"),
                status="not_ok: no simulated target received yet",
            )
            return

        if self.real_target is None:
            self.publish_result(
                ok=False,
                error_m=float("inf"),
                status="not_ok: no real target received yet",
            )
            return

        sim_age = now - self.sim_target_time
        real_age = now - self.real_target_time

        if sim_age > MAX_TARGET_AGE_S:
            self.publish_result(
                ok=False,
                error_m=float("inf"),
                status=f"not_ok: simulated target too old, age={sim_age:.3f}s",
            )
            return

        if real_age > MAX_TARGET_AGE_S:
            self.publish_result(
                ok=False,
                error_m=float("inf"),
                status=f"not_ok: real target too old, age={real_age:.3f}s",
            )
            return

        dx = self.real_target.x - self.sim_target.x
        dy = self.real_target.y - self.sim_target.y
        dz = self.real_target.z - self.sim_target.z

        error = math.sqrt(dx * dx + dy * dy + dz * dz)

        ok = error <= MAX_ALLOWED_ERROR_M

        if ok:
            status = (
                f"ok: error={error:.4f} m, "
                f"dx={dx:.4f}, dy={dy:.4f}, dz={dz:.4f}, "
                f"sim=({self.sim_target.x:.3f}, {self.sim_target.y:.3f}, {self.sim_target.z:.3f}), "
                f"real=({self.real_target.x:.3f}, {self.real_target.y:.3f}, {self.real_target.z:.3f})"
            )
        else:
            status = (
                f"not_ok: error={error:.4f} m exceeds limit {MAX_ALLOWED_ERROR_M:.4f} m, "
                f"dx={dx:.4f}, dy={dy:.4f}, dz={dz:.4f}, "
                f"sim=({self.sim_target.x:.3f}, {self.sim_target.y:.3f}, {self.sim_target.z:.3f}), "
                f"real=({self.real_target.x:.3f}, {self.real_target.y:.3f}, {self.real_target.z:.3f})"
            )

        self.publish_result(
            ok=ok,
            error_m=error,
            status=status,
        )

        if now - self.last_print_time > PRINT_PERIOD_S:
            self.last_print_time = now

            if ok:
                self.get_logger().info(status)
            else:
                self.get_logger().warn(status)


def main(args=None):
    rclpy.init(args=args)
    node = RealSimTargetComparator()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
