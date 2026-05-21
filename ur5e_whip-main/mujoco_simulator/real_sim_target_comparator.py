#!/usr/bin/env python3

import math

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Point
from std_msgs.msg import Bool, Float64, String


SIM_TARGET_TOPIC = "/target_position_world"
REAL_TARGET_TOPIC = "/real/target_position_world"

MATCH_OK_TOPIC = "/whip_vision/target_match_ok"
MATCH_ERROR_TOPIC = "/whip_vision/target_match_error"
MATCH_STATUS_TOPIC = "/whip_vision/target_match_status"

MAX_ALLOWED_ERROR_M = 0.10

# Required because the real robot trajectory was flipped 180 degrees.
FLIP_SIM_TARGET_180 = True

LOG_EVERY_N_MESSAGES = 5


class RealSimTargetComparator(Node):
    def __init__(self):
        super().__init__("real_sim_target_comparator")

        self.latest_sim = None
        self.latest_real = None
        self.msg_count = 0

        self.sim_sub = self.create_subscription(
            Point,
            SIM_TARGET_TOPIC,
            self.sim_callback,
            10,
        )

        self.real_sub = self.create_subscription(
            Point,
            REAL_TARGET_TOPIC,
            self.real_callback,
            10,
        )

        self.ok_pub = self.create_publisher(Bool, MATCH_OK_TOPIC, 10)
        self.error_pub = self.create_publisher(Float64, MATCH_ERROR_TOPIC, 10)
        self.status_pub = self.create_publisher(String, MATCH_STATUS_TOPIC, 10)

        self.get_logger().info("Real/sim target comparator started.")
        self.get_logger().info(f"Sim target topic:  {SIM_TARGET_TOPIC}")
        self.get_logger().info(f"Real target topic: {REAL_TARGET_TOPIC}")
        self.get_logger().info(f"Match OK topic:    {MATCH_OK_TOPIC}")
        self.get_logger().info(f"Max allowed error: {MAX_ALLOWED_ERROR_M:.3f} m")
        self.get_logger().info(f"FLIP_SIM_TARGET_180: {FLIP_SIM_TARGET_180}")

    def sim_callback(self, msg: Point):
        self.latest_sim = msg
        self.compare_if_ready()

    def real_callback(self, msg: Point):
        self.latest_real = msg
        self.compare_if_ready()

    def compare_if_ready(self):
        if self.latest_sim is None or self.latest_real is None:
            return

        sim_raw_x = float(self.latest_sim.x)
        sim_raw_y = float(self.latest_sim.y)
        sim_raw_z = float(self.latest_sim.z)

        if FLIP_SIM_TARGET_180:
            sim_cmp_x = -sim_raw_x
            sim_cmp_y = -sim_raw_y
            sim_cmp_z =  sim_raw_z
        else:
            sim_cmp_x = sim_raw_x
            sim_cmp_y = sim_raw_y
            sim_cmp_z = sim_raw_z

        real_x = float(self.latest_real.x)
        real_y = float(self.latest_real.y)
        real_z = float(self.latest_real.z)

        dx = real_x - sim_cmp_x
        dy = real_y - sim_cmp_y
        dz = real_z - sim_cmp_z

        error = math.sqrt(dx * dx + dy * dy + dz * dz)
        ok = error <= MAX_ALLOWED_ERROR_M

        ok_msg = Bool()
        ok_msg.data = ok
        self.ok_pub.publish(ok_msg)

        err_msg = Float64()
        err_msg.data = error
        self.error_pub.publish(err_msg)

        status_msg = String()
        status_msg.data = (
            f"{'ok' if ok else 'not_ok'}: "
            f"error={error:.4f} m, "
            f"dx={dx:.4f}, dy={dy:.4f}, dz={dz:.4f}, "
            f"sim_raw=({sim_raw_x:.3f}, {sim_raw_y:.3f}, {sim_raw_z:.3f}), "
            f"sim_compared=({sim_cmp_x:.3f}, {sim_cmp_y:.3f}, {sim_cmp_z:.3f}), "
            f"real=({real_x:.3f}, {real_y:.3f}, {real_z:.3f})"
        )
        self.status_pub.publish(status_msg)

        self.msg_count += 1
        if self.msg_count % LOG_EVERY_N_MESSAGES == 1:
            if ok:
                self.get_logger().info(status_msg.data)
            else:
                self.get_logger().warn(status_msg.data)


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
