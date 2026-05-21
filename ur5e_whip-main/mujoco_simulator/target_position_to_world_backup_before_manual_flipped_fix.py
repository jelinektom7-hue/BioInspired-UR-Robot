#!/usr/bin/env python3

import time
import numpy as np

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Point, PoseStamped


# ============================================================
# USER CONFIGURATION
# ============================================================

# Published by image_object_locator/image_subscriber.py
CAMERA_TARGET_TOPIC = "/target_position"

# Published by real_camera_pose_publisher.py
REAL_CAMERA_POSE_TOPIC = "/real/camera_pose_world"

# Output used by real_sim_target_comparator.py
REAL_TARGET_WORLD_TOPIC = "/real/target_position_world"

MAX_TARGET_AGE_S = 1.0
MAX_CAMERA_POSE_AGE_S = 5.0

PRINT_PERIOD_S = 0.5


def quaternion_to_rotation_matrix(qx, qy, qz, qw):
    x = qx
    y = qy
    z = qz
    w = qw

    R = np.array([
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - z * w),
            2.0 * (x * z + y * w),
        ],
        [
            2.0 * (x * y + z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - x * w),
        ],
        [
            2.0 * (x * z - y * w),
            2.0 * (y * z + x * w),
            1.0 - 2.0 * (x * x + y * y),
        ],
    ], dtype=float)

    return R


class TargetPositionToWorld(Node):
    def __init__(self):
        super().__init__("target_position_to_world")

        self.latest_target_camera = None
        self.latest_camera_pose = None

        self.latest_target_time = None
        self.latest_pose_time = None

        self.last_print_time = 0.0

        self.target_sub = self.create_subscription(
            Point,
            CAMERA_TARGET_TOPIC,
            self.target_callback,
            10,
        )

        self.pose_sub = self.create_subscription(
            PoseStamped,
            REAL_CAMERA_POSE_TOPIC,
            self.pose_callback,
            10,
        )

        self.world_pub = self.create_publisher(
            Point,
            REAL_TARGET_WORLD_TOPIC,
            10,
        )

        self.timer = self.create_timer(0.05, self.timer_callback)

        self.get_logger().info("Target-position-to-world adapter started.")
        self.get_logger().info(f"Input camera target: {CAMERA_TARGET_TOPIC}")
        self.get_logger().info(f"Input camera pose:   {REAL_CAMERA_POSE_TOPIC}")
        self.get_logger().info(f"Output world target: {REAL_TARGET_WORLD_TOPIC}")

    def target_callback(self, msg):
        self.latest_target_camera = msg
        self.latest_target_time = time.time()

    def pose_callback(self, msg):
        self.latest_camera_pose = msg
        self.latest_pose_time = time.time()

    def timer_callback(self):
        now = time.time()

        if self.latest_target_camera is None:
            return

        if self.latest_camera_pose is None:
            return

        if now - self.latest_target_time > MAX_TARGET_AGE_S:
            return

        if now - self.latest_pose_time > MAX_CAMERA_POSE_AGE_S:
            return

        target_camera = np.array([
            self.latest_target_camera.x,
            self.latest_target_camera.y,
            self.latest_target_camera.z,
        ], dtype=float)

        pose = self.latest_camera_pose.pose

        t_world_camera = np.array([
            pose.position.x,
            pose.position.y,
            pose.position.z,
        ], dtype=float)

        R_world_camera = quaternion_to_rotation_matrix(
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        )

        target_world = t_world_camera + R_world_camera @ target_camera

        msg = Point()
        msg.x = float(target_world[0])
        msg.y = float(target_world[1])
        msg.z = float(target_world[2])

        self.world_pub.publish(msg)

        if now - self.last_print_time > PRINT_PERIOD_S:
            self.last_print_time = now
            self.get_logger().info(
                f"camera target=({target_camera[0]:.3f}, {target_camera[1]:.3f}, {target_camera[2]:.3f}) "
                f"-> world target=({msg.x:.3f}, {msg.y:.3f}, {msg.z:.3f})"
            )


def main(args=None):
    rclpy.init(args=args)
    node = TargetPositionToWorld()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
