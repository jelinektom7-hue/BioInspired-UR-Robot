#!/usr/bin/env python3

import math

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Point, Pose, PoseStamped


INPUT_TARGET_TOPIC = "/target_position"
INPUT_POSE_TOPIC = "/real/camera_pose_world"
OUTPUT_WORLD_TARGET_TOPIC = "/real/target_position_world"

# Expected real target for the 180-degree flipped robot trajectory.
EXPECTED_TARGET_X = -0.85
EXPECTED_TARGET_Y =  0.40
EXPECTED_TARGET_Z =  0.70

# This is the important fix for the current physical flipped setup.
#
# Assumption from observed data:
#   /target_position.x = camera left/right coordinate, in meters
#   /target_position.y = camera vertical coordinate, in meters
#   /target_position.z = camera depth/forward coordinate, in meters
#
# Real camera is placed at roughly:
#   world x = 0.0
#   world y = +2.30
#   world z = +0.50
#
# Since the camera looks from +Y back toward the robot/target,
# camera depth must reduce world Y, not increase it.
USE_MANUAL_FLIPPED_CAMERA_MAPPING = True

LOG_EVERY_N_MESSAGES = 10


class TargetPositionToWorld(Node):
    def __init__(self):
        super().__init__("target_position_to_world")

        self.latest_pose = None
        self.msg_count = 0

        self.target_sub = self.create_subscription(
            Point,
            INPUT_TARGET_TOPIC,
            self.target_callback,
            10,
        )

        # Support either PoseStamped or Pose, depending on what the pose publisher uses.
        self.pose_stamped_sub = self.create_subscription(
            PoseStamped,
            INPUT_POSE_TOPIC,
            self.pose_stamped_callback,
            10,
        )

        self.pose_sub = self.create_subscription(
            Pose,
            INPUT_POSE_TOPIC,
            self.pose_callback,
            10,
        )

        self.world_pub = self.create_publisher(
            Point,
            OUTPUT_WORLD_TARGET_TOPIC,
            10,
        )

        self.get_logger().info("Target-position-to-world adapter started.")
        self.get_logger().info(f"Input camera target: {INPUT_TARGET_TOPIC}")
        self.get_logger().info(f"Input camera pose:   {INPUT_POSE_TOPIC}")
        self.get_logger().info(f"Output world target: {OUTPUT_WORLD_TARGET_TOPIC}")
        self.get_logger().info(
            "Using MANUAL FLIPPED mapping: "
            "world_x=pose_x+cam_x, world_y=pose_y-cam_z, world_z=pose_z+cam_y"
        )

    def pose_stamped_callback(self, msg: PoseStamped):
        self.latest_pose = msg.pose

    def pose_callback(self, msg: Pose):
        self.latest_pose = msg

    def target_callback(self, msg: Point):
        if self.latest_pose is None:
            self.get_logger().warn(
                f"Received {INPUT_TARGET_TOPIC}, but no {INPUT_POSE_TOPIC} yet."
            )
            return

        p = self.latest_pose.position

        cam_x = float(msg.x)
        cam_y = float(msg.y)
        cam_z = float(msg.z)

        if USE_MANUAL_FLIPPED_CAMERA_MAPPING:
            world_x = float(p.x) + cam_x
            world_y = float(p.y) - cam_z
            world_z = float(p.z) + cam_y
        else:
            # Fallback should not be used right now.
            world_x = float(p.x) + cam_x
            world_y = float(p.y) + cam_y
            world_z = float(p.z) + cam_z

        out = Point()
        out.x = world_x
        out.y = world_y
        out.z = world_z
        self.world_pub.publish(out)

        self.msg_count += 1

        if self.msg_count % LOG_EVERY_N_MESSAGES == 1:
            err = math.sqrt(
                (world_x - EXPECTED_TARGET_X) ** 2
                + (world_y - EXPECTED_TARGET_Y) ** 2
                + (world_z - EXPECTED_TARGET_Z) ** 2
            )

            self.get_logger().info(
                "camera target=({:.3f}, {:.3f}, {:.3f}) -> "
                "world target=({:.3f}, {:.3f}, {:.3f}), "
                "expected=({:.3f}, {:.3f}, {:.3f}), "
                "error_to_expected={:.3f} m".format(
                    cam_x,
                    cam_y,
                    cam_z,
                    world_x,
                    world_y,
                    world_z,
                    EXPECTED_TARGET_X,
                    EXPECTED_TARGET_Y,
                    EXPECTED_TARGET_Z,
                    err,
                )
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
