#!/usr/bin/env python3

import math

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Point, Pose, PoseStamped


INPUT_TARGET_TOPIC = "/target_position"
INPUT_POSE_TOPIC = "/real/camera_pose_world"
OUTPUT_WORLD_TARGET_TOPIC = "/real/target_position_world"

EXPECTED_TARGET_X = -0.85
EXPECTED_TARGET_Y =  0.40
EXPECTED_TARGET_Z =  0.70

LOG_EVERY_N_MESSAGES = 10


class TargetPositionToWorld(Node):
    def __init__(self):
        super().__init__("target_position_to_world")

        self.latest_pose = None
        self.pose_subscription_created = False
        self.pose_sub = None
        self.msg_count = 0

        self.target_sub = self.create_subscription(
            Point,
            INPUT_TARGET_TOPIC,
            self.target_callback,
            10,
        )

        self.world_pub = self.create_publisher(
            Point,
            OUTPUT_WORLD_TARGET_TOPIC,
            10,
        )

        self.pose_type_timer = self.create_timer(1.0, self.try_create_pose_subscription)

        self.get_logger().info("Target-position-to-world adapter started.")
        self.get_logger().info(f"Input camera target: {INPUT_TARGET_TOPIC}")
        self.get_logger().info(f"Input camera pose:   {INPUT_POSE_TOPIC}")
        self.get_logger().info(f"Output world target: {OUTPUT_WORLD_TARGET_TOPIC}")
        self.get_logger().info("Waiting to detect /real/camera_pose_world topic type...")

    def try_create_pose_subscription(self):
        if self.pose_subscription_created:
            return

        topic_names_and_types = self.get_topic_names_and_types()
        topic_type_list = []

        for topic_name, type_list in topic_names_and_types:
            if topic_name == INPUT_POSE_TOPIC:
                topic_type_list = type_list
                break

        if not topic_type_list:
            self.get_logger().warn(
                f"{INPUT_POSE_TOPIC} not visible yet. Start real_camera_pose_publisher.py first."
            )
            return

        self.get_logger().info(f"Detected {INPUT_POSE_TOPIC} type(s): {topic_type_list}")

        if "geometry_msgs/msg/PoseStamped" in topic_type_list:
            self.pose_sub = self.create_subscription(
                PoseStamped,
                INPUT_POSE_TOPIC,
                self.pose_stamped_callback,
                10,
            )
            self.pose_subscription_created = True
            self.get_logger().info("Subscribed to camera pose as geometry_msgs/msg/PoseStamped.")
            self.pose_type_timer.cancel()

        elif "geometry_msgs/msg/Pose" in topic_type_list:
            self.pose_sub = self.create_subscription(
                Pose,
                INPUT_POSE_TOPIC,
                self.pose_callback,
                10,
            )
            self.pose_subscription_created = True
            self.get_logger().info("Subscribed to camera pose as geometry_msgs/msg/Pose.")
            self.pose_type_timer.cancel()

        else:
            self.get_logger().error(
                f"Unsupported pose topic type for {INPUT_POSE_TOPIC}: {topic_type_list}"
            )

    def pose_stamped_callback(self, msg: PoseStamped):
        self.latest_pose = msg.pose

    def pose_callback(self, msg: Pose):
        self.latest_pose = msg

    def target_callback(self, msg: Point):
        if not self.pose_subscription_created:
            self.get_logger().warn("No valid camera pose subscription yet.")
            return

        if self.latest_pose is None:
            self.get_logger().warn(
                f"Received {INPUT_TARGET_TOPIC}, but no camera pose message yet."
            )
            return

        p = self.latest_pose.position

        cam_x = float(msg.x)
        cam_y = float(msg.y)
        cam_z = float(msg.z)

        # Manual mapping for your flipped real setup:
        #
        # Camera target appears to be:
        #   x = left/right
        #   y = vertical
        #   z = depth forward
        #
        # Real camera is placed around:
        #   world x = 0.0
        #   world y = +2.30
        #   world z = +0.50
        #
        # Since the camera looks from +Y back toward the robot,
        # depth should REDUCE world Y.
        world_x = float(p.x) + cam_x
        world_y = float(p.y) - cam_z
        world_z = float(p.z) + cam_y

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
