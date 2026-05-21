#!/usr/bin/env python3

import math
import time
from typing import Optional, Tuple

import cv2
import numpy as np

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Point, PoseStamped
from cv_bridge import CvBridge


# ============================================================
# USER CONFIGURATION
# ============================================================

COLOR_TOPIC = "/camera/camera/color/image_raw"
DEPTH_TOPIC = "/camera/camera/aligned_depth_to_color/image_raw"
CAMERA_INFO_TOPIC = "/camera/camera/color/camera_info"
REAL_CAMERA_POSE_TOPIC = "/real/camera_pose_world"

REAL_TARGET_WORLD_TOPIC = "/real/target_position_world"

# HSV threshold for a green target.
# Tune these if your target is not detected.
HSV_LOWER = np.array([35, 60, 40])
HSV_UPPER = np.array([90, 255, 255])

MIN_TARGET_AREA_PX = 80

MAX_IMAGE_AGE_S = 1.0
MAX_DEPTH_AGE_S = 1.0
MAX_POSE_AGE_S = 5.0

PRINT_PERIOD_S = 0.5


# ============================================================
# QUATERNION / TRANSFORM HELPERS
# ============================================================

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


# ============================================================
# NODE
# ============================================================

class RealTargetLocator(Node):
    def __init__(self):
        super().__init__("real_target_locator")

        self.bridge = CvBridge()

        self.latest_color = None
        self.latest_depth = None
        self.latest_camera_info = None
        self.latest_camera_pose = None

        self.latest_color_time = None
        self.latest_depth_time = None
        self.latest_pose_time = None

        self.last_print_time = 0.0

        self.color_sub = self.create_subscription(
            Image,
            COLOR_TOPIC,
            self.color_callback,
            10,
        )

        self.depth_sub = self.create_subscription(
            Image,
            DEPTH_TOPIC,
            self.depth_callback,
            10,
        )

        self.info_sub = self.create_subscription(
            CameraInfo,
            CAMERA_INFO_TOPIC,
            self.camera_info_callback,
            10,
        )

        self.pose_sub = self.create_subscription(
            PoseStamped,
            REAL_CAMERA_POSE_TOPIC,
            self.camera_pose_callback,
            10,
        )

        self.target_pub = self.create_publisher(
            Point,
            REAL_TARGET_WORLD_TOPIC,
            10,
        )

        self.timer = self.create_timer(0.05, self.timer_callback)

        self.get_logger().info("Real target locator started.")
        self.get_logger().info(f"Color topic: {COLOR_TOPIC}")
        self.get_logger().info(f"Depth topic: {DEPTH_TOPIC}")
        self.get_logger().info(f"Camera info topic: {CAMERA_INFO_TOPIC}")
        self.get_logger().info(f"Camera pose topic: {REAL_CAMERA_POSE_TOPIC}")
        self.get_logger().info(f"Publishing: {REAL_TARGET_WORLD_TOPIC}")

    def color_callback(self, msg):
        self.latest_color = self.bridge.imgmsg_to_cv2(
            msg,
            desired_encoding="bgr8",
        )
        self.latest_color_time = time.time()

    def depth_callback(self, msg):
        depth = self.bridge.imgmsg_to_cv2(
            msg,
            desired_encoding="passthrough",
        )

        # RealSense aligned depth is usually uint16 in millimeters.
        if depth.dtype == np.uint16:
            depth = depth.astype(np.float32) / 1000.0
        else:
            depth = depth.astype(np.float32)

        self.latest_depth = depth
        self.latest_depth_time = time.time()

    def camera_info_callback(self, msg):
        self.latest_camera_info = msg

    def camera_pose_callback(self, msg):
        self.latest_camera_pose = msg
        self.latest_pose_time = time.time()

    def detect_target_pixel(self, color_img) -> Optional[Tuple[float, float, float]]:
        hsv = cv2.cvtColor(color_img, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv, HSV_LOWER, HSV_UPPER)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        if not contours:
            return None

        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)

        if area < MIN_TARGET_AREA_PX:
            return None

        M = cv2.moments(largest)

        if M["m00"] == 0:
            return None

        u = M["m10"] / M["m00"]
        v = M["m01"] / M["m00"]

        return u, v, area

    def get_depth_at_pixel(self, depth_img, u, v) -> Optional[float]:
        h, w = depth_img.shape[:2]

        u_i = int(round(u))
        v_i = int(round(v))

        if u_i < 0 or u_i >= w or v_i < 0 or v_i >= h:
            return None

        window = depth_img[
            max(0, v_i - 2):min(h, v_i + 3),
            max(0, u_i - 2):min(w, u_i + 3),
        ]

        valid = window[np.isfinite(window)]
        valid = valid[valid > 0.05]

        if valid.size == 0:
            return None

        return float(np.median(valid))

    def pixel_depth_to_camera_xyz(self, u, v, z):
        K = self.latest_camera_info.k

        fx = K[0]
        fy = K[4]
        cx = K[2]
        cy = K[5]

        x = (u - cx) * z / fx
        y = (v - cy) * z / fy

        return np.array([x, y, z], dtype=float)

    def camera_xyz_to_world_xyz(self, camera_xyz):
        pose = self.latest_camera_pose.pose

        t = np.array([
            pose.position.x,
            pose.position.y,
            pose.position.z,
        ], dtype=float)

        R = quaternion_to_rotation_matrix(
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        )

        world_xyz = t + R @ camera_xyz

        return world_xyz

    def timer_callback(self):
        now = time.time()

        if self.latest_color is None:
            return

        if self.latest_depth is None:
            return

        if self.latest_camera_info is None:
            return

        if self.latest_camera_pose is None:
            return

        if now - self.latest_color_time > MAX_IMAGE_AGE_S:
            return

        if now - self.latest_depth_time > MAX_DEPTH_AGE_S:
            return

        if now - self.latest_pose_time > MAX_POSE_AGE_S:
            return

        detection = self.detect_target_pixel(self.latest_color)

        if detection is None:
            if now - self.last_print_time > PRINT_PERIOD_S:
                self.last_print_time = now
                self.get_logger().warn("No target detected in RGB image.")
            return

        u, v, area = detection

        z = self.get_depth_at_pixel(self.latest_depth, u, v)

        if z is None:
            if now - self.last_print_time > PRINT_PERIOD_S:
                self.last_print_time = now
                self.get_logger().warn("Target detected, but no valid depth at target pixel.")
            return

        camera_xyz = self.pixel_depth_to_camera_xyz(u, v, z)
        world_xyz = self.camera_xyz_to_world_xyz(camera_xyz)

        msg = Point()
        msg.x = float(world_xyz[0])
        msg.y = float(world_xyz[1])
        msg.z = float(world_xyz[2])

        self.target_pub.publish(msg)

        if now - self.last_print_time > PRINT_PERIOD_S:
            self.last_print_time = now
            self.get_logger().info(
                f"Real target world: "
                f"({msg.x:.3f}, {msg.y:.3f}, {msg.z:.3f}) | "
                f"pixel=({u:.1f}, {v:.1f}), depth={z:.3f} m, area={area:.1f}"
            )


def main(args=None):
    rclpy.init(args=args)
    node = RealTargetLocator()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
