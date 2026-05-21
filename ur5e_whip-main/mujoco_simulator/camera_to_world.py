#!/usr/bin/env python3
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from geometry_msgs.msg import Point, PoseStamped


def quaternion_to_rotation_matrix(x, y, z, w):
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    return np.array([
        [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz),       2.0 * (xz + wy)],
        [2.0 * (xy + wz),       1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
        [2.0 * (xz - wy),       2.0 * (yz + wx),       1.0 - 2.0 * (xx + yy)],
    ], dtype=float)


class CameraToWorld(Node):
    def __init__(self):
        super().__init__("camera_to_world")

        self.t_cam_world = None
        self.R_cam_to_world = None

        qos_latched = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.pose_sub = self.create_subscription(
            PoseStamped,
            "/camera_pose_world",
            self.pose_callback,
            qos_latched,
        )

        self.target_sub = self.create_subscription(
            Point,
            "/target_position",
            self.target_callback,
            10,
        )

        self.world_pub = self.create_publisher(
            Point,
            "/target_position_world",
            10,
        )

        self.cam_pos_pub = self.create_publisher(
            Point,
            "/camera_position_world",
            10,
        )

        self.get_logger().info("camera_to_world started, waiting for /camera_pose_world")

    def pose_callback(self, msg: PoseStamped):
        self.t_cam_world = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z,
        ], dtype=float)

        qx = msg.pose.orientation.x
        qy = msg.pose.orientation.y
        qz = msg.pose.orientation.z
        qw = msg.pose.orientation.w

        self.R_cam_to_world = quaternion_to_rotation_matrix(qx, qy, qz, qw)

        cam_msg = Point()
        cam_msg.x = float(self.t_cam_world[0])
        cam_msg.y = float(self.t_cam_world[1])
        cam_msg.z = float(self.t_cam_world[2])
        self.cam_pos_pub.publish(cam_msg)

        self.get_logger().info(
            f"Received camera pose: "
            f"pos=({cam_msg.x:.3f}, {cam_msg.y:.3f}, {cam_msg.z:.3f})"
        )

    def target_callback(self, msg: Point):
        if self.t_cam_world is None or self.R_cam_to_world is None:
            self.get_logger().warn("No camera pose received yet")
            return

        # Detector output frame
        p_det = np.array([msg.x, msg.y, msg.z], dtype=float)

        # Detector frame -> MuJoCo camera frame
        p_cam = np.array([
            -p_det[0],   # flip x
             p_det[1],   # keep y
            -p_det[2],   # keep z
        ], dtype=float)

        # Camera frame -> world frame
        p_world = self.t_cam_world + self.R_cam_to_world @ p_cam

        out = Point()
        out.x = float(p_world[0])
        out.y = float(p_world[1])
        out.z = float(p_world[2])

        self.world_pub.publish(out)

        self.get_logger().info(
            f"det=({msg.x:.3f}, {msg.y:.3f}, {msg.z:.3f}) -> "
            f"world=({out.x:.3f}, {out.y:.3f}, {out.z:.3f})"
        )


def main(args=None):
    rclpy.init(args=args)
    node = CameraToWorld()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()