#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy


class RealCameraPosePublisher(Node):
    def __init__(self):
        super().__init__("real_camera_pose_publisher")

        qos_latched = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.pose_pub = self.create_publisher(
            PoseStamped,
            "/real/camera_pose_world",
            qos_latched
        )

        # TODO: replace with calibrated real camera pose in world / robot base frame.
        self.x = 0.0
        self.y = 2.30
        self.z = 0.50

        self.qx = 0.0
        self.qy = 0.615386
        self.qz = 0.788226
        self.qw = 0.0

        self.timer = self.create_timer(1.0, self.publish_pose)

        self.get_logger().info("Publishing /real/camera_pose_world")

    def publish_pose(self):
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "world"

        msg.pose.position.x = float(self.x)
        msg.pose.position.y = float(self.y)
        msg.pose.position.z = float(self.z)

        msg.pose.orientation.x = float(self.qx)
        msg.pose.orientation.y = float(self.qy)
        msg.pose.orientation.z = float(self.qz)
        msg.pose.orientation.w = float(self.qw)

        self.pose_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = RealCameraPosePublisher()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()