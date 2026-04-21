#!/usr/bin/env python3

import math

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
from std_msgs.msg import Float64MultiArray


class FakeTargetPublisher(Node):
    def __init__(self):
        super().__init__('fake_target_publisher')

        self.pos_pub = self.create_publisher(Point, '/target_position', 10)
        self.transform_pub = self.create_publisher(Float64MultiArray, '/target_reference_frame', 10)

        self.t = 0.0
        self.timer = self.create_timer(0.1, self.publish_target)

        self.get_logger().info('Fake target publisher started')

    def publish_target(self):
        x = 0.35 + 0.02 * math.sin(self.t)
        y = 0.10 + 0.02 * math.cos(self.t)
        z = 0.25

        pos_msg = Point()
        pos_msg.x = x
        pos_msg.y = y
        pos_msg.z = z
        self.pos_pub.publish(pos_msg)

        tf_msg = Float64MultiArray()
        tf_msg.data = [
            1.0, 0.0, 0.0, x,
            0.0, 1.0, 0.0, y,
            0.0, 0.0, 1.0, z,
            0.0, 0.0, 0.0, 1.0
        ]
        self.transform_pub.publish(tf_msg)

        self.get_logger().info(f'Published target: x={x:.3f}, y={y:.3f}, z={z:.3f}')
        self.t += 0.1


def main(args=None):
    rclpy.init(args=args)
    node = FakeTargetPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
