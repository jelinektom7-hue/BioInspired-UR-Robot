#!/usr/bin/env python3

import csv
import os
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint


CSV_FILE = os.path.expanduser(
    "~/ros2_ws/src/BioInspired-UR-Robot/ur5e_whip-main/reinforcement_learning/exported_trajectory_best_fixed_target_slow.csv"
)

ACTION_NAME = "/scaled_joint_trajectory_controller/follow_joint_trajectory"

JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]


class CsvTrajectorySender(Node):
    def __init__(self):
        super().__init__("csv_trajectory_sender")
        self.client = ActionClient(self, FollowJointTrajectory, ACTION_NAME)

    def load_csv(self):
        points = []

        with open(CSV_FILE, "r") as f:
            reader = csv.DictReader(f)

            for row in reader:
                p = JointTrajectoryPoint()
                p.positions = [float(row[name]) for name in JOINT_NAMES]

                t = float(row["elapsed_time"])
                sec = int(t)
                nanosec = int((t - sec) * 1e9)
                p.time_from_start.sec = sec
                p.time_from_start.nanosec = nanosec

                points.append(p)

        return points

    def send(self):
        self.get_logger().info(f"Waiting for action server: {ACTION_NAME}")
        self.client.wait_for_server()

        points = self.load_csv()

        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = JOINT_NAMES
        goal.trajectory.points = points

        self.get_logger().info(f"Sending trajectory with {len(points)} points")
        future = self.client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)

        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error("Trajectory rejected")
            return

        self.get_logger().info("Trajectory accepted")
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)

        result = result_future.result().result
        self.get_logger().info(f"Done. Error code: {result.error_code}")


def main():
    rclpy.init()
    node = CsvTrajectorySender()
    node.send()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()