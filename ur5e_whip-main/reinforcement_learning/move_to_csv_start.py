#!/usr/bin/env python3

import csv
import os
import math

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from sensor_msgs.msg import JointState
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

MOVE_DURATION = 15.0  # seconds, slow and safe
MAX_ALLOWED_JOINT_DELTA = 2.0  # rad; warning threshold, not a hard robot limit


class MoveToCsvStart(Node):
    def __init__(self):
        super().__init__("move_to_csv_start")

        self.current_joint_state = None

        self.joint_sub = self.create_subscription(
            JointState,
            "/joint_states",
            self.joint_state_callback,
            10,
        )

        self.client = ActionClient(
            self,
            FollowJointTrajectory,
            ACTION_NAME,
        )

    def joint_state_callback(self, msg):
        self.current_joint_state = msg

    def wait_for_joint_state(self):
        self.get_logger().info("Waiting for /joint_states...")

        while rclpy.ok() and self.current_joint_state is None:
            rclpy.spin_once(self, timeout_sec=0.1)

        self.get_logger().info("Received /joint_states.")

    def get_current_positions_ordered(self):
        name_to_position = dict(
            zip(self.current_joint_state.name, self.current_joint_state.position)
        )

        missing = [name for name in JOINT_NAMES if name not in name_to_position]
        if missing:
            raise RuntimeError(f"Missing joints in /joint_states: {missing}")

        return [float(name_to_position[name]) for name in JOINT_NAMES]

    def load_csv_start_positions(self):
        with open(CSV_FILE, "r") as f:
            reader = csv.DictReader(f)
            first_row = next(reader)

        return [float(first_row[name]) for name in JOINT_NAMES]

    def send_move(self):
        self.wait_for_joint_state()

        current_positions = self.get_current_positions_ordered()
        target_positions = self.load_csv_start_positions()

        deltas = [
            abs(t - c)
            for c, t in zip(current_positions, target_positions)
        ]

        max_delta = max(deltas)

        print("\nCurrent robot pose:")
        for name, value in zip(JOINT_NAMES, current_positions):
            print(f"  {name:22s}: {value: .4f} rad")

        print("\nCSV start pose:")
        for name, value in zip(JOINT_NAMES, target_positions):
            print(f"  {name:22s}: {value: .4f} rad")

        print("\nJoint deltas:")
        for name, value in zip(JOINT_NAMES, deltas):
            print(f"  {name:22s}: {value: .4f} rad")

        print(f"\nMax joint delta: {max_delta:.4f} rad")

        if max_delta > MAX_ALLOWED_JOINT_DELTA:
            print("\nWARNING:")
            print("The start pose is far from the current robot pose.")
            print("This may still be okay because the move is slow, but inspect it carefully.")

        print("\nSafety checklist:")
        print("  - No whip attached")
        print("  - Workspace clear")
        print("  - Robot speed slider low, around 5–10%")
        print("  - Emergency stop accessible")
        print("  - You are ready for slow motion to the CSV start pose")

        confirm = input("\nType MOVE to send the slow move-to-start trajectory: ")
        if confirm.strip() != "MOVE":
            print("Cancelled.")
            return

        self.get_logger().info(f"Waiting for action server: {ACTION_NAME}")
        self.client.wait_for_server()

        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = JOINT_NAMES

        # Point 1: current pose at t=0
        p0 = JointTrajectoryPoint()
        p0.positions = current_positions
        p0.time_from_start.sec = 0
        p0.time_from_start.nanosec = 0

        # Point 2: CSV start pose after MOVE_DURATION seconds
        p1 = JointTrajectoryPoint()
        p1.positions = target_positions

        sec = int(MOVE_DURATION)
        nanosec = int((MOVE_DURATION - sec) * 1e9)
        p1.time_from_start.sec = sec
        p1.time_from_start.nanosec = nanosec

        goal.trajectory.points = [p0, p1]

        self.get_logger().info(
            f"Sending slow move-to-start trajectory over {MOVE_DURATION:.1f} seconds"
        )

        future = self.client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)

        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error("Move-to-start trajectory rejected.")
            return

        self.get_logger().info("Move-to-start trajectory accepted.")

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)

        result = result_future.result().result
        self.get_logger().info(f"Done. Error code: {result.error_code}")


def main():
    rclpy.init()
    node = MoveToCsvStart()

    try:
        node.send_move()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()