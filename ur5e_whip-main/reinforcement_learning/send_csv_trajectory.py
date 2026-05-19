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

# Very safe dry-run limits.
# This is intentionally slow. It is for validating robot motion, not whip performance.
MAX_JOINT_SPEED = 0.20  # rad/s
MIN_TIME_BETWEEN_POINTS = 0.25  # seconds
MOVE_TO_FIRST_POINT_DURATION = 10.0  # seconds
HOLD_CURRENT_POSE_DURATION = 1.0  # seconds


class SafeCsvTrajectorySender(Node):
    def __init__(self):
        super().__init__("safe_csv_trajectory_sender")

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

    def load_csv_positions(self):
        rows = []

        with open(CSV_FILE, "r") as f:
            reader = csv.DictReader(f)

            for row in reader:
                positions = [float(row[name]) for name in JOINT_NAMES]
                elapsed_time = float(row["elapsed_time"])

                rows.append(
                    {
                        "elapsed_time": elapsed_time,
                        "positions": positions,
                    }
                )

        if len(rows) < 2:
            raise RuntimeError("CSV trajectory has fewer than 2 rows.")

        return rows

    def build_safe_points(self, current_positions, csv_rows):
        points = []

        # Point 0: hold current pose briefly.
        p0 = JointTrajectoryPoint()
        p0.positions = current_positions
        p0.time_from_start.sec = int(HOLD_CURRENT_POSE_DURATION)
        p0.time_from_start.nanosec = int(
            (HOLD_CURRENT_POSE_DURATION - int(HOLD_CURRENT_POSE_DURATION)) * 1e9
        )
        points.append(p0)

        first_csv_positions = csv_rows[0]["positions"]

        # Point 1: move from current robot pose to first CSV pose slowly.
        p1 = JointTrajectoryPoint()
        p1.positions = first_csv_positions

        t = HOLD_CURRENT_POSE_DURATION + MOVE_TO_FIRST_POINT_DURATION
        p1.time_from_start.sec = int(t)
        p1.time_from_start.nanosec = int((t - int(t)) * 1e9)
        points.append(p1)

        previous_positions = first_csv_positions
        current_time = t

        max_computed_joint_speed = 0.0
        largest_joint_jump = 0.0

        for i in range(1, len(csv_rows)):
            target_positions = csv_rows[i]["positions"]

            joint_deltas = [
                abs(target - previous)
                for target, previous in zip(target_positions, previous_positions)
            ]

            largest_delta = max(joint_deltas)
            largest_joint_jump = max(largest_joint_jump, largest_delta)

            # Time needed to keep every joint under MAX_JOINT_SPEED.
            required_dt_from_velocity = largest_delta / MAX_JOINT_SPEED

            # Also respect a minimum spacing.
            dt = max(
                required_dt_from_velocity,
                MIN_TIME_BETWEEN_POINTS,
            )

            computed_speed = largest_delta / dt if dt > 0 else float("inf")
            max_computed_joint_speed = max(max_computed_joint_speed, computed_speed)

            current_time += dt

            p = JointTrajectoryPoint()
            p.positions = target_positions
            p.time_from_start.sec = int(current_time)
            p.time_from_start.nanosec = int((current_time - int(current_time)) * 1e9)
            points.append(p)

            previous_positions = target_positions

        return points, current_time, largest_joint_jump, max_computed_joint_speed

    def send(self):
        self.wait_for_joint_state()

        current_positions = self.get_current_positions_ordered()
        csv_rows = self.load_csv_positions()

        points, total_duration, largest_jump, max_speed = self.build_safe_points(
            current_positions,
            csv_rows,
        )

        print("\nCurrent robot pose:")
        for name, value in zip(JOINT_NAMES, current_positions):
            print(f"  {name:22s}: {value: .4f} rad")

        print("\nFirst CSV pose:")
        for name, value in zip(JOINT_NAMES, csv_rows[0]["positions"]):
            print(f"  {name:22s}: {value: .4f} rad")

        print("\nSafe trajectory summary:")
        print(f"  CSV file:                    {CSV_FILE}")
        print(f"  CSV rows:                    {len(csv_rows)}")
        print(f"  Points sent:                 {len(points)}")
        print(f"  Total planned duration:      {total_duration:.2f} s")
        print(f"  Largest point-to-point jump: {largest_jump:.4f} rad")
        print(f"  Max planned joint speed:     {max_speed:.4f} rad/s")
        print(f"  Safety speed limit used:     {MAX_JOINT_SPEED:.4f} rad/s")

        print("\nSafety checklist:")
        print("  - No whip attached")
        print("  - Workspace clear")
        print("  - Robot speed slider low, around 5–10%")
        print("  - Emergency stop accessible")
        print("  - External Control program running")
        print("  - This is only a slow dry-run trajectory")

        confirm = input("\nType SEND to send this safe slowed trajectory: ")
        if confirm.strip() != "SEND":
            print("Cancelled.")
            return

        self.get_logger().info(f"Waiting for action server: {ACTION_NAME}")
        self.client.wait_for_server()

        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = JOINT_NAMES
        goal.trajectory.points = points

        self.get_logger().info(
            f"Sending safe CSV trajectory with {len(points)} points over {total_duration:.2f} s"
        )

        future = self.client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)

        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error("Trajectory rejected.")
            return

        self.get_logger().info("Trajectory accepted.")

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)

        result = result_future.result().result
        self.get_logger().info(f"Done. Error code: {result.error_code}")


def main():
    rclpy.init()
    node = SafeCsvTrajectorySender()

    try:
        node.send()
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()