#!/usr/bin/env python3

import math
import time
from typing import Dict, List, Tuple

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node

from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.action import FollowJointTrajectory
from builtin_interfaces.msg import Duration

# ============================================================
# INITIAL UPWARD POSE CONFIGURATION
# ============================================================

MOVE_TO_INITIAL_UPWARD_POSE = True

# A conservative upward / ceiling-pointing diagnostic pose.
# Joint order:
# shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3
INITIAL_UPWARD_POSE = [
    0.0,
    -math.pi / 2.0,
    0.0,
    -math.pi / 2.0,
    0.0,
    0.0,
]

INITIAL_MOVE_MAX_SPEED_RAD_S = 0.15
INITIAL_MOVE_ACCEL_RAD_S2 = 0.10
INITIAL_MOVE_START_HOLD_S = 1.0
INITIAL_MOVE_END_HOLD_S = 2.0

# ============================================================
# USER CONFIGURATION
# ============================================================

TEST_ANGLE_RAD = 0.15          # Safer than 0.20 for first full diagnostic
MAX_SPEED_RAD_S = 0.20         # Safer than 0.30
ACCEL_RAD_S2 = 0.15
DT = 0.05

START_DELAY_S = 0.75           # First trajectory point is not at t=0
START_HOLD_S = 1.0
HOLD_AT_EXTREME_S = 0.75
HOLD_BETWEEN_JOINTS_S = 2.0

SETTLE_BEFORE_EACH_JOINT_S = 2.0
REPETITIONS_PER_JOINT = 1

ACTION_NAME = "/scaled_joint_trajectory_controller/follow_joint_trajectory"

JOINT_LIMIT_LOW = -2.0 * math.pi
JOINT_LIMIT_HIGH = 2.0 * math.pi
JOINT_LIMIT_MARGIN = 0.05

UR5E_MAX_SPEED_RAD_S = math.pi

GOAL_TIME_TOLERANCE_S = 3.0


# ============================================================
# UR5e joint order expected by the trajectory controller
# ============================================================

JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]


def sec_to_duration(seconds: float) -> Duration:
    msg = Duration()
    msg.sec = int(seconds)
    msg.nanosec = int((seconds - msg.sec) * 1e9)
    return msg


def clamp_to_joint_limits(q: float) -> float:
    low = JOINT_LIMIT_LOW + JOINT_LIMIT_MARGIN
    high = JOINT_LIMIT_HIGH - JOINT_LIMIT_MARGIN
    return max(low, min(high, q))


def make_point(
    positions: List[float],
    velocities: List[float],
    t: float,
) -> JointTrajectoryPoint:
    point = JointTrajectoryPoint()
    point.positions = list(positions)
    point.velocities = list(velocities)
    point.time_from_start = sec_to_duration(t)
    return point


def trapezoid_1d(
    q_start: float,
    q_goal: float,
    max_speed: float,
    accel: float,
    dt: float,
) -> List[Tuple[float, float, float]]:
    distance_signed = q_goal - q_start
    distance = abs(distance_signed)

    if distance < 1e-9:
        return [(dt, q_goal, 0.0)]

    direction = 1.0 if distance_signed >= 0.0 else -1.0

    max_speed = min(abs(max_speed), UR5E_MAX_SPEED_RAD_S)
    accel = abs(accel)

    if max_speed <= 0.0:
        raise ValueError("MAX_SPEED_RAD_S must be > 0")

    if accel <= 0.0:
        raise ValueError("ACCEL_RAD_S2 must be > 0")

    t_accel = max_speed / accel
    d_accel = 0.5 * accel * t_accel * t_accel

    if 2.0 * d_accel >= distance:
        # Triangular profile
        t_accel = math.sqrt(distance / accel)
        t_flat = 0.0
        v_peak = accel * t_accel
    else:
        # Trapezoidal profile
        d_flat = distance - 2.0 * d_accel
        t_flat = d_flat / max_speed
        v_peak = max_speed

    total_time = 2.0 * t_accel + t_flat

    samples = []
    n_steps = max(1, int(math.ceil(total_time / dt)))

    for i in range(1, n_steps + 1):
        t = min(i * dt, total_time)

        if t <= t_accel:
            s = 0.5 * accel * t * t
            v = accel * t

        elif t <= t_accel + t_flat:
            t2 = t - t_accel
            s = 0.5 * accel * t_accel * t_accel + v_peak * t2
            v = v_peak

        else:
            t3 = t - t_accel - t_flat
            d_before_decel = 0.5 * accel * t_accel * t_accel + v_peak * t_flat
            s = d_before_decel + v_peak * t3 - 0.5 * accel * t3 * t3
            v = max(0.0, v_peak - accel * t3)

        q = q_start + direction * s
        qd = direction * v

        samples.append((t, q, qd))

    samples[-1] = (total_time, q_goal, 0.0)
    return samples


class RealJointDiagnostic(Node):
    def __init__(self):
        super().__init__("real_joint_diagnostic")

        self.latest_joint_state = None

        self.joint_sub = self.create_subscription(
            JointState,
            "/joint_states",
            self.joint_state_callback,
            10,
        )

        self.action_client = ActionClient(
            self,
            FollowJointTrajectory,
            ACTION_NAME,
        )

    def joint_state_callback(self, msg: JointState):
        self.latest_joint_state = msg

    def wait_for_joint_state(self, timeout_s: float = 10.0) -> Dict[str, float]:
        start = time.time()

        while rclpy.ok() and time.time() - start < timeout_s:
            rclpy.spin_once(self, timeout_sec=0.1)

            if self.latest_joint_state is None:
                continue

            joint_map = {
                name: pos
                for name, pos in zip(
                    self.latest_joint_state.name,
                    self.latest_joint_state.position,
                )
            }

            if all(j in joint_map for j in JOINT_NAMES):
                return joint_map

        raise RuntimeError("Timed out waiting for /joint_states with all UR5e joints.")

    def wait_until_robot_settled(self, settle_s: float):
        start = time.time()

        while rclpy.ok() and time.time() - start < settle_s:
            rclpy.spin_once(self, timeout_sec=0.1)

    def send_trajectory(self, trajectory: JointTrajectory):
        if not self.action_client.wait_for_server(timeout_sec=10.0):
            raise RuntimeError(f"Action server not available: {ACTION_NAME}")

        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory = trajectory
        goal_msg.goal_time_tolerance = sec_to_duration(GOAL_TIME_TOLERANCE_S)

        send_future = self.action_client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self, send_future)

        goal_handle = send_future.result()

        if goal_handle is None:
            raise RuntimeError("No goal handle returned by action server.")

        if not goal_handle.accepted:
            raise RuntimeError(
                "Trajectory goal was rejected by the controller. "
                "This usually means the first point, timing, or current robot state "
                "was not acceptable to the scaled_joint_trajectory_controller."
            )

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)

        result = result_future.result().result

        if result.error_code != FollowJointTrajectory.Result.SUCCESSFUL:
            raise RuntimeError(
                f"Trajectory failed. error_code={result.error_code}, "
                f"error_string='{result.error_string}'"
            )
    
    def build_move_to_initial_pose(
            self,
            current_positions: List[float],
            target_positions: List[float],
        ) -> JointTrajectory:
            trajectory = JointTrajectory()
            trajectory.joint_names = list(JOINT_NAMES)

            if len(target_positions) != len(JOINT_NAMES):
                raise ValueError("INITIAL_UPWARD_POSE must contain exactly 6 joint values.")

            safe_target = [
                clamp_to_joint_limits(q)
                for q in target_positions
            ]

            points = []

            zero_vel = [0.0] * len(JOINT_NAMES)

            t_global = START_DELAY_S

            # Hold current pose first.
            points.append(
                make_point(
                    positions=current_positions,
                    velocities=zero_vel,
                    t=t_global,
                )
            )

            t_global += INITIAL_MOVE_START_HOLD_S
            points.append(
                make_point(
                    positions=current_positions,
                    velocities=zero_vel,
                    t=t_global,
                )
            )

            # Build one trapezoid profile for each joint.
            joint_profiles = []
            max_profile_time = 0.0

            for q_start, q_goal in zip(current_positions, safe_target):
                profile = trapezoid_1d(
                    q_start=q_start,
                    q_goal=q_goal,
                    max_speed=INITIAL_MOVE_MAX_SPEED_RAD_S,
                    accel=INITIAL_MOVE_ACCEL_RAD_S2,
                    dt=DT,
                )

                joint_profiles.append(profile)
                max_profile_time = max(max_profile_time, profile[-1][0])

            n_steps = max(1, int(math.ceil(max_profile_time / DT)))

            for step in range(1, n_steps + 1):
                rel_t = min(step * DT, max_profile_time)

                positions = []
                velocities = []

                for joint_index, profile in enumerate(joint_profiles):
                    q_start = current_positions[joint_index]
                    q_goal = safe_target[joint_index]

                    # If this joint's own profile is finished, hold target.
                    if rel_t >= profile[-1][0]:
                        positions.append(q_goal)
                        velocities.append(0.0)
                        continue

                    # Otherwise find nearest sample in this joint profile.
                    sample_index = min(
                        int(rel_t / DT),
                        len(profile) - 1,
                    )

                    _, q, qd = profile[sample_index]
                    positions.append(q)
                    velocities.append(qd)

                points.append(
                    make_point(
                        positions=positions,
                        velocities=velocities,
                        t=t_global + rel_t,
                    )
                )

            t_global += max_profile_time

            # Force exact final pose.
            t_global += INITIAL_MOVE_END_HOLD_S
            points.append(
                make_point(
                    positions=safe_target,
                    velocities=zero_vel,
                    t=t_global,
                )
            )

            trajectory.points = points
            return trajectory

    def move_to_initial_upward_pose(self):
        print("\nMoving to initial upward pose.")

        joint_map = self.wait_for_joint_state()
        current_positions = [joint_map[name] for name in JOINT_NAMES]

        print("Current position before initial move:")
        self.print_positions(current_positions)

        print("Target initial upward pose:")
        self.print_positions(INITIAL_UPWARD_POSE)

        trajectory = self.build_move_to_initial_pose(
            current_positions=current_positions,
            target_positions=INITIAL_UPWARD_POSE,
        )

        total_time = (
            trajectory.points[-1].time_from_start.sec
            + trajectory.points[-1].time_from_start.nanosec * 1e-9
        )

        print(
            f"Sending initial upward-pose trajectory with "
            f"{len(trajectory.points)} points over {total_time:.2f} s"
        )

        self.send_trajectory(trajectory)

        print("Initial upward pose reached.")

        self.wait_until_robot_settled(SETTLE_BEFORE_EACH_JOINT_S)

    def build_single_joint_test(
        self,
        base_positions: List[float],
        joint_index: int,
    ) -> JointTrajectory:
        trajectory = JointTrajectory()
        trajectory.joint_names = list(JOINT_NAMES)

        points = []

        q_center = list(base_positions)
        current = list(q_center)

        zero_vel = [0.0] * len(JOINT_NAMES)

        t_global = START_DELAY_S

        # First point is delayed, not immediate.
        points.append(make_point(q_center, zero_vel, t_global))

        t_global += START_HOLD_S
        points.append(make_point(q_center, zero_vel, t_global))

        center_angle = q_center[joint_index]

        plus_target = clamp_to_joint_limits(center_angle + TEST_ANGLE_RAD)
        minus_target = clamp_to_joint_limits(center_angle - TEST_ANGLE_RAD)
        final_target = center_angle

        targets = []

        for _ in range(REPETITIONS_PER_JOINT):
            targets.extend([
                plus_target,
                minus_target,
                final_target,
            ])

        for target_angle in targets:
            q_start = current[joint_index]
            q_goal = target_angle

            profile = trapezoid_1d(
                q_start=q_start,
                q_goal=q_goal,
                max_speed=MAX_SPEED_RAD_S,
                accel=ACCEL_RAD_S2,
                dt=DT,
            )

            profile_start_time = t_global

            for rel_t, q, qd in profile:
                positions = list(current)
                velocities = [0.0] * len(JOINT_NAMES)

                positions[joint_index] = q
                velocities[joint_index] = qd

                points.append(
                    make_point(
                        positions=positions,
                        velocities=velocities,
                        t=profile_start_time + rel_t,
                    )
                )

            t_global = profile_start_time + profile[-1][0]

            current[joint_index] = q_goal

            t_global += HOLD_AT_EXTREME_S
            points.append(
                make_point(
                    positions=current,
                    velocities=[0.0] * len(JOINT_NAMES),
                    t=t_global,
                )
            )

        t_global += HOLD_BETWEEN_JOINTS_S
        points.append(
            make_point(
                positions=current,
                velocities=[0.0] * len(JOINT_NAMES),
                t=t_global,
            )
        )

        trajectory.points = points
        return trajectory

    def print_positions(self, positions: List[float]):
        for name, pos in zip(JOINT_NAMES, positions):
            print(f"  {name}: {pos:.6f} rad")

    def run_test(self):
        joint_map = self.wait_for_joint_state()
        base_positions = [joint_map[name] for name in JOINT_NAMES]

        print("Starting real UR5e joint diagnostic.")
        print("Initial joint positions:")
        self.print_positions(base_positions)

        if MOVE_TO_INITIAL_UPWARD_POSE:
            self.move_to_initial_upward_pose()

        joint_map = self.wait_for_joint_state()
        base_positions = [joint_map[name] for name in JOINT_NAMES]

        print("\nJoint positions after initial upward move:")
        self.print_positions(base_positions)

        for joint_index, joint_name in enumerate(JOINT_NAMES):
            print(f"\nTesting joint {joint_index}: {joint_name}")

            self.wait_until_robot_settled(SETTLE_BEFORE_EACH_JOINT_S)

            joint_map = self.wait_for_joint_state()
            current_positions = [joint_map[name] for name in JOINT_NAMES]

            print("Current robot position before this joint test:")
            self.print_positions(current_positions)

            q = current_positions[joint_index]
            q_plus = clamp_to_joint_limits(q + TEST_ANGLE_RAD)
            q_minus = clamp_to_joint_limits(q - TEST_ANGLE_RAD)

            print(f"Commanded range for {joint_name}:")
            print(f"  center: {q:.6f}")
            print(f"  plus:   {q_plus:.6f}")
            print(f"  minus:  {q_minus:.6f}")

            if abs(q_plus - q) < 1e-4 and abs(q_minus - q) < 1e-4:
                print(f"Skipping {joint_name}: too close to configured joint limits.")
                continue

            trajectory = self.build_single_joint_test(
                base_positions=current_positions,
                joint_index=joint_index,
            )

            print(
                f"Sending {len(trajectory.points)} points over "
                f"{trajectory.points[-1].time_from_start.sec + trajectory.points[-1].time_from_start.nanosec * 1e-9:.2f} s"
            )

            self.send_trajectory(trajectory)

            print(f"Finished {joint_name}")

        print("\nJoint diagnostic finished successfully.")


def main():
    rclpy.init()

    node = RealJointDiagnostic()

    try:
        node.run_test()
    except KeyboardInterrupt:
        print("\nInterrupted.")
    except Exception as exc:
        print(f"\nERROR: {exc}")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()