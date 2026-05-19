#!/usr/bin/env python3

import csv
import math
import os
from typing import List, Tuple


# ============================================================
# CONFIGURATION
# ============================================================

CSV_PATH = "/home/dragos/ros2_ws/src/BioInspired-UR-Robot/ur5e_whip-main/reinforcement_learning/exported_trajectory_best_fixed_target_slow_flipped_180.csv"

JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

JOINT_LIMIT_LOW = -2.0 * math.pi
JOINT_LIMIT_HIGH = 2.0 * math.pi
JOINT_LIMIT_MARGIN = 0.05

# Conservative dry-run limit.
# If the CSV requires more than this, do not send it to the real robot yet.
SAFE_MAX_VELOCITY_RAD_S = 0.25

# If the CSV has tiny dt, this will reveal it.
MIN_REASONABLE_DT = 0.02


# ============================================================
# CSV READING
# ============================================================

def try_float(x):
    try:
        return float(x)
    except Exception:
        return None


def read_csv(path: str) -> Tuple[List[float], List[List[float]], List[str]]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    with open(path, "r", newline="") as f:
        sample = f.read(4096)
        f.seek(0)

        has_header = csv.Sniffer().has_header(sample)
        reader = csv.reader(f)

        rows = list(reader)

    if not rows:
        raise RuntimeError("CSV is empty.")

    if has_header:
        header = rows[0]
        data_rows = rows[1:]
    else:
        header = []
        data_rows = rows

    numeric_rows = []

    for row in data_rows:
        if not row:
            continue

        values = [try_float(x.strip()) for x in row]

        if any(v is None for v in values):
            continue

        numeric_rows.append(values)

    if len(numeric_rows) < 2:
        raise RuntimeError("CSV does not contain enough numeric rows.")

    num_cols = len(numeric_rows[0])

    for r in numeric_rows:
        if len(r) != num_cols:
            raise RuntimeError("CSV has inconsistent number of columns.")

    # Try to identify time column.
    time_col = None

    if header:
        lower_header = [h.strip().lower() for h in header]

        for candidate in ["time", "t", "timestamp", "time_s", "seconds"]:
            if candidate in lower_header:
                time_col = lower_header.index(candidate)
                break

    if time_col is None:
        # If first column is monotonically increasing, treat it as time.
        first_col = [r[0] for r in numeric_rows]
        if all(first_col[i + 1] > first_col[i] for i in range(len(first_col) - 1)):
            time_col = 0

    # Try to identify joint columns.
    joint_cols = []

    if header:
        lower_header = [h.strip().lower() for h in header]

        # First try exact UR joint names.
        for name in JOINT_NAMES:
            if name.lower() in lower_header:
                joint_cols.append(lower_header.index(name.lower()))

        # Then try common short names.
        if len(joint_cols) != 6:
            joint_cols = []

            candidates = [
                ["q0", "q1", "q2", "q3", "q4", "q5"],
                ["joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5"],
                ["j0", "j1", "j2", "j3", "j4", "j5"],
                ["position_0", "position_1", "position_2", "position_3", "position_4", "position_5"],
            ]

            for candidate_set in candidates:
                if all(c in lower_header for c in candidate_set):
                    joint_cols = [lower_header.index(c) for c in candidate_set]
                    break

    if len(joint_cols) != 6:
        # Fallback:
        # If there is a time column, use the first 6 numeric columns after excluding time.
        # Otherwise use the first 6 columns.
        all_cols = list(range(num_cols))

        if time_col is not None:
            all_cols.remove(time_col)

        if len(all_cols) < 6:
            raise RuntimeError("Could not find 6 joint columns in CSV.")

        joint_cols = all_cols[:6]

    positions = []

    for r in numeric_rows:
        positions.append([r[c] for c in joint_cols])

    if time_col is not None:
        times = [r[time_col] for r in numeric_rows]
    else:
        # No time column. Use index as fake time.
        times = [float(i) for i in range(len(numeric_rows))]

    used_columns = []

    if header:
        used_columns = [header[c] for c in joint_cols]
    else:
        used_columns = [f"column_{c}" for c in joint_cols]

    return times, positions, used_columns


# ============================================================
# ANALYSIS
# ============================================================

def analyze(times: List[float], positions: List[List[float]], used_columns: List[str]):
    print("\n========== CSV TRAJECTORY CHECK ==========")
    print(f"Number of points: {len(positions)}")
    print(f"Used joint columns: {used_columns}")

    print("\nFirst position:")
    for name, q in zip(JOINT_NAMES, positions[0]):
        print(f"  {name}: {q:.6f}")

    print("\nLast position:")
    for name, q in zip(JOINT_NAMES, positions[-1]):
        print(f"  {name}: {q:.6f}")

    # Time checks
    dts = [times[i + 1] - times[i] for i in range(len(times) - 1)]

    print("\nTime check:")
    print(f"  first time: {times[0]:.6f}")
    print(f"  last time:  {times[-1]:.6f}")
    print(f"  min dt:     {min(dts):.6f}")
    print(f"  max dt:     {max(dts):.6f}")
    print(f"  mean dt:    {sum(dts) / len(dts):.6f}")

    non_positive_dt_count = sum(1 for dt in dts if dt <= 0.0)
    tiny_dt_count = sum(1 for dt in dts if 0.0 < dt < MIN_REASONABLE_DT)

    print(f"  non-positive dt count: {non_positive_dt_count}")
    print(f"  tiny dt count < {MIN_REASONABLE_DT}s: {tiny_dt_count}")

    # Joint limit checks
    print("\nJoint limit check:")

    limit_problem_count = 0

    low = JOINT_LIMIT_LOW + JOINT_LIMIT_MARGIN
    high = JOINT_LIMIT_HIGH - JOINT_LIMIT_MARGIN

    for i, q_list in enumerate(positions):
        for j, q in enumerate(q_list):
            if q < low or q > high:
                print(
                    f"  LIMIT VIOLATION at row {i}, {JOINT_NAMES[j]}: "
                    f"{q:.6f} not in [{low:.6f}, {high:.6f}]"
                )
                limit_problem_count += 1

    if limit_problem_count == 0:
        print("  OK: no joint limit violations found.")

    # Velocity checks
    print("\nImplied velocity check:")

    max_vel = 0.0
    max_vel_info = None

    max_step = 0.0
    max_step_info = None

    for i in range(len(positions) - 1):
        dt = dts[i]

        if dt <= 0.0:
            continue

        for j in range(6):
            dq = positions[i + 1][j] - positions[i][j]
            step = abs(dq)
            vel = step / dt

            if step > max_step:
                max_step = step
                max_step_info = (i, j, positions[i][j], positions[i + 1][j], dt)

            if vel > max_vel:
                max_vel = vel
                max_vel_info = (i, j, positions[i][j], positions[i + 1][j], dt, vel)

    if max_step_info:
        i, j, q0, q1, dt = max_step_info
        print(
            f"  max joint step: {max_step:.6f} rad "
            f"at row {i}->{i+1}, {JOINT_NAMES[j]}, "
            f"{q0:.6f}->{q1:.6f}, dt={dt:.6f}s"
        )

    if max_vel_info:
        i, j, q0, q1, dt, vel = max_vel_info
        print(
            f"  max implied velocity: {vel:.6f} rad/s "
            f"at row {i}->{i+1}, {JOINT_NAMES[j]}, "
            f"{q0:.6f}->{q1:.6f}, dt={dt:.6f}s"
        )

    print("\nSafety verdict:")

    unsafe = False

    if non_positive_dt_count > 0:
        print("  FAIL: CSV has duplicate or backwards timestamps.")
        unsafe = True

    if max_vel > SAFE_MAX_VELOCITY_RAD_S:
        print(
            f"  FAIL: max implied velocity {max_vel:.6f} rad/s "
            f"is above safe dry-run limit {SAFE_MAX_VELOCITY_RAD_S:.6f} rad/s."
        )
        unsafe = True

    if limit_problem_count > 0:
        print("  FAIL: CSV has joint limit violations.")
        unsafe = True

    if not unsafe:
        print("  OK: CSV looks safe according to this checker.")

    print("\nImportant:")
    print("  This only checks the CSV internally.")
    print("  The sender must still handle the jump from the robot's current pose to the first CSV pose.")
    print("==========================================\n")


def main():
    times, positions, used_columns = read_csv(CSV_PATH)
    analyze(times, positions, used_columns)


if __name__ == "__main__":
    main()