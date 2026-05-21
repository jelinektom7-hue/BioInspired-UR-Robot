#!/usr/bin/env python3

import csv
import math
import os


# ============================================================
# USER CONFIGURATION
# ============================================================

INPUT_CSV = "/home/dragos/ros2_ws/src/BioInspired-UR-Robot/ur5e_whip-main/reinforcement_learning/exported_trajectory_best_fixed_target.csv"

OUTPUT_CSV = "/home/dragos/ros2_ws/src/BioInspired-UR-Robot/ur5e_whip-main/reinforcement_learning/exported_trajectory_best_fixed_target_flipped_180.csv"

# Try +math.pi first.
# If the robot points the wrong opposite direction or violates limits,
# change this to -math.pi.
SHOULDER_PAN_OFFSET_RAD = math.pi

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


# ============================================================
# HELPERS
# ============================================================

def try_float(x):
    try:
        return float(x)
    except Exception:
        return None


def normalize_near_original(q_new, q_old):
    """
    UR joints can often represent similar physical directions with angles
    shifted by 2*pi. This keeps the modified value close to the original
    numerical range where possible.
    """
    while q_new - q_old > math.pi:
        q_new -= 2.0 * math.pi

    while q_new - q_old < -math.pi:
        q_new += 2.0 * math.pi

    return q_new


def within_limits(q):
    low = JOINT_LIMIT_LOW + JOINT_LIMIT_MARGIN
    high = JOINT_LIMIT_HIGH - JOINT_LIMIT_MARGIN
    return low <= q <= high


def find_shoulder_pan_column(header, num_cols):
    if header:
        lower_header = [h.strip().lower() for h in header]

        if "shoulder_pan_joint" in lower_header:
            return lower_header.index("shoulder_pan_joint")

        candidates = [
            "q0",
            "joint_0",
            "j0",
            "position_0",
            "shoulder_pan",
        ]

        for c in candidates:
            if c in lower_header:
                return lower_header.index(c)

    # Fallback:
    # If first column looks like time, shoulder_pan is probably column 1.
    # Otherwise shoulder_pan is probably column 0.
    return None


def main():
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(INPUT_CSV)

    with open(INPUT_CSV, "r", newline="") as f:
        sample = f.read(4096)
        f.seek(0)

        try:
            has_header = csv.Sniffer().has_header(sample)
        except Exception:
            has_header = True

        reader = csv.reader(f)
        rows = list(reader)

    if not rows:
        raise RuntimeError("Input CSV is empty.")

    if has_header:
        header = rows[0]
        data_rows = rows[1:]
    else:
        header = []
        data_rows = rows

    if not data_rows:
        raise RuntimeError("Input CSV has no data rows.")

    num_cols = len(data_rows[0])

    shoulder_col = find_shoulder_pan_column(header, num_cols)

    if shoulder_col is None:
        # Fallback detection:
        # If first numeric column is monotonically increasing, treat it as time,
        # so shoulder_pan is column 1. Otherwise shoulder_pan is column 0.
        numeric_rows = []

        for row in data_rows:
            values = [try_float(x.strip()) for x in row]
            if any(v is None for v in values):
                continue
            numeric_rows.append(values)

        if len(numeric_rows) < 2:
            raise RuntimeError("Could not parse numeric rows.")

        first_col = [r[0] for r in numeric_rows]
        first_col_is_time = all(
            first_col[i + 1] > first_col[i]
            for i in range(len(first_col) - 1)
        )

        if first_col_is_time:
            shoulder_col = 1
        else:
            shoulder_col = 0

    print(f"Input CSV:  {INPUT_CSV}")
    print(f"Output CSV: {OUTPUT_CSV}")
    print(f"Using shoulder_pan column index: {shoulder_col}")
    if header:
        print(f"Column name: {header[shoulder_col]}")
    print(f"Applying shoulder_pan offset: {SHOULDER_PAN_OFFSET_RAD:.6f} rad")

    output_rows = []

    if has_header:
        output_rows.append(header)

    min_q = float("inf")
    max_q = -float("inf")
    violation_count = 0
    changed_count = 0

    first_old = None
    first_new = None
    last_old = None
    last_new = None

    for row_index, row in enumerate(data_rows):
        new_row = list(row)

        q_old = try_float(row[shoulder_col])

        if q_old is None:
            output_rows.append(new_row)
            continue

        q_new = q_old + SHOULDER_PAN_OFFSET_RAD

        # Keep the angle numerically reasonable.
        q_new = normalize_near_original(q_new, q_old)

        if not within_limits(q_new):
            violation_count += 1

        min_q = min(min_q, q_new)
        max_q = max(max_q, q_new)

        if first_old is None:
            first_old = q_old
            first_new = q_new

        last_old = q_old
        last_new = q_new

        new_row[shoulder_col] = f"{q_new:.10f}"
        output_rows.append(new_row)
        changed_count += 1

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(output_rows)

    print("\nDone.")
    print(f"Rows changed: {changed_count}")
    print(f"First shoulder_pan: {first_old:.6f} -> {first_new:.6f}")
    print(f"Last shoulder_pan:  {last_old:.6f} -> {last_new:.6f}")
    print(f"New shoulder_pan min/max: {min_q:.6f}, {max_q:.6f}")

    if violation_count > 0:
        print(f"\nWARNING: {violation_count} rows violate configured joint limits.")
        print("Try changing SHOULDER_PAN_OFFSET_RAD from +math.pi to -math.pi.")
    else:
        print("\nNo shoulder_pan joint limit violations found.")

    print("\nNext:")
    print("1. Set your safe sender CSV_PATH to the flipped CSV.")
    print("2. Run check_csv_trajectory.py on the flipped CSV.")
    print("3. Move to the flipped CSV start pose.")
    print("4. Run send_csv_trajectory_safe.py slowly, without whip first.")


if __name__ == "__main__":
    main()
