import csv
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path.home() / "ros2_ws/src/BioInspired-UR-Robot/ur5e_whip-main"

INPUT_FILE = PROJECT_ROOT / "reinforcement_learning/exported_trajectory_best_fixed_target.csv"
OUTPUT_FILE = PROJECT_ROOT / "reinforcement_learning/exported_trajectory_best_fixed_target_slow.csv"

SLOWDOWN_FACTOR = 4.0


JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Could not find input trajectory: {INPUT_FILE}")

    rows = []

    with open(INPUT_FILE, "r") as f:
        reader = csv.DictReader(f)

        for row in reader:
            elapsed_time = float(row["elapsed_time"]) * SLOWDOWN_FACTOR

            joint_values = [
                float(row["shoulder_pan_joint"]),
                float(row["shoulder_lift_joint"]),
                float(row["elbow_joint"]),
                float(row["wrist_1_joint"]),
                float(row["wrist_2_joint"]),
                float(row["wrist_3_joint"]),
            ]

            reward = float(row.get("reward", 0.0))
            terminated = int(float(row.get("terminated", 0)))
            truncated = int(float(row.get("truncated", 0)))

            rows.append([elapsed_time] + joint_values + [reward, terminated, truncated])

    data = np.array(rows, dtype=float)

    # Unwrap joints to avoid artificial jumps around ±pi.
    data[:, 1:7] = np.unwrap(data[:, 1:7], axis=0)

    with open(OUTPUT_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["elapsed_time"] + JOINT_NAMES + ["reward", "terminated", "truncated"])
        writer.writerows(data.tolist())

    print("Saved slowed trajectory:")
    print(OUTPUT_FILE)
    print(f"Slowdown factor: {SLOWDOWN_FACTOR}x")
    print(f"Original final time: {rows[-1][0] / SLOWDOWN_FACTOR:.3f} s")
    print(f"Slowed final time: {data[-1, 0]:.3f} s")


if __name__ == "__main__":
    main()