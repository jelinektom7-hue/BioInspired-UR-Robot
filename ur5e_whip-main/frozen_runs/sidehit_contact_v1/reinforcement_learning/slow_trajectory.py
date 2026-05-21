import csv
import os
import numpy as np


INPUT_FILE = os.path.expanduser(
    "~/ros2_ws/src/BioInspired-UR-Robot/ur5e_whip-main/reinforcement_learning/exported_trajectory.csv"
)

OUTPUT_FILE = os.path.expanduser(
    "~/ros2_ws/src/BioInspired-UR-Robot/ur5e_whip-main/reinforcement_learning/exported_trajectory_slow.csv"
)

SLOWDOWN_FACTOR = 4.0  # 4x slower for first real robot test


def main():
    rows = []

    with open(INPUT_FILE, "r") as f:
        reader = csv.reader(f)
        header = next(reader)

        for row in reader:
            values = [float(x) for x in row]
            values[-1] *= SLOWDOWN_FACTOR
            rows.append(values)

    rows = np.array(rows, dtype=float)

    # Basic unwrap for joint angles to avoid artificial jumps around ±pi
    joint_data = rows[:, :6]
    joint_data = np.unwrap(joint_data, axis=0)
    rows[:, :6] = joint_data

    with open(OUTPUT_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows.tolist())

    print("Saved slowed trajectory:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()