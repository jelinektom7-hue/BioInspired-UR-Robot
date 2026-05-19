#!/usr/bin/env python3
"""
Smooth a trajectory CSV and slow down the last part so the robot does not stop abruptly.

Default use:
    cd /home/dragos/ros2_ws/src/BioInspired-UR-Robot/ur5e_whip-main/reinforcement_learning
    python3 smooth_decelerate_trajectory.py

By default this script reads the CSV_PATH currently used by send_csv_trajectory_safe.py,
creates a new CSV with "_smooth_decel" added to the filename, and updates:
    - move_to_whip_start_pose.py
    - send_csv_trajectory_safe.py
    - send_csv_trajectory_preserved_speed.py

Recommended first try:
    python3 smooth_decelerate_trajectory.py --decel-fraction 0.25 --final-dt-multiplier 5.0 --smooth-passes 2

More aggressive deceleration:
    python3 smooth_decelerate_trajectory.py --decel-fraction 0.35 --final-dt-multiplier 7.0 --smooth-passes 2
"""

import argparse
import csv
import re
from pathlib import Path
import numpy as np


PROJECT_ROOT_DEFAULT = Path("/home/dragos/ros2_ws/src/BioInspired-UR-Robot/ur5e_whip-main")

JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

SENDER_SCRIPTS = [
    "reinforcement_learning/move_to_whip_start_pose.py",
    "reinforcement_learning/send_csv_trajectory_safe.py",
    "reinforcement_learning/send_csv_trajectory_preserved_speed.py",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Smooth a UR5e CSV trajectory and decelerate the final part."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Input CSV. If omitted, reads CSV_PATH from send_csv_trajectory_safe.py.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV. If omitted, appends _smooth_decel to input filename.",
    )
    parser.add_argument(
        "--project-root",
        type=str,
        default=str(PROJECT_ROOT_DEFAULT),
    )
    parser.add_argument(
        "--decel-fraction",
        type=float,
        default=0.25,
        help="Fraction of the motion part to decelerate at the end. Default: 0.25.",
    )
    parser.add_argument(
        "--final-dt-multiplier",
        type=float,
        default=5.0,
        help=(
            "How much larger the final timestep becomes compared to normal. "
            "Higher = stronger slowdown near end. Default: 5.0."
        ),
    )
    parser.add_argument(
        "--smooth-passes",
        type=int,
        default=2,
        help="Number of smoothing passes on joint positions. Default: 2.",
    )
    parser.add_argument(
        "--hold-seconds",
        type=float,
        default=1.0,
        help="Final hold duration at the final pose. Default: 1.0 s.",
    )
    parser.add_argument(
        "--no-update-scripts",
        action="store_true",
        help="Create CSV only; do not update robot sender scripts.",
    )
    return parser.parse_args()


def read_csv_path_from_sender(project_root: Path) -> Path:
    sender = project_root / "reinforcement_learning/send_csv_trajectory_safe.py"
    text = sender.read_text()

    match = re.search(r'^\s*CSV_PATH\s*=\s*["\']([^"\']+\.csv)["\']', text, flags=re.MULTILINE)
    if not match:
        raise RuntimeError(f"Could not find CSV_PATH in {sender}")

    p = Path(match.group(1)).expanduser()
    if not p.is_absolute():
        p = project_root / "reinforcement_learning" / p

    return p.resolve()


def load_rows(path: Path):
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames

    if not rows:
        raise RuntimeError(f"Input CSV is empty: {path}")

    for name in ["elapsed_time"] + JOINT_NAMES:
        if name not in fieldnames:
            raise RuntimeError(f"Missing required column {name} in {path}")

    return rows, fieldnames


def rows_to_arrays(rows):
    t = np.array([float(r["elapsed_time"]) for r in rows], dtype=float)
    q = np.array([[float(r[j]) for j in JOINT_NAMES] for r in rows], dtype=float)
    return t, q


def strip_existing_hold(t, q, rows):
    """
    If the final row is a pure hold row, remove it before processing.
    Detection:
      - last and second-last joint commands are nearly identical
      - final dt is much larger than the median motion dt
    """
    if len(rows) < 4:
        return t, q, rows, False

    dt = np.diff(t)
    positive_dt = dt[dt > 1e-9]
    if len(positive_dt) == 0:
        return t, q, rows, False

    median_dt = float(np.median(positive_dt))
    last_dt = float(t[-1] - t[-2])
    same_pose = float(np.linalg.norm(q[-1] - q[-2])) < 1e-8

    if same_pose and last_dt > max(0.2, 5.0 * median_dt):
        return t[:-1], q[:-1], rows[:-1], True

    return t, q, rows, False


def smooth_positions(q, passes):
    if passes <= 0 or len(q) < 5:
        return q.copy()

    out = q.copy()
    kernel = np.array([1.0, 2.0, 3.0, 2.0, 1.0], dtype=float)
    kernel /= kernel.sum()

    for _ in range(passes):
        padded = np.pad(out, ((2, 2), (0, 0)), mode="edge")
        smoothed = out.copy()

        for i in range(len(out)):
            smoothed[i] = (
                kernel[0] * padded[i + 0]
                + kernel[1] * padded[i + 1]
                + kernel[2] * padded[i + 2]
                + kernel[3] * padded[i + 3]
                + kernel[4] * padded[i + 4]
            )

        # Preserve exact start and final pose.
        smoothed[0] = q[0]
        smoothed[-1] = q[-1]
        out = smoothed

    return out


def retime_with_deceleration(t, decel_fraction, final_dt_multiplier):
    if not (0.0 < decel_fraction <= 1.0):
        raise ValueError("--decel-fraction must be > 0 and <= 1")

    if final_dt_multiplier < 1.0:
        raise ValueError("--final-dt-multiplier must be >= 1")

    if len(t) < 3:
        return t.copy()

    dt = np.diff(t)
    dt = np.maximum(dt, 1e-6)

    nseg = len(dt)
    decel_start = max(0, int(np.floor((1.0 - decel_fraction) * nseg)))

    new_dt = dt.copy()

    for i in range(decel_start, nseg):
        if nseg - decel_start <= 1:
            s = 1.0
        else:
            s = (i - decel_start) / float((nseg - decel_start) - 1)

        # Smoothstep easing from 0 to 1.
        smooth_s = s * s * (3.0 - 2.0 * s)

        multiplier = 1.0 + (final_dt_multiplier - 1.0) * smooth_s
        new_dt[i] = dt[i] * multiplier

    new_t = np.zeros_like(t)
    new_t[1:] = np.cumsum(new_dt)

    return new_t


def write_output(input_rows, fieldnames, q_new, t_new, output_path: Path, hold_seconds: float):
    out_rows = []

    for i, row in enumerate(input_rows):
        r = row.copy()
        r["elapsed_time"] = f"{float(t_new[i]):.6f}"
        for j, name in enumerate(JOINT_NAMES):
            r[name] = f"{float(q_new[i, j]):.12f}"
        out_rows.append(r)

    # Add final hold row at same final pose.
    hold_row = out_rows[-1].copy()
    hold_row["elapsed_time"] = f"{float(t_new[-1] + hold_seconds):.6f}"
    out_rows.append(hold_row)

    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    return out_rows


def compute_stats(rows):
    t, q = rows_to_arrays(rows)
    if len(t) < 2:
        return {}

    dt = np.diff(t)
    dq = np.diff(q, axis=0)

    valid = dt > 1e-9
    if not np.any(valid):
        return {}

    vel = dq[valid] / dt[valid, None]
    max_abs_vel = np.max(np.abs(vel), axis=0)

    # Acceleration estimate from velocity differences.
    acc_stats = None
    if len(vel) >= 2:
        dt_mid = dt[valid][1:]
        dv = np.diff(vel, axis=0)
        valid_acc = dt_mid > 1e-9
        if np.any(valid_acc):
            acc = dv[valid_acc] / dt_mid[valid_acc, None]
            acc_stats = np.max(np.abs(acc), axis=0)

    return {
        "points": len(rows),
        "duration": float(t[-1] - t[0]),
        "max_abs_vel": max_abs_vel,
        "max_abs_acc": acc_stats,
    }


def update_sender_script(script_path: Path, wanted_csv: Path):
    original = script_path.read_text()
    text = original

    pattern = re.compile(
        r'(^\s*CSV_PATH\s*=\s*)(["\'])(.*?\.csv)(["\'])',
        flags=re.MULTILINE,
    )

    if pattern.search(text):
        text = pattern.sub(lambda m: f'{m.group(1)}"{wanted_csv}"', text, count=1)
    else:
        text = re.sub(
            r'(["\'])(?:[^"\']*/)?exported_trajectory[^"\']*\.csv(["\'])',
            f'"{wanted_csv}"',
            text,
        )

    if text == original:
        print(f"WARNING: no CSV path replaced in {script_path}")
        return

    backup = script_path.with_suffix(script_path.suffix + ".backup_before_smooth_decel")
    if not backup.exists():
        backup.write_text(original)

    script_path.write_text(text)
    print(f"Updated {script_path} -> {wanted_csv}")


def main():
    args = parse_args()

    project_root = Path(args.project_root).expanduser().resolve()

    if args.input is None:
        input_path = read_csv_path_from_sender(project_root)
    else:
        input_path = Path(args.input).expanduser()
        if not input_path.is_absolute():
            input_path = (project_root / "reinforcement_learning" / input_path).resolve()
        else:
            input_path = input_path.resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    if args.output is None:
        output_path = input_path.with_name(input_path.stem + "_smooth_decel.csv")
    else:
        output_path = Path(args.output).expanduser()
        if not output_path.is_absolute():
            output_path = (project_root / "reinforcement_learning" / output_path).resolve()
        else:
            output_path = output_path.resolve()

    rows, fieldnames = load_rows(input_path)
    t, q = rows_to_arrays(rows)

    t, q, rows_without_hold, stripped_hold = strip_existing_hold(t, q, rows)

    q_smooth = smooth_positions(q, args.smooth_passes)
    t_decel = retime_with_deceleration(t, args.decel_fraction, args.final_dt_multiplier)

    output_rows = write_output(rows_without_hold, fieldnames, q_smooth, t_decel, output_path, args.hold_seconds)

    print("=== Smooth + decelerate trajectory ===")
    print(f"Input:                    {input_path}")
    print(f"Output:                   {output_path}")
    print(f"Stripped existing hold:    {stripped_hold}")
    print(f"Smooth passes:            {args.smooth_passes}")
    print(f"Decel fraction:           {args.decel_fraction}")
    print(f"Final dt multiplier:      {args.final_dt_multiplier}")
    print(f"Hold seconds:             {args.hold_seconds}")
    print("")

    stats = compute_stats(output_rows)
    print("Output stats:")
    print(f"  points:   {stats.get('points')}")
    print(f"  duration: {stats.get('duration'):.6f} s")
    print("  max velocity:")
    for name, value in zip(JOINT_NAMES, stats.get("max_abs_vel")):
        print(f"    {name:25s}: {value:.3f} rad/s")

    acc = stats.get("max_abs_acc")
    if acc is not None:
        print("  max accel estimate:")
        for name, value in zip(JOINT_NAMES, acc):
            print(f"    {name:25s}: {value:.3f} rad/s^2")

    if not args.no_update_scripts:
        print("")
        print("Updating robot scripts:")
        for rel in SENDER_SCRIPTS:
            update_sender_script(project_root / rel, output_path)

    print("")
    print("Done.")
    print(f"Active CSV should now be: {output_path}")


if __name__ == "__main__":
    main()
