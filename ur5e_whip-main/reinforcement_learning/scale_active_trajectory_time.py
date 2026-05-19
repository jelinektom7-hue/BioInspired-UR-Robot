#!/usr/bin/env python3
"""
Scale/compress the timing of the currently active preserved-speed CSV.

Use this after you have a dense acceleration-limited trajectory that works but is too slow.
It keeps the same joint path and same intermediate points, but compresses the motion time.

Example:
    cd /home/dragos/ros2_ws/src/BioInspired-UR-Robot/ur5e_whip-main/reinforcement_learning

    # 25% faster
    python3 scale_active_trajectory_time.py --time-scale 0.80

    # 43% faster
    python3 scale_active_trajectory_time.py --time-scale 0.70

    # 67% faster
    python3 scale_active_trajectory_time.py --time-scale 0.60

Then run:
    python3 move_to_whip_start_pose.py
    python3 send_csv_trajectory_preserved_speed.py

Meaning:
    time-scale < 1.0  -> faster
    time-scale = 1.0  -> same speed
    time-scale > 1.0  -> slower

Important:
    Velocity scales by 1 / time_scale.
    Acceleration scales by 1 / time_scale^2.
So do not jump too aggressively.
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

SCRIPTS_TO_UPDATE = [
    "reinforcement_learning/move_to_whip_start_pose.py",
    "reinforcement_learning/send_csv_trajectory_preserved_speed.py",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--time-scale",
        type=float,
        required=True,
        help="Scale factor for moving-section time. <1 faster, >1 slower. Example: 0.8",
    )
    p.add_argument(
        "--input",
        default=None,
        help="Input CSV. If omitted, reads CSV_PATH from send_csv_trajectory_preserved_speed.py.",
    )
    p.add_argument(
        "--project-root",
        default=str(PROJECT_ROOT_DEFAULT),
    )
    p.add_argument(
        "--hold-seconds",
        type=float,
        default=None,
        help="Override final hold seconds. If omitted, preserves detected final hold if present.",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Output CSV. If omitted, appends _timescaleX to the input name.",
    )
    p.add_argument(
        "--no-update-scripts",
        action="store_true",
    )
    return p.parse_args()


def get_active_csv(project_root: Path) -> Path:
    sender = project_root / "reinforcement_learning/send_csv_trajectory_preserved_speed.py"
    text = sender.read_text()
    m = re.search(r'^\s*CSV_PATH\s*=\s*["\']([^"\']+\.csv)["\']', text, flags=re.MULTILINE)
    if not m:
        raise RuntimeError(f"Could not find CSV_PATH in {sender}")
    p = Path(m.group(1)).expanduser()
    if not p.is_absolute():
        p = project_root / "reinforcement_learning" / p
    return p.resolve()


def load_csv(path: Path):
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames

    if not rows:
        raise RuntimeError(f"CSV is empty: {path}")

    for col in ["elapsed_time"] + JOINT_NAMES:
        if col not in fieldnames:
            raise RuntimeError(f"Missing required column: {col}")

    return rows, fieldnames


def write_csv(path: Path, rows, fieldnames):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def q_from_row(row):
    return np.array([float(row[j]) for j in JOINT_NAMES], dtype=float)


def detect_final_hold(rows):
    """
    Returns (moving_rows, hold_seconds).
    If final row is a hold row, it is removed from moving_rows and hold_seconds is detected.
    """
    if len(rows) < 3:
        return rows, 0.0

    q_last = q_from_row(rows[-1])
    q_prev = q_from_row(rows[-2])
    same_pose = np.linalg.norm(q_last - q_prev) < 1e-9

    t_last = float(rows[-1]["elapsed_time"])
    t_prev = float(rows[-2]["elapsed_time"])
    final_dt = t_last - t_prev

    # Large same-pose final dt = hold.
    if same_pose and final_dt > 0.1:
        return rows[:-1], final_dt

    return rows, 0.0


def scale_times(rows, time_scale, hold_seconds):
    if time_scale <= 0.0:
        raise ValueError("--time-scale must be positive")

    moving_rows, detected_hold = detect_final_hold(rows)

    if hold_seconds is None:
        hold_seconds_used = detected_hold
    else:
        hold_seconds_used = hold_seconds

    t0 = float(moving_rows[0]["elapsed_time"])

    out = []
    for r in moving_rows:
        nr = r.copy()
        t = float(r["elapsed_time"])
        nr["elapsed_time"] = f"{(t - t0) * time_scale:.6f}"
        out.append(nr)

    if hold_seconds_used and hold_seconds_used > 0.0:
        hold = out[-1].copy()
        hold["elapsed_time"] = f"{float(out[-1]['elapsed_time']) + hold_seconds_used:.6f}"
        out.append(hold)

    return out, hold_seconds_used


def compute_stats(rows):
    t = np.array([float(r["elapsed_time"]) for r in rows], dtype=float)
    q = np.array([[float(r[j]) for j in JOINT_NAMES] for r in rows], dtype=float)

    dt = np.diff(t)
    dq = np.diff(q, axis=0)
    valid = dt > 1e-9

    vel = dq[valid] / dt[valid, None]
    max_v = np.max(np.abs(vel), axis=0) if len(vel) else np.zeros(6)

    max_a = np.zeros(6)
    if len(vel) >= 2:
        dtv = dt[valid]
        dt_mid = 0.5 * (dtv[:-1] + dtv[1:])
        acc = np.diff(vel, axis=0) / dt_mid[:, None]
        max_a = np.max(np.abs(acc), axis=0) if len(acc) else np.zeros(6)

    moving_duration = float(t[-1] - t[0])
    if len(rows) >= 2:
        q_last = q[-1]
        q_prev = q[-2]
        if np.linalg.norm(q_last - q_prev) < 1e-9:
            moving_duration = float(t[-2] - t[0])

    return {
        "points": len(rows),
        "duration": float(t[-1] - t[0]),
        "moving_duration": moving_duration,
        "max_v": max_v,
        "max_a": max_a,
    }


def update_script(script_path: Path, wanted_csv: Path):
    if not script_path.exists():
        print(f"WARNING: missing script: {script_path}")
        return

    old = script_path.read_text()
    pattern = re.compile(
        r'(^\s*CSV_PATH\s*=\s*)(["\'])(.*?\.csv)(["\'])',
        flags=re.MULTILINE,
    )

    if pattern.search(old):
        new = pattern.sub(lambda m: f'{m.group(1)}"{wanted_csv}"', old, count=1)
    else:
        new = re.sub(
            r'(["\'])(?:[^"\']*/)?exported_trajectory[^"\']*\.csv(["\'])',
            f'"{wanted_csv}"',
            old,
        )

    if new == old:
        print(f"WARNING: no CSV path replaced in {script_path}")
        return

    backup = script_path.with_suffix(script_path.suffix + ".backup_before_timescale")
    if not backup.exists():
        backup.write_text(old)

    script_path.write_text(new)
    print(f"Updated {script_path} -> {wanted_csv}")


def label_for_scale(x: float) -> str:
    return str(x).replace(".", "p").replace("-", "m")


def main():
    args = parse_args()
    project_root = Path(args.project_root).expanduser().resolve()

    if args.input is None:
        input_path = get_active_csv(project_root)
    else:
        input_path = Path(args.input).expanduser()
        if not input_path.is_absolute():
            input_path = (project_root / "reinforcement_learning" / input_path).resolve()
        else:
            input_path = input_path.resolve()

    if not input_path.exists():
        raise FileNotFoundError(input_path)

    rows, fieldnames = load_csv(input_path)

    output_rows, hold_used = scale_times(rows, args.time_scale, args.hold_seconds)

    if args.output is None:
        output_path = input_path.with_name(input_path.stem + f"_timescale{label_for_scale(args.time_scale)}.csv")
    else:
        output_path = Path(args.output).expanduser()
        if not output_path.is_absolute():
            output_path = (project_root / "reinforcement_learning" / output_path).resolve()
        else:
            output_path = output_path.resolve()

    write_csv(output_path, output_rows, fieldnames)

    s = compute_stats(output_rows)

    print("=== Time-scaled active trajectory ===")
    print(f"Input:            {input_path}")
    print(f"Output:           {output_path}")
    print(f"Time scale:       {args.time_scale}")
    print(f"Final hold used:  {hold_used:.6f} s")
    print(f"Points:           {s['points']}")
    print(f"Moving duration:  {s['moving_duration']:.6f} s")
    print(f"Total duration:   {s['duration']:.6f} s")
    print("")
    print("Max joint velocity:")
    for name, v in zip(JOINT_NAMES, s["max_v"]):
        print(f"  {name:25s}: {v:.3f} rad/s")
    print("Max joint acceleration estimate:")
    for name, a in zip(JOINT_NAMES, s["max_a"]):
        print(f"  {name:25s}: {a:.3f} rad/s^2")

    if not args.no_update_scripts:
        print("")
        print("Updating scripts:")
        for rel in SCRIPTS_TO_UPDATE:
            update_script(project_root / rel, output_path)

    print("")
    print("Run:")
    print("  python3 move_to_whip_start_pose.py")
    print("  python3 send_csv_trajectory_preserved_speed.py")


if __name__ == "__main__":
    main()
