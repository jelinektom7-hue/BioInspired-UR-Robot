#!/usr/bin/env python3
"""
Create a 40%-cut trajectory that keeps the first part fast, then brakes hard,
while capping implied joint velocity by retiming the CSV itself.

This is intended for preserved-speed execution, not the safe sender.

Typical command:
    cd /home/dragos/ros2_ws/src/BioInspired-UR-Robot/ur5e_whip-main/reinforcement_learning

    python3 make_fast_braked_velocity_capped_trajectory.py \
      --input exported_trajectory_SAC17_120k_command_clean.csv \
      --keep-percent 40 \
      --fast-fraction 0.60 \
      --fast-speed-limit 4.2 \
      --final-speed-limit 0.25 \
      --curve-power 3.0 \
      --smooth-passes 1 \
      --hold-seconds 1.0

This keeps first 40% of the source trajectory.
Inside that 40%:
    - first 60% is allowed to be fast up to fast-speed-limit
    - final 40% is exponentially retimed toward final-speed-limit
"""

import argparse
import csv
import math
import re
from pathlib import Path

import numpy as np


PROJECT_ROOT_DEFAULT = Path("/home/dragos/ros2_ws/src/BioInspired-UR-Robot/ur5e_whip-main")
RL_DIR_DEFAULT = PROJECT_ROOT_DEFAULT / "reinforcement_learning"

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
    "reinforcement_learning/send_csv_trajectory_preserved_speed_smooth.py",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="exported_trajectory_SAC17_120k_command_clean.csv")
    p.add_argument("--project-root", default=str(PROJECT_ROOT_DEFAULT))
    p.add_argument("--keep-percent", type=float, default=40.0)
    p.add_argument("--fast-fraction", type=float, default=0.60)
    p.add_argument("--fast-speed-limit", type=float, default=4.2)
    p.add_argument("--final-speed-limit", type=float, default=0.25)
    p.add_argument("--curve-power", type=float, default=3.0)
    p.add_argument("--smooth-passes", type=int, default=1)
    p.add_argument("--min-dt", type=float, default=0.012)
    p.add_argument("--hold-seconds", type=float, default=1.0)
    p.add_argument("--output-prefix", default=None)
    p.add_argument("--no-update-scripts", action="store_true")
    return p.parse_args()


def resolve_input(s, project_root):
    p = Path(s).expanduser()
    if p.is_absolute():
        return p.resolve()
    return (project_root / "reinforcement_learning" / p).resolve()


def read_csv(path):
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames
    if not rows:
        raise RuntimeError("CSV is empty.")
    for name in ["elapsed_time"] + JOINT_NAMES:
        if name not in fieldnames:
            raise RuntimeError(f"Missing column {name}")
    return rows, fieldnames


def write_csv(path, rows, fieldnames):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def rows_to_arrays(rows):
    t = np.array([float(r["elapsed_time"]) for r in rows], dtype=float)
    q = np.array([[float(r[j]) for j in JOINT_NAMES] for r in rows], dtype=float)
    return t, q


def smooth_positions(q, passes):
    if passes <= 0 or len(q) < 5:
        return q.copy()

    out = q.copy()
    kernel = np.array([1.0, 2.0, 3.0, 2.0, 1.0], dtype=float)
    kernel /= kernel.sum()

    for _ in range(passes):
        padded = np.pad(out, ((2, 2), (0, 0)), mode="edge")
        new = out.copy()
        for i in range(len(out)):
            new[i] = (
                kernel[0] * padded[i + 0]
                + kernel[1] * padded[i + 1]
                + kernel[2] * padded[i + 2]
                + kernel[3] * padded[i + 3]
                + kernel[4] * padded[i + 4]
            )
        new[0] = q[0]
        new[-1] = q[-1]
        out = new
    return out


def retime_velocity_capped(t, q, fast_fraction, fast_limit, final_limit, curve_power, min_dt):
    if not (0.0 <= fast_fraction < 1.0):
        raise ValueError("--fast-fraction must be >=0 and <1")
    if fast_limit <= 0.0 or final_limit <= 0.0:
        raise ValueError("speed limits must be positive")
    if final_limit > fast_limit:
        raise ValueError("final speed limit must be <= fast speed limit")
    if curve_power <= 0.0:
        raise ValueError("--curve-power must be positive")

    t = t - t[0]
    base_dt = np.diff(t)
    base_dt = np.maximum(base_dt, min_dt)

    dq = np.diff(q, axis=0)
    max_dq = np.max(np.abs(dq), axis=1)

    nseg = len(base_dt)
    decel_start = int(np.floor(fast_fraction * nseg))
    decel_start = max(0, min(decel_start, nseg - 1))

    new_dt = np.zeros_like(base_dt)

    for i in range(nseg):
        if i < decel_start:
            speed_limit = fast_limit
        else:
            if nseg - decel_start <= 1:
                s = 1.0
            else:
                s = (i - decel_start) / float((nseg - decel_start) - 1)

            # Exponential-like speed falloff:
            # s=0 -> fast_limit
            # s=1 -> final_limit
            shaped = s ** curve_power
            speed_limit = fast_limit * math.exp(math.log(final_limit / fast_limit) * shaped)

        dt_by_speed = max_dq[i] / max(speed_limit, 1e-9)
        new_dt[i] = max(base_dt[i], dt_by_speed, min_dt)

    new_t = np.zeros_like(t)
    new_t[1:] = np.cumsum(new_dt)
    return new_t


def flip_rows(rows, fieldnames):
    joint_name = "shoulder_pan_joint"
    if joint_name not in fieldnames:
        raise RuntimeError("Missing shoulder_pan_joint")
    out = [r.copy() for r in rows]
    for r in out:
        r[joint_name] = f"{float(r[joint_name]) + math.pi:.12f}"
    return out


def compute_vel_stats(rows):
    t, q = rows_to_arrays(rows)
    dt = np.diff(t)
    dq = np.diff(q, axis=0)
    valid = dt > 1e-9
    vel = dq[valid] / dt[valid, None]
    max_v = np.max(np.abs(vel), axis=0) if len(vel) else np.zeros(6)
    return t, q, max_v


def update_script(script_path, wanted):
    if not script_path.exists():
        print("WARNING: missing", script_path)
        return
    old = script_path.read_text()
    pattern = re.compile(r'(^\s*CSV_PATH\s*=\s*)(["\'])(.*?\.csv)(["\'])', flags=re.MULTILINE)
    if pattern.search(old):
        new = pattern.sub(lambda m: f'{m.group(1)}"{wanted}"', old, count=1)
    else:
        new = re.sub(r'(["\'])(?:[^"\']*/)?exported_trajectory[^"\']*\.csv(["\'])', f'"{wanted}"', old)

    if new == old:
        print("WARNING: no CSV path replaced in", script_path)
        return

    backup = script_path.with_suffix(script_path.suffix + ".backup_before_velcap")
    if not backup.exists():
        backup.write_text(old)
    script_path.write_text(new)
    print("Updated", script_path, "->", wanted)


def main():
    args = parse_args()
    project_root = Path(args.project_root).expanduser().resolve()
    input_path = resolve_input(args.input, project_root)

    rows, fieldnames = read_csv(input_path)

    keep_n = max(2, int(len(rows) * args.keep_percent / 100.0))
    keep_n = min(keep_n, len(rows))

    kept_source = [r.copy() for r in rows[:keep_n]]
    t, q = rows_to_arrays(kept_source)

    q_smooth = smooth_positions(q, args.smooth_passes)

    new_t = retime_velocity_capped(
        t=t,
        q=q_smooth,
        fast_fraction=args.fast_fraction,
        fast_limit=args.fast_speed_limit,
        final_limit=args.final_speed_limit,
        curve_power=args.curve_power,
        min_dt=args.min_dt,
    )

    out_rows = []
    for i, r_old in enumerate(kept_source):
        r = r_old.copy()
        r["elapsed_time"] = f"{float(new_t[i]):.6f}"
        for j, name in enumerate(JOINT_NAMES):
            r[name] = f"{float(q_smooth[i, j]):.12f}"
        out_rows.append(r)

    # Hold at final pose.
    hold = out_rows[-1].copy()
    hold["elapsed_time"] = f"{float(new_t[-1] + args.hold_seconds):.6f}"
    out_rows.append(hold)

    label = str(args.keep_percent).replace(".", "p")
    fastlabel = str(args.fast_speed_limit).replace(".", "p")
    finallabel = str(args.final_speed_limit).replace(".", "p")
    if args.output_prefix:
        prefix = Path(args.output_prefix)
        if not prefix.is_absolute():
            prefix = (project_root / "reinforcement_learning" / prefix).resolve()
    else:
        prefix = input_path.with_name(
            f"{input_path.stem}_keep{label}_velcap{fastlabel}_final{finallabel}"
        )

    unflipped = prefix.with_suffix(".csv")
    flipped = prefix.with_name(prefix.name + "_flipped_180").with_suffix(".csv")

    write_csv(unflipped, out_rows, fieldnames)
    flipped_rows = flip_rows(out_rows, fieldnames)
    write_csv(flipped, flipped_rows, fieldnames)

    t_out, _, max_v = compute_vel_stats(flipped_rows)

    print("=== Created velocity-capped fast/braked trajectory ===")
    print("Input:", input_path)
    print("Kept rows:", keep_n, "of", len(rows), f"({args.keep_percent}%)")
    print("Fast fraction:", args.fast_fraction)
    print("Fast speed limit:", args.fast_speed_limit, "rad/s")
    print("Final speed limit:", args.final_speed_limit, "rad/s")
    print("Curve power:", args.curve_power)
    print("Min dt:", args.min_dt)
    print("Unflipped:", unflipped)
    print("Flipped:", flipped)
    print("Duration:", float(t_out[-1] - t_out[0]), "s")
    print("Max joint velocities:")
    for name, v in zip(JOINT_NAMES, max_v):
        print(f"  {name:25s}: {v:.3f} rad/s")

    if not args.no_update_scripts:
        print("\nUpdating scripts:")
        for rel in SENDER_SCRIPTS:
            update_script(project_root / rel, flipped)

    print("\nRun:")
    print("  python3 move_to_whip_start_pose.py")
    print("  python3 send_csv_trajectory_preserved_speed_smooth.py")


if __name__ == "__main__":
    main()
