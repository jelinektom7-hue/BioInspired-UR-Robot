#!/usr/bin/env python3
"""
Create a dense, acceleration-limited, fast-start / braked-end trajectory.

This solves the issue where the UR controller sees a very large internal velocity
between sparse waypoints. It does three important things:

1. Keeps only the requested first part of the original cleaned trajectory.
2. Inserts many intermediate joint-space waypoints.
3. Retimes those waypoints so both velocity and acceleration are bounded.

Default intended use:
    cd /home/dragos/ros2_ws/src/BioInspired-UR-Robot/ur5e_whip-main/reinforcement_learning

    python3 make_dense_accel_limited_trajectory.py \
      --input exported_trajectory_SAC17_120k_command_clean.csv \
      --keep-percent 40 \
      --fast-fraction 0.60 \
      --fast-speed-limit 3.4 \
      --final-speed-limit 0.55 \
      --max-accel 7.0 \
      --max-step-rad 0.006 \
      --min-dt 0.008 \
      --curve-power 2.2 \
      --hold-seconds 1.0

Then run:
    python3 move_to_whip_start_pose.py
    python3 send_csv_trajectory_preserved_speed.py

Do NOT use send_csv_trajectory_safe.py for this timing-shaped trajectory.
"""

import argparse
import csv
import math
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
    "reinforcement_learning/send_csv_trajectory_preserved_speed.py",
]


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--input", default="exported_trajectory_SAC17_120k_command_clean.csv")
    p.add_argument("--project-root", default=str(PROJECT_ROOT_DEFAULT))
    p.add_argument("--keep-percent", type=float, default=40.0)

    p.add_argument(
        "--fast-fraction",
        type=float,
        default=0.60,
        help="Fraction of the kept trajectory before braking begins. Default 0.60.",
    )
    p.add_argument(
        "--fast-speed-limit",
        type=float,
        default=3.4,
        help="Approx max joint speed during fast phase, rad/s. Default 3.4.",
    )
    p.add_argument(
        "--final-speed-limit",
        type=float,
        default=0.55,
        help="Approx max joint speed at final moving segment, rad/s. Default 0.55.",
    )
    p.add_argument(
        "--max-accel",
        type=float,
        default=7.0,
        help="Approx max joint acceleration after iterative retiming, rad/s^2. Default 7.0.",
    )
    p.add_argument(
        "--max-step-rad",
        type=float,
        default=0.006,
        help="Maximum joint-space step before densification, rad. Smaller = more points. Default 0.006.",
    )
    p.add_argument(
        "--min-dt",
        type=float,
        default=0.008,
        help="Minimum time between generated points, seconds. Default 0.008.",
    )
    p.add_argument(
        "--curve-power",
        type=float,
        default=2.2,
        help="Brake curve. Higher keeps fast speed longer, then brakes harder. Default 2.2.",
    )
    p.add_argument(
        "--smooth-passes",
        type=int,
        default=1,
        help="Light smoothing passes before densification. Default 1.",
    )
    p.add_argument("--hold-seconds", type=float, default=1.0)
    p.add_argument("--output-prefix", default=None)
    p.add_argument("--no-update-scripts", action="store_true")

    return p.parse_args()


def resolve_path(path_str: str, project_root: Path) -> Path:
    p = Path(path_str).expanduser()
    if p.is_absolute():
        return p.resolve()
    return (project_root / "reinforcement_learning" / p).resolve()


def read_csv(path: Path):
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


def rows_to_q(rows):
    return np.array([[float(r[j]) for j in JOINT_NAMES] for r in rows], dtype=float)


def smooth_positions(q, passes):
    if passes <= 0 or len(q) < 5:
        return q.copy()

    out = q.copy()
    kernel = np.array([1, 2, 3, 2, 1], dtype=float)
    kernel /= kernel.sum()

    for _ in range(passes):
        padded = np.pad(out, ((2, 2), (0, 0)), mode="edge")
        new = out.copy()
        for i in range(len(out)):
            new[i] = sum(kernel[k] * padded[i + k] for k in range(5))

        # Preserve exact start and final pose.
        new[0] = q[0]
        new[-1] = q[-1]
        out = new

    return out


def densify_q(q, max_step_rad):
    dense = [q[0].copy()]
    source_indices = [0]

    for i in range(len(q) - 1):
        a = q[i]
        b = q[i + 1]
        dq = b - a
        max_abs = float(np.max(np.abs(dq)))
        n = max(1, int(math.ceil(max_abs / max_step_rad)))

        for k in range(1, n + 1):
            s = k / float(n)
            dense.append(a + s * dq)
            source_indices.append(i + s)

    return np.array(dense, dtype=float), source_indices


def velocity_limit_profile(progress, fast_fraction, fast_speed, final_speed, curve_power):
    if progress <= fast_fraction:
        return fast_speed

    s = (progress - fast_fraction) / max(1e-9, 1.0 - fast_fraction)
    s = min(1.0, max(0.0, s))

    # Exponential speed decay:
    # progress at fast_fraction -> fast_speed
    # progress at 1.0 -> final_speed
    shaped = s ** curve_power
    return fast_speed * math.exp(math.log(final_speed / fast_speed) * shaped)


def initial_dt_from_velocity_profile(q_dense, fast_fraction, fast_speed, final_speed, curve_power, min_dt):
    dq = np.diff(q_dense, axis=0)
    max_dq = np.max(np.abs(dq), axis=1)
    nseg = len(max_dq)

    dt = np.zeros(nseg, dtype=float)

    for i in range(nseg):
        progress = i / max(1, nseg - 1)
        limit = velocity_limit_profile(progress, fast_fraction, fast_speed, final_speed, curve_power)
        dt[i] = max(min_dt, max_dq[i] / max(limit, 1e-9))

    return dt


def stretch_for_accel_limit(q_dense, dt, max_accel, iterations=80):
    if len(dt) < 2:
        return dt

    dt = dt.copy()

    for _ in range(iterations):
        dq = np.diff(q_dense, axis=0)
        v = dq / dt[:, None]

        dt_mid = 0.5 * (dt[:-1] + dt[1:])
        acc = np.diff(v, axis=0) / dt_mid[:, None]

        abs_acc = np.abs(acc)
        worst = float(np.max(abs_acc)) if abs_acc.size else 0.0

        if worst <= max_accel:
            break

        # Stretch local segments around acceleration violations.
        bad = np.where(np.max(abs_acc, axis=1) > max_accel)[0]
        for j in bad:
            local_worst = float(np.max(abs_acc[j]))
            scale = min(1.35, max(1.03, math.sqrt(local_worst / max_accel)))
            dt[j] *= scale
            dt[j + 1] *= scale

    return dt


def make_rows_from_q_and_dt(template_row, q_dense, dt, hold_seconds):
    times = np.zeros(len(q_dense), dtype=float)
    if len(dt):
        times[1:] = np.cumsum(dt)

    rows = []
    for i, q in enumerate(q_dense):
        r = template_row.copy()
        r["elapsed_time"] = f"{float(times[i]):.6f}"
        for j, name in enumerate(JOINT_NAMES):
            r[name] = f"{float(q[j]):.12f}"
        rows.append(r)

    hold = rows[-1].copy()
    hold["elapsed_time"] = f"{float(times[-1] + hold_seconds):.6f}"
    rows.append(hold)

    return rows


def flip_shoulder(rows):
    out = [r.copy() for r in rows]
    for r in out:
        r["shoulder_pan_joint"] = f"{float(r['shoulder_pan_joint']) + math.pi:.12f}"
    return out


def stats(rows):
    q = rows_to_q(rows)
    t = np.array([float(r["elapsed_time"]) for r in rows], dtype=float)

    dt = np.diff(t)
    dq = np.diff(q, axis=0)
    valid = dt > 1e-9

    v = dq[valid] / dt[valid, None]
    max_v = np.max(np.abs(v), axis=0) if len(v) else np.zeros(6)

    max_a = np.zeros(6)
    if len(v) >= 2:
        dt_valid = dt[valid]
        dt_mid = 0.5 * (dt_valid[:-1] + dt_valid[1:])
        a = np.diff(v, axis=0) / dt_mid[:, None]
        max_a = np.max(np.abs(a), axis=0) if len(a) else np.zeros(6)

    return {
        "points": len(rows),
        "duration": float(t[-1] - t[0]),
        "moving_duration": float(t[-2] - t[0]) if len(t) >= 2 else 0.0,
        "max_v": max_v,
        "max_a": max_a,
    }


def update_script(script_path: Path, wanted_csv: Path):
    if not script_path.exists():
        print(f"WARNING: script not found: {script_path}")
        return

    old = script_path.read_text()
    pattern = re.compile(r'(^\s*CSV_PATH\s*=\s*)(["\'])(.*?\.csv)(["\'])', flags=re.MULTILINE)

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

    backup = script_path.with_suffix(script_path.suffix + ".backup_before_dense_accel")
    if not backup.exists():
        backup.write_text(old)

    script_path.write_text(new)
    print(f"Updated {script_path} -> {wanted_csv}")


def main():
    args = parse_args()
    project_root = Path(args.project_root).expanduser().resolve()
    input_path = resolve_path(args.input, project_root)

    if not input_path.exists():
        raise FileNotFoundError(input_path)

    if not (0.0 < args.keep_percent <= 100.0):
        raise ValueError("--keep-percent must be > 0 and <= 100")

    if not (0.0 <= args.fast_fraction < 1.0):
        raise ValueError("--fast-fraction must be >= 0 and < 1")

    rows, fieldnames = read_csv(input_path)

    keep_n = max(2, int(len(rows) * args.keep_percent / 100.0))
    keep_n = min(keep_n, len(rows))

    kept = [r.copy() for r in rows[:keep_n]]
    q = rows_to_q(kept)
    q = smooth_positions(q, args.smooth_passes)

    q_dense, _ = densify_q(q, args.max_step_rad)

    dt0 = initial_dt_from_velocity_profile(
        q_dense=q_dense,
        fast_fraction=args.fast_fraction,
        fast_speed=args.fast_speed_limit,
        final_speed=args.final_speed_limit,
        curve_power=args.curve_power,
        min_dt=args.min_dt,
    )

    dt = stretch_for_accel_limit(q_dense, dt0, args.max_accel)

    out_rows = make_rows_from_q_and_dt(kept[0], q_dense, dt, args.hold_seconds)
    flipped_rows = flip_shoulder(out_rows)

    if args.output_prefix:
        prefix = Path(args.output_prefix).expanduser()
        if not prefix.is_absolute():
            prefix = (project_root / "reinforcement_learning" / prefix).resolve()
    else:
        label = str(args.keep_percent).replace(".", "p")
        fast_label = str(args.fast_speed_limit).replace(".", "p")
        accel_label = str(args.max_accel).replace(".", "p")
        prefix = input_path.with_name(
            f"{input_path.stem}_keep{label}_dense_v{fast_label}_a{accel_label}"
        )

    unflipped = prefix.with_suffix(".csv")
    flipped = prefix.with_name(prefix.name + "_flipped_180").with_suffix(".csv")

    write_csv(unflipped, out_rows, fieldnames)
    write_csv(flipped, flipped_rows, fieldnames)

    s = stats(flipped_rows)

    print("=== Dense acceleration-limited trajectory created ===")
    print(f"Input:                 {input_path}")
    print(f"Kept original rows:    {keep_n}/{len(rows)} ({args.keep_percent}%)")
    print(f"Dense points + hold:   {s['points']}")
    print(f"Moving duration:       {s['moving_duration']:.6f} s")
    print(f"Total duration:        {s['duration']:.6f} s")
    print(f"Fast fraction:         {args.fast_fraction}")
    print(f"Fast speed limit:      {args.fast_speed_limit:.3f} rad/s")
    print(f"Final speed limit:     {args.final_speed_limit:.3f} rad/s")
    print(f"Max accel target:      {args.max_accel:.3f} rad/s^2")
    print(f"Max step rad:          {args.max_step_rad:.6f}")
    print(f"Min dt:                {args.min_dt:.6f}")
    print(f"Unflipped CSV:         {unflipped}")
    print(f"Flipped CSV:           {flipped}")
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
        for rel in SENDER_SCRIPTS:
            update_script(project_root / rel, flipped)

    print("")
    print("Run:")
    print("  python3 move_to_whip_start_pose.py")
    print("  python3 send_csv_trajectory_preserved_speed.py")
    print("")
    print("Do not use send_csv_trajectory_safe.py for this timing-shaped trajectory.")


if __name__ == "__main__":
    main()
