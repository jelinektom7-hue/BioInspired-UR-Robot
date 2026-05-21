#!/usr/bin/env python3
"""
Create a dense trajectory with:
  1. a gentle initial speed ramp,
  2. a fast middle section,
  3. an exponential late brake only in the last part of the motion.

This is intended for the real UR5e preserved-speed sender.

Typical use:
    cd /home/dragos/ros2_ws/src/BioInspired-UR-Robot/ur5e_whip-main/reinforcement_learning

    python3 make_ramp_fast_latebrake_trajectory.py \
      --input exported_trajectory_SAC17_120k_command_clean.csv \
      --keep-percent 40 \
      --ramp-fraction 0.15 \
      --brake-start-fraction 0.75 \
      --start-speed-limit 0.9 \
      --fast-speed-limit 7.0 \
      --final-speed-limit 0.25 \
      --max-accel 70.0 \
      --max-step-rad 0.003 \
      --min-dt 0.0025 \
      --brake-curve-power 3.5 \
      --ramp-curve-power 1.3 \
      --smooth-passes 1 \
      --hold-seconds 1.0

Meaning:
  - keep-percent 40:
      use only first 40% of the original cleaned trajectory.
  - ramp-fraction 0.15:
      first 15% of the cut motion ramps from start-speed-limit to fast-speed-limit.
  - brake-start-fraction 0.75:
      first 75% of the cut motion is not braking.
      final 25% brakes exponentially.
  - brake-curve-power > 1:
      keeps speed high early in braking phase, then brakes harder near the end.
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

    p.add_argument("--ramp-fraction", type=float, default=0.15)
    p.add_argument("--brake-start-fraction", type=float, default=0.75)

    p.add_argument("--start-speed-limit", type=float, default=0.9)
    p.add_argument("--fast-speed-limit", type=float, default=7.0)
    p.add_argument("--final-speed-limit", type=float, default=0.25)

    p.add_argument("--max-accel", type=float, default=70.0)
    p.add_argument("--max-step-rad", type=float, default=0.003)
    p.add_argument("--min-dt", type=float, default=0.0025)

    p.add_argument("--ramp-curve-power", type=float, default=1.3)
    p.add_argument("--brake-curve-power", type=float, default=3.5)

    p.add_argument("--smooth-passes", type=int, default=1)
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

        # Preserve exact start and end of cut.
        new[0] = q[0]
        new[-1] = q[-1]
        out = new

    return out


def densify_q(q, max_step_rad):
    dense = [q[0].copy()]

    for i in range(len(q) - 1):
        a = q[i]
        b = q[i + 1]
        dq = b - a
        max_abs = float(np.max(np.abs(dq)))
        n = max(1, int(math.ceil(max_abs / max_step_rad)))

        for k in range(1, n + 1):
            s = k / float(n)
            dense.append(a + s * dq)

    return np.array(dense, dtype=float)


def smoothstep(x):
    x = min(1.0, max(0.0, x))
    return x * x * (3.0 - 2.0 * x)


def speed_limit_profile(progress, ramp_fraction, brake_start_fraction,
                        start_speed, fast_speed, final_speed,
                        ramp_power, brake_power):
    """
    progress: 0..1 along the dense trajectory.

    Phase 1: 0 -> ramp_fraction
        speed ramps from start_speed to fast_speed.

    Phase 2: ramp_fraction -> brake_start_fraction
        speed stays at fast_speed.

    Phase 3: brake_start_fraction -> 1
        speed exponentially decreases from fast_speed to final_speed.
    """
    if progress < ramp_fraction:
        s = progress / max(1e-9, ramp_fraction)
        s = s ** ramp_power
        s = smoothstep(s)
        return start_speed + (fast_speed - start_speed) * s

    if progress < brake_start_fraction:
        return fast_speed

    s = (progress - brake_start_fraction) / max(1e-9, 1.0 - brake_start_fraction)
    s = min(1.0, max(0.0, s))

    # brake_power > 1 keeps fast speed longer, then brakes harder near the end.
    shaped = s ** brake_power
    return fast_speed * math.exp(math.log(final_speed / fast_speed) * shaped)


def initial_dt_from_profile(q_dense, args):
    dq = np.diff(q_dense, axis=0)
    max_dq = np.max(np.abs(dq), axis=1)
    nseg = len(max_dq)

    dt = np.zeros(nseg, dtype=float)

    for i in range(nseg):
        progress = i / max(1, nseg - 1)
        limit = speed_limit_profile(
            progress=progress,
            ramp_fraction=args.ramp_fraction,
            brake_start_fraction=args.brake_start_fraction,
            start_speed=args.start_speed_limit,
            fast_speed=args.fast_speed_limit,
            final_speed=args.final_speed_limit,
            ramp_power=args.ramp_curve_power,
            brake_power=args.brake_curve_power,
        )

        dt[i] = max(args.min_dt, max_dq[i] / max(limit, 1e-9))

    return dt


def stretch_for_accel_limit(q_dense, dt, max_accel, iterations=100):
    if len(dt) < 2:
        return dt

    dt = dt.copy()

    for _ in range(iterations):
        dq = np.diff(q_dense, axis=0)
        v = dq / dt[:, None]

        dt_mid = 0.5 * (dt[:-1] + dt[1:])
        acc = np.diff(v, axis=0) / dt_mid[:, None]

        if acc.size == 0:
            break

        abs_acc = np.abs(acc)
        worst = float(np.max(abs_acc))

        if worst <= max_accel:
            break

        bad = np.where(np.max(abs_acc, axis=1) > max_accel)[0]

        for j in bad:
            local_worst = float(np.max(abs_acc[j]))
            scale = math.sqrt(local_worst / max_accel)
            scale = min(1.25, max(1.02, scale))

            dt[j] *= scale
            dt[j + 1] *= scale

    return dt


def make_rows(template_row, q_dense, dt, hold_seconds):
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

    vel = dq[valid] / dt[valid, None]
    max_v = np.max(np.abs(vel), axis=0) if len(vel) else np.zeros(6)

    max_a = np.zeros(6)
    if len(vel) >= 2:
        dt_valid = dt[valid]
        dt_mid = 0.5 * (dt_valid[:-1] + dt_valid[1:])
        acc = np.diff(vel, axis=0) / dt_mid[:, None]
        max_a = np.max(np.abs(acc), axis=0) if len(acc) else np.zeros(6)

    moving_duration = float(t[-2] - t[0]) if len(t) >= 2 else float(t[-1] - t[0])

    return {
        "points": len(rows),
        "duration": float(t[-1] - t[0]),
        "moving_duration": moving_duration,
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

    backup = script_path.with_suffix(script_path.suffix + ".backup_before_ramp_latebrake")
    if not backup.exists():
        backup.write_text(old)

    script_path.write_text(new)
    print(f"Updated {script_path} -> {wanted_csv}")


def validate_args(args):
    if not (0.0 < args.keep_percent <= 100.0):
        raise ValueError("--keep-percent must be > 0 and <= 100")
    if not (0.0 <= args.ramp_fraction < 1.0):
        raise ValueError("--ramp-fraction must be >= 0 and < 1")
    if not (0.0 < args.brake_start_fraction < 1.0):
        raise ValueError("--brake-start-fraction must be > 0 and < 1")
    if args.ramp_fraction >= args.brake_start_fraction:
        raise ValueError("--ramp-fraction must be smaller than --brake-start-fraction")
    if args.start_speed_limit <= 0 or args.fast_speed_limit <= 0 or args.final_speed_limit <= 0:
        raise ValueError("speed limits must be positive")
    if args.start_speed_limit > args.fast_speed_limit:
        raise ValueError("--start-speed-limit must be <= --fast-speed-limit")
    if args.final_speed_limit > args.fast_speed_limit:
        raise ValueError("--final-speed-limit must be <= --fast-speed-limit")
    if args.max_accel <= 0:
        raise ValueError("--max-accel must be positive")
    if args.max_step_rad <= 0:
        raise ValueError("--max-step-rad must be positive")
    if args.min_dt <= 0:
        raise ValueError("--min-dt must be positive")


def main():
    args = parse_args()
    validate_args(args)

    project_root = Path(args.project_root).expanduser().resolve()
    input_path = resolve_path(args.input, project_root)

    if not input_path.exists():
        raise FileNotFoundError(input_path)

    rows, fieldnames = read_csv(input_path)

    keep_n = max(2, int(len(rows) * args.keep_percent / 100.0))
    keep_n = min(keep_n, len(rows))

    kept = [r.copy() for r in rows[:keep_n]]
    q = rows_to_q(kept)
    q = smooth_positions(q, args.smooth_passes)

    q_dense = densify_q(q, args.max_step_rad)
    dt0 = initial_dt_from_profile(q_dense, args)
    dt = stretch_for_accel_limit(q_dense, dt0, args.max_accel)

    out_rows = make_rows(kept[0], q_dense, dt, args.hold_seconds)
    flipped_rows = flip_shoulder(out_rows)

    if args.output_prefix:
        prefix = Path(args.output_prefix).expanduser()
        if not prefix.is_absolute():
            prefix = (project_root / "reinforcement_learning" / prefix).resolve()
    else:
        kp = str(args.keep_percent).replace(".", "p")
        fs = str(args.fast_speed_limit).replace(".", "p")
        acc = str(args.max_accel).replace(".", "p")
        bs = str(args.brake_start_fraction).replace(".", "p")
        rp = str(args.ramp_fraction).replace(".", "p")
        prefix = input_path.with_name(
            f"{input_path.stem}_keep{kp}_ramp{rp}_brake{bs}_v{fs}_a{acc}"
        )

    unflipped = prefix.with_suffix(".csv")
    flipped = prefix.with_name(prefix.name + "_flipped_180").with_suffix(".csv")

    write_csv(unflipped, out_rows, fieldnames)
    write_csv(flipped, flipped_rows, fieldnames)

    s = stats(flipped_rows)

    print("=== Ramp + fast middle + late exponential brake trajectory created ===")
    print(f"Input:                    {input_path}")
    print(f"Kept original rows:       {keep_n}/{len(rows)} ({args.keep_percent}%)")
    print(f"Dense points + hold:      {s['points']}")
    print(f"Moving duration:          {s['moving_duration']:.6f} s")
    print(f"Total duration:           {s['duration']:.6f} s")
    print(f"Ramp fraction:            {args.ramp_fraction}")
    print(f"Brake start fraction:     {args.brake_start_fraction}")
    print(f"Brake fraction:           {1.0 - args.brake_start_fraction:.3f}")
    print(f"Start speed limit:        {args.start_speed_limit:.3f} rad/s")
    print(f"Fast speed limit:         {args.fast_speed_limit:.3f} rad/s")
    print(f"Final speed limit:        {args.final_speed_limit:.3f} rad/s")
    print(f"Max accel target:         {args.max_accel:.3f} rad/s^2")
    print(f"Max step rad:             {args.max_step_rad:.6f}")
    print(f"Min dt:                   {args.min_dt:.6f}")
    print(f"Ramp curve power:         {args.ramp_curve_power:.3f}")
    print(f"Brake curve power:        {args.brake_curve_power:.3f}")
    print(f"Unflipped CSV:            {unflipped}")
    print(f"Flipped CSV:              {flipped}")
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
