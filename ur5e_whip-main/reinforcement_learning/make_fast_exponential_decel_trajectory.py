#!/usr/bin/env python3
"""
Create a trajectory that keeps the first part at original simulation speed,
then exponentially decelerates near the end, flips it for the real setup,
and updates the robot sender scripts.

Default intended use for the current SAC17 trajectory:
    cd /home/dragos/ros2_ws/src/BioInspired-UR-Robot/ur5e_whip-main/reinforcement_learning

    python3 make_fast_exponential_decel_trajectory.py \
      --input exported_trajectory_SAC17_120k_command_clean.csv \
      --keep-percent 40 \
      --fast-fraction 0.60 \
      --final-dt-multiplier 80 \
      --curve-power 2.5 \
      --hold-seconds 1.0

Meaning:
    - Use the first 40% of the original cleaned trajectory.
    - Keep the first 60% of that cut trajectory at original timing.
    - Exponentially slow the final 40%.
    - Flip shoulder_pan_joint by +pi.
    - Update move_to_whip_start_pose.py and send_csv_trajectory_preserved_speed.py.
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
    "reinforcement_learning/send_csv_trajectory_preserved_speed.py",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Make fast-start / exponential-deceleration CSV for preserved-speed execution."
    )

    parser.add_argument(
        "--input",
        default="exported_trajectory_SAC17_120k_command_clean.csv",
        help="Input CSV. Relative paths are resolved inside reinforcement_learning/.",
    )

    parser.add_argument(
        "--keep-percent",
        type=float,
        default=40.0,
        help="Keep the first N percent of the input trajectory rows. Default: 40.",
    )

    parser.add_argument(
        "--fast-fraction",
        type=float,
        default=0.60,
        help=(
            "Fraction of the kept trajectory to preserve at original timing before deceleration starts. "
            "Default: 0.60 means first 60%% full speed, final 40%% decelerates."
        ),
    )

    parser.add_argument(
        "--final-dt-multiplier",
        type=float,
        default=80.0,
        help=(
            "Final timestep multiplier at the last moving segment. "
            "Higher gives much stronger final slowdown. Default: 80."
        ),
    )

    parser.add_argument(
        "--curve-power",
        type=float,
        default=2.5,
        help=(
            "Shape of deceleration curve. Higher keeps speed high longer, then brakes harder near the end. "
            "Default: 2.5."
        ),
    )

    parser.add_argument(
        "--smooth-passes",
        type=int,
        default=1,
        help="Small smoothing passes on joint positions. Use 0 to disable. Default: 1.",
    )

    parser.add_argument(
        "--hold-seconds",
        type=float,
        default=1.0,
        help="Final hold duration after deceleration. Default: 1.0 s.",
    )

    parser.add_argument(
        "--project-root",
        default=str(PROJECT_ROOT_DEFAULT),
        help="Project root.",
    )

    parser.add_argument(
        "--output-prefix",
        default=None,
        help="Optional output prefix without .csv.",
    )

    parser.add_argument(
        "--no-update-scripts",
        action="store_true",
        help="Create CSVs only; do not update sender scripts.",
    )

    parser.add_argument(
        "--update-safe-too",
        action="store_true",
        help=(
            "Also update send_csv_trajectory_safe.py. Normally not needed because safe sender "
            "does not preserve CSV timing."
        ),
    )

    return parser.parse_args()


def resolve_input(path_str: str, project_root: Path) -> Path:
    p = Path(path_str).expanduser()
    if p.is_absolute():
        return p.resolve()
    return (project_root / "reinforcement_learning" / p).resolve()


def load_csv(path: Path):
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames

    if not rows:
        raise RuntimeError(f"Input CSV is empty: {path}")

    for name in ["elapsed_time"] + JOINT_NAMES:
        if name not in fieldnames:
            raise RuntimeError(f"Missing required column {name}")

    return rows, fieldnames


def write_csv(path: Path, rows, fieldnames):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def rows_to_arrays(rows):
    t = np.array([float(r["elapsed_time"]) for r in rows], dtype=float)
    q = np.array([[float(r[j]) for j in JOINT_NAMES] for r in rows], dtype=float)
    return t, q


def smooth_positions(q: np.ndarray, passes: int):
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

        # Preserve exact beginning and end of the cut.
        smoothed[0] = q[0]
        smoothed[-1] = q[-1]
        out = smoothed

    return out


def make_fast_exponential_time(t: np.ndarray, fast_fraction: float, final_dt_multiplier: float, curve_power: float):
    if not (0.0 <= fast_fraction < 1.0):
        raise ValueError("--fast-fraction must be >= 0 and < 1")

    if final_dt_multiplier < 1.0:
        raise ValueError("--final-dt-multiplier must be >= 1")

    if curve_power <= 0.0:
        raise ValueError("--curve-power must be > 0")

    if len(t) < 3:
        return t - t[0]

    t = t - t[0]
    dt = np.diff(t)
    dt = np.maximum(dt, 1e-6)

    nseg = len(dt)
    decel_start = int(np.floor(fast_fraction * nseg))
    decel_start = max(0, min(decel_start, nseg - 1))

    new_dt = dt.copy()

    for i in range(decel_start, nseg):
        if nseg - decel_start <= 1:
            s = 1.0
        else:
            s = (i - decel_start) / float((nseg - decel_start) - 1)

        # Exponential-like multiplier:
        #   s = 0 => multiplier = 1
        #   s = 1 => multiplier = final_dt_multiplier
        # curve_power > 1 keeps the first part faster and brakes harder near the end.
        shaped = s ** curve_power
        multiplier = math.exp(math.log(final_dt_multiplier) * shaped)
        new_dt[i] = dt[i] * multiplier

    new_t = np.zeros_like(t)
    new_t[1:] = np.cumsum(new_dt)
    return new_t


def flip_shoulder(input_rows, fieldnames, output_path: Path):
    joint_name = "shoulder_pan_joint"
    if joint_name not in fieldnames:
        raise RuntimeError(f"Missing column: {joint_name}")

    rows = [r.copy() for r in input_rows]
    for r in rows:
        r[joint_name] = f"{float(r[joint_name]) + math.pi:.12f}"

    write_csv(output_path, rows, fieldnames)
    return rows


def update_script(script_path: Path, wanted_csv: Path):
    if not script_path.exists():
        print(f"WARNING: missing script: {script_path}")
        return

    old = script_path.read_text()
    new = old

    pattern = re.compile(
        r'(^\s*CSV_PATH\s*=\s*)(["\'])(.*?\.csv)(["\'])',
        flags=re.MULTILINE,
    )

    if pattern.search(new):
        new = pattern.sub(lambda m: f'{m.group(1)}"{wanted_csv}"', new, count=1)
    else:
        new = re.sub(
            r'(["\'])(?:[^"\']*/)?exported_trajectory[^"\']*\.csv(["\'])',
            f'"{wanted_csv}"',
            new,
        )

    if new == old:
        print(f"WARNING: no CSV path replaced in {script_path}")
        return

    backup = script_path.with_suffix(script_path.suffix + ".backup_before_fast_exp_decel")
    if not backup.exists():
        backup.write_text(old)

    script_path.write_text(new)
    print(f"Updated {script_path} -> {wanted_csv}")


def compute_stats(rows):
    t, q = rows_to_arrays(rows)
    dt = np.diff(t)
    dq = np.diff(q, axis=0)

    valid = dt > 1e-9
    vel = dq[valid] / dt[valid, None]

    max_abs_vel = np.max(np.abs(vel), axis=0) if len(vel) else np.zeros(6)

    return {
        "points": len(rows),
        "duration": float(t[-1] - t[0]),
        "max_abs_vel": max_abs_vel,
        "first_moving_duration": float(t[-2] - t[0]) if len(t) >= 2 else 0.0,
    }


def main():
    args = parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    input_path = resolve_input(args.input, project_root)

    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    rows, fieldnames = load_csv(input_path)

    keep_percent = args.keep_percent
    if keep_percent <= 0.0 or keep_percent > 100.0:
        raise ValueError("--keep-percent must be > 0 and <= 100")

    keep_n = max(2, int(len(rows) * (keep_percent / 100.0)))
    keep_n = min(keep_n, len(rows))

    kept_rows = [r.copy() for r in rows[:keep_n]]
    t, q = rows_to_arrays(kept_rows)

    q_smooth = smooth_positions(q, args.smooth_passes)
    t_new = make_fast_exponential_time(
        t,
        fast_fraction=args.fast_fraction,
        final_dt_multiplier=args.final_dt_multiplier,
        curve_power=args.curve_power,
    )

    output_rows = []
    for i, r_old in enumerate(kept_rows):
        r = r_old.copy()
        r["elapsed_time"] = f"{float(t_new[i]):.6f}"
        for j, name in enumerate(JOINT_NAMES):
            r[name] = f"{float(q_smooth[i, j]):.12f}"
        output_rows.append(r)

    # Final hold at the final decelerated pose.
    hold_row = output_rows[-1].copy()
    hold_row["elapsed_time"] = f"{float(t_new[-1] + args.hold_seconds):.6f}"
    output_rows.append(hold_row)

    if args.output_prefix is None:
        label = str(args.keep_percent).replace(".", "p")
        fast_label = str(args.fast_fraction).replace(".", "p")
        output_prefix = input_path.with_name(
            f"{input_path.stem}_keep{label}_fast{fast_label}_expdecel"
        )
    else:
        output_prefix = Path(args.output_prefix).expanduser()
        if not output_prefix.is_absolute():
            output_prefix = (project_root / "reinforcement_learning" / output_prefix).resolve()

    cut_csv = output_prefix.with_suffix(".csv")
    flipped_csv = output_prefix.with_name(output_prefix.name + "_flipped_180").with_suffix(".csv")

    write_csv(cut_csv, output_rows, fieldnames)
    flipped_rows = flip_shoulder(output_rows, fieldnames, flipped_csv)

    print("=== Fast-start exponential deceleration CSV created ===")
    print(f"Input:                    {input_path}")
    print(f"Kept rows:                {keep_n}/{len(rows)} ({keep_percent}%)")
    print(f"Fast fraction of kept:    {args.fast_fraction}")
    print(f"Decel fraction of kept:   {1.0 - args.fast_fraction}")
    print(f"Final dt multiplier:      {args.final_dt_multiplier}")
    print(f"Curve power:              {args.curve_power}")
    print(f"Smooth passes:            {args.smooth_passes}")
    print(f"Hold seconds:             {args.hold_seconds}")
    print(f"Unflipped CSV:            {cut_csv}")
    print(f"Flipped CSV:              {flipped_csv}")
    print("")

    stats = compute_stats(flipped_rows)
    print("Flipped output stats:")
    print(f"  points:                {stats['points']}")
    print(f"  duration:              {stats['duration']:.6f} s")
    print(f"  duration before hold:  {stats['first_moving_duration']:.6f} s")
    print("  max joint velocity:")
    for name, v in zip(JOINT_NAMES, stats["max_abs_vel"]):
        print(f"    {name:25s}: {v:.3f} rad/s")
    print("")

    if not args.no_update_scripts:
        scripts = list(SENDER_SCRIPTS)
        if args.update_safe_too:
            scripts.append("reinforcement_learning/send_csv_trajectory_safe.py")

        print("Updating sender scripts:")
        for rel in scripts:
            update_script(project_root / rel, flipped_csv)

    print("")
    print("Use preserved timing sender:")
    print("  python3 move_to_whip_start_pose.py")
    print("  python3 send_csv_trajectory_preserved_speed.py")
    print("")
    print("Do not use send_csv_trajectory_safe.py for this timing-shaped trajectory unless you only want constant-speed retiming.")


if __name__ == "__main__":
    main()
