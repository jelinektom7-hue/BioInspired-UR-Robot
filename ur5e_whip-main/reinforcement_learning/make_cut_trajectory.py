#!/usr/bin/env python3
"""
Create a shortened real-robot trajectory CSV and point the robot sender scripts to it.

Example:
    cd /home/dragos/ros2_ws/src/BioInspired-UR-Robot/ur5e_whip-main/reinforcement_learning
    python3 make_cut_trajectory.py --keep-percent 33

This will:
    1. Take the first 33% of the input CSV rows.
    2. Delete everything after that.
    3. Add a hold point at the final kept pose.
    4. Flip shoulder_pan_joint by +pi for the real robot setup.
    5. Update move_to_whip_start_pose.py, send_csv_trajectory_safe.py,
       and send_csv_trajectory_preserved_speed.py to use the newly generated CSV.
"""

import argparse
import csv
import math
import re
from pathlib import Path


DEFAULT_PROJECT_ROOT = Path("/home/dragos/ros2_ws/src/BioInspired-UR-Robot/ur5e_whip-main")
DEFAULT_INPUT_CSV = (
    DEFAULT_PROJECT_ROOT
    / "reinforcement_learning/exported_trajectory_SAC17_120k_command_clean_retimed_1p8x.csv"
)

SENDER_SCRIPTS = [
    "reinforcement_learning/move_to_whip_start_pose.py",
    "reinforcement_learning/send_csv_trajectory_safe.py",
    "reinforcement_learning/send_csv_trajectory_preserved_speed.py",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Cut a trajectory to the first N percent, add a hold, flip for the real setup, "
            "and update the robot sender scripts."
        )
    )

    parser.add_argument(
        "--keep-percent",
        type=float,
        required=True,
        help="Percentage of original CSV rows to keep. Example: 33 for first third, 50 for half.",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(DEFAULT_INPUT_CSV),
        help=f"Input CSV to cut. Default: {DEFAULT_INPUT_CSV}",
    )
    parser.add_argument(
        "--hold-seconds",
        type=float,
        default=1.0,
        help="Seconds to hold at the final kept pose. Default: 1.0",
    )
    parser.add_argument(
        "--project-root",
        type=str,
        default=str(DEFAULT_PROJECT_ROOT),
        help=f"Project root. Default: {DEFAULT_PROJECT_ROOT}",
    )
    parser.add_argument(
        "--no-flip",
        action="store_true",
        help="Do not create/use the +pi shoulder_pan_joint flipped CSV.",
    )
    parser.add_argument(
        "--no-update-scripts",
        action="store_true",
        help="Only create the CSV files; do not update robot sender scripts.",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default=None,
        help=(
            "Optional output prefix without .csv. "
            "If omitted, a name is generated from the input file and keep percentage."
        ),
    )

    return parser.parse_args()


def percent_label(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return str(value).replace(".", "p")


def load_csv(path: Path):
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames

    if not rows:
        raise RuntimeError(f"Input CSV is empty: {path}")
    if not fieldnames:
        raise RuntimeError(f"Input CSV has no header: {path}")
    if "elapsed_time" not in fieldnames:
        raise RuntimeError("CSV must contain an elapsed_time column.")

    return rows, fieldnames


def write_csv(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def cut_trajectory(input_csv: Path, output_csv: Path, keep_percent: float, hold_seconds: float):
    if keep_percent <= 0.0 or keep_percent > 100.0:
        raise ValueError("--keep-percent must be > 0 and <= 100")

    rows, fieldnames = load_csv(input_csv)

    keep_fraction = keep_percent / 100.0
    keep_n = max(2, int(len(rows) * keep_fraction))
    keep_n = min(keep_n, len(rows))

    kept = [r.copy() for r in rows[:keep_n]]

    hold_row = kept[-1].copy()
    hold_row["elapsed_time"] = f"{float(kept[-1]['elapsed_time']) + hold_seconds:.6f}"
    kept.append(hold_row)

    # Re-zero time so the new trajectory starts at 0.
    t0 = float(kept[0]["elapsed_time"])
    for row in kept:
        row["elapsed_time"] = f"{float(row['elapsed_time']) - t0:.6f}"

    write_csv(output_csv, kept, fieldnames)

    return {
        "input_rows": len(rows),
        "kept_original_rows": keep_n,
        "output_rows_including_hold": len(kept),
        "final_kept_original_row_index": keep_n - 1,
        "new_duration": float(kept[-1]["elapsed_time"]),
    }


def flip_trajectory(input_csv: Path, output_csv: Path):
    rows, fieldnames = load_csv(input_csv)

    joint_name = "shoulder_pan_joint"
    if joint_name not in fieldnames:
        raise RuntimeError(f"Missing required column: {joint_name}")

    flipped = [r.copy() for r in rows]
    for row in flipped:
        row[joint_name] = f"{float(row[joint_name]) + math.pi:.12f}"

    write_csv(output_csv, flipped, fieldnames)


def update_sender_script(script_path: Path, wanted_csv: Path):
    if not script_path.exists():
        print(f"WARNING: script not found, skipping: {script_path}")
        return False

    original = script_path.read_text()
    text = original

    # Preferred replacement: replace the CSV_PATH assignment.
    csv_path_pattern = re.compile(
        r'(^\s*CSV_PATH\s*=\s*)(["\'])(.*?\.csv)(["\'])',
        flags=re.MULTILINE,
    )

    if csv_path_pattern.search(text):
        text = csv_path_pattern.sub(
            lambda m: f'{m.group(1)}"{wanted_csv}"',
            text,
            count=1,
        )
    else:
        # Fallback: replace any quoted exported_trajectory*.csv path.
        text = re.sub(
            r'(["\'])(?:[^"\']*/)?exported_trajectory[^"\']*\.csv(["\'])',
            f'"{wanted_csv}"',
            text,
        )

    if text == original:
        print(f"WARNING: no CSV path replaced in {script_path}")
        return False

    backup_path = script_path.with_suffix(script_path.suffix + ".backup_before_cut_update")
    if not backup_path.exists():
        backup_path.write_text(original)

    script_path.write_text(text)
    print(f"Updated {script_path} -> {wanted_csv}")
    return True


def main():
    args = parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    input_csv = Path(args.input).expanduser()
    if not input_csv.is_absolute():
        input_csv = (project_root / "reinforcement_learning" / input_csv).resolve()
    else:
        input_csv = input_csv.resolve()

    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    label = percent_label(args.keep_percent)

    if args.output_prefix is None:
        output_base = input_csv.with_name(f"{input_csv.stem}_keep{label}_hold")
    else:
        output_base = Path(args.output_prefix).expanduser()
        if not output_base.is_absolute():
            output_base = (project_root / "reinforcement_learning" / output_base).resolve()

    cut_csv = output_base.with_suffix(".csv")
    flipped_csv = output_base.with_name(output_base.name + "_flipped_180").with_suffix(".csv")

    print("=== Creating cut trajectory ===")
    print(f"Input CSV:      {input_csv}")
    print(f"Keep percent:   {args.keep_percent}%")
    print(f"Hold seconds:   {args.hold_seconds}")
    print(f"Cut CSV:        {cut_csv}")

    stats = cut_trajectory(input_csv, cut_csv, args.keep_percent, args.hold_seconds)

    print("")
    print("Cut stats:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    if args.no_flip:
        final_csv = cut_csv
        print("")
        print("Skipping flip because --no-flip was used.")
    else:
        print("")
        print("=== Creating flipped trajectory ===")
        print(f"Flipped CSV:    {flipped_csv}")
        flip_trajectory(cut_csv, flipped_csv)
        final_csv = flipped_csv

    if args.no_update_scripts:
        print("")
        print("Skipping script updates because --no-update-scripts was used.")
    else:
        print("")
        print("=== Updating robot scripts ===")
        for rel_script in SENDER_SCRIPTS:
            update_sender_script(project_root / rel_script, final_csv)

    print("")
    print("Done.")
    print(f"Use this CSV: {final_csv}")
    print("")
    print("Recommended next check:")
    print(f'  grep -n "CSV_PATH\\|exported_trajectory" {project_root}/reinforcement_learning/send_csv_trajectory_safe.py')
    print("")
    print("Then run:")
    print("  python3 move_to_whip_start_pose.py")
    print("  python3 send_csv_trajectory_safe.py")


if __name__ == "__main__":
    main()
