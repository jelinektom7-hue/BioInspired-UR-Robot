#!/usr/bin/env python3

import argparse
import csv
import math
import re
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC

import gymnasium_env


PROJECT_ROOT = Path.home() / "ros2_ws/src/BioInspired-UR-Robot/ur5e_whip-main"
ENVIRONMENT = "gymnasium_env/WhipWorld-v0"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "reinforcement_learning/SAC_models_my_run"
DEFAULT_SAVE_FILE = PROJECT_ROOT / "reinforcement_learning/exported_trajectory_detected_target.csv"

JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

# Must match whip_world.py.
TARGET_CENTER = np.array([1.10, -0.45, 0.80], dtype=np.float64)
TARGET_BOX_SIZE = np.array([0.50, 0.50, 0.50], dtype=np.float64)
TARGET_LOW = TARGET_CENTER - TARGET_BOX_SIZE / 2.0
TARGET_HIGH = TARGET_CENTER + TARGET_BOX_SIZE / 2.0
TARGET_MIN_RADIUS_XY = 0.80
TARGET_MAX_RADIUS_XY = 1.55


def checkpoint_step(path: Path) -> int:
    match = re.search(r"rl_model_(\d+)_steps\.zip", path.name)
    return int(match.group(1)) if match else -1


def find_latest_model(model_dir: Path) -> Path:
    final_model = model_dir / "final_model.zip"
    if final_model.exists():
        return final_model
    checkpoints = sorted(model_dir.glob("rl_model_*_steps.zip"), key=checkpoint_step)
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {model_dir}")
    return checkpoints[-1]


def resolve_model_path(model_arg: str | None) -> Path:
    if model_arg is None:
        return find_latest_model(DEFAULT_MODEL_DIR)
    p = Path(model_arg).expanduser()
    if not p.is_absolute():
        candidate_a = DEFAULT_MODEL_DIR / p
        candidate_b = PROJECT_ROOT / "reinforcement_learning" / p
        p = candidate_a if candidate_a.exists() else candidate_b
    return p


def real_flipped_to_sim_target(real_target):
    real_target = np.array(real_target, dtype=np.float64)
    return np.array([-real_target[0], -real_target[1], real_target[2]], dtype=np.float64)


def target_radius_xy(target):
    target = np.array(target, dtype=np.float64)
    return float(math.sqrt(target[0] ** 2 + target[1] ** 2))


def target_inside_training_region(target):
    target = np.array(target, dtype=np.float64)
    inside_box = bool(np.all(target >= TARGET_LOW - 1e-9) and np.all(target <= TARGET_HIGH + 1e-9))
    radius = target_radius_xy(target)
    inside_radius = TARGET_MIN_RADIUS_XY - 1e-9 <= radius <= TARGET_MAX_RADIUS_XY + 1e-9
    return inside_box and inside_radius


def describe_training_region():
    return (
        f"sim box low={TARGET_LOW.tolist()}, high={TARGET_HIGH.tolist()}, "
        f"radius_xy=[{TARGET_MIN_RADIUS_XY:.3f}, {TARGET_MAX_RADIUS_XY:.3f}]"
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Export a target-conditioned SAC policy rollout to CSV.")
    parser.add_argument("--x", type=float, help="Target x in SIM frame [m].")
    parser.add_argument("--y", type=float, help="Target y in SIM frame [m].")
    parser.add_argument("--z", type=float, help="Target z in SIM frame [m].")
    parser.add_argument("--real-x", type=float, help="Target x in flipped REAL frame [m].")
    parser.add_argument("--real-y", type=float, help="Target y in flipped REAL frame [m].")
    parser.add_argument("--real-z", type=float, help="Target z in flipped REAL frame [m].")
    parser.add_argument("--model", type=str, default=None, help="Model/checkpoint zip. Default: final_model.zip if present, else latest checkpoint from SAC_models_my_run.")
    parser.add_argument("--output", type=Path, default=DEFAULT_SAVE_FILE)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--allow-outside-training-region", action="store_true")
    return parser.parse_args()


def resolve_target(args):
    sim_values = [args.x, args.y, args.z]
    real_values = [args.real_x, args.real_y, args.real_z]
    has_sim = any(v is not None for v in sim_values)
    has_real = any(v is not None for v in real_values)

    if has_sim and has_real:
        raise ValueError("Use either --x/--y/--z OR --real-x/--real-y/--real-z, not both.")
    if has_sim:
        if any(v is None for v in sim_values):
            raise ValueError("Simulation-frame target requires all of --x --y --z.")
        return "sim", np.array(sim_values, dtype=np.float64), None
    if has_real:
        if any(v is None for v in real_values):
            raise ValueError("Real-frame target requires all of --real-x --real-y --real-z.")
        real_target = np.array(real_values, dtype=np.float64)
        return "real_flipped", real_flipped_to_sim_target(real_target), real_target
    raise ValueError("Provide either --x --y --z or --real-x --real-y --real-z.")


def main():
    args = parse_args()
    model_file = resolve_model_path(args.model)
    save_file = args.output.expanduser()

    source, sim_target, real_target = resolve_target(args)
    radius = target_radius_xy(sim_target)
    inside_region = target_inside_training_region(sim_target)

    print("Target source:", source)
    if real_target is not None:
        print(f"Real flipped target: x={real_target[0]:.3f}, y={real_target[1]:.3f}, z={real_target[2]:.3f}")
    print(f"Simulation target: x={sim_target[0]:.3f}, y={sim_target[1]:.3f}, z={sim_target[2]:.3f}, radius_xy={radius:.3f}")
    print("Training region:", describe_training_region())
    print("Inside training region:", inside_region)

    if not inside_region and not args.allow_outside_training_region:
        raise ValueError("Target is outside the trained/reachable target region.")
    if not model_file.exists():
        raise FileNotFoundError(f"Could not find model checkpoint: {model_file}")

    save_file.parent.mkdir(parents=True, exist_ok=True)

    env = gym.make(ENVIRONMENT)
    model = SAC.load(str(model_file), env=env)
    obs, info = env.reset(options={"target_position": sim_target.tolist()})

    done = False
    rows = []
    step_index = 0
    total_reward = 0.0
    terminated = False
    truncated = False

    while not done:
        action, _ = model.predict(obs, deterministic=not args.stochastic)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += float(reward)

        joint_values = obs["agent_joint_values"]
        elapsed = info.get("elapsed time [s]", step_index * 0.01)
        rows.append([
            float(elapsed),
            float(joint_values[0]), float(joint_values[1]), float(joint_values[2]),
            float(joint_values[3]), float(joint_values[4]), float(joint_values[5]),
            float(sim_target[0]), float(sim_target[1]), float(sim_target[2]),
            float(radius), int(inside_region), float(reward), int(terminated), int(truncated),
        ])
        step_index += 1

    with open(save_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["elapsed_time"] + JOINT_NAMES + [
                "target_x", "target_y", "target_z", "target_radius_xy",
                "target_inside_training_region", "reward", "terminated", "truncated",
            ]
        )
        writer.writerows(rows)

    print("Export complete.")
    print(f"Model: {model_file}")
    print(f"Saved trajectory with {len(rows)} points:")
    print(save_file)
    print(f"Total reward: {total_reward:.2f}")
    print(f"Final info: {info}")
    print(f"Terminated: {terminated}")
    print(f"Truncated: {truncated}")

    env.close()


if __name__ == "__main__":
    main()
