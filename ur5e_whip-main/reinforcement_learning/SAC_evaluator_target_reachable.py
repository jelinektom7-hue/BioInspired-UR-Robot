#!/usr/bin/env python3

import argparse
import re
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import SAC

import gymnasium_env


PROJECT_ROOT = Path.home() / "ros2_ws/src/BioInspired-UR-Robot/ur5e_whip-main"

ENVIRONMENT = "gymnasium_env/WhipWorld-v0"

DEFAULT_MODEL_DIR = (
    PROJECT_ROOT
    / "reinforcement_learning/SAC_models_target_reachable_50cm_cube"
)


def checkpoint_step(path: Path) -> int:
    match = re.search(r"rl_model_(\d+)_steps\.zip", path.name)
    if match:
        return int(match.group(1))
    return -1


def find_latest_model(model_dir: Path) -> Path:
    final_model = model_dir / "final_model.zip"

    if final_model.exists():
        return final_model

    checkpoints = sorted(
        model_dir.glob("rl_model_*_steps.zip"),
        key=checkpoint_step,
    )

    if not checkpoints:
        raise FileNotFoundError(
            f"No checkpoints found in {model_dir}\n"
            "Expected files like rl_model_10000_steps.zip"
        )

    return checkpoints[-1]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visual SAC checkpoint evaluator for target-reachable whip policy."
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=(
            "Path to checkpoint/model zip. "
            "If omitted, uses final_model.zip if available, otherwise latest rl_model_*_steps.zip."
        ),
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of visual episodes to run.",
    )

    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic SAC actions.",
    )

    parser.add_argument(
        "--x",
        type=float,
        default=None,
        help="Fixed sim target x. If omitted, environment randomizes target.",
    )

    parser.add_argument(
        "--y",
        type=float,
        default=None,
        help="Fixed sim target y. If omitted, environment randomizes target.",
    )

    parser.add_argument(
        "--z",
        type=float,
        default=None,
        help="Fixed sim target z. If omitted, environment randomizes target.",
    )

    parser.add_argument(
        "--pause-between",
        type=float,
        default=0.0,
        help="Pause between episodes, seconds.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.model is None:
        model_path = find_latest_model(DEFAULT_MODEL_DIR)
    else:
        model_path = Path(args.model).expanduser()

        if not model_path.is_absolute():
            model_path = PROJECT_ROOT / "reinforcement_learning" / model_path

    if not model_path.exists():
        raise FileNotFoundError(f"Could not find model: {model_path}")

    fixed_target_requested = (
        args.x is not None
        or args.y is not None
        or args.z is not None
    )

    if fixed_target_requested and not (
        args.x is not None and args.y is not None and args.z is not None
    ):
        raise ValueError("For fixed target mode, provide all three: --x --y --z")

    print("\n========== SAC VISUAL EVALUATOR ==========")
    print(f"Environment:   {ENVIRONMENT}")
    print(f"Model:         {model_path}")
    print(f"Episodes:      {args.episodes}")
    print(f"Deterministic: {args.deterministic}")

    if fixed_target_requested:
        print(f"Fixed target:  x={args.x:.3f}, y={args.y:.3f}, z={args.z:.3f}")
    else:
        print("Target mode:   randomized by environment reset()")

    print("==========================================\n")

    env = gym.make(ENVIRONMENT, render_mode="human")
    model = SAC.load(str(model_path), env=env)

    for ep in range(args.episodes):
        if fixed_target_requested:
            obs, info = env.reset(
                options={
                    "target_position": [args.x, args.y, args.z],
                }
            )
        else:
            obs, info = env.reset(seed=ep)

        done = False
        total_reward = 0.0
        steps = 0

        print(f"\nEpisode {ep + 1}/{args.episodes}")

        tx = info.get("target_x", None)
        ty = info.get("target_y", None)
        tz = info.get("target_z", None)

        if tx is not None:
            print(f"Target: x={tx:.3f}, y={ty:.3f}, z={tz:.3f}")

        while not done:
            action, _ = model.predict(obs, deterministic=args.deterministic)
            obs, reward, terminated, truncated, info = env.step(action)

            total_reward += float(reward)
            steps += 1
            done = bool(terminated or truncated)

        print(f"Steps:        {steps}")
        print(f"Total reward: {total_reward:.2f}")
        print(f"Terminated:   {terminated}")
        print(f"Truncated:    {truncated}")

        print("Final info:")
        for key in [
            "elapsed time [s]",
            "distance",
            "whip_target_contact",
            "valid_side_hit",
            "bad_top_down_hit",
            "arm_tip_ground_hit",
            "robot_ground_contact",
            "arm_self_contact",
            "target_x",
            "target_y",
            "target_z",
            "target_radius_xy",
            "target_inside_training_region",
        ]:
            if key in info:
                print(f"  {key}: {info[key]}")

    env.close()


if __name__ == "__main__":
    main()
