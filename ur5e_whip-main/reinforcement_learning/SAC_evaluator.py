#!/usr/bin/env python3

import argparse
import re
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import SAC

import gymnasium_env


PROJECT_ROOT = Path.home() / "ros2_ws/src/BioInspired-UR-Robot/ur5e_whip-main"
ENVIRONMENT = "gymnasium_env/WhipWorld-v0"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "reinforcement_learning/SAC_models_my_run"


def checkpoint_step(path: Path) -> int:
    match = re.search(r"rl_model_(\d+)_steps\.zip", path.name)
    return int(match.group(1)) if match else -1


def find_latest_model(model_dir: Path) -> Path:
    final_model = model_dir / "final_model.zip"
    if final_model.exists():
        return final_model

    checkpoints = sorted(model_dir.glob("rl_model_*_steps.zip"), key=checkpoint_step)
    if not checkpoints:
        raise FileNotFoundError(
            f"No checkpoints found in {model_dir}\n"
            "Expected files like rl_model_1000_steps.zip. If training just started, wait until the first save_freq interval."
        )
    return checkpoints[-1]


def resolve_model_path(model_arg: str | None) -> Path:
    if model_arg is None:
        return find_latest_model(DEFAULT_MODEL_DIR)

    model_path = Path(model_arg).expanduser()
    if not model_path.is_absolute():
        # Accept either "rl_model_1000_steps.zip" or "SAC_models_my_run/rl_model_1000_steps.zip".
        candidate_a = DEFAULT_MODEL_DIR / model_path
        candidate_b = PROJECT_ROOT / "reinforcement_learning" / model_path
        if candidate_a.exists():
            model_path = candidate_a
        else:
            model_path = candidate_b
    return model_path


def parse_args():
    parser = argparse.ArgumentParser(description="Visual SAC checkpoint evaluator for the current WhipWorld-v0 environment.")
    parser.add_argument("--model", type=str, default=None, help="Checkpoint/model zip. Default: final_model.zip if present, else latest rl_model_*_steps.zip from SAC_models_my_run.")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic SAC actions.")
    parser.add_argument("--x", type=float, default=None, help="Fixed SIM target x [m]. If omitted, target is randomized.")
    parser.add_argument("--y", type=float, default=None, help="Fixed SIM target y [m]. If omitted, target is randomized.")
    parser.add_argument("--z", type=float, default=None, help="Fixed SIM target z [m]. If omitted, target is randomized.")
    return parser.parse_args()


def main():
    args = parse_args()
    model_path = resolve_model_path(args.model)

    if not model_path.exists():
        raise FileNotFoundError(f"Could not find model: {model_path}")

    fixed_target = args.x is not None or args.y is not None or args.z is not None
    if fixed_target and not (args.x is not None and args.y is not None and args.z is not None):
        raise ValueError("For fixed target mode, provide all three: --x --y --z")

    print("\n========== SAC VISUAL EVALUATOR ==========")
    print(f"Environment:   {ENVIRONMENT}")
    print(f"Model:         {model_path}")
    print(f"Episodes:      {args.episodes}")
    print(f"Deterministic: {args.deterministic}")
    if fixed_target:
        print(f"Fixed target:  x={args.x:.3f}, y={args.y:.3f}, z={args.z:.3f}")
    else:
        print("Target mode:   randomized by environment reset()")
    print("==========================================\n")

    env = gym.make(ENVIRONMENT, render_mode="human")
    model = SAC.load(str(model_path), env=env)

    for ep in range(args.episodes):
        if fixed_target:
            obs, info = env.reset(options={"target_position": [args.x, args.y, args.z]})
        else:
            obs, info = env.reset(seed=ep)

        done = False
        total_reward = 0.0
        steps = 0
        terminated = False
        truncated = False

        print(f"\nEpisode {ep + 1}/{args.episodes}")
        if "target_x" in info:
            print(f"Target: x={info['target_x']:.3f}, y={info['target_y']:.3f}, z={info['target_z']:.3f}")

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
            "elapsed time [s]", "distance", "whip_target_contact", "valid_side_hit",
            "bad_top_down_hit", "arm_tip_ground_hit", "robot_ground_contact", "arm_self_contact",
            "target_x", "target_y", "target_z", "target_radius_xy", "target_inside_training_region",
            "max_command_velocity", "max_command_acceleration", "max_command_jerk",
            "active_joint_count", "shoulder_speed_fraction",
        ]:
            if key in info:
                print(f"  {key}: {info[key]}")

    env.close()


if __name__ == "__main__":
    main()
