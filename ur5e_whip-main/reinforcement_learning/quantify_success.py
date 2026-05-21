#!/usr/bin/env python3

import argparse
import csv
import json
import re
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC

import gymnasium_env


PROJECT_ROOT = Path.home() / "ros2_ws/src/BioInspired-UR-Robot/ur5e_whip-main"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "reinforcement_learning/SAC_models_my_run"
OUT_DIR = PROJECT_ROOT / "reinforcement_learning/metrics_my_run"
OUT_CSV = OUT_DIR / "episode_metrics.csv"
OUT_JSON = OUT_DIR / "summary_metrics.json"
ENVIRONMENT = "gymnasium_env/WhipWorld-v0"


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


def get_env_attr(env, name, default=None):
    try:
        return getattr(env.unwrapped, name, default)
    except Exception:
        return default


def get_info_value(info, key, fallback=None):
    return info.get(key, fallback)


def parse_args():
    parser = argparse.ArgumentParser(description="Quantify current SAC target-conditioned checkpoint success.")
    parser.add_argument("--model", type=str, default=None, help="Model/checkpoint zip. Default: final_model.zip if present, else latest checkpoint from SAC_models_my_run.")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--stochastic", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    model_file = resolve_model_path(args.model)
    deterministic = not args.stochastic

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not model_file.exists():
        raise FileNotFoundError(f"Could not find model checkpoint: {model_file}")

    env = gym.make(ENVIRONMENT)
    model = SAC.load(str(model_file), env=env)

    episode_rows = []

    for ep in range(args.episodes):
        obs, info = env.reset(seed=ep)

        target_x = get_info_value(info, "target_x", get_env_attr(env, "_target_position", np.zeros(3))[0])
        target_y = get_info_value(info, "target_y", get_env_attr(env, "_target_position", np.zeros(3))[1])
        target_z = get_info_value(info, "target_z", get_env_attr(env, "_target_position", np.zeros(3))[2])

        done = False
        terminated_final = False
        truncated_final = False
        total_reward = 0.0
        steps = 0
        min_distance = float("inf")
        time_to_first_contact = None
        had_target_contact = False
        had_valid_side_hit = False
        had_arm_tip_ground_hit = False
        had_arm_self_contact = False
        had_robot_ground_contact = False
        had_bad_top_down_hit = False
        final_distance = None
        final_elapsed_time = None
        max_command_velocity = 0.0
        max_command_acceleration = 0.0
        max_command_jerk = 0.0
        max_measured_joint_velocity = 0.0
        min_active_joint_count = 6
        max_shoulder_speed_fraction = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1
            total_reward += float(reward)

            distance = get_info_value(info, "distance", get_env_attr(env, "_distance_to_target", np.nan))
            if distance is not None and np.isfinite(distance):
                min_distance = min(min_distance, float(distance))
                final_distance = float(distance)

            elapsed_time = get_info_value(info, "elapsed time [s]", steps * 0.01)
            final_elapsed_time = float(elapsed_time)

            whip_target_contact = bool(get_info_value(info, "whip_target_contact", get_env_attr(env, "_whip_target_contact", False)))
            valid_side_hit = bool(get_info_value(info, "valid_side_hit", get_env_attr(env, "_valid_side_hit", False)))
            arm_tip_ground_hit = bool(get_info_value(info, "arm_tip_ground_hit", get_env_attr(env, "_arm_tip_ground_hit", False)))
            arm_self_contact = bool(get_info_value(info, "arm_self_contact", get_env_attr(env, "_arm_self_contact", False)))
            robot_ground_contact = bool(get_info_value(info, "robot_ground_contact", get_env_attr(env, "_robot_ground_contact", False)))
            bad_top_down_hit = bool(get_info_value(info, "bad_top_down_hit", get_env_attr(env, "_bad_top_down_hit", False)))

            if whip_target_contact and time_to_first_contact is None:
                time_to_first_contact = final_elapsed_time

            had_target_contact = had_target_contact or whip_target_contact
            had_valid_side_hit = had_valid_side_hit or valid_side_hit
            had_arm_tip_ground_hit = had_arm_tip_ground_hit or arm_tip_ground_hit
            had_arm_self_contact = had_arm_self_contact or arm_self_contact
            had_robot_ground_contact = had_robot_ground_contact or robot_ground_contact
            had_bad_top_down_hit = had_bad_top_down_hit or bad_top_down_hit

            max_command_velocity = max(max_command_velocity, float(get_info_value(info, "max_command_velocity", 0.0)))
            max_command_acceleration = max(max_command_acceleration, float(get_info_value(info, "max_command_acceleration", 0.0)))
            max_command_jerk = max(max_command_jerk, float(get_info_value(info, "max_command_jerk", 0.0)))
            max_measured_joint_velocity = max(max_measured_joint_velocity, float(get_info_value(info, "max_measured_joint_velocity", 0.0)))
            min_active_joint_count = min(min_active_joint_count, int(get_info_value(info, "active_joint_count", 6)))
            max_shoulder_speed_fraction = max(max_shoulder_speed_fraction, float(get_info_value(info, "shoulder_speed_fraction", 0.0)))

            terminated_final = bool(terminated)
            truncated_final = bool(truncated)
            done = terminated or truncated

        if time_to_first_contact is None:
            time_to_first_contact = float("nan")
        if not np.isfinite(min_distance):
            min_distance = float("nan")
        if final_distance is None:
            final_distance = float("nan")
        if final_elapsed_time is None:
            final_elapsed_time = float("nan")

        if had_valid_side_hit:
            outcome = "side_hit"
        elif had_target_contact:
            outcome = "contact_not_side"
        elif had_arm_tip_ground_hit or had_robot_ground_contact:
            outcome = "robot_ground_failure"
        elif had_arm_self_contact:
            outcome = "self_contact_failure"
        elif truncated_final:
            outcome = "timeout_or_miss"
        elif terminated_final:
            outcome = "terminated_other"
        else:
            outcome = "unknown"

        row = {
            "episode": ep,
            "outcome": outcome,
            "total_reward": total_reward,
            "steps": steps,
            "elapsed_time": final_elapsed_time,
            "time_to_first_contact": time_to_first_contact,
            "min_distance": min_distance,
            "final_distance": final_distance,
            "target_contact": int(had_target_contact),
            "valid_side_hit": int(had_valid_side_hit),
            "bad_top_down_hit": int(had_bad_top_down_hit),
            "arm_tip_ground_hit": int(had_arm_tip_ground_hit),
            "robot_ground_contact": int(had_robot_ground_contact),
            "arm_self_contact": int(had_arm_self_contact),
            "terminated": int(terminated_final),
            "truncated": int(truncated_final),
            "target_x": float(target_x),
            "target_y": float(target_y),
            "target_z": float(target_z),
            "max_command_velocity": max_command_velocity,
            "max_command_acceleration": max_command_acceleration,
            "max_command_jerk": max_command_jerk,
            "max_measured_joint_velocity": max_measured_joint_velocity,
            "min_active_joint_count": min_active_joint_count,
            "max_shoulder_speed_fraction": max_shoulder_speed_fraction,
        }
        episode_rows.append(row)

        print(
            f"Episode {ep + 1:03d}/{args.episodes} | "
            f"outcome={outcome:22s} | reward={total_reward:9.2f} | "
            f"steps={steps:4d} | min_dist={min_distance:.4f} | "
            f"target=({target_x:.2f},{target_y:.2f},{target_z:.2f})"
        )

    env.close()

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(episode_rows[0].keys()))
        writer.writeheader()
        writer.writerows(episode_rows)

    n = len(episode_rows)

    def mean_of(key):
        values = [float(r[key]) for r in episode_rows]
        values = [v for v in values if np.isfinite(v)]
        return float(np.mean(values)) if values else float("nan")

    def rate_of(key):
        return float(np.mean([int(r[key]) for r in episode_rows]))

    outcome_counts = {}
    for r in episode_rows:
        outcome_counts[r["outcome"]] = outcome_counts.get(r["outcome"], 0) + 1

    summary = {
        "model_file": str(model_file),
        "episodes": n,
        "deterministic": deterministic,
        "contact_success_rate": rate_of("target_contact"),
        "valid_side_hit_rate": rate_of("valid_side_hit"),
        "bad_top_down_hit_rate": rate_of("bad_top_down_hit"),
        "arm_tip_ground_failure_rate": rate_of("arm_tip_ground_hit"),
        "robot_ground_contact_rate": rate_of("robot_ground_contact"),
        "arm_self_contact_rate": rate_of("arm_self_contact"),
        "truncation_rate": rate_of("truncated"),
        "mean_total_reward": mean_of("total_reward"),
        "mean_steps": mean_of("steps"),
        "mean_elapsed_time": mean_of("elapsed_time"),
        "mean_time_to_first_contact": mean_of("time_to_first_contact"),
        "mean_min_distance": mean_of("min_distance"),
        "mean_final_distance": mean_of("final_distance"),
        "mean_max_command_velocity": mean_of("max_command_velocity"),
        "mean_max_command_acceleration": mean_of("max_command_acceleration"),
        "mean_max_command_jerk": mean_of("max_command_jerk"),
        "mean_max_measured_joint_velocity": mean_of("max_measured_joint_velocity"),
        "mean_min_active_joint_count": mean_of("min_active_joint_count"),
        "mean_max_shoulder_speed_fraction": mean_of("max_shoulder_speed_fraction"),
        "outcome_counts": outcome_counts,
    }

    with open(OUT_JSON, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n========== SUMMARY ==========")
    print(json.dumps(summary, indent=2))
    print(f"\nSaved episode metrics: {OUT_CSV}")
    print(f"Saved summary metrics: {OUT_JSON}")


if __name__ == "__main__":
    main()
