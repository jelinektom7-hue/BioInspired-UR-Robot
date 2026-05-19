#!/usr/bin/env python3

import csv
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC

import gymnasium_env


PROJECT_ROOT = Path.home() / "ros2_ws/src/BioInspired-UR-Robot/ur5e_whip-main"

MODEL_FILE = (
    PROJECT_ROOT
    / "reinforcement_learning/SAC_models_target_reachable_fast_static/final_model.zip"
)

OUT_DIR = PROJECT_ROOT / "reinforcement_learning/metrics_target_reachable_fast_static"
OUT_CSV = OUT_DIR / "episode_metrics.csv"
OUT_JSON = OUT_DIR / "summary_metrics.json"

ENVIRONMENT = "gymnasium_env/WhipWorld-v0"

N_EPISODES = 100
DETERMINISTIC = True


def get_env_attr(env, name, default=None):
    try:
        return getattr(env.unwrapped, name, default)
    except Exception:
        return default


def get_info_value(info, key, fallback=None):
    return info.get(key, fallback)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not MODEL_FILE.exists():
        raise FileNotFoundError(f"Could not find model checkpoint: {MODEL_FILE}")

    env = gym.make(ENVIRONMENT)
    model = SAC.load(str(MODEL_FILE), env=env)

    episode_rows = []

    for ep in range(N_EPISODES):
        # The environment itself randomizes target position at reset().
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
            action, _ = model.predict(obs, deterministic=DETERMINISTIC)
            obs, reward, terminated, truncated, info = env.step(action)

            steps += 1
            total_reward += float(reward)

            distance = get_info_value(
                info,
                "distance",
                get_env_attr(env, "_distance_to_target", np.nan),
            )

            if distance is not None and np.isfinite(distance):
                min_distance = min(min_distance, float(distance))
                final_distance = float(distance)

            elapsed_time = get_info_value(info, "elapsed time [s]", steps * 0.01)
            final_elapsed_time = float(elapsed_time)

            whip_target_contact = bool(
                get_info_value(info, "whip_target_contact", get_env_attr(env, "_whip_target_contact", False))
            )
            valid_side_hit = bool(
                get_info_value(info, "valid_side_hit", get_env_attr(env, "_valid_side_hit", False))
            )
            arm_tip_ground_hit = bool(
                get_info_value(info, "arm_tip_ground_hit", get_env_attr(env, "_arm_tip_ground_hit", False))
            )
            arm_self_contact = bool(
                get_info_value(info, "arm_self_contact", get_env_attr(env, "_arm_self_contact", False))
            )
            robot_ground_contact = bool(
                get_info_value(info, "robot_ground_contact", get_env_attr(env, "_robot_ground_contact", False))
            )
            bad_top_down_hit = bool(
                get_info_value(info, "bad_top_down_hit", get_env_attr(env, "_bad_top_down_hit", False))
            )

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
            "target_radius_xy": float(get_info_value(info, "target_radius_xy", np.nan)),
            "target_inside_training_region": int(bool(get_info_value(info, "target_inside_training_region", True))),
            "max_command_velocity": max_command_velocity,
            "max_command_acceleration": max_command_acceleration,
            "max_command_jerk": max_command_jerk,
            "max_measured_joint_velocity": max_measured_joint_velocity,
            "min_active_joint_count": min_active_joint_count,
            "max_shoulder_speed_fraction": max_shoulder_speed_fraction,
        }

        episode_rows.append(row)

        print(
            f"Episode {ep + 1:03d}/{N_EPISODES} | "
            f"outcome={outcome:22s} | "
            f"reward={total_reward:9.2f} | "
            f"steps={steps:4d} | "
            f"min_dist={min_distance:.4f} | "
            f"target=({float(target_x):.3f}, {float(target_y):.3f}, {float(target_z):.3f})"
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
        "model_file": str(MODEL_FILE),
        "episodes": n,
        "target_box_center": [1.30, -0.50, 0.80],
        "target_box_size_m": [0.50, 0.50, 0.50],
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
        "mean_max_measured_joint_velocity": mean_of("max_measured_joint_velocity"),
        "mean_max_shoulder_speed_fraction": mean_of("max_shoulder_speed_fraction"),
        "outcome_counts": outcome_counts,
    }

    with open(OUT_JSON, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n================ SUMMARY ================")
    print(f"Episodes:                  {summary['episodes']}")
    print(f"Contact success rate:       {summary['contact_success_rate'] * 100:.1f}%")
    print(f"Valid side-hit rate:        {summary['valid_side_hit_rate'] * 100:.1f}%")
    print(f"Bad top-down hit rate:      {summary['bad_top_down_hit_rate'] * 100:.1f}%")
    print(f"Robot ground contact rate:  {summary['robot_ground_contact_rate'] * 100:.1f}%")
    print(f"Self-contact rate:          {summary['arm_self_contact_rate'] * 100:.1f}%")
    print(f"Truncation rate:            {summary['truncation_rate'] * 100:.1f}%")
    print(f"Mean closest distance:      {summary['mean_min_distance']:.4f} m")
    print(f"Mean final distance:        {summary['mean_final_distance']:.4f} m")
    print(f"Outcome counts:             {summary['outcome_counts']}")
    print("=========================================")

    print(f"\nSaved per-episode metrics to:\n{OUT_CSV}")
    print(f"\nSaved summary metrics to:\n{OUT_JSON}")


if __name__ == "__main__":
    main()
