#!/usr/bin/env python3

import argparse
import csv
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC

import gymnasium_env


PROJECT_ROOT = Path.home() / "ros2_ws/src/BioInspired-UR-Robot/ur5e_whip-main"
ENVIRONMENT = "gymnasium_env/WhipWorld-v0"

JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]


def resolve_path(path_str: str) -> Path:
    p = Path(path_str).expanduser()
    if p.is_absolute():
        return p
    return PROJECT_ROOT / "reinforcement_learning" / p


def get_flag(env, info, key, default=False):
    if key in info:
        return bool(info[key])
    return bool(getattr(env.unwrapped, f"_{key}", default))


def make_teacher_obs(full_obs, teacher_keys):
    return {k: full_obs[k] for k in teacher_keys if k in full_obs}


def save_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["elapsed_time"]
            + JOINT_NAMES
            + [
                "reward",
                "terminated",
                "truncated",
                "distance",
                "whip_target_contact",
                "valid_side_hit",
                "arm_tip_ground_hit",
                "robot_ground_contact",
                "arm_self_contact",
            ]
        )
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Export a real-robot CSV from an old SAC checkpoint with incompatible observation keys."
    )

    parser.add_argument(
        "--teacher",
        required=True,
        help="Old checkpoint path, e.g. SAC_19/rl_model_630000_steps.zip",
    )
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--output", default="exported_trajectory_SAC19_630k_best.csv")
    parser.add_argument("--summary", default="exported_trajectory_SAC19_630k_best_summary.json")
    parser.add_argument("--include-close-misses", type=float, default=0.0)

    parser.add_argument("--x", type=float, default=None)
    parser.add_argument("--y", type=float, default=None)
    parser.add_argument("--z", type=float, default=None)

    args = parser.parse_args()

    teacher_path = resolve_path(args.teacher)
    output_path = resolve_path(args.output)
    summary_path = resolve_path(args.summary)

    if not teacher_path.exists():
        raise FileNotFoundError(f"Teacher checkpoint not found: {teacher_path}")

    fixed_target = args.x is not None or args.y is not None or args.z is not None
    if fixed_target and not (args.x is not None and args.y is not None and args.z is not None):
        raise ValueError("For fixed target, provide all three: --x --y --z")

    print("Loading old teacher checkpoint without attaching current env:")
    print(teacher_path)

    teacher = SAC.load(str(teacher_path), env=None)
    teacher_keys = list(teacher.observation_space.spaces.keys())

    print("Teacher observation keys:", teacher_keys)

    env = gym.make(ENVIRONMENT)

    best_score = -1e30
    best_rows = None
    best_summary = None

    accepted_count = 0

    for ep in range(args.episodes):
        if fixed_target:
            obs, info = env.reset(
                seed=ep,
                options={"target_position": [args.x, args.y, args.z]},
            )
        else:
            obs, info = env.reset(seed=ep)

        done = False
        rows = []
        total_reward = 0.0
        min_distance = float("inf")
        steps = 0

        had_contact = False
        had_side_hit = False
        had_arm_tip_ground = False
        had_robot_ground = False
        had_self_contact = False
        terminated_final = False
        truncated_final = False

        while not done:
            teacher_obs = make_teacher_obs(obs, teacher_keys)

            action, _ = teacher.predict(
                teacher_obs,
                deterministic=args.deterministic,
            )

            obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)

            distance = float(info.get("distance", np.nan))
            if np.isfinite(distance):
                min_distance = min(min_distance, distance)

            contact = get_flag(env, info, "whip_target_contact", False)
            side_hit = get_flag(env, info, "valid_side_hit", False)
            arm_tip_ground = get_flag(env, info, "arm_tip_ground_hit", False)
            robot_ground = get_flag(env, info, "robot_ground_contact", False)
            self_contact = get_flag(env, info, "arm_self_contact", False)

            had_contact = had_contact or contact
            had_side_hit = had_side_hit or side_hit
            had_arm_tip_ground = had_arm_tip_ground or arm_tip_ground
            had_robot_ground = had_robot_ground or robot_ground
            had_self_contact = had_self_contact or self_contact

            total_reward += float(reward)
            terminated_final = bool(terminated)
            truncated_final = bool(truncated)

            q = obs["agent_joint_values"]
            elapsed = float(info.get("elapsed time [s]", steps * 0.01))

            rows.append(
                [
                    elapsed,
                    float(q[0]),
                    float(q[1]),
                    float(q[2]),
                    float(q[3]),
                    float(q[4]),
                    float(q[5]),
                    float(reward),
                    int(terminated),
                    int(truncated),
                    float(distance) if np.isfinite(distance) else np.nan,
                    int(contact),
                    int(side_hit),
                    int(arm_tip_ground),
                    int(robot_ground),
                    int(self_contact),
                ]
            )

            steps += 1

        unsafe = had_arm_tip_ground or had_robot_ground or had_self_contact
        safe_contact = had_contact and not unsafe
        safe_close_miss = (
            args.include_close_misses > 0.0
            and min_distance <= args.include_close_misses
            and not unsafe
        )

        accepted = safe_contact or safe_close_miss

        # Strongly prefer safe contact, then side hit, then close distance.
        score = 0.0
        score += 100000.0 if safe_contact else 0.0
        score += 30000.0 if had_side_hit and not unsafe else 0.0
        score += 10000.0 if safe_close_miss else 0.0
        score += -50000.0 if unsafe else 0.0
        score += -1000.0 * min_distance
        score += 0.01 * total_reward

        if accepted:
            accepted_count += 1

        if score > best_score:
            best_score = score
            best_rows = rows
            best_summary = {
                "episode": ep,
                "teacher": str(teacher_path),
                "output": str(output_path),
                "fixed_target": [args.x, args.y, args.z] if fixed_target else None,
                "deterministic": bool(args.deterministic),
                "safe_contact": bool(safe_contact),
                "safe_close_miss": bool(safe_close_miss),
                "had_contact": bool(had_contact),
                "had_side_hit": bool(had_side_hit),
                "unsafe": bool(unsafe),
                "had_arm_tip_ground": bool(had_arm_tip_ground),
                "had_robot_ground": bool(had_robot_ground),
                "had_self_contact": bool(had_self_contact),
                "min_distance": float(min_distance),
                "total_reward": float(total_reward),
                "steps": int(steps),
                "terminated": bool(terminated_final),
                "truncated": bool(truncated_final),
                "score": float(score),
                "accepted_count_so_far": int(accepted_count),
            }

        if (ep + 1) % 25 == 0:
            print(
                f"episode {ep+1:5d}/{args.episodes} | "
                f"accepted={accepted_count:4d} | "
                f"best_score={best_score:10.1f} | "
                f"best_min_dist={best_summary['min_distance']:.4f} | "
                f"best_contact={best_summary['safe_contact']} | "
                f"best_unsafe={best_summary['unsafe']}"
            )

    env.close()

    if best_rows is None:
        raise RuntimeError("No rollout was produced.")

    save_csv(output_path, best_rows)

    with open(summary_path, "w") as f:
        json.dump(best_summary, f, indent=2)

    print("")
    print("Exported best teacher trajectory:")
    print(output_path)
    print("")
    print("Summary:")
    print(json.dumps(best_summary, indent=2))

    if not best_summary["safe_contact"]:
        print("")
        print("WARNING: Best exported rollout was not a safe target contact.")
        print("Do not send this to the real robot unless you visually inspect and accept it.")


if __name__ == "__main__":
    main()
