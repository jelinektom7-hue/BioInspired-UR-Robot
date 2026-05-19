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


def resolve_path(path_str):
    p = Path(path_str).expanduser()
    if p.is_absolute():
        return p
    return PROJECT_ROOT / "reinforcement_learning" / p


def filter_obs_for_teacher(obs, teacher_keys):
    return {k: obs[k] for k in teacher_keys if k in obs}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--x", type=float, required=True)
    parser.add_argument("--y", type=float, required=True)
    parser.add_argument("--z", type=float, required=True)
    parser.add_argument("--output", default="exported_teacher_robot_only.csv")
    parser.add_argument("--summary", default="exported_teacher_robot_only_summary.json")
    args = parser.parse_args()

    teacher_path = resolve_path(args.teacher)
    output_path = resolve_path(args.output)
    summary_path = resolve_path(args.summary)

    print("Loading teacher without env check:")
    print(teacher_path)

    teacher = SAC.load(str(teacher_path), env=None)
    teacher_keys = list(teacher.observation_space.spaces.keys())
    print("Teacher observation keys:", teacher_keys)

    env = gym.make(ENVIRONMENT)

    best_rows = None
    best_summary = None
    best_score = -1e30

    for ep in range(args.episodes):
        obs, info = env.reset(
            seed=ep,
            options={"target_position": [args.x, args.y, args.z]},
        )

        done = False
        rows = []
        total_reward = 0.0
        min_distance = float("inf")
        steps = 0
        had_nonfinite_robot_state = False

        final_info = {}

        while not done:
            teacher_obs = filter_obs_for_teacher(obs, teacher_keys)
            action, _ = teacher.predict(teacher_obs, deterministic=args.deterministic)

            obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)

            q = np.array(obs["agent_joint_values"], dtype=float)

            if not np.all(np.isfinite(q)):
                had_nonfinite_robot_state = True
                break

            elapsed = float(info.get("elapsed time [s]", steps * 0.01))
            distance = float(info.get("distance", np.nan))

            if np.isfinite(distance):
                min_distance = min(min_distance, distance)

            total_reward += float(reward)
            final_info = info

            rows.append([
                elapsed,
                float(q[0]),
                float(q[1]),
                float(q[2]),
                float(q[3]),
                float(q[4]),
                float(q[5]),
                float(reward),
                float(distance) if np.isfinite(distance) else np.nan,
                int(bool(info.get("whip_target_contact", False))),
                int(bool(info.get("valid_side_hit", False))),
                int(bool(info.get("arm_tip_ground_hit", False))),
                int(bool(info.get("robot_ground_contact", False))),
                int(bool(info.get("arm_self_contact", False))),
            ])

            steps += 1

        if not rows or had_nonfinite_robot_state:
            score = -1e30
        else:
            # Robot-only export score:
            # choose the trajectory that got closest, then reward.
            score = -1000.0 * min_distance + 0.01 * total_reward

        if score > best_score:
            best_score = score
            best_rows = rows
            best_summary = {
                "episode": ep,
                "teacher": str(teacher_path),
                "target": [args.x, args.y, args.z],
                "deterministic": bool(args.deterministic),
                "output": str(output_path),
                "min_distance": float(min_distance),
                "total_reward": float(total_reward),
                "steps": int(steps),
                "score": float(score),
                "had_nonfinite_robot_state": bool(had_nonfinite_robot_state),
                "final_info": {k: str(v) for k, v in final_info.items()},
                "note": (
                    "Robot-only export. MuJoCo rope QACC warnings are not used "
                    "to reject this trajectory. You must visually inspect and "
                    "check UR5e joint speed/acceleration before real execution."
                ),
            }

        print(
            f"episode {ep+1:4d}/{args.episodes} | "
            f"best_min_dist={best_summary['min_distance']:.4f} | "
            f"this_min_dist={min_distance:.4f} | "
            f"steps={steps}"
        )

    env.close()

    if best_rows is None:
        raise RuntimeError("No valid robot joint trajectory was produced.")

    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["elapsed_time"]
            + JOINT_NAMES
            + [
                "reward",
                "distance",
                "whip_target_contact",
                "valid_side_hit",
                "arm_tip_ground_hit",
                "robot_ground_contact",
                "arm_self_contact",
            ]
        )
        writer.writerows(best_rows)

    with summary_path.open("w") as f:
        json.dump(best_summary, f, indent=2)

    print("")
    print("Exported robot-only trajectory:")
    print(output_path)
    print("")
    print("Summary:")
    print(json.dumps(best_summary, indent=2))


if __name__ == "__main__":
    main()
