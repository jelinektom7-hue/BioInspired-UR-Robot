import os
import csv
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import SAC

import gymnasium_env


PROJECT_ROOT = Path.home() / "ros2_ws/src/BioInspired-UR-Robot/ur5e_whip-main"

ENVIRONMENT = "gymnasium_env/WhipWorld-v0"

MODEL_FILE = (
    PROJECT_ROOT
    / "frozen_runs/sidehit_contact_v1/checkpoints/rl_model_970000_steps.zip"
)

SAVE_FILE = (
    PROJECT_ROOT
    / "reinforcement_learning/exported_trajectory_best_fixed_target.csv"
)

JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]


def main():
    if not MODEL_FILE.exists():
        raise FileNotFoundError(f"Could not find checkpoint: {MODEL_FILE}")

    env = gym.make(ENVIRONMENT)
    model = SAC.load(str(MODEL_FILE), env=env)

    obs, info = env.reset()
    done = False

    rows = []
    step_index = 0
    total_reward = 0.0

    while not done:
        action, _ = model.predict(obs, deterministic=True)

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += float(reward)

        joint_values = obs["agent_joint_values"]

        elapsed = info.get("elapsed time [s]", step_index * 0.01)

        rows.append([
            float(elapsed),
            float(joint_values[0]),
            float(joint_values[1]),
            float(joint_values[2]),
            float(joint_values[3]),
            float(joint_values[4]),
            float(joint_values[5]),
            float(reward),
            int(terminated),
            int(truncated),
        ])

        step_index += 1

    with open(SAVE_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["elapsed_time"] + JOINT_NAMES + ["reward", "terminated", "truncated"]
        )
        writer.writerows(rows)

    print("Export complete.")
    print(f"Saved trajectory with {len(rows)} points:")
    print(SAVE_FILE)
    print(f"Total reward: {total_reward:.2f}")
    print(f"Final info: {info}")
    print(f"Terminated: {terminated}")
    print(f"Truncated: {truncated}")

    env.close()


if __name__ == "__main__":
    main()