import os
import csv
import gymnasium as gym
from stable_baselines3 import SAC
import gymnasium_env


ENVIRONMENT = "gymnasium_env/WhipWorld-v0"

MODEL_FILE = os.path.expanduser(
    "~/ros2_ws/src/BioInspired-UR-Robot/ur5e_whip-main/reinforcement_learning/"
    "SAC_models_my_run/rl_model_970000_steps.zip"
)

SAVE_FILE = os.path.expanduser(
    "~/ros2_ws/src/BioInspired-UR-Robot/ur5e_whip-main/reinforcement_learning/"
    "exported_trajectory.csv"
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
    # IMPORTANT:
    # Do not use render_mode="human" here.
    # The viewer/OpenGL cleanup can segfault after saving.
    env = gym.make(ENVIRONMENT)

    model = SAC.load(MODEL_FILE, env=env)

    obs, info = env.reset()
    done = False

    rows = []
    step_index = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        joint_values = obs["agent_joint_values"]
        elapsed = info.get("elapsed time [s]", step_index * 0.01)

        rows.append([
            float(joint_values[0]),
            float(joint_values[1]),
            float(joint_values[2]),
            float(joint_values[3]),
            float(joint_values[4]),
            float(joint_values[5]),
            float(elapsed),
        ])

        step_index += 1

    with open(SAVE_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(JOINT_NAMES + ["elapsed_time"])
        writer.writerows(rows)

    print(f"Saved trajectory with {len(rows)} points:")
    print(SAVE_FILE)

    env.close()


if __name__ == "__main__":
    main()