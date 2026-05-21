import os
from pathlib import Path

import gymnasium as gym
import imageio
import mujoco
from stable_baselines3 import SAC

import gymnasium_env


PROJECT_ROOT = Path.home() / "ros2_ws/src/BioInspired-UR-Robot/ur5e_whip-main"

ENVIRONMENT = "gymnasium_env/WhipWorld-v0"

MODEL_FILE = (
    PROJECT_ROOT
    / "frozen_runs/sidehit_contact_v1/checkpoints/rl_model_970000_steps.zip"
)

OUTPUT_VIDEO = (
    PROJECT_ROOT
    / "reinforcement_learning"
    / "fixed_target_success_run.mp4"
)

CAMERA_NAME = "fixed"

WIDTH = 1280
HEIGHT = 720
FPS = 60

DETERMINISTIC = True


def main():
    if not MODEL_FILE.exists():
        raise FileNotFoundError(f"Could not find model: {MODEL_FILE}")

    env = gym.make(ENVIRONMENT)
    model = SAC.load(str(MODEL_FILE), env=env)

    base_env = env.unwrapped

    renderer = mujoco.Renderer(
        base_env.model,
        height=HEIGHT,
        width=WIDTH,
    )

    obs, info = env.reset()

    frames = []
    done = False
    total_reward = 0.0
    step_count = 0

    # Record initial frame
    renderer.update_scene(base_env.data, camera=CAMERA_NAME)
    frames.append(renderer.render())

    while not done:
        action, _ = model.predict(obs, deterministic=DETERMINISTIC)

        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += float(reward)
        step_count += 1
        done = terminated or truncated

        renderer.update_scene(base_env.data, camera=CAMERA_NAME)
        frames.append(renderer.render())

    imageio.mimsave(
        OUTPUT_VIDEO,
        frames,
        fps=FPS,
        quality=8,
    )

    print("Saved video:")
    print(OUTPUT_VIDEO)
    print(f"Steps: {step_count}")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Terminated: {terminated}")
    print(f"Truncated: {truncated}")
    print(f"Final info: {info}")

    env.close()


if __name__ == "__main__":
    main()