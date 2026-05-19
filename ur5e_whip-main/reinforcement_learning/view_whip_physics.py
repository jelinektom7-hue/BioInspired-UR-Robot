#!/usr/bin/env python3
"""
View the whip/rope physics without training.

Edit the rope constants at the top of whip_world.py, then run this script again.
It opens MuJoCo and applies a simple repeatable robot motion so you can judge
whether the whip is too stiff/soft before starting SAC training.
"""

import argparse
import math
import time

import gymnasium as gym
import numpy as np

import gymnasium_env


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-x", type=float, default=1.10)
    parser.add_argument("--target-y", type=float, default=-0.45)
    parser.add_argument("--target-z", type=float, default=0.80)
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument("--mode", choices=["settle", "wiggle", "whip"], default="wiggle")
    parser.add_argument("--sleep", type=float, default=0.0, help="Optional wall-clock delay per policy step.")
    return parser.parse_args()


def action_for_mode(mode: str, t: float):
    action = np.zeros(6, dtype=np.float64)

    if mode == "settle":
        return action

    if mode == "wiggle":
        # Small periodic motion to excite rope flexibility without training.
        action[0] = 0.010 * math.sin(2.0 * math.pi * 0.60 * t)
        action[1] = 0.006 * math.sin(2.0 * math.pi * 0.45 * t + 0.7)
        action[2] = 0.006 * math.sin(2.0 * math.pi * 0.55 * t + 1.2)
        action[3] = 0.010 * math.sin(2.0 * math.pi * 0.80 * t)
        action[4] = 0.010 * math.sin(2.0 * math.pi * 0.75 * t + 1.0)
        return action

    if mode == "whip":
        # One stronger open-loop pulse. This is only for visual physics testing.
        if 0.25 < t < 0.70:
            action[0] = 0.020
            action[1] = -0.020
            action[2] = 0.015
            action[3] = -0.018
            action[4] = 0.012
        elif 0.70 <= t < 1.05:
            action[0] = -0.018
            action[1] = 0.014
            action[2] = -0.012
            action[3] = 0.018
            action[4] = -0.010
        return action

    return action


def main():
    args = parse_args()

    env = gym.make("gymnasium_env/WhipWorld-v0", render_mode="human")

    try:
        obs, info = env.reset(
            options={"target_position": [args.target_x, args.target_y, args.target_z]}
        )
    except Exception:
        # Older/fixed-target env versions may not support reset options.
        obs, info = env.reset()

    sim_t = 0.0
    step_dt = 0.01
    max_steps = int(args.seconds / step_dt)

    print("Physics viewer running.")
    print(f"mode={args.mode}, seconds={args.seconds}")
    print("Close the MuJoCo viewer window or press Ctrl+C to stop.")

    for step in range(max_steps):
        action = action_for_mode(args.mode, sim_t)
        obs, reward, terminated, truncated, info = env.step(action)
        sim_t = float(info.get("elapsed time [s]", sim_t + step_dt))

        if step % 50 == 0:
            print(
                f"t={sim_t:5.2f}s  "
                f"dist={float(info.get('distance', float('nan'))):.3f}  "
                f"whip_z={float(info.get('whip_z', float('nan'))):.3f}"
            )

        if args.sleep > 0.0:
            time.sleep(args.sleep)

        if terminated or truncated:
            # Reset and keep viewing instead of exiting immediately.
            try:
                obs, info = env.reset(
                    options={"target_position": [args.target_x, args.target_y, args.target_z]}
                )
            except Exception:
                obs, info = env.reset()
            sim_t = 0.0

    env.close()


if __name__ == "__main__":
    main()
