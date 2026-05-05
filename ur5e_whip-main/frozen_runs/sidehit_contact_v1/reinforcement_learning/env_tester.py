import gymnasium as gym
import numpy as np
import gymnasium_env


ENVIRONMENT = 'gymnasium_env/WhipWorld-v0'


env = gym.make(ENVIRONMENT, render_mode="human")
observaton, info = env.reset()

for i in range (100_000):

    # action = env.action_space.sample()
    action = [0.0, 0, 0, 0, 0, 0]
    # action = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    # action = [1.0, -np.pi*3/4, np.pi/2, 0.0, 0.0, 0.0]

    observaton, reward, terminated, truncated, info = env.step(action)

    if terminated:
        print("reset: terminated")
        env.reset()
    elif truncated:
        print("reset: truncated")
        env.reset()

    if i % 100 == 0:
        print(
            "reward:", reward, "\n" \
            # "action:", action, "\n" \
            # "observation:", observaton, "\n" \
            # "info:" , info
            )

env.close()

"""import gymnasium as gym
import gymnasium_env
import time

ENVIRONMENT = "gymnasium_env/WhipWorld-v0"

env = gym.make(ENVIRONMENT, render_mode="human")
observation, info = env.reset()

print("Initial observation:", observation)
print("Initial info:", info)
print("Viewer is open. Press Ctrl+C when done.")

while True:
    env.render()
    time.sleep(0.05)"""
