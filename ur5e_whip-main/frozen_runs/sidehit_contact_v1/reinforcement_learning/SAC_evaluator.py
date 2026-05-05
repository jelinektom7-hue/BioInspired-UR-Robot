import gymnasium as gym
import gymnasium_robotics  # Needed to register FetchReach
from stable_baselines3 import SAC
import time
import os
import gymnasium_env


ENVIRONMENT = 'gymnasium_env/WhipWorld-v0'

MODEL_FILE = os.path.expanduser("~") + "/ros2_ws/src/BioInspired-UR-Robot/ur5e_whip-main/reinforcement_learning/SAC_models_my_run/rl_model_970000_steps.zip"


# Load environment
env = gym.make(ENVIRONMENT, render_mode="human")

# Load trained SAC model
model = SAC.load(MODEL_FILE)

# Run evaluation
EPISODES = 20

for ep in range(EPISODES):
    obs, info = env.reset()
    done = False
    total_reward = 0

    print(f"\nEpisode {ep+1}")
    while not done:
        action, info = model.predict(obs, deterministic=False)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward

    print(f"Total Reward: {total_reward:.2f}")

env.close()