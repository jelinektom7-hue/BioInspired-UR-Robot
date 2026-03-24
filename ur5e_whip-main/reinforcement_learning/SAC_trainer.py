import gymnasium as gym
import gymnasium_robotics
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
import os
import gymnasium_env


# For training a new model
MODEL_SAVE_FOLDER = os.path.expanduser("~") + "/ros2_ws/src/ur5e_whip/reinforcement_learning/SAC_models_09_whip_02/"
TENSOR_LOG_FOLDER = os.path.expanduser("~") +   "/ros2_ws/src/ur5e_whip/reinforcement_learning/SAC_tensorboard/"

# For loading a pretrained model
MODEL_TO_LOAD = os.path.expanduser("~") + "/ros2_ws/src/ur5e_whip/reinforcement_learning/SAC_models_09_whip_02/l_model_710000_steps"

ENVIRONMENT = 'gymnasium_env/WhipWorld-v0'


def main():
    # Ensure checkpoint directory exists
    os.makedirs(MODEL_SAVE_FOLDER, exist_ok=True)

    # Create the environment
    env = gym.make(ENVIRONMENT)

    # Create a new model
    # model = SAC(
    #     policy="MultiInputPolicy",  # For environments with dict obs (like Fetch)
    #     env=env,
    #     verbose=1,
    #     tensorboard_log=TENSOR_LOG_FOLDER
    # )

    # Load a pretrained model
    model = SAC.load(MODEL_TO_LOAD, env=env)

    # Callback to save model every 'save_freq' steps
    checkpointCallback = CheckpointCallback(save_freq=10_000, save_path=MODEL_SAVE_FOLDER)

    # Continue training
    model.learn(
        total_timesteps=100_000_000,
        log_interval=1,
        callback=[checkpointCallback],
    )


if __name__ == "__main__":
    main()