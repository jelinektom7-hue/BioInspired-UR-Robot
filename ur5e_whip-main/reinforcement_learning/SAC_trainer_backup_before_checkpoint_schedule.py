import gymnasium as gym
import gymnasium_robotics
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
import os
import gymnasium_env


MODEL_SAVE_FOLDER = os.path.expanduser("~") + "/ros2_ws/src/BioInspired-UR-Robot/ur5e_whip-main/reinforcement_learning/SAC_models_my_run/"
TENSOR_LOG_FOLDER = os.path.expanduser("~") + "/ros2_ws/src/BioInspired-UR-Robot/ur5e_whip-main/reinforcement_learning/SAC_tensorboard/"
ENVIRONMENT = "gymnasium_env/WhipWorld-v0"

# 80-segment rope + randomized target training is slower than the old fixed-target setup.
# Save more often than before so you can visually inspect checkpoints early.
CHECKPOINT_SAVE_FREQ = 1_000
TOTAL_TIMESTEPS = 2_000_000


def main():
    os.makedirs(MODEL_SAVE_FOLDER, exist_ok=True)
    os.makedirs(TENSOR_LOG_FOLDER, exist_ok=True)

    print("=== SAC target-conditioned training ===")
    print(f"Environment:        {ENVIRONMENT}")
    print(f"Model save folder:  {MODEL_SAVE_FOLDER}")
    print(f"TensorBoard folder: {TENSOR_LOG_FOLDER}")
    print(f"Checkpoint freq:    {CHECKPOINT_SAVE_FREQ} timesteps")
    print(f"Total timesteps:    {TOTAL_TIMESTEPS}")
    print("======================================")

    env = gym.make(ENVIRONMENT)

    model = SAC(
        policy="MultiInputPolicy",
        env=env,
        verbose=1,
        tensorboard_log=TENSOR_LOG_FOLDER,
        learning_starts=10_000,
        batch_size=256,
        buffer_size=300_000,
        train_freq=1,
        gradient_steps=1,
        gamma=0.99,
        tau=0.005,
    )

    # To continue training from an existing model, uncomment and edit this block.
    """MODEL_TO_LOAD = os.path.expanduser("~") + "/ros2_ws/src/BioInspired-UR-Robot/ur5e_whip-main/reinforcement_learning/SAC_models_my_run/rl_model_100000_steps.zip"

    model = SAC.load(
        MODEL_TO_LOAD,
        env=env,
        tensorboard_log=TENSOR_LOG_FOLDER,
        device="auto",
    )"""

    checkpoint_callback = CheckpointCallback(
        save_freq=CHECKPOINT_SAVE_FREQ,
        save_path=MODEL_SAVE_FOLDER,
        name_prefix="rl_model"
    )

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        log_interval=10,
        callback=[checkpoint_callback],
        # reset_num_timesteps=False,
    )

    model.save(os.path.join(MODEL_SAVE_FOLDER, "final_model"))
    env.close()


if __name__ == "__main__":
    main()
