import gymnasium as gym
import gymnasium_robotics
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
import os
import gymnasium_env


PROJECT_ROOT = os.path.expanduser("~") + "/ros2_ws/src/BioInspired-UR-Robot/ur5e_whip-main"

# Separate folders so the reachable target-conditioned training does not overwrite
# the previous fixed-target or earlier randomized runs.
MODEL_SAVE_FOLDER = PROJECT_ROOT + "/reinforcement_learning/SAC_models_target_reachable_fast_static/"
TENSOR_LOG_FOLDER = PROJECT_ROOT + "/reinforcement_learning/SAC_tensorboard_target_reachable_fast_static/"
ENVIRONMENT = "gymnasium_env/WhipWorld-v0"


def main():
    os.makedirs(MODEL_SAVE_FOLDER, exist_ok=True)
    os.makedirs(TENSOR_LOG_FOLDER, exist_ok=True)

    env = gym.make(ENVIRONMENT)

    model = SAC(
        policy="MultiInputPolicy",
        env=env,
        verbose=1,
        tensorboard_log=TENSOR_LOG_FOLDER,
        learning_starts=20_000,
        batch_size=256,
        buffer_size=300_000,
        train_freq=1,
        gradient_steps=1,
        gamma=0.99,
        tau=0.005,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=10_000,
        save_path=MODEL_SAVE_FOLDER,
        name_prefix="rl_model"
    )

    model.learn(
        total_timesteps=2_000_000,
        log_interval=10,
        callback=[checkpoint_callback],
    )

    model.save(os.path.join(MODEL_SAVE_FOLDER, "final_model"))
    env.close()


if __name__ == "__main__":
    main()
