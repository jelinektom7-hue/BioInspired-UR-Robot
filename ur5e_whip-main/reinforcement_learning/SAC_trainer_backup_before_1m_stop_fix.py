import gymnasium as gym
import gymnasium_robotics
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
import os
import gymnasium_env


MODEL_SAVE_FOLDER = os.path.expanduser("~") + "/ros2_ws/src/BioInspired-UR-Robot/ur5e_whip-main/reinforcement_learning/SAC_models_my_run/"
TENSOR_LOG_FOLDER = os.path.expanduser("~") + "/ros2_ws/src/BioInspired-UR-Robot/ur5e_whip-main/reinforcement_learning/SAC_tensorboard/"
ENVIRONMENT = "gymnasium_env/WhipWorld-v0"

# Save often at the beginning so you can confirm the pipeline is running,
# then reduce checkpoint frequency to avoid filling the folder with files.
EARLY_CHECKPOINT_UNTIL_STEPS = 10_000
EARLY_CHECKPOINT_SAVE_FREQ = 1_000
NORMAL_CHECKPOINT_SAVE_FREQ = 10_000
TOTAL_TIMESTEPS = 2_000_000


class TwoStageCheckpointCallback(BaseCallback):
    """
    Save checkpoints every 1,000 steps for the first 10,000 steps,
    then every 10,000 steps after that.

    This replaces a single CheckpointCallback because Stable-Baselines3's
    standard callback only supports one fixed save frequency.
    """

    def __init__(self, save_path, name_prefix="rl_model", verbose=1):
        super().__init__(verbose=verbose)
        self.save_path = save_path
        self.name_prefix = name_prefix
        self._last_saved_step = None

    def _init_callback(self) -> None:
        os.makedirs(self.save_path, exist_ok=True)

    def _should_save(self, step: int) -> bool:
        if step <= 0:
            return False

        if step <= EARLY_CHECKPOINT_UNTIL_STEPS:
            return step % EARLY_CHECKPOINT_SAVE_FREQ == 0

        return step % NORMAL_CHECKPOINT_SAVE_FREQ == 0

    def _on_step(self) -> bool:
        step = int(self.num_timesteps)

        if self._should_save(step) and step != self._last_saved_step:
            path = os.path.join(self.save_path, f"{self.name_prefix}_{step}_steps.zip")
            self.model.save(path)
            self._last_saved_step = step

            if self.verbose > 0:
                print(f"Saved checkpoint: {path}")

        return True


def main():
    os.makedirs(MODEL_SAVE_FOLDER, exist_ok=True)
    os.makedirs(TENSOR_LOG_FOLDER, exist_ok=True)

    print("=== SAC target-conditioned training ===")
    print(f"Environment:        {ENVIRONMENT}")
    print(f"Model save folder:  {MODEL_SAVE_FOLDER}")
    print(f"TensorBoard folder: {TENSOR_LOG_FOLDER}")
    print(f"Early checkpoints:  every {EARLY_CHECKPOINT_SAVE_FREQ} steps until {EARLY_CHECKPOINT_UNTIL_STEPS} steps")
    print(f"Normal checkpoints: every {NORMAL_CHECKPOINT_SAVE_FREQ} steps after {EARLY_CHECKPOINT_UNTIL_STEPS} steps")
    print(f"Total timesteps:    {TOTAL_TIMESTEPS}")
    print("Note: SAC learning_starts is 10,000, so checkpoints before 10,000 steps are mostly for confirming saving/pipeline health, not judging learned behavior.")
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

    checkpoint_callback = TwoStageCheckpointCallback(
        save_path=MODEL_SAVE_FOLDER,
        name_prefix="rl_model",
        verbose=1,
    )

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        log_interval=10,
        callback=checkpoint_callback,
        # reset_num_timesteps=False,
    )

    model.save(os.path.join(MODEL_SAVE_FOLDER, "final_model"))
    env.close()


if __name__ == "__main__":
    main()
