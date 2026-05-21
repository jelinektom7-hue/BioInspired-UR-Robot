import csv
import json
import os
from pathlib import Path

import gymnasium as gym
import gymnasium_robotics  # Keeps old dependency/import behavior intact.
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback

import gymnasium_env


# ============================================================
# PATHS / BASIC CONFIGURATION
# ============================================================

PROJECT_ROOT = Path.home() / "ros2_ws/src/BioInspired-UR-Robot/ur5e_whip-main"

MODEL_SAVE_FOLDER = str(PROJECT_ROOT / "reinforcement_learning/SAC_models_my_run/")
TENSOR_LOG_FOLDER = str(PROJECT_ROOT / "reinforcement_learning/SAC_tensorboard/")
ENVIRONMENT = "gymnasium_env/WhipWorld-v0"

TOTAL_TIMESTEPS = 1_000_000

# Save often at the beginning so you can confirm training is running,
# then save every 10k steps.
EARLY_CHECKPOINT_UNTIL_STEPS = 10_000
EARLY_CHECKPOINT_SAVE_FREQ = 1_000
NORMAL_CHECKPOINT_SAVE_FREQ = 10_000

# ============================================================
# BEST-MODEL EVALUATION CONFIGURATION
# ============================================================

# Evaluate periodically and save SAC_models_my_run/best_model.zip whenever a
# checkpoint performs better on the same fixed target set.
BEST_MODEL_EVAL_ENABLED = True
BEST_MODEL_EVAL_FREQ = 20_000
BEST_MODEL_N_EVAL_TARGETS = 13
BEST_MODEL_DETERMINISTIC = True

BEST_MODEL_PATH = Path(MODEL_SAVE_FOLDER) / "best_model.zip"
BEST_EVAL_SUMMARY_PATH = Path(MODEL_SAVE_FOLDER) / "best_eval_summary.json"
EVAL_HISTORY_CSV_PATH = Path(MODEL_SAVE_FOLDER) / "eval_history.csv"
BEST_ROLLOUT_CSV_PATH = Path(MODEL_SAVE_FOLDER) / "best_rollout.csv"

# Fixed target set in the SIMULATION frame. These are intentionally deterministic
# so model A and model B are judged on the same target locations.
# Training cube: center [1.10, -0.45, 0.80], size [0.50, 0.50, 0.50].
EVAL_TARGETS = np.array(
    [
        [1.10, -0.45, 0.80],  # center
        [0.85, -0.45, 0.80],  # near x edge, old fixed-hit side included in cube
        [1.35, -0.45, 0.80],  # far x edge
        [1.10, -0.70, 0.80],  # y low edge
        [1.10, -0.20, 0.80],  # y high edge
        [1.10, -0.45, 0.55],  # low z
        [1.10, -0.45, 1.05],  # high z
        [0.90, -0.65, 0.65],
        [0.90, -0.25, 0.95],
        [1.30, -0.65, 0.95],
        [1.30, -0.25, 0.65],
        [1.25, -0.55, 0.75],
        [0.95, -0.35, 0.85],
    ],
    dtype=np.float64,
)

# Score used only for selecting best_model.zip.
# This is a true "high score" system: any evaluation that beats the previous
# score overwrites best_model.zip. Safe hits dominate. Reward is deliberately
# secondary so the evaluator does not select floor-smashing reward hacks.
BEST_SCORE_SAFE_CONTACT_WEIGHT = 12_000.0
BEST_SCORE_SIDE_HIT_WEIGHT = 6_000.0
BEST_SCORE_CLOSE_HIT_WEIGHT = 1_000.0
BEST_SCORE_REWARD_WEIGHT = 0.05
BEST_SCORE_MIN_DISTANCE_WEIGHT = 150.0
BEST_SCORE_UNSAFE_FAILURE_PENALTY = 18_000.0
BEST_SCORE_TOP_DOWN_PENALTY = 2_000.0

# If the target contact sensor is too strict, this still counts as a useful
# close hit for selecting a best model. It does not change the environment reward.
CLOSE_HIT_DISTANCE_M = 0.08


# ============================================================
# CALLBACKS
# ============================================================

class TwoStageCheckpointCallback(BaseCallback):
    """
    Save checkpoints every 1,000 steps for the first 10,000 steps,
    then every 10,000 steps after that.
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


class BestWhipModelCallback(BaseCallback):
    """
    Periodically evaluates the current policy on a fixed target set and saves
    best_model.zip when it performs better than all previous evaluations.

    This is the closest useful equivalent of "remember the best try" for SAC:
    it preserves the best policy network found during training instead of only
    relying on the final checkpoint.
    """

    def __init__(
        self,
        eval_env_id: str,
        eval_targets: np.ndarray,
        eval_freq: int,
        save_dir: str,
        deterministic: bool = True,
        verbose: int = 1,
    ):
        super().__init__(verbose=verbose)
        self.eval_env_id = eval_env_id
        self.eval_targets = np.array(eval_targets, dtype=np.float64)
        self.eval_freq = int(eval_freq)
        self.save_dir = Path(save_dir)
        self.deterministic = bool(deterministic)
        self.best_score = -np.inf
        self.eval_env = None
        self._last_eval_step = None

    def _init_callback(self) -> None:
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.eval_env = gym.make(self.eval_env_id)

        if not EVAL_HISTORY_CSV_PATH.exists():
            with open(EVAL_HISTORY_CSV_PATH, "w", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "timesteps",
                        "score",
                        "safe_contact_rate",
                        "contact_success_rate",
                        "close_hit_rate",
                        "side_hit_rate",
                        "top_down_contact_rate",
                        "unsafe_failure_rate",
                        "failure_rate",
                        "mean_reward",
                        "mean_min_distance",
                        "mean_final_distance",
                        "mean_steps",
                    ],
                )
                writer.writeheader()

    def _on_training_end(self) -> None:
        if self.eval_env is not None:
            self.eval_env.close()
            self.eval_env = None

    def _on_step(self) -> bool:
        step = int(self.num_timesteps)

        if step <= 0:
            return True

        if step % self.eval_freq != 0:
            return True

        if step == self._last_eval_step:
            return True

        self._last_eval_step = step
        summary, best_rollout_rows = self.evaluate_current_policy(step)

        with open(EVAL_HISTORY_CSV_PATH, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
            writer.writerow(summary)

        if self.verbose > 0:
            print(
                "\n========== FIXED-TARGET BEST-MODEL EVAL =========="
                f"\nsteps:                {step}"
                f"\nscore:                {summary['score']:.2f}"
                f"\nsafe_contact_rate:    {summary['safe_contact_rate']:.3f}"
                f"\ncontact_success_rate: {summary['contact_success_rate']:.3f}"
                f"\nclose_hit_rate:       {summary['close_hit_rate']:.3f}"
                f"\nside_hit_rate:        {summary['side_hit_rate']:.3f}"
                f"\ntop_down_rate:        {summary['top_down_contact_rate']:.3f}"
                f"\nunsafe_failure_rate:  {summary['unsafe_failure_rate']:.3f}"
                f"\nmean_reward:          {summary['mean_reward']:.2f}"
                f"\nmean_min_distance:    {summary['mean_min_distance']:.4f} m"
                f"\nbest_score_so_far:    {self.best_score:.2f}"
                "\n==================================================\n"
            )

        if summary["score"] > self.best_score:
            self.best_score = float(summary["score"])
            self.model.save(str(BEST_MODEL_PATH))

            with open(BEST_EVAL_SUMMARY_PATH, "w") as f:
                json.dump(summary, f, indent=2)

            self.write_best_rollout_csv(best_rollout_rows)

            if self.verbose > 0:
                print(f"New best model saved: {BEST_MODEL_PATH}")
                print(f"Best eval summary:    {BEST_EVAL_SUMMARY_PATH}")
                print(f"Best rollout CSV:     {BEST_ROLLOUT_CSV_PATH}\n")

        return True

    def evaluate_current_policy(self, step: int):
        rows = []
        best_rollout_rows = []
        best_episode_reward = -np.inf

        for target_index, target in enumerate(self.eval_targets):
            obs, info = self.eval_env.reset(
                options={"target_position": target.tolist()}
            )

            done = False
            total_reward = 0.0
            episode_rows = []
            min_distance = float("inf")
            final_distance = float("inf")
            steps = 0
            had_contact = False
            had_safe_contact = False
            had_side_hit = False
            had_top_down_contact = False
            had_failure = False
            final_info = {}

            while not done:
                action, _ = self.model.predict(obs, deterministic=self.deterministic)
                obs, reward, terminated, truncated, info = self.eval_env.step(action)

                total_reward += float(reward)
                steps += 1
                final_info = info

                distance = float(info.get("distance", np.nan))
                if np.isfinite(distance):
                    min_distance = min(min_distance, distance)
                    final_distance = distance

                had_contact = bool(
                    had_contact
                    or getattr(self.eval_env.unwrapped, "_whip_target_contact", False)
                )
                had_safe_contact = bool(had_safe_contact or info.get("safe_target_contact", False))
                had_side_hit = bool(had_side_hit or info.get("valid_side_hit", False))
                had_top_down_contact = bool(had_top_down_contact or info.get("top_down_contact", False))
                had_failure = bool(
                    had_failure
                    or info.get("unsafe_robot_failure", False)
                    or info.get("arm_tip_ground_hit", False)
                    or info.get("robot_ground_contact", False)
                    or info.get("arm_self_contact", False)
                )

                joint_values = obs.get("agent_joint_values", np.zeros(6))
                elapsed = float(info.get("elapsed time [s]", steps * 0.01))

                episode_rows.append(
                    {
                        "timesteps": step,
                        "target_index": target_index,
                        "elapsed_time": elapsed,
                        "target_x": float(target[0]),
                        "target_y": float(target[1]),
                        "target_z": float(target[2]),
                        "distance": distance,
                        "reward": float(reward),
                        "shoulder_pan_joint": float(joint_values[0]),
                        "shoulder_lift_joint": float(joint_values[1]),
                        "elbow_joint": float(joint_values[2]),
                        "wrist_1_joint": float(joint_values[3]),
                        "wrist_2_joint": float(joint_values[4]),
                        "wrist_3_joint": float(joint_values[5]),
                        "valid_side_hit": int(info.get("valid_side_hit", False)),
                        "top_down_contact": int(info.get("top_down_contact", False)),
                        "unsafe_robot_failure": int(info.get("unsafe_robot_failure", False)),
                        "safe_target_contact": int(info.get("safe_target_contact", False)),
                        "whip_target_contact": int(had_contact),
                    }
                )

                done = bool(terminated or truncated)

            close_hit = bool(min_distance <= CLOSE_HIT_DISTANCE_M and not had_failure)
            success = bool(had_safe_contact or close_hit)

            rows.append(
                {
                    "target_index": target_index,
                    "target_x": float(target[0]),
                    "target_y": float(target[1]),
                    "target_z": float(target[2]),
                    "total_reward": total_reward,
                    "steps": steps,
                    "min_distance": min_distance,
                    "final_distance": final_distance,
                    "contact_success": int(had_contact),
                    "safe_contact": int(had_safe_contact),
                    "close_hit": int(close_hit),
                    "success": int(success),
                    "side_hit": int(had_side_hit),
                    "top_down_contact": int(had_top_down_contact),
                    "failure": int(had_failure),
                    "terminated": int(final_info.get("terminated", 0)),
                    "truncated": int(final_info.get("truncated", 0)),
                }
            )

            # Store the best-looking safe single rollout from this evaluation cycle.
            # Avoid choosing a floor-smashing rollout just because its reward is high.
            episode_selection_score = (
                10_000.0 * int(had_safe_contact)
                + 5_000.0 * int(had_side_hit)
                + 1_000.0 * int(close_hit)
                - 20_000.0 * int(had_failure)
                - 2_000.0 * int(had_top_down_contact)
                + 0.05 * total_reward
                - 100.0 * min_distance
            )
            if episode_selection_score > best_episode_reward:
                best_episode_reward = episode_selection_score
                best_rollout_rows = episode_rows

        rewards = np.array([r["total_reward"] for r in rows], dtype=np.float64)
        min_distances = np.array([r["min_distance"] for r in rows], dtype=np.float64)
        final_distances = np.array([r["final_distance"] for r in rows], dtype=np.float64)
        contact_success = np.array([r["contact_success"] for r in rows], dtype=np.float64)
        safe_contacts = np.array([r["safe_contact"] for r in rows], dtype=np.float64)
        close_hits = np.array([r["close_hit"] for r in rows], dtype=np.float64)
        side_hits = np.array([r["side_hit"] for r in rows], dtype=np.float64)
        top_down_contacts = np.array([r["top_down_contact"] for r in rows], dtype=np.float64)
        failures = np.array([r["failure"] for r in rows], dtype=np.float64)
        steps_arr = np.array([r["steps"] for r in rows], dtype=np.float64)

        contact_success_rate = float(np.mean(contact_success))
        safe_contact_rate = float(np.mean(safe_contacts))
        close_hit_rate = float(np.mean(close_hits))
        side_hit_rate = float(np.mean(side_hits))
        top_down_contact_rate = float(np.mean(top_down_contacts))
        unsafe_failure_rate = float(np.mean(failures))
        failure_rate = unsafe_failure_rate
        mean_reward = float(np.mean(rewards))
        mean_min_distance = float(np.mean(min_distances))
        mean_final_distance = float(np.mean(final_distances))
        mean_steps = float(np.mean(steps_arr))

        # Model-selection high score. Safe contact and side hits dominate. Reward
        # contributes only weakly, so reward-hacked unsafe hits should not win.
        score = (
            BEST_SCORE_SAFE_CONTACT_WEIGHT * safe_contact_rate
            + BEST_SCORE_SIDE_HIT_WEIGHT * side_hit_rate
            + BEST_SCORE_CLOSE_HIT_WEIGHT * close_hit_rate
            + BEST_SCORE_REWARD_WEIGHT * mean_reward
            - BEST_SCORE_MIN_DISTANCE_WEIGHT * mean_min_distance
            - BEST_SCORE_UNSAFE_FAILURE_PENALTY * unsafe_failure_rate
            - BEST_SCORE_TOP_DOWN_PENALTY * top_down_contact_rate
        )

        summary = {
            "timesteps": step,
            "score": float(score),
            "safe_contact_rate": safe_contact_rate,
            "contact_success_rate": contact_success_rate,
            "close_hit_rate": close_hit_rate,
            "side_hit_rate": side_hit_rate,
            "top_down_contact_rate": top_down_contact_rate,
            "unsafe_failure_rate": unsafe_failure_rate,
            "failure_rate": failure_rate,
            "mean_reward": mean_reward,
            "mean_min_distance": mean_min_distance,
            "mean_final_distance": mean_final_distance,
            "mean_steps": mean_steps,
        }

        return summary, best_rollout_rows

    def write_best_rollout_csv(self, rows):
        if not rows:
            return

        with open(BEST_ROLLOUT_CSV_PATH, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


# ============================================================
# TRAINING ENTRYPOINT
# ============================================================

def main():
    os.makedirs(MODEL_SAVE_FOLDER, exist_ok=True)
    os.makedirs(TENSOR_LOG_FOLDER, exist_ok=True)

    print("=== SAC target-conditioned training ===")
    print(f"Environment:        {ENVIRONMENT}")
    print(f"Model save folder:  {MODEL_SAVE_FOLDER}")
    print(f"TensorBoard folder: {TENSOR_LOG_FOLDER}")
    print(f"Total timesteps:    {TOTAL_TIMESTEPS}")
    print(f"Periodic saves:     1k until 10k, then every 10k")
    print(f"Best-model eval:    every {BEST_MODEL_EVAL_FREQ} steps on {BEST_MODEL_N_EVAL_TARGETS} fixed targets")
    print(f"Best model path:    {BEST_MODEL_PATH}")
    print("Note: SAC learning_starts is 10,000, so early checkpoints are mostly for pipeline health.")
    print("======================================")

    env = gym.make(ENVIRONMENT)

    model = SAC(
        policy="MultiInputPolicy",
        env=env,
        verbose=1,
        tensorboard_log=TENSOR_LOG_FOLDER,
        learning_starts=10_000,
        batch_size=256,
        buffer_size=1_000_000,
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

    callbacks = [
        TwoStageCheckpointCallback(
            save_path=MODEL_SAVE_FOLDER,
            name_prefix="rl_model",
            verbose=1,
        )
    ]

    if BEST_MODEL_EVAL_ENABLED:
        callbacks.append(
            BestWhipModelCallback(
                eval_env_id=ENVIRONMENT,
                eval_targets=EVAL_TARGETS[:BEST_MODEL_N_EVAL_TARGETS],
                eval_freq=BEST_MODEL_EVAL_FREQ,
                save_dir=MODEL_SAVE_FOLDER,
                deterministic=BEST_MODEL_DETERMINISTIC,
                verbose=1,
            )
        )

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        log_interval=10,
        callback=callbacks,
        # reset_num_timesteps=False,
    )

    model.save(os.path.join(MODEL_SAVE_FOLDER, "final_model"))
    env.close()


if __name__ == "__main__":
    main()
