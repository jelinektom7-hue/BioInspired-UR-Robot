import os
import csv
import json
from pathlib import Path
import mujoco

import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC

import gymnasium_env


PROJECT_ROOT = Path.home() / "ros2_ws/src/BioInspired-UR-Robot/ur5e_whip-main"

MODEL_FILE = (
    PROJECT_ROOT
    / "frozen_runs/sidehit_contact_v1/checkpoints/rl_model_970000_steps.zip"
)

OUT_DIR = PROJECT_ROOT / "reinforcement_learning/metrics_sidehit_contact_v1"
OUT_CSV = OUT_DIR / "episode_metrics.csv"
OUT_JSON = OUT_DIR / "summary_metrics.json"

ENVIRONMENT = "gymnasium_env/WhipWorld-v0"

N_EPISODES = 100
DETERMINISTIC = True


def get_env_attr(env, name, default=None):
    """
    Safely read internal variables from the unwrapped Gym environment.
    This lets us read flags like _whip_target_contact, _valid_side_hit, etc.
    """
    try:
        return getattr(env.unwrapped, name, default)
    except Exception:
        return default


def get_info_value(info, key, fallback=None):
    value = info.get(key, fallback)
    return value

def set_bottle_position(env, x, y, z):
    """
    Move the physical target body named 'bottle' during evaluation.

    Works even if the bottle has a free joint.
    """
    base_env = env.unwrapped

    bottle_id = base_env.model.body("bottle").id
    jnt_adr = base_env.model.body_jntadr[bottle_id]

    if jnt_adr >= 0:
        qpos_adr = base_env.model.jnt_qposadr[jnt_adr]
        joint_type = base_env.model.jnt_type[jnt_adr]

        if joint_type == mujoco.mjtJoint.mjJNT_FREE:
            # Free joint qpos format: x, y, z, qw, qx, qy, qz
            base_env.data.qpos[qpos_adr:qpos_adr + 3] = [x, y, z]
            base_env.data.qpos[qpos_adr + 3:qpos_adr + 7] = [1.0, 0.0, 0.0, 0.0]
        else:
            base_env.model.body_pos[bottle_id] = [x, y, z]
    else:
        base_env.model.body_pos[bottle_id] = [x, y, z]

    mujoco.mj_forward(base_env.model, base_env.data)

    base_env._target_position = base_env.data.xpos[bottle_id].copy()
    base_env._distance_to_target = np.linalg.norm(
        base_env._whip_position - base_env._target_position
    )
    base_env._shortest_distance_to_target = base_env._distance_to_target

    if hasattr(base_env, "_prev_distance_to_target"):
        base_env._prev_distance_to_target = base_env._distance_to_target

    if hasattr(base_env, "_whip_target_contact"):
        base_env._whip_target_contact = False

    return base_env._target_position.copy()

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not MODEL_FILE.exists():
        raise FileNotFoundError(f"Could not find model checkpoint: {MODEL_FILE}")

    env = gym.make(ENVIRONMENT)
    model = SAC.load(str(MODEL_FILE), env=env)

    rng = np.random.default_rng(12345)

    episode_rows = []

    for ep in range(N_EPISODES):
        obs, info = env.reset(seed=ep)

        # Randomize target around the trained target area.
        # Keep the range small at first.
        target_x = rng.uniform(0.80, 0.90)
        target_y = rng.uniform(0.30, 0.40)
        target_z = rng.uniform(0.60, 0.70)

        target_position = set_bottle_position(env, target_x, target_y, target_z)

        obs = env.unwrapped._get_obs()

        done = False
        terminated_final = False
        truncated_final = False

        total_reward = 0.0
        steps = 0

        min_distance = float("inf")
        time_to_first_contact = None

        had_target_contact = False
        had_valid_side_hit = False
        had_arm_tip_ground_hit = False
        had_arm_self_contact = False
        had_robot_ground_contact = False
        had_bad_top_down_hit = False

        final_distance = None
        final_elapsed_time = None

        while not done:
            action, _ = model.predict(obs, deterministic=DETERMINISTIC)

            obs, reward, terminated, truncated, info = env.step(action)

            steps += 1
            total_reward += float(reward)

            distance = get_info_value(
                info,
                "distance",
                get_env_attr(env, "_distance_to_target", np.nan),
            )

            if distance is not None and np.isfinite(distance):
                min_distance = min(min_distance, float(distance))
                final_distance = float(distance)

            elapsed_time = get_info_value(
                info,
                "elapsed time [s]",
                steps * 0.01,
            )
            final_elapsed_time = float(elapsed_time)

            # Read flags from info first, then fallback to environment internals.
            whip_target_contact = bool(
                get_info_value(
                    info,
                    "whip_target_contact",
                    get_env_attr(env, "_whip_target_contact", False),
                )
            )

            valid_side_hit = bool(
                get_info_value(
                    info,
                    "valid_side_hit",
                    get_env_attr(env, "_valid_side_hit", False),
                )
            )

            arm_tip_ground_hit = bool(
                get_info_value(
                    info,
                    "arm_tip_ground_hit",
                    get_env_attr(env, "_arm_tip_ground_hit", False),
                )
            )

            arm_self_contact = bool(
                get_info_value(
                    info,
                    "arm_self_contact",
                    get_env_attr(env, "_arm_self_contact", False),
                )
            )

            robot_ground_contact = bool(
                get_info_value(
                    info,
                    "robot_ground_contact",
                    get_env_attr(env, "_robot_ground_contact", False),
                )
            )

            bad_top_down_hit = bool(
                get_info_value(
                    info,
                    "bad_top_down_hit",
                    get_env_attr(env, "_bad_top_down_hit", False),
                )
            )

            if whip_target_contact and time_to_first_contact is None:
                time_to_first_contact = final_elapsed_time

            had_target_contact = had_target_contact or whip_target_contact
            had_valid_side_hit = had_valid_side_hit or valid_side_hit
            had_arm_tip_ground_hit = had_arm_tip_ground_hit or arm_tip_ground_hit
            had_arm_self_contact = had_arm_self_contact or arm_self_contact
            had_robot_ground_contact = had_robot_ground_contact or robot_ground_contact
            had_bad_top_down_hit = had_bad_top_down_hit or bad_top_down_hit

            terminated_final = bool(terminated)
            truncated_final = bool(truncated)
            done = terminated or truncated

        if time_to_first_contact is None:
            time_to_first_contact = float("nan")

        if not np.isfinite(min_distance):
            min_distance = float("nan")

        if final_distance is None:
            final_distance = float("nan")

        if final_elapsed_time is None:
            final_elapsed_time = float("nan")

        # Useful episode category.
        if had_valid_side_hit:
            outcome = "side_hit"
        elif had_target_contact:
            outcome = "contact_not_side"
        elif had_arm_tip_ground_hit or had_robot_ground_contact:
            outcome = "robot_ground_failure"
        elif had_arm_self_contact:
            outcome = "self_contact_failure"
        elif truncated_final:
            outcome = "timeout_or_miss"
        elif terminated_final:
            outcome = "terminated_other"
        else:
            outcome = "unknown"

        row = {
            "episode": ep,
            "outcome": outcome,
            "total_reward": total_reward,
            "steps": steps,
            "elapsed_time": final_elapsed_time,
            "time_to_first_contact": time_to_first_contact,
            "min_distance": min_distance,
            "final_distance": final_distance,
            "target_contact": int(had_target_contact),
            "valid_side_hit": int(had_valid_side_hit),
            "bad_top_down_hit": int(had_bad_top_down_hit),
            "arm_tip_ground_hit": int(had_arm_tip_ground_hit),
            "robot_ground_contact": int(had_robot_ground_contact),
            "arm_self_contact": int(had_arm_self_contact),
            "terminated": int(terminated_final),
            "truncated": int(truncated_final),
            "target_x": float(target_x),
            "target_y": float(target_y),
            "target_z": float(target_z),
        }

        episode_rows.append(row)

        print(
            f"Episode {ep + 1:03d}/{N_EPISODES} | "
            f"outcome={outcome:22s} | "
            f"reward={total_reward:9.2f} | "
            f"steps={steps:4d} | "
            f"min_dist={min_distance:.4f}"
        )

    env.close()

    # Save per-episode CSV.
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(episode_rows[0].keys()))
        writer.writeheader()
        writer.writerows(episode_rows)

    # Compute summary.
    n = len(episode_rows)

    def mean_of(key):
        values = [float(r[key]) for r in episode_rows]
        values = [v for v in values if np.isfinite(v)]
        return float(np.mean(values)) if values else float("nan")

    def rate_of(key):
        return float(np.mean([int(r[key]) for r in episode_rows]))

    outcome_counts = {}
    for r in episode_rows:
        outcome_counts[r["outcome"]] = outcome_counts.get(r["outcome"], 0) + 1

    summary = {
        "model_file": str(MODEL_FILE),
        "episodes": n,
        "contact_success_rate": rate_of("target_contact"),
        "valid_side_hit_rate": rate_of("valid_side_hit"),
        "bad_top_down_hit_rate": rate_of("bad_top_down_hit"),
        "arm_tip_ground_failure_rate": rate_of("arm_tip_ground_hit"),
        "robot_ground_contact_rate": rate_of("robot_ground_contact"),
        "arm_self_contact_rate": rate_of("arm_self_contact"),
        "truncation_rate": rate_of("truncated"),
        "mean_total_reward": mean_of("total_reward"),
        "mean_steps": mean_of("steps"),
        "mean_elapsed_time": mean_of("elapsed_time"),
        "mean_time_to_first_contact": mean_of("time_to_first_contact"),
        "mean_min_distance": mean_of("min_distance"),
        "mean_final_distance": mean_of("final_distance"),
        "outcome_counts": outcome_counts,
    }

    with open(OUT_JSON, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n================ SUMMARY ================")
    print(f"Episodes:                  {summary['episodes']}")
    print(f"Contact success rate:       {summary['contact_success_rate'] * 100:.1f}%")
    print(f"Valid side-hit rate:        {summary['valid_side_hit_rate'] * 100:.1f}%")
    print(f"Bad top-down hit rate:      {summary['bad_top_down_hit_rate'] * 100:.1f}%")
    print(f"Arm tip ground fail rate:   {summary['arm_tip_ground_failure_rate'] * 100:.1f}%")
    print(f"Robot ground contact rate:  {summary['robot_ground_contact_rate'] * 100:.1f}%")
    print(f"Self-contact rate:          {summary['arm_self_contact_rate'] * 100:.1f}%")
    print(f"Truncation rate:            {summary['truncation_rate'] * 100:.1f}%")
    print(f"Mean reward:                {summary['mean_total_reward']:.2f}")
    print(f"Mean steps:                 {summary['mean_steps']:.2f}")
    print(f"Mean time to contact:       {summary['mean_time_to_first_contact']:.3f} s")
    print(f"Mean closest distance:      {summary['mean_min_distance']:.4f} m")
    print(f"Mean final distance:        {summary['mean_final_distance']:.4f} m")
    print(f"Outcome counts:             {summary['outcome_counts']}")
    print("=========================================")

    print(f"\nSaved per-episode metrics to:\n{OUT_CSV}")
    print(f"\nSaved summary metrics to:\n{OUT_JSON}")


if __name__ == "__main__":
    main()