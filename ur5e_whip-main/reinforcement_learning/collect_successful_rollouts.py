#!/usr/bin/env python3
"""
Collect successful/safe SAC rollouts for behavior cloning.

This script runs a trained SAC policy in the current WhipWorld environment and
stores state-action pairs from episodes that are useful to imitate.

Good episodes are, by default:
  - safe target contacts, including safe side hits and safe top/down hits, OR
  - optional close safe misses if --include-close-misses is used.

Bad episodes are excluded if they contain robot floor contact, arm-tip ground
contact, or self-contact.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC

import gymnasium_env  # noqa: F401, registers WhipWorld-v0


PROJECT_ROOT = Path.home() / "ros2_ws/src/BioInspired-UR-Robot/ur5e_whip-main"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "reinforcement_learning/SAC_models_my_run"
ENVIRONMENT = "gymnasium_env/WhipWorld-v0"


def resolve_model_path(model_arg: str) -> Path:
    p = Path(model_arg).expanduser()
    if p.is_absolute():
        return p
    return DEFAULT_MODEL_DIR / p


def obs_append(storage: Dict[str, List[np.ndarray]], obs: Dict[str, np.ndarray]) -> None:
    for key, value in obs.items():
        storage.setdefault(key, []).append(np.asarray(value, dtype=np.float32).copy())


def stack_obs(storage: Dict[str, List[np.ndarray]]) -> Dict[str, np.ndarray]:
    return {key: np.stack(values, axis=0).astype(np.float32) for key, values in storage.items()}


def get_flag(info: dict, env, name: str, default: bool = False) -> bool:
    if name in info:
        return bool(info[name])
    return bool(getattr(env.unwrapped, f"_{name}", default))


def run_episode(env, model: SAC, deterministic: bool, target: List[float] | None):
    if target is None:
        obs, info = env.reset()
    else:
        obs, info = env.reset(options={"target_position": target})

    episode_obs: Dict[str, List[np.ndarray]] = {}
    episode_actions: List[np.ndarray] = []
    episode_rewards: List[float] = []

    done = False
    total_reward = 0.0
    min_distance = float("inf")
    final_distance = float("inf")
    steps = 0

    had_contact = False
    had_safe_contact = False
    had_side_hit = False
    had_top_down_contact = False
    had_unsafe_failure = False
    final_info = {}

    while not done:
        obs_append(episode_obs, obs)
        action, _ = model.predict(obs, deterministic=deterministic)
        action = np.asarray(action, dtype=np.float32).copy()
        episode_actions.append(action)

        obs, reward, terminated, truncated, info = env.step(action)

        reward = float(reward)
        episode_rewards.append(reward)
        total_reward += reward
        steps += 1
        final_info = dict(info)

        distance = float(info.get("distance", np.nan))
        if np.isfinite(distance):
            min_distance = min(min_distance, distance)
            final_distance = distance

        # These names match the newer environment versions, with fallbacks for old ones.
        env_contact = bool(getattr(env.unwrapped, "_whip_target_contact", False))
        had_contact = bool(had_contact or env_contact or info.get("whip_target_contact", False))
        had_safe_contact = bool(had_safe_contact or info.get("safe_target_contact", False))
        had_side_hit = bool(had_side_hit or info.get("valid_side_hit", False))
        had_top_down_contact = bool(had_top_down_contact or info.get("top_down_contact", False) or info.get("bad_top_down_hit", False))

        unsafe_now = bool(
            info.get("unsafe_robot_failure", False)
            or info.get("arm_tip_ground_hit", False)
            or info.get("robot_ground_contact", False)
            or info.get("arm_self_contact", False)
            or getattr(env.unwrapped, "_arm_tip_ground_hit", False)
            or getattr(env.unwrapped, "_robot_ground_contact", False)
            or getattr(env.unwrapped, "_arm_self_contact", False)
        )
        had_unsafe_failure = bool(had_unsafe_failure or unsafe_now)

        done = bool(terminated or truncated)

    obs_stacked = stack_obs(episode_obs)
    actions = np.stack(episode_actions, axis=0).astype(np.float32)
    rewards = np.asarray(episode_rewards, dtype=np.float32)

    summary = {
        "total_reward": float(total_reward),
        "steps": int(steps),
        "min_distance": float(min_distance),
        "final_distance": float(final_distance),
        "had_contact": bool(had_contact),
        "had_safe_contact": bool(had_safe_contact),
        "had_side_hit": bool(had_side_hit),
        "had_top_down_contact": bool(had_top_down_contact),
        "had_unsafe_failure": bool(had_unsafe_failure),
        "target_x": float(final_info.get("target_x", np.nan)),
        "target_y": float(final_info.get("target_y", np.nan)),
        "target_z": float(final_info.get("target_z", np.nan)),
    }

    return obs_stacked, actions, rewards, summary


def should_keep_episode(summary: dict, include_close_misses: float | None) -> Tuple[bool, str]:
    if summary["had_unsafe_failure"]:
        return False, "unsafe_failure"

    if summary["had_side_hit"]:
        return True, "side_hit"

    if summary["had_safe_contact"]:
        return True, "safe_contact"

    # Fallback for older envs that do not expose safe_target_contact.
    if summary["had_contact"] and not summary["had_unsafe_failure"]:
        return True, "contact_no_unsafe_flag"

    if include_close_misses is not None and summary["min_distance"] <= include_close_misses:
        return True, "close_safe_miss"

    return False, "miss"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="best_model.zip", help="Model zip inside SAC_models_my_run/ or absolute path.")
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--output", default="successful_rollouts.npz")
    parser.add_argument("--summary", default="successful_rollouts_summary.json")
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic policy actions. Default is stochastic to discover more successes.")
    parser.add_argument("--include-close-misses", type=float, default=None, help="Also keep safe episodes with min distance <= this value in meters, e.g. 0.06.")
    parser.add_argument("--fixed-target", nargs=3, type=float, default=None, metavar=("X", "Y", "Z"), help="Optional fixed sim target for all episodes.")
    parser.add_argument("--max-kept-episodes", type=int, default=1000)
    args = parser.parse_args()

    model_path = resolve_model_path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    output_path = Path(args.output).expanduser()
    if not output_path.is_absolute():
        output_path = DEFAULT_MODEL_DIR / output_path

    summary_path = Path(args.summary).expanduser()
    if not summary_path.is_absolute():
        summary_path = DEFAULT_MODEL_DIR / summary_path

    print("=== Collect successful rollouts ===")
    print(f"Environment:     {ENVIRONMENT}")
    print(f"Model:           {model_path}")
    print(f"Episodes:        {args.episodes}")
    print(f"Deterministic:   {args.deterministic}")
    print(f"Close misses:    {args.include_close_misses}")
    print(f"Fixed target:    {args.fixed_target}")
    print(f"Output:          {output_path}")
    print("===================================")

    env = gym.make(ENVIRONMENT)
    try:
        env.unwrapped.set_training_progress(1.0)
    except Exception:
        pass

    model = SAC.load(str(model_path), env=env)

    all_obs: Dict[str, List[np.ndarray]] = {}
    all_actions: List[np.ndarray] = []
    all_episode_ids: List[int] = []
    kept_summaries: List[dict] = []
    all_summaries: List[dict] = []

    kept_count = 0

    for ep in range(args.episodes):
        obs, actions, rewards, summary = run_episode(
            env=env,
            model=model,
            deterministic=args.deterministic,
            target=args.fixed_target,
        )

        keep, reason = should_keep_episode(summary, args.include_close_misses)
        summary["episode"] = int(ep)
        summary["kept"] = bool(keep)
        summary["keep_reason"] = reason
        all_summaries.append(summary)

        if keep and kept_count < args.max_kept_episodes:
            for key, arr in obs.items():
                all_obs.setdefault(key, []).append(arr)
            all_actions.append(actions)
            all_episode_ids.extend([kept_count] * len(actions))
            kept_summaries.append(summary)
            kept_count += 1
            status = "KEEP"
        else:
            status = "skip"

        print(
            f"ep={ep + 1:04d}/{args.episodes} | {status:4s} | {reason:22s} | "
            f"safe={int(summary['had_safe_contact'])} side={int(summary['had_side_hit'])} "
            f"unsafe={int(summary['had_unsafe_failure'])} min_dist={summary['min_distance']:.4f} "
            f"reward={summary['total_reward']:.1f}"
        )

    env.close()

    if not all_actions:
        raise RuntimeError(
            "No successful/safe rollouts were collected. Try more episodes, use stochastic collection "
            "by omitting --deterministic, or add --include-close-misses 0.06."
        )

    obs_concat = {key: np.concatenate(chunks, axis=0).astype(np.float32) for key, chunks in all_obs.items()}
    actions_concat = np.concatenate(all_actions, axis=0).astype(np.float32)
    episode_ids = np.asarray(all_episode_ids, dtype=np.int32)

    save_dict = {
        "actions": actions_concat,
        "episode_ids": episode_ids,
        "obs_keys": np.asarray(list(obs_concat.keys()), dtype=object),
    }
    for key, arr in obs_concat.items():
        save_dict[f"obs__{key}"] = arr

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **save_dict)

    summary_data = {
        "model": str(model_path),
        "environment": ENVIRONMENT,
        "episodes_requested": int(args.episodes),
        "episodes_kept": int(kept_count),
        "transitions_saved": int(len(actions_concat)),
        "output_dataset": str(output_path),
        "kept_summaries": kept_summaries,
        "all_summaries": all_summaries,
    }
    with open(summary_path, "w") as f:
        json.dump(summary_data, f, indent=2)

    print("\nSaved dataset:", output_path)
    print("Saved summary:", summary_path)
    print("Kept episodes:", kept_count)
    print("Saved transitions:", len(actions_concat))
    print("Observation keys:", list(obs_concat.keys()))


if __name__ == "__main__":
    main()
