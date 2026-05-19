#!/usr/bin/env python3
"""
Use an older SAC checkpoint whose observation space is missing newer keys
(e.g. no swing_hint) as a teacher for the current environment.

The script:
  1. Loads the old teacher checkpoint WITHOUT binding it to the current env.
  2. Runs the teacher in the current env using only the observation keys it knows.
  3. Keeps successful/safe rollouts or close safe misses.
  4. Behavior-clones a new current-observation SAC actor from those rollouts.
  5. Saves a current-compatible warm-start model, usually bc_warmstart_model.zip.

This lets you use useful behavior from an older checkpoint even after the
observation space changed, for example after adding swing_hint.
"""

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from stable_baselines3 import SAC

import gymnasium_env  # noqa: F401, needed to register env


PROJECT_ROOT = Path.home() / "ros2_ws/src/BioInspired-UR-Robot/ur5e_whip-main"
RL_DIR = PROJECT_ROOT / "reinforcement_learning"
MODEL_DIR = RL_DIR / "SAC_models_my_run"
ENVIRONMENT = "gymnasium_env/WhipWorld-v0"

DEFAULT_TEACHER = MODEL_DIR / "best_model.zip"
DEFAULT_DATASET = MODEL_DIR / "distilled_success_rollouts.npz"
DEFAULT_OUTPUT = MODEL_DIR / "bc_warmstart_model.zip"
DEFAULT_SUMMARY = MODEL_DIR / "distillation_summary.json"
DEFAULT_HISTORY = MODEL_DIR / "distillation_rollout_history.csv"


def resolve_model_path(path_text: str) -> Path:
    p = Path(path_text).expanduser()
    if p.is_absolute():
        return p

    candidates = [
        RL_DIR / p,
        MODEL_DIR / p,
        PROJECT_ROOT / p,
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    # Return the most likely path for a useful error message.
    return RL_DIR / p


def resolve_output_path(path_text: str) -> Path:
    p = Path(path_text).expanduser()
    if p.is_absolute():
        return p
    return MODEL_DIR / p


def dict_obs_to_teacher_obs(obs: Dict[str, np.ndarray], teacher_keys: List[str]) -> Dict[str, np.ndarray]:
    missing = [k for k in teacher_keys if k not in obs]
    if missing:
        raise KeyError(
            "Current environment does not contain keys needed by teacher checkpoint: "
            f"{missing}. Current keys are: {list(obs.keys())}"
        )
    return {k: obs[k] for k in teacher_keys}


def rollout_is_accepted(
    episode_infos: List[dict],
    min_distance: float,
    include_close_misses: float,
) -> Tuple[bool, str]:
    unsafe = False
    contact = False
    side_hit = False

    for info in episode_infos:
        if bool(info.get("arm_tip_ground_hit", False)):
            unsafe = True
        if bool(info.get("robot_ground_contact", False)):
            unsafe = True
        if bool(info.get("arm_self_contact", False)):
            unsafe = True

        if bool(info.get("whip_target_contact", False)):
            contact = True
        if bool(info.get("valid_side_hit", False)):
            side_hit = True
            contact = True

    if unsafe:
        return False, "unsafe_rejected"

    if side_hit:
        return True, "safe_side_hit"

    if contact:
        return True, "safe_contact"

    if include_close_misses > 0.0 and min_distance <= include_close_misses:
        return True, "safe_close_miss"

    return False, "miss_rejected"


def collect_teacher_rollouts(args, env, teacher_model):
    teacher_keys = list(teacher_model.observation_space.spaces.keys())
    current_keys = list(env.observation_space.spaces.keys())

    print("\n========== TEACHER/CURRENT OBSERVATION KEYS ==========")
    print("Teacher keys:", teacher_keys)
    print("Current keys:", current_keys)
    extra_current = [k for k in current_keys if k not in teacher_keys]
    print("Current-only keys that teacher will ignore:", extra_current)
    print("======================================================\n")

    obs_storage: Dict[str, List[np.ndarray]] = {k: [] for k in current_keys}
    action_storage: List[np.ndarray] = []

    history_rows = []
    accepted_count = 0
    safe_contact_count = 0
    safe_side_hit_count = 0
    close_miss_count = 0
    unsafe_count = 0

    for ep in range(args.episodes):
        reset_kwargs = {}
        if args.x is not None or args.y is not None or args.z is not None:
            if args.x is None or args.y is None or args.z is None:
                raise ValueError("Provide all of --x --y --z, or none of them.")
            reset_kwargs["options"] = {"target_position": [args.x, args.y, args.z]}
        else:
            reset_kwargs["seed"] = ep

        obs, info = env.reset(**reset_kwargs)
        done = False
        total_reward = 0.0
        steps = 0
        min_distance = float("inf")
        episode_infos: List[dict] = []
        episode_obs: Dict[str, List[np.ndarray]] = {k: [] for k in current_keys}
        episode_actions: List[np.ndarray] = []

        while not done:
            teacher_obs = dict_obs_to_teacher_obs(obs, teacher_keys)
            action, _ = teacher_model.predict(teacher_obs, deterministic=args.deterministic)
            action = np.asarray(action, dtype=np.float32)

            # Store the CURRENT full observation, including keys the old teacher did not have.
            for k in current_keys:
                episode_obs[k].append(np.asarray(obs[k], dtype=np.float32).copy())
            episode_actions.append(action.copy())

            obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            total_reward += float(reward)
            steps += 1
            episode_infos.append(dict(info))

            d = info.get("distance", None)
            if d is not None and np.isfinite(d):
                min_distance = min(min_distance, float(d))

        accepted, reason = rollout_is_accepted(
            episode_infos=episode_infos,
            min_distance=min_distance,
            include_close_misses=args.include_close_misses,
        )

        if reason == "unsafe_rejected":
            unsafe_count += 1
        if reason == "safe_contact":
            safe_contact_count += 1
        if reason == "safe_side_hit":
            safe_side_hit_count += 1
        if reason == "safe_close_miss":
            close_miss_count += 1

        if accepted:
            accepted_count += 1
            for k in current_keys:
                obs_storage[k].extend(episode_obs[k])
            action_storage.extend(episode_actions)

        tx = episode_infos[-1].get("target_x", np.nan) if episode_infos else np.nan
        ty = episode_infos[-1].get("target_y", np.nan) if episode_infos else np.nan
        tz = episode_infos[-1].get("target_z", np.nan) if episode_infos else np.nan

        history_rows.append({
            "episode": ep,
            "accepted": int(accepted),
            "reason": reason,
            "steps": steps,
            "total_reward": total_reward,
            "min_distance": min_distance,
            "target_x": tx,
            "target_y": ty,
            "target_z": tz,
        })

        if (ep + 1) % max(1, args.print_every) == 0 or accepted:
            print(
                f"episode {ep + 1:5d}/{args.episodes} | "
                f"accepted={accepted_count:4d} | reason={reason:18s} | "
                f"min_dist={min_distance:.4f} | reward={total_reward:.1f} | steps={steps}"
            )

    if not action_storage:
        raise RuntimeError(
            "No acceptable rollouts were collected. Try more episodes, a larger "
            "--include-close-misses value, a fixed center target with --x --y --z, "
            "or a different teacher checkpoint."
        )

    dataset_path = resolve_output_path(args.dataset)
    dataset_path.parent.mkdir(parents=True, exist_ok=True)

    arrays = {f"obs_{k}": np.asarray(v, dtype=np.float32) for k, v in obs_storage.items()}
    arrays["actions"] = np.asarray(action_storage, dtype=np.float32)
    arrays["obs_keys"] = np.asarray(current_keys, dtype=object)
    np.savez_compressed(dataset_path, **arrays)

    history_path = resolve_output_path(args.history)
    with history_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history_rows[0].keys()))
        writer.writeheader()
        writer.writerows(history_rows)

    summary = {
        "teacher_model": str(args.teacher_resolved),
        "episodes": args.episodes,
        "accepted_episodes": accepted_count,
        "accepted_rate": accepted_count / max(args.episodes, 1),
        "safe_contact_episodes": safe_contact_count,
        "safe_side_hit_episodes": safe_side_hit_count,
        "safe_close_miss_episodes": close_miss_count,
        "unsafe_rejected_episodes": unsafe_count,
        "transitions": int(arrays["actions"].shape[0]),
        "dataset": str(dataset_path),
        "history": str(history_path),
        "teacher_keys": teacher_keys,
        "current_keys": current_keys,
    }

    print("\n========== COLLECTION SUMMARY ==========")
    print(json.dumps(summary, indent=2))
    print("========================================\n")

    return dataset_path, summary


def load_dataset(dataset_path: Path):
    data = np.load(dataset_path, allow_pickle=True)
    obs_keys = [str(k) for k in data["obs_keys"].tolist()]
    observations = {k: data[f"obs_{k}"].astype(np.float32) for k in obs_keys}
    actions = data["actions"].astype(np.float32)
    return observations, actions, obs_keys


def train_student_actor(args, env, dataset_path: Path):
    observations, actions, obs_keys = load_dataset(dataset_path)
    n = actions.shape[0]

    if n < args.min_transitions:
        raise RuntimeError(
            f"Dataset has only {n} transitions, below --min-transitions={args.min_transitions}. "
            "Collect more rollouts before behavior cloning."
        )

    print("\n========== BEHAVIOR CLONING DATASET ==========")
    print("Dataset:", dataset_path)
    print("Transitions:", n)
    print("Observation keys:", obs_keys)
    print("Action shape:", actions.shape)
    print("==============================================\n")

    if args.student_start is not None:
        student_start = resolve_model_path(args.student_start)
        print("Loading student start model:", student_start)
        student = SAC.load(str(student_start), env=env, device=args.device)
    else:
        print("Creating fresh current-compatible SAC student.")
        student = SAC(
            policy="MultiInputPolicy",
            env=env,
            verbose=1,
            tensorboard_log=str(RL_DIR / "SAC_tensorboard"),
            learning_starts=10_000,
            batch_size=256,
            buffer_size=1_000_000,
            train_freq=1,
            gradient_steps=1,
            gamma=0.99,
            tau=0.005,
            device=args.device,
        )

    # Convert target actions to SAC actor's internal scaled action range [-1, 1].
    scaled_actions = student.policy.scale_action(actions).astype(np.float32)
    scaled_actions = np.clip(scaled_actions, -1.0, 1.0)

    device = student.device
    actor = student.policy.actor
    actor.train()

    rng = np.random.default_rng(args.seed)
    losses = []

    for epoch in range(args.epochs):
        indices = rng.permutation(n)
        epoch_losses = []

        for start in range(0, n, args.batch_size):
            batch_idx = indices[start:start + args.batch_size]
            obs_batch_np = {k: observations[k][batch_idx] for k in obs_keys}
            target_batch = torch.as_tensor(scaled_actions[batch_idx], dtype=torch.float32, device=device)

            obs_tensor, _ = student.policy.obs_to_tensor(obs_batch_np)
            pred_scaled = actor(obs_tensor, deterministic=True)

            loss = F.mse_loss(pred_scaled, target_batch)

            actor.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), args.max_grad_norm)
            actor.optimizer.step()

            epoch_losses.append(float(loss.detach().cpu().item()))

        mean_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        losses.append(mean_loss)

        if (epoch + 1) % max(1, args.print_every_epochs) == 0 or epoch == 0:
            print(f"BC epoch {epoch + 1:4d}/{args.epochs} | loss={mean_loss:.8f}")

    output_path = resolve_output_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    student.save(str(output_path))

    report = {
        "output_model": str(output_path),
        "dataset": str(dataset_path),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "final_loss": losses[-1] if losses else None,
        "min_loss": min(losses) if losses else None,
        "transitions": int(n),
    }

    report_path = output_path.with_suffix(".json")
    with report_path.open("w") as f:
        json.dump(report, f, indent=2)

    print("\n========== BC SUMMARY ==========")
    print(json.dumps(report, indent=2))
    print("===============================\n")

    return output_path, report


def parse_args():
    parser = argparse.ArgumentParser(
        description="Distill an older SAC checkpoint into the current observation-space model."
    )

    parser.add_argument("--teacher", default=str(DEFAULT_TEACHER), help="Old teacher checkpoint path.")
    parser.add_argument("--episodes", type=int, default=1000, help="Teacher rollout episodes to try.")
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic teacher actions.")
    parser.add_argument("--include-close-misses", type=float, default=0.06, help="Accept safe misses under this min distance. 0 disables close misses.")

    parser.add_argument("--x", type=float, default=None, help="Fixed sim target x for collection.")
    parser.add_argument("--y", type=float, default=None, help="Fixed sim target y for collection.")
    parser.add_argument("--z", type=float, default=None, help="Fixed sim target z for collection.")

    parser.add_argument("--dataset", default=str(DEFAULT_DATASET.name), help="Output dataset .npz path/name.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT.name), help="Output current-compatible model path/name.")
    parser.add_argument("--summary", default=str(DEFAULT_SUMMARY.name), help="Collection summary JSON path/name.")
    parser.add_argument("--history", default=str(DEFAULT_HISTORY.name), help="Rollout history CSV path/name.")

    parser.add_argument("--student-start", default=None, help="Optional current-compatible model to start BC from.")
    parser.add_argument("--epochs", type=int, default=50, help="Behavior cloning epochs.")
    parser.add_argument("--batch-size", type=int, default=256, help="BC batch size.")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="BC actor learning rate.")
    parser.add_argument("--max-grad-norm", type=float, default=1.0, help="BC gradient clipping norm.")
    parser.add_argument("--min-transitions", type=int, default=500, help="Minimum transitions required for BC.")
    parser.add_argument("--device", default="auto", help="Torch device for Stable-Baselines3.")
    parser.add_argument("--seed", type=int, default=12345, help="Random seed for BC minibatches.")
    parser.add_argument("--print-every", type=int, default=25, help="Print collection progress every N episodes.")
    parser.add_argument("--print-every-epochs", type=int, default=5, help="Print BC progress every N epochs.")

    args = parser.parse_args()
    args.teacher_resolved = resolve_model_path(args.teacher)
    return args


def main():
    args = parse_args()

    if not args.teacher_resolved.exists():
        raise FileNotFoundError(f"Teacher checkpoint not found: {args.teacher_resolved}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading old teacher checkpoint without current env:")
    print(args.teacher_resolved)
    teacher_model = SAC.load(str(args.teacher_resolved), device=args.device)

    env = gym.make(ENVIRONMENT)

    try:
        dataset_path, collection_summary = collect_teacher_rollouts(args, env, teacher_model)

        summary_path = resolve_output_path(args.summary)
        with summary_path.open("w") as f:
            json.dump(collection_summary, f, indent=2)

        # Set actor optimizer LR before BC if possible.
        output_path, report = train_student_actor(args, env, dataset_path)

        print("Current-compatible warm-start model saved to:")
        print(output_path)
        print("Use it with:")
        print("  python SAC_evaluator.py --model bc_warmstart_model.zip --episodes 10 --deterministic")
        print("Then continue training with:")
        print("  python SAC_trainer.py")

    finally:
        env.close()


if __name__ == "__main__":
    main()
