#!/usr/bin/env python3
"""
Behavior-clone a SAC actor from successful rollout state-action pairs.

This script does supervised learning on the SAC actor only:
  observation -> action
using the dataset created by collect_successful_rollouts.py.

The critic is not behavior-cloned. After this, continue SAC fine-tuning from the
saved bc_warmstart_model.zip so the critic can adapt to the improved actor.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from stable_baselines3 import SAC

import gymnasium_env  # noqa: F401


PROJECT_ROOT = Path.home() / "ros2_ws/src/BioInspired-UR-Robot/ur5e_whip-main"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "reinforcement_learning/SAC_models_my_run"
ENVIRONMENT = "gymnasium_env/WhipWorld-v0"


def resolve_path(path_arg: str, base: Path = DEFAULT_MODEL_DIR) -> Path:
    p = Path(path_arg).expanduser()
    if p.is_absolute():
        return p
    return base / p


def load_dataset(dataset_path: Path):
    data = np.load(dataset_path, allow_pickle=True)
    actions = data["actions"].astype(np.float32)
    obs_keys = [str(k) for k in data["obs_keys"].tolist()]
    obs = {}
    for key in obs_keys:
        obs_key = f"obs__{key}"
        if obs_key not in data:
            raise KeyError(f"Missing {obs_key} in dataset")
        obs[key] = data[obs_key].astype(np.float32)
    return obs, actions, obs_keys


def scale_actions(actions: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    # Convert env-space Box actions to SAC actor output space [-1, 1].
    scaled = 2.0 * ((actions - low) / (high - low)) - 1.0
    return np.clip(scaled, -1.0, 1.0).astype(np.float32)


def make_batch(obs: Dict[str, np.ndarray], actions_scaled: np.ndarray, indices: np.ndarray):
    batch_obs = {key: arr[indices] for key, arr in obs.items()}
    batch_actions = actions_scaled[indices]
    return batch_obs, batch_actions


def evaluate_bc_loss(model: SAC, obs: Dict[str, np.ndarray], actions_scaled: np.ndarray, indices: np.ndarray, batch_size: int) -> float:
    if len(indices) == 0:
        return float("nan")

    model.policy.set_training_mode(False)
    losses = []
    device = model.device

    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            idx = indices[start:start + batch_size]
            batch_obs, batch_actions = make_batch(obs, actions_scaled, idx)
            obs_tensor, _ = model.policy.obs_to_tensor(batch_obs)
            pred_scaled = model.policy.actor(obs_tensor, deterministic=True)
            target = torch.as_tensor(batch_actions, device=device)
            loss = F.mse_loss(pred_scaled, target)
            losses.append(float(loss.item()))

    model.policy.set_training_mode(True)
    return float(np.mean(losses)) if losses else float("nan")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="best_model.zip", help="Source SAC model zip inside SAC_models_my_run/ or absolute path.")
    parser.add_argument("--dataset", default="successful_rollouts.npz", help="Dataset made by collect_successful_rollouts.py.")
    parser.add_argument("--output", default="bc_warmstart_model.zip", help="Output SAC model zip.")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--validation-split", type=float, default=0.1)
    parser.add_argument("--max-samples", type=int, default=0, help="Optional cap. 0 means use all samples.")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    model_path = resolve_path(args.model)
    dataset_path = resolve_path(args.dataset)
    output_path = resolve_path(args.output)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    env = gym.make(ENVIRONMENT)
    model = SAC.load(str(model_path), env=env)
    model.policy.set_training_mode(True)

    obs, actions, obs_keys = load_dataset(dataset_path)

    n = len(actions)
    if args.max_samples and args.max_samples > 0 and args.max_samples < n:
        rng = np.random.default_rng(args.seed)
        keep = rng.choice(n, size=args.max_samples, replace=False)
        actions = actions[keep]
        obs = {key: arr[keep] for key, arr in obs.items()}
        n = len(actions)

    low = np.asarray(env.action_space.low, dtype=np.float32)
    high = np.asarray(env.action_space.high, dtype=np.float32)
    actions_scaled = scale_actions(actions, low, high)

    rng = np.random.default_rng(args.seed)
    indices = np.arange(n)
    rng.shuffle(indices)

    n_val = int(round(args.validation_split * n))
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    if len(train_idx) == 0:
        raise RuntimeError("Dataset too small after validation split.")

    optimizer = model.policy.actor.optimizer
    for group in optimizer.param_groups:
        group["lr"] = args.learning_rate

    print("=== Behavior cloning from successful rollouts ===")
    print(f"Environment:    {ENVIRONMENT}")
    print(f"Source model:   {model_path}")
    print(f"Dataset:        {dataset_path}")
    print(f"Output model:   {output_path}")
    print(f"Samples:        {n}")
    print(f"Train / val:    {len(train_idx)} / {len(val_idx)}")
    print(f"Obs keys:       {obs_keys}")
    print("=================================================")

    device = model.device
    history = []

    for epoch in range(1, args.epochs + 1):
        rng.shuffle(train_idx)
        batch_losses = []

        model.policy.set_training_mode(True)
        for start in range(0, len(train_idx), args.batch_size):
            idx = train_idx[start:start + args.batch_size]
            batch_obs, batch_actions = make_batch(obs, actions_scaled, idx)

            obs_tensor, _ = model.policy.obs_to_tensor(batch_obs)
            pred_scaled = model.policy.actor(obs_tensor, deterministic=True)
            target = torch.as_tensor(batch_actions, device=device)

            loss = F.mse_loss(pred_scaled, target)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.policy.actor.parameters(), max_norm=5.0)
            optimizer.step()

            batch_losses.append(float(loss.item()))

        train_loss = float(np.mean(batch_losses)) if batch_losses else float("nan")
        val_loss = evaluate_bc_loss(model, obs, actions_scaled, val_idx, args.batch_size)

        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
        history.append(row)

        print(f"epoch {epoch:03d}/{args.epochs} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(output_path))

    history_path = output_path.with_suffix(".json")
    with open(history_path, "w") as f:
        json.dump(
            {
                "source_model": str(model_path),
                "dataset": str(dataset_path),
                "output_model": str(output_path),
                "samples": int(n),
                "obs_keys": obs_keys,
                "epochs": int(args.epochs),
                "batch_size": int(args.batch_size),
                "learning_rate": float(args.learning_rate),
                "history": history,
            },
            f,
            indent=2,
        )

    env.close()
    print("\nSaved BC warm-start model:", output_path)
    print("Saved BC training history:", history_path)


if __name__ == "__main__":
    main()
