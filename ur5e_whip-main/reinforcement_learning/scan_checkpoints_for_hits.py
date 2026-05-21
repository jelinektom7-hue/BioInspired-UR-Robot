#!/usr/bin/env python3
"""
Scan many SAC checkpoints and rank them by side-hit / close-hit / directed whip speed.

It loads each checkpoint with env=None and feeds it only the observation keys it was
trained with, so it can evaluate older checkpoints against the current environment.

Run from reinforcement_learning, for example:

python3 scan_checkpoints_for_hits.py \
  --folders SAC_15 SAC_16 SAC_17 SAC_18 SAC_19 SAC_models_my_run \
  --episodes 5 \
  --x 1.10 --y -0.45 --z 0.80 \
  --top-k 25 \
  --copy-top
"""

import argparse
import csv
import math
import re
import shutil
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC

import gymnasium_env

PROJECT_ROOT = Path("/home/dragos/ros2_ws/src/BioInspired-UR-Robot/ur5e_whip-main")
RL_DIR = PROJECT_ROOT / "reinforcement_learning"
ENVIRONMENT = "gymnasium_env/WhipWorld-v0"


def parse_args():
    parser = argparse.ArgumentParser(description="Scan SAC checkpoints for good target-directed whip motions.")
    parser.add_argument("--folders", nargs="+", default=["SAC_16", "SAC_17", "SAC_18", "SAC_19", "SAC_models_my_run"])
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--x", type=float, default=1.10)
    parser.add_argument("--y", type=float, default=-0.45)
    parser.add_argument("--z", type=float, default=0.80)
    parser.add_argument("--random-targets", action="store_true")
    parser.add_argument("--output", default="checkpoint_scan_results.csv")
    parser.add_argument("--commands-output", default="checkpoint_scan_top_commands.txt")
    parser.add_argument("--top-k", type=int, default=25)
    parser.add_argument("--copy-top", action="store_true")
    parser.add_argument("--candidate-dir", default="candidate_checkpoints")
    parser.add_argument("--max-checkpoints", type=int, default=None)
    return parser.parse_args()


def resolve_folder(folder: str) -> Path:
    p = Path(folder).expanduser()
    if p.is_absolute():
        return p
    return RL_DIR / p


def find_checkpoints(folders):
    checkpoints = []
    for folder in folders:
        root = resolve_folder(folder)
        if not root.exists():
            print(f"WARNING: folder not found, skipping: {root}")
            continue
        for p in sorted(root.rglob("*.zip")):
            name = p.name
            if re.match(r"rl_model_\d+_steps\.zip$", name) or name in ["best_model.zip", "final_model.zip", "bc_warmstart_model.zip"]:
                checkpoints.append(p)

    seen = set()
    unique = []
    for p in checkpoints:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            unique.append(p)
    return unique


def checkpoint_step(path: Path):
    m = re.search(r"rl_model_(\d+)_steps\.zip", path.name)
    if m:
        return int(m.group(1))
    if path.name == "best_model.zip":
        return 10**12
    if path.name == "final_model.zip":
        return 10**12 - 1
    return -1


def rel(path: Path):
    try:
        return str(path.resolve().relative_to(RL_DIR))
    except Exception:
        return str(path)


def filter_obs_for_model(obs, model_keys):
    missing = [k for k in model_keys if k not in obs]
    if missing:
        raise KeyError(f"Current env observation is missing keys required by checkpoint: {missing}")
    return {k: obs[k] for k in model_keys}


def bool_from_info_or_env(env, info, key, default=False):
    if key in info:
        return bool(info[key])
    attr = "_" + key
    if hasattr(env.unwrapped, attr):
        return bool(getattr(env.unwrapped, attr))
    return default


def float_from_info_or_env(env, info, key, default=float("nan")):
    if key in info:
        try:
            return float(info[key])
        except Exception:
            pass
    attr = "_" + key
    if hasattr(env.unwrapped, attr):
        try:
            return float(getattr(env.unwrapped, attr))
        except Exception:
            pass
    return default


def unit_vector(v):
    n = float(np.linalg.norm(v))
    if n < 1e-9 or not np.isfinite(n):
        return np.zeros_like(v), 0.0
    return v / n, n


def evaluate_checkpoint(path: Path, episodes: int, deterministic: bool, fixed_target, random_targets: bool):
    model = SAC.load(str(path), env=None)
    model_keys = list(model.observation_space.spaces.keys())

    env = gym.make(ENVIRONMENT)
    episode_results = []

    for ep in range(episodes):
        if random_targets:
            obs, info = env.reset(seed=ep)
            target_used = [float(info.get("target_x", np.nan)), float(info.get("target_y", np.nan)), float(info.get("target_z", np.nan))]
        else:
            obs, info = env.reset(seed=ep, options={"target_position": fixed_target})
            target_used = list(fixed_target)

        done = False
        steps = 0
        total_reward = 0.0
        min_distance = float("inf")
        had_contact = False
        had_side_hit = False
        had_bad_top_down = False
        had_arm_tip_ground = False
        had_robot_ground = False
        had_self_contact = False
        max_whip_speed = 0.0
        max_directed_speed = -1e9
        max_directed_speed_near = -1e9
        prev_whip_pos = None
        prev_time = None
        nonfinite = False
        exception_text = ""

        try:
            while not done:
                model_obs = filter_obs_for_model(obs, model_keys)
                action, _ = model.predict(model_obs, deterministic=deterministic)
                obs, reward, terminated, truncated, info = env.step(action)
                done = bool(terminated or truncated)
                total_reward += float(reward)
                steps += 1

                distance = float_from_info_or_env(env, info, "distance", float("nan"))
                if np.isfinite(distance):
                    min_distance = min(min_distance, distance)

                contact = bool_from_info_or_env(env, info, "whip_target_contact", False)
                side_hit = bool_from_info_or_env(env, info, "valid_side_hit", False)
                bad_top_down = bool_from_info_or_env(env, info, "bad_top_down_hit", False)
                arm_tip_ground = bool_from_info_or_env(env, info, "arm_tip_ground_hit", False)
                robot_ground = bool_from_info_or_env(env, info, "robot_ground_contact", False)
                self_contact = bool_from_info_or_env(env, info, "arm_self_contact", False)

                had_contact = had_contact or contact
                had_side_hit = had_side_hit or side_hit
                had_bad_top_down = had_bad_top_down or bad_top_down
                had_arm_tip_ground = had_arm_tip_ground or arm_tip_ground
                had_robot_ground = had_robot_ground or robot_ground
                had_self_contact = had_self_contact or self_contact

                if "whip_position" in obs:
                    whip_pos = np.array(obs["whip_position"], dtype=float)
                else:
                    whip_pos = None

                if "target_relative_to_whip" in obs:
                    rel_to_target = np.array(obs["target_relative_to_whip"], dtype=float)
                elif "target_position" in obs and whip_pos is not None:
                    rel_to_target = np.array(obs["target_position"], dtype=float) - whip_pos
                else:
                    rel_to_target = None

                if "whip_velocity" in obs:
                    whip_vel = np.array(obs["whip_velocity"], dtype=float)
                elif whip_pos is not None:
                    now = float(info.get("elapsed time [s]", steps * 0.01))
                    if prev_whip_pos is not None and prev_time is not None:
                        dt = max(1e-6, now - prev_time)
                        whip_vel = (whip_pos - prev_whip_pos) / dt
                    else:
                        whip_vel = np.zeros(3)
                    prev_whip_pos = whip_pos.copy()
                    prev_time = now
                else:
                    whip_vel = np.zeros(3)

                if not np.all(np.isfinite(whip_vel)):
                    nonfinite = True
                    break

                speed = float(np.linalg.norm(whip_vel))
                max_whip_speed = max(max_whip_speed, speed)

                if rel_to_target is not None:
                    direction, dist = unit_vector(rel_to_target)
                    directed_speed = float(np.dot(whip_vel, direction))
                    max_directed_speed = max(max_directed_speed, directed_speed)
                    if dist < 0.50:
                        max_directed_speed_near = max(max_directed_speed_near, directed_speed)

        except Exception as e:
            nonfinite = True
            exception_text = repr(e)

        unsafe = had_arm_tip_ground or had_robot_ground or had_self_contact
        safe_contact = had_contact and not unsafe
        safe_side_hit = had_side_hit and not unsafe
        close_08 = (min_distance <= 0.08) and not unsafe
        close_12 = (min_distance <= 0.12) and not unsafe

        if max_directed_speed_near < -1e8:
            max_directed_speed_near = float("nan")

        score = 0.0
        score += 200000.0 if safe_side_hit else 0.0
        score += 100000.0 if safe_contact else 0.0
        score += 40000.0 if close_08 else 0.0
        score += 15000.0 if close_12 else 0.0
        score -= 150000.0 if unsafe else 0.0
        score -= 20000.0 if had_bad_top_down and not safe_contact else 0.0
        score -= 3000.0 * min(min_distance, 10.0)
        score += 2500.0 * max(0.0, max_directed_speed)
        if np.isfinite(max_directed_speed_near):
            score += 4000.0 * max(0.0, max_directed_speed_near)
        score += 0.01 * total_reward
        score -= 50000.0 if nonfinite else 0.0

        episode_results.append({
            "episode": ep,
            "target": target_used,
            "steps": steps,
            "total_reward": total_reward,
            "min_distance": min_distance,
            "had_contact": had_contact,
            "had_side_hit": had_side_hit,
            "had_bad_top_down": had_bad_top_down,
            "had_arm_tip_ground": had_arm_tip_ground,
            "had_robot_ground": had_robot_ground,
            "had_self_contact": had_self_contact,
            "unsafe": unsafe,
            "safe_contact": safe_contact,
            "safe_side_hit": safe_side_hit,
            "close_08": close_08,
            "close_12": close_12,
            "max_whip_speed": max_whip_speed,
            "max_directed_speed": max_directed_speed,
            "max_directed_speed_near": max_directed_speed_near,
            "nonfinite": nonfinite,
            "exception": exception_text,
            "score": score,
        })

    env.close()

    best = max(episode_results, key=lambda r: r["score"])
    safe_contacts = sum(1 for r in episode_results if r["safe_contact"])
    side_hits = sum(1 for r in episode_results if r["safe_side_hit"])
    close_08s = sum(1 for r in episode_results if r["close_08"])
    unsafe_count = sum(1 for r in episode_results if r["unsafe"])
    nonfinite_count = sum(1 for r in episode_results if r["nonfinite"])
    aggregate_score = max(r["score"] for r in episode_results)

    return {
        "checkpoint": rel(path),
        "checkpoint_abs": str(path.resolve()),
        "folder": path.parent.name,
        "filename": path.name,
        "step": checkpoint_step(path),
        "episodes": episodes,
        "deterministic": deterministic,
        "random_targets": random_targets,
        "target_x": "" if random_targets else fixed_target[0],
        "target_y": "" if random_targets else fixed_target[1],
        "target_z": "" if random_targets else fixed_target[2],
        "model_obs_keys": "|".join(model_keys),
        "safe_contact_count": safe_contacts,
        "side_hit_count": side_hits,
        "close_08_count": close_08s,
        "unsafe_count": unsafe_count,
        "nonfinite_count": nonfinite_count,
        "best_episode": best["episode"],
        "best_score": aggregate_score,
        "best_min_distance": best["min_distance"],
        "best_total_reward": best["total_reward"],
        "best_max_whip_speed": best["max_whip_speed"],
        "best_max_directed_speed": best["max_directed_speed"],
        "best_max_directed_speed_near": best["max_directed_speed_near"],
        "best_safe_contact": best["safe_contact"],
        "best_safe_side_hit": best["safe_side_hit"],
        "best_unsafe": best["unsafe"],
        "best_bad_top_down": best["had_bad_top_down"],
        "best_target": best["target"],
        "best_exception": best["exception"],
    }


def write_results_csv(path: Path, rows):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_commands(path: Path, rows, top_k):
    lines = []
    lines.append("# Top checkpoint visual-test commands")
    lines.append("# Run from:")
    lines.append("#   cd /home/dragos/ros2_ws/src/BioInspired-UR-Robot/ur5e_whip-main/reinforcement_learning")
    lines.append("")
    lines.append("# Use SAC_evaluator_teacher_visual.py because it handles observation-space mismatches.")
    lines.append("")

    for i, row in enumerate(rows[:top_k], start=1):
        ckpt = row["checkpoint"]
        lines.append(
            f"# Rank {i}: score={float(row['best_score']):.1f}, "
            f"min_dist={float(row['best_min_distance']):.4f}, "
            f"side_hits={row['side_hit_count']}/{row['episodes']}, "
            f"safe_contacts={row['safe_contact_count']}/{row['episodes']}, "
            f"unsafe={row['unsafe_count']}/{row['episodes']}"
        )
        if row["random_targets"]:
            lines.append(f"python SAC_evaluator_teacher_visual.py --model {ckpt} --episodes 10")
        else:
            lines.append(
                f"python SAC_evaluator_teacher_visual.py --model {ckpt} "
                f"--x {row['target_x']} --y {row['target_y']} --z {row['target_z']} "
                f"--episodes 10"
            )
        lines.append("")
    path.write_text("\n".join(lines))


def copy_top_candidates(rows, top_k, candidate_dir: Path):
    candidate_dir.mkdir(parents=True, exist_ok=True)
    for i, row in enumerate(rows[:top_k], start=1):
        src = Path(row["checkpoint_abs"])
        safe_name = row["checkpoint"].replace("/", "__")
        score_int = int(float(row["best_score"]))
        dst = candidate_dir / f"rank_{i:02d}__score_{score_int}__{safe_name}"
        shutil.copy2(src, dst)


def main():
    args = parse_args()
    checkpoints = find_checkpoints(args.folders)
    checkpoints = sorted(checkpoints, key=lambda p: (str(p.parent), checkpoint_step(p), p.name))
    if args.max_checkpoints is not None:
        checkpoints = checkpoints[:args.max_checkpoints]
    if not checkpoints:
        raise RuntimeError("No checkpoints found.")

    fixed_target = [args.x, args.y, args.z]

    print("Found checkpoints:", len(checkpoints))
    print("Episodes per checkpoint:", args.episodes)
    print("Deterministic:", args.deterministic)
    print("Target mode:", "random" if args.random_targets else fixed_target)
    print("")

    rows = []
    for idx, ckpt in enumerate(checkpoints, start=1):
        print(f"[{idx}/{len(checkpoints)}] Evaluating {rel(ckpt)}")
        try:
            result = evaluate_checkpoint(
                ckpt,
                episodes=args.episodes,
                deterministic=args.deterministic,
                fixed_target=fixed_target,
                random_targets=args.random_targets,
            )
            rows.append(result)
            print(
                f"  score={float(result['best_score']):.1f} | "
                f"min_dist={float(result['best_min_distance']):.4f} | "
                f"side={result['side_hit_count']}/{args.episodes} | "
                f"safe_contact={result['safe_contact_count']}/{args.episodes} | "
                f"unsafe={result['unsafe_count']}/{args.episodes}"
            )
        except Exception as e:
            print(f"  FAILED: {repr(e)}")
            rows.append({
                "checkpoint": rel(ckpt),
                "checkpoint_abs": str(ckpt.resolve()),
                "folder": ckpt.parent.name,
                "filename": ckpt.name,
                "step": checkpoint_step(ckpt),
                "episodes": args.episodes,
                "deterministic": args.deterministic,
                "random_targets": args.random_targets,
                "target_x": "" if args.random_targets else fixed_target[0],
                "target_y": "" if args.random_targets else fixed_target[1],
                "target_z": "" if args.random_targets else fixed_target[2],
                "model_obs_keys": "",
                "safe_contact_count": 0,
                "side_hit_count": 0,
                "close_08_count": 0,
                "unsafe_count": 999,
                "nonfinite_count": 999,
                "best_episode": -1,
                "best_score": -1e30,
                "best_min_distance": math.inf,
                "best_total_reward": -math.inf,
                "best_max_whip_speed": 0.0,
                "best_max_directed_speed": 0.0,
                "best_max_directed_speed_near": 0.0,
                "best_safe_contact": False,
                "best_safe_side_hit": False,
                "best_unsafe": True,
                "best_bad_top_down": False,
                "best_target": "",
                "best_exception": repr(e),
            })

    rows_sorted = sorted(rows, key=lambda r: float(r["best_score"]), reverse=True)
    output_path = Path(args.output)
    commands_path = Path(args.commands_output)
    write_results_csv(output_path, rows_sorted)
    write_commands(commands_path, rows_sorted, args.top_k)

    if args.copy_top:
        copy_top_candidates(rows_sorted, args.top_k, Path(args.candidate_dir))

    print("")
    print("Done.")
    print("Results CSV:", output_path.resolve())
    print("Top commands:", commands_path.resolve())
    if args.copy_top:
        print("Copied candidates to:", Path(args.candidate_dir).resolve())

    print("")
    print("Top results:")
    for i, row in enumerate(rows_sorted[:args.top_k], start=1):
        print(
            f"{i:02d}. {row['checkpoint']} | "
            f"score={float(row['best_score']):.1f} | "
            f"min_dist={float(row['best_min_distance']):.4f} | "
            f"side={row['side_hit_count']}/{row['episodes']} | "
            f"safe_contact={row['safe_contact_count']}/{row['episodes']} | "
            f"unsafe={row['unsafe_count']}/{row['episodes']}"
        )


if __name__ == "__main__":
    main()
