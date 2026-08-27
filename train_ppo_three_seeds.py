"""Train three reproducible PPO models without overwriting earlier results.

The default seeds (101, 202 and 303) create three independent training runs.
Every run uses deterministic software seeds and a deterministic sequence of
different procedural scenes.  Models, monitor data, reward histories and
metadata are written to separate seed folders.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import time
from pathlib import Path

import gymnasium
import matplotlib.pyplot as plt
import numpy as np
import scipy
import stable_baselines3
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

from rl_environment import (
    CONFIRM_THRESHOLD,
    HEIGHT_BACKGROUND_NOISE_STD,
    HEIGHT_DETECTION_THRESHOLD,
    RESIDUE_HEIGHT_MAX,
    RESIDUE_HEIGHT_MIN,
    ReproducibleElectrodeInspectionEnv,
)


DEFAULT_SEEDS = [101, 202, 303]
MOVING_AVERAGE_WINDOW = 20


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_reward_history(
    path: Path,
    rewards: list[float],
    lengths: list[int],
    elapsed_times: list[float],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["episode", "reward", "length", "elapsed_seconds"])
        for episode, (reward, length, elapsed) in enumerate(
            zip(rewards, lengths, elapsed_times), start=1
        ):
            writer.writerow([episode, reward, length, elapsed])


def moving_average(values: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    if len(values) < window:
        return np.arange(len(values)), values
    averages = np.convolve(values, np.ones(window) / window, mode="valid")
    episodes = np.arange(window, len(values) + 1)
    return episodes, averages


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--timesteps", type=int, default=35_000)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments_reproducible_v3"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {args.output_dir.resolve()}\n"
            "Choose a new --output-dir so existing evidence is not overwritten."
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)

    source_environment = Path("rl_environment.py")
    source_training = Path(__file__)
    run_summaries: list[dict] = []
    reward_histories: dict[int, np.ndarray] = {}

    for seed in args.seeds:
        print(f"Training reproducible PPO seed {seed} ...", flush=True)
        set_random_seed(seed, using_cuda=False)

        run_directory = args.output_dir / f"seed_{seed}"
        run_directory.mkdir(parents=True, exist_ok=False)
        monitor_path = run_directory / "monitor.csv"

        base_environment = ReproducibleElectrodeInspectionEnv(
            max_steps=15,
            base_seed=seed,
        )
        environment = Monitor(base_environment, filename=str(monitor_path))

        model = PPO(
            "MlpPolicy",
            environment,
            verbose=0,
            learning_rate=3e-4,
            n_steps=256,
            batch_size=64,
            gamma=0.99,
            seed=seed,
            device="cpu",
        )

        start_time = time.perf_counter()
        model.learn(total_timesteps=args.timesteps)
        training_seconds = time.perf_counter() - start_time

        model_path = run_directory / "ppo_electrode_agent"
        model.save(model_path)

        rewards = [float(value) for value in environment.get_episode_rewards()]
        lengths = [int(value) for value in environment.get_episode_lengths()]
        elapsed_times = [
            float(value) for value in environment.get_episode_times()
        ]
        write_reward_history(
            run_directory / "episode_rewards.csv",
            rewards,
            lengths,
            elapsed_times,
        )
        reward_histories[seed] = np.asarray(rewards, dtype=float)

        first_window = rewards[:MOVING_AVERAGE_WINDOW]
        final_window = rewards[-MOVING_AVERAGE_WINDOW:]
        model_zip_path = model_path.with_suffix(".zip")
        run_summary = {
            "seed": seed,
            "requested_timesteps": args.timesteps,
            "stored_timesteps": int(model.num_timesteps),
            "episodes": len(rewards),
            "first_20_episode_mean_reward": float(np.mean(first_window)),
            "final_20_episode_mean_reward": float(np.mean(final_window)),
            "training_seconds": float(training_seconds),
            "model_path": str(model_zip_path.resolve()),
            "model_sha256": file_sha256(model_zip_path),
        }
        run_summaries.append(run_summary)

        (run_directory / "training_metadata.json").write_text(
            json.dumps(
                {
                    **run_summary,
                    "algorithm": "Stable-Baselines3 PPO with MlpPolicy",
                    "policy_network": "two 64-unit Tanh hidden layers for actor and critic (framework default)",
                    "hyperparameters": {
                        "learning_rate": 3e-4,
                        "n_steps": 256,
                        "batch_size": 64,
                        "gamma": 0.99,
                        "device": "cpu",
                    },
                    "environment": {
                        "actions": 4,
                        "observation_values": 7,
                        "maximum_episode_steps": 15,
                        "confirmation_coverage_threshold": CONFIRM_THRESHOLD,
                        "height_units": "dimensionless synthetic relative-height units (SRHU)",
                        "background_height_noise_std": HEIGHT_BACKGROUND_NOISE_STD,
                        "residue_height_range": [
                            RESIDUE_HEIGHT_MIN,
                            RESIDUE_HEIGHT_MAX,
                        ],
                        "height_detection_threshold": HEIGHT_DETECTION_THRESHOLD,
                    },
                    "software": {
                        "python": platform.python_version(),
                        "numpy": np.__version__,
                        "scipy": scipy.__version__,
                        "gymnasium": gymnasium.__version__,
                        "stable_baselines3": stable_baselines3.__version__,
                        "torch": torch.__version__,
                    },
                    "source_files": {
                        "environment": {
                            "path": str(source_environment.resolve()),
                            "sha256": file_sha256(source_environment),
                        },
                        "training": {
                            "path": str(source_training.resolve()),
                            "sha256": file_sha256(source_training),
                        },
                    },
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        environment.close()
        print(
            f"Seed {seed}: {len(rewards)} episodes, stored {model.num_timesteps} "
            f"timesteps, final-20 reward {np.mean(final_window):.3f}",
            flush=True,
        )

    figure, axis = plt.subplots(figsize=(9.2, 5.2))
    colours = ["#2e86c1", "#d35400", "#239b56", "#7d3c98", "#566573"]
    for index, seed in enumerate(args.seeds):
        episodes, averages = moving_average(
            reward_histories[seed],
            MOVING_AVERAGE_WINDOW,
        )
        axis.plot(
            episodes,
            averages,
            linewidth=1.5,
            color=colours[index % len(colours)],
            label=f"Seed {seed}",
        )
    axis.set_xlabel("Training episode")
    axis.set_ylabel("20-episode moving-average reward")
    axis.set_title("Reproducible PPO training across three random seeds")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(
        args.output_dir / "training_curves_three_seeds.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)

    summary_path = args.output_dir / "training_runs_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(run_summaries[0].keys()))
        writer.writeheader()
        writer.writerows(run_summaries)

    (args.output_dir / "experiment_metadata.json").write_text(
        json.dumps(
            {
                "purpose": "three-seed reproducible PPO training for the pure-simulation report",
                "seeds": args.seeds,
                "moving_average_window_episodes": MOVING_AVERAGE_WINDOW,
                "runs": run_summaries,
                "interpretation_boundary": (
                    "All inputs and rewards are procedural. Results do not measure "
                    "real RGB-D camera or electrode performance."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Training outputs: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
