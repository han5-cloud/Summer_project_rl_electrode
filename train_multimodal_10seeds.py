"""Train 10 seeds for combined, RGB-only and height-only PPO policies.

The three ablation conditions use the same seven-value observation shape,
network, reward, action space, procedural scene distribution and training
budget.  RGB-only zeros the two height-derived values; height-only zeros the
two RGB-derived values.  This isolates input availability without changing
model capacity.  Every run is timed and stored in its own directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import gymnasium
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
import stable_baselines3
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

from step3_rl_environment_reproducible import (
    CONFIRM_THRESHOLD,
    HEIGHT_BACKGROUND_NOISE_STD,
    HEIGHT_DETECTION_THRESHOLD,
    OBSERVATION_MODES,
    RESIDUE_HEIGHT_MAX,
    RESIDUE_HEIGHT_MIN,
    ReproducibleElectrodeInspectionEnv,
)


DEFAULT_SEEDS = [101, 202, 303, 404, 505, 606, 707, 808, 909, 1010]
DEFAULT_MODES = ["combined", "rgb_only", "height_only"]
MOVING_AVERAGE_WINDOW = 20


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_reward_history(
    path: Path,
    rewards: list[float],
    lengths: list[int],
    elapsed_times: list[float],
) -> None:
    rows = [
        {
            "episode": episode,
            "reward": reward,
            "length": length,
            "elapsed_seconds": elapsed,
        }
        for episode, (reward, length, elapsed) in enumerate(
            zip(rewards, lengths, elapsed_times), start=1
        )
    ]
    write_csv(path, rows)


def train_one(
    observation_mode: str,
    seed: int,
    timesteps: int,
    output_dir_text: str,
) -> dict:
    """Train and persist one independent run; safe for a worker process."""

    output_dir = Path(output_dir_text)
    run_directory = output_dir / observation_mode / f"seed_{seed}"
    run_directory.mkdir(parents=True, exist_ok=False)

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    set_random_seed(seed, using_cuda=False)

    base_environment = ReproducibleElectrodeInspectionEnv(
        max_steps=15,
        base_seed=seed,
        observation_mode=observation_mode,
    )
    environment = Monitor(
        base_environment,
        filename=str(run_directory / "monitor.csv"),
    )
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
    model.learn(total_timesteps=timesteps)
    training_seconds = time.perf_counter() - start_time

    model_path = run_directory / "ppo_electrode_agent"
    model.save(model_path)
    model_zip_path = model_path.with_suffix(".zip")

    rewards = [float(value) for value in environment.get_episode_rewards()]
    lengths = [int(value) for value in environment.get_episode_lengths()]
    elapsed_times = [float(value) for value in environment.get_episode_times()]
    write_reward_history(
        run_directory / "episode_rewards.csv",
        rewards,
        lengths,
        elapsed_times,
    )

    first_window = rewards[:MOVING_AVERAGE_WINDOW]
    final_window = rewards[-MOVING_AVERAGE_WINDOW:]
    run_summary = {
        "observation_mode": observation_mode,
        "seed": seed,
        "requested_timesteps": timesteps,
        "stored_timesteps": int(model.num_timesteps),
        "episodes": len(rewards),
        "first_20_episode_mean_reward": float(np.mean(first_window)),
        "final_20_episode_mean_reward": float(np.mean(final_window)),
        "training_seconds": float(training_seconds),
        "model_relative_path": str(
            Path(observation_mode) / f"seed_{seed}" / model_zip_path.name
        ),
        "model_sha256": file_sha256(model_zip_path),
    }
    metadata = {
        **run_summary,
        "algorithm": "Stable-Baselines3 PPO with MlpPolicy",
        "policy_network": (
            "two 64-unit Tanh hidden layers for actor and critic "
            "(framework default)"
        ),
        "ablation_control": (
            "The observation remains seven-dimensional. Combined exposes all "
            "features; rgb_only zeros indices 4-5; height_only zeros indices 2-3."
        ),
        "hyperparameters": {
            "learning_rate": 3e-4,
            "n_steps": 256,
            "batch_size": 64,
            "gamma": 0.99,
            "device": "cpu",
            "torch_threads": 1,
            "deterministic_torch_algorithms": True,
        },
        "environment": {
            "actions": 4,
            "observation_values": 7,
            "observation_mode": observation_mode,
            "maximum_episode_steps": 15,
            "confirmation_coverage_threshold": CONFIRM_THRESHOLD,
            "height_units": (
                "dimensionless synthetic relative-height units (SRHU)"
            ),
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
                "path": "step3_rl_environment_reproducible.py",
                "sha256": file_sha256(
                    Path("step3_rl_environment_reproducible.py")
                ),
            },
            "training": {
                "path": Path(__file__).name,
                "sha256": file_sha256(Path(__file__)),
            },
        },
    }
    (run_directory / "training_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    environment.close()
    return run_summary


def load_completed_run(
    output_dir: Path,
    observation_mode: str,
    seed: int,
) -> dict | None:
    metadata_path = (
        output_dir / observation_mode / f"seed_{seed}" / "training_metadata.json"
    )
    model_path = (
        output_dir / observation_mode / f"seed_{seed}" / "ppo_electrode_agent.zip"
    )
    rewards_path = (
        output_dir / observation_mode / f"seed_{seed}" / "episode_rewards.csv"
    )
    if metadata_path.exists() and model_path.exists() and rewards_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        keys = [
            "observation_mode",
            "seed",
            "requested_timesteps",
            "stored_timesteps",
            "episodes",
            "first_20_episode_mean_reward",
            "final_20_episode_mean_reward",
            "training_seconds",
            "model_relative_path",
            "model_sha256",
        ]
        return {key: metadata[key] for key in keys}
    run_directory = output_dir / observation_mode / f"seed_{seed}"
    if run_directory.exists():
        raise RuntimeError(
            f"Incomplete run directory cannot be resumed safely: {run_directory}"
        )
    return None


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if len(values) < window:
        return values
    return np.convolve(values, np.ones(window) / window, mode="valid")


def plot_training_curves(
    output_dir: Path,
    modes: list[str],
    seeds: list[int],
) -> None:
    colours = {
        "combined": "#1f77b4",
        "rgb_only": "#d55e00",
        "height_only": "#009e73",
    }
    titles = {
        "combined": "Combined RGB and relative height",
        "rgb_only": "RGB-only",
        "height_only": "Relative-height-only",
    }
    figure, axes = plt.subplots(1, len(modes), figsize=(5.2 * len(modes), 4.6))
    axes = np.atleast_1d(axes)
    grid = np.linspace(0.0, 100.0, 201)
    for axis, mode in zip(axes, modes):
        interpolated: list[np.ndarray] = []
        for seed in seeds:
            rows = read_csv(
                output_dir / mode / f"seed_{seed}" / "episode_rewards.csv"
            )
            rewards = np.array([float(row["reward"]) for row in rows])
            average = moving_average(rewards, MOVING_AVERAGE_WINDOW)
            progress = np.linspace(0.0, 100.0, len(average))
            curve = np.interp(grid, progress, average)
            interpolated.append(curve)
            axis.plot(
                grid,
                curve,
                color=colours[mode],
                linewidth=0.55,
                alpha=0.20,
            )
        matrix = np.vstack(interpolated)
        mean = matrix.mean(axis=0)
        sample_sd = matrix.std(axis=0, ddof=1)
        axis.fill_between(
            grid,
            mean - sample_sd,
            mean + sample_sd,
            color=colours[mode],
            alpha=0.18,
            label="Mean +/- sample SD",
        )
        axis.plot(grid, mean, color=colours[mode], linewidth=2.2, label="Mean")
        axis.axhline(0.0, color="black", linewidth=0.7, alpha=0.5)
        axis.set_title(titles[mode])
        axis.set_xlabel("Normalised training progress (%)")
        axis.grid(alpha=0.22)
        axis.legend(loc="lower right", fontsize=8)
    axes[0].set_ylabel("20-episode moving-average reward")
    figure.suptitle("PPO training across 10 seeds per observation condition")
    figure.tight_layout()
    figure.savefig(
        output_dir / "training_curves_multimodal_10seeds.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(figure)


def cost_summary(run_summaries: list[dict]) -> list[dict]:
    output: list[dict] = []
    for mode in DEFAULT_MODES + ["all_conditions"]:
        subset = (
            run_summaries
            if mode == "all_conditions"
            else [row for row in run_summaries if row["observation_mode"] == mode]
        )
        if not subset:
            continue
        seconds = np.array([float(row["training_seconds"]) for row in subset])
        output.append(
            {
                "observation_mode": mode,
                "runs": len(subset),
                "requested_timesteps_total": sum(
                    int(row["requested_timesteps"]) for row in subset
                ),
                "stored_timesteps_total": sum(
                    int(row["stored_timesteps"]) for row in subset
                ),
                "training_seconds_total": float(seconds.sum()),
                "training_hours_total": float(seconds.sum() / 3600.0),
                "training_seconds_mean_per_run": float(seconds.mean()),
                "training_seconds_sample_sd": (
                    float(seconds.std(ddof=1)) if len(seconds) > 1 else 0.0
                ),
                "training_seconds_min": float(seconds.min()),
                "training_seconds_max": float(seconds.max()),
            }
        )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument(
        "--modes",
        choices=OBSERVATION_MODES,
        nargs="+",
        default=DEFAULT_MODES,
    )
    parser.add_argument("--timesteps", type=int, default=35_000)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments_multimodal_10seeds"),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse complete run directories and train only missing runs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Training seeds must be unique.")
    if len(set(args.modes)) != len(args.modes):
        raise ValueError("Observation modes must be unique.")
    if args.workers < 1:
        raise ValueError("--workers must be at least 1.")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.resume:
        raise FileExistsError(
            f"Output directory is not empty: {args.output_dir.resolve()}\n"
            "Use --resume to retain complete runs and train only missing runs."
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    run_summaries: list[dict] = []
    pending: list[tuple[str, int]] = []
    for mode in args.modes:
        for seed in args.seeds:
            completed = load_completed_run(args.output_dir, mode, seed)
            if completed is None:
                pending.append((mode, seed))
            else:
                run_summaries.append(completed)
                print(f"Reusing completed {mode} seed {seed}", flush=True)

    experiment_start = time.perf_counter()
    if pending:
        print(
            f"Training {len(pending)} runs with {min(args.workers, len(pending))} "
            "worker process(es) ...",
            flush=True,
        )
        with ProcessPoolExecutor(
            max_workers=min(args.workers, len(pending))
        ) as executor:
            futures = {
                executor.submit(
                    train_one,
                    mode,
                    seed,
                    args.timesteps,
                    str(args.output_dir),
                ): (mode, seed)
                for mode, seed in pending
            }
            for future in as_completed(futures):
                mode, seed = futures[future]
                summary = future.result()
                run_summaries.append(summary)
                print(
                    f"Completed {mode} seed {seed}: "
                    f"{summary['training_seconds']:.1f} s, "
                    f"final-20 reward {summary['final_20_episode_mean_reward']:.3f}",
                    flush=True,
                )
    experiment_wall_seconds = time.perf_counter() - experiment_start

    mode_rank = {mode: index for index, mode in enumerate(DEFAULT_MODES)}
    run_summaries.sort(
        key=lambda row: (mode_rank[str(row["observation_mode"])], int(row["seed"]))
    )
    expected_count = len(args.modes) * len(args.seeds)
    if len(run_summaries) != expected_count:
        raise RuntimeError(
            f"Expected {expected_count} completed runs, found {len(run_summaries)}."
        )

    write_csv(args.output_dir / "training_runs_summary.csv", run_summaries)
    costs = cost_summary(run_summaries)
    write_csv(args.output_dir / "training_cost_summary.csv", costs)
    plot_training_curves(args.output_dir, args.modes, args.seeds)

    metadata = {
        "experiment_type": "three_observation_conditions_by_ten_training_seeds",
        "observation_modes": args.modes,
        "training_seeds": args.seeds,
        "runs": expected_count,
        "requested_timesteps_per_run": args.timesteps,
        "worker_processes": min(args.workers, max(1, len(pending))),
        "wall_seconds_for_this_invocation": experiment_wall_seconds,
        "timing_definition": (
            "Per-run wall-clock seconds measured around model.learn using "
            "time.perf_counter. Concurrent workers may share CPU resources."
        ),
        "cost_summary": costs,
    }
    (args.output_dir / "experiment_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    print(f"Outputs: {args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
