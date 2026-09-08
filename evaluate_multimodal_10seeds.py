"""Evaluate 30 PPO policies on shared scenes with paired uncertainty.

This script compares combined, RGB-only and relative-height-only PPO policies
trained with the same 10 seeds.  Every policy and baseline receives the same
1,000 scene/starting-position seeds.  It records evaluation time, estimates
scene-level confidence intervals, reports variation across training seeds and
uses a hierarchical paired bootstrap over training seeds and shared scenes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import scipy
import torch
from scipy.stats import t as student_t
from stable_baselines3 import PPO

from evaluate_three_seed_models import (
    EpisodeResult,
    calibrate_sweep_threshold,
    evaluate_modalities,
    evaluate_random,
    evaluate_sweep,
    plot_example,
    run_episode,
    summarise_results,
)
from step3_rl_environment_reproducible import (
    BIG_SIZE,
    CONFIRM_THRESHOLD,
    WINDOW_SIZE,
    ReproducibleElectrodeInspectionEnv,
)


DEFAULT_SEEDS = [101, 202, 303, 404, 505, 606, 707, 808, 909, 1010]
DEFAULT_MODES = ["combined", "rgb_only", "height_only"]
BOOTSTRAP_RESAMPLES = 5000
FAILURE_SCENE_SEED = 2_215_982
MODE_LABELS = {
    "combined": "Combined",
    "rgb_only": "RGB-only",
    "height_only": "Relative-height-only",
}
MODE_COLOURS = {
    "combined": "#1f77b4",
    "rgb_only": "#d55e00",
    "height_only": "#009e73",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def policy_label(mode: str, seed: int) -> str:
    return f"ppo_{mode}_seed_{seed}"


def evaluate_model_task(
    mode: str,
    seed: int,
    model_path_text: str,
    scene_seeds: np.ndarray,
) -> tuple[list[EpisodeResult], dict, float]:
    """Load and evaluate one model in a worker process."""

    torch.set_num_threads(1)
    model_path = Path(model_path_text)
    model = PPO.load(model_path, device="cpu")
    if tuple(model.observation_space.shape) != (7,):
        raise ValueError(f"Unexpected observation space in {model_path}")
    if int(model.action_space.n) != 4:
        raise ValueError(f"Unexpected action space in {model_path}")

    environment = ReproducibleElectrodeInspectionEnv(
        max_steps=15,
        base_seed=0,
        observation_mode=mode,
    )
    label = policy_label(mode, seed)
    started = time.perf_counter()
    rows = [
        run_episode(environment, int(scene_seed), label, model)
        for scene_seed in scene_seeds
    ]
    elapsed = time.perf_counter() - started
    environment.close()
    metadata = {
        "observation_mode": mode,
        "training_seed": seed,
        "model_relative_path": str(
            Path(mode) / f"seed_{seed}" / model_path.name
        ),
        "model_sha256": file_sha256(model_path),
        "stored_timesteps": int(model.num_timesteps),
    }
    return rows, metadata, elapsed


def metric_values(rows: list[EpisodeResult], metric: str) -> np.ndarray:
    if metric == "success_rate":
        return np.array([row.success for row in rows], dtype=float)
    if metric == "false_confirmation_rate":
        return np.array([row.false_confirmation for row in rows], dtype=float)
    if metric == "timeout_rate":
        return np.array([row.timeout for row in rows], dtype=float)
    if metric == "mean_total_reward":
        return np.array([row.total_reward for row in rows], dtype=float)
    if metric == "mean_steps":
        return np.array([row.steps for row in rows], dtype=float)
    raise KeyError(metric)


def across_seed_summaries(
    policy_summary: list[dict],
    modes: list[str],
    seeds: list[int],
) -> tuple[list[dict], list[dict]]:
    detailed: list[dict] = []
    compact: list[dict] = []
    metrics = (
        "success_rate",
        "false_confirmation_rate",
        "timeout_rate",
        "mean_total_reward",
        "mean_steps",
    )
    by_policy = {str(row["policy"]): row for row in policy_summary}
    for mode in modes:
        compact_row: dict[str, str | int | float] = {
            "observation_mode": mode,
            "training_seed_count": len(seeds),
        }
        for metric in metrics:
            values = np.array(
                [
                    float(by_policy[policy_label(mode, seed)][metric])
                    for seed in seeds
                ],
                dtype=float,
            )
            mean = float(values.mean())
            sample_sd = float(values.std(ddof=1))
            half_width = float(
                student_t.ppf(0.975, df=len(values) - 1)
                * sample_sd
                / np.sqrt(len(values))
            )
            detailed.append(
                {
                    "observation_mode": mode,
                    "metric": metric,
                    "training_seed_count": len(values),
                    "mean_across_training_seeds": mean,
                    "sample_sd_across_training_seeds": sample_sd,
                    "minimum_training_seed_result": float(values.min()),
                    "maximum_training_seed_result": float(values.max()),
                    "t_95_ci_low": mean - half_width,
                    "t_95_ci_high": mean + half_width,
                }
            )
            compact_row[f"{metric}_mean"] = mean
            compact_row[f"{metric}_sample_sd"] = sample_sd
            compact_row[f"{metric}_min"] = float(values.min())
            compact_row[f"{metric}_max"] = float(values.max())
            compact_row[f"{metric}_t_ci_low"] = mean - half_width
            compact_row[f"{metric}_t_ci_high"] = mean + half_width
        compact.append(compact_row)
    return detailed, compact


def success_matrix(
    episode_rows: list[EpisodeResult],
    mode: str,
    seeds: list[int],
    scene_seeds: np.ndarray,
) -> np.ndarray:
    matrix = np.empty((len(seeds), len(scene_seeds)), dtype=float)
    scene_index = {int(seed): index for index, seed in enumerate(scene_seeds)}
    seed_index = {seed: index for index, seed in enumerate(seeds)}
    expected_labels = {policy_label(mode, seed): seed for seed in seeds}
    counts = np.zeros_like(matrix, dtype=int)
    for row in episode_rows:
        if row.policy not in expected_labels:
            continue
        i = seed_index[expected_labels[row.policy]]
        j = scene_index[row.scene_seed]
        matrix[i, j] = row.success
        counts[i, j] += 1
    if not np.all(counts == 1):
        raise RuntimeError(f"Incomplete shared-scene matrix for {mode}")
    return matrix


def baseline_success_vector(
    episode_rows: list[EpisodeResult],
    label: str,
    scene_seeds: np.ndarray,
) -> np.ndarray:
    by_scene = {
        row.scene_seed: float(row.success)
        for row in episode_rows
        if row.policy == label
    }
    if len(by_scene) != len(scene_seeds):
        raise RuntimeError(f"Incomplete baseline results for {label}")
    return np.array([by_scene[int(seed)] for seed in scene_seeds], dtype=float)


def paired_bootstrap(
    matrices: dict[str, np.ndarray],
    fixed_success: np.ndarray,
    seed: int,
) -> list[dict]:
    """Hierarchical paired bootstrap over training seeds and shared scenes."""

    comparisons = [
        ("combined", "fixed_sweep"),
        ("rgb_only", "fixed_sweep"),
        ("height_only", "fixed_sweep"),
        ("combined", "rgb_only"),
        ("combined", "height_only"),
        ("height_only", "rgb_only"),
    ]
    rng = np.random.default_rng(seed)
    training_seed_count, scene_count = matrices["combined"].shape
    output: list[dict] = []
    for left, right in comparisons:
        if right == "fixed_sweep":
            estimate = float(matrices[left].mean() - fixed_success.mean())
        else:
            estimate = float(matrices[left].mean() - matrices[right].mean())
        values = np.empty(BOOTSTRAP_RESAMPLES, dtype=float)
        for index in range(BOOTSTRAP_RESAMPLES):
            sampled_training = rng.integers(
                0, training_seed_count, size=training_seed_count
            )
            sampled_scenes = rng.integers(0, scene_count, size=scene_count)
            left_value = float(
                matrices[left][sampled_training][:, sampled_scenes].mean()
            )
            if right == "fixed_sweep":
                right_value = float(fixed_success[sampled_scenes].mean())
            else:
                right_value = float(
                    matrices[right][sampled_training][:, sampled_scenes].mean()
                )
            values[index] = left_value - right_value
        low, high = np.percentile(values, [2.5, 97.5])
        output.append(
            {
                "comparison": f"{left}_minus_{right}",
                "left": left,
                "right": right,
                "training_seed_count": training_seed_count,
                "shared_scene_count": scene_count,
                "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
                "mean_paired_difference": estimate,
                "difference_percentage_points": 100.0 * estimate,
                "ci_low": float(low),
                "ci_high": float(high),
                "ci_low_percentage_points": 100.0 * float(low),
                "ci_high_percentage_points": 100.0 * float(high),
            }
        )
    return output


def plot_ablation(
    path: Path,
    policy_summary: list[dict],
    paired_summary: list[dict],
    modes: list[str],
    seeds: list[int],
) -> None:
    by_policy = {str(row["policy"]): row for row in policy_summary}
    fixed = by_policy["fixed_sweep"]
    random = by_policy["random"]
    figure, axes = plt.subplots(2, 2, figsize=(11.4, 8.2))
    metrics = [
        ("success_rate", "Success rate"),
        ("false_confirmation_rate", "False-confirmation rate"),
        ("timeout_rate", "Timeout rate"),
    ]
    rng = np.random.default_rng(7129)
    for axis, (metric, title) in zip(axes.flat[:3], metrics):
        values_by_mode = [
            [
                100.0 * float(by_policy[policy_label(mode, seed)][metric])
                for seed in seeds
            ]
            for mode in modes
        ]
        box = axis.boxplot(
            values_by_mode,
            positions=np.arange(len(modes)),
            widths=0.48,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "black", "linewidth": 1.4},
        )
        for patch, mode in zip(box["boxes"], modes):
            patch.set_facecolor(MODE_COLOURS[mode])
            patch.set_alpha(0.27)
        for mode_index, (mode, values) in enumerate(zip(modes, values_by_mode)):
            jitter = rng.uniform(-0.10, 0.10, size=len(values))
            axis.scatter(
                mode_index + jitter,
                values,
                s=27,
                color=MODE_COLOURS[mode],
                edgecolor="white",
                linewidth=0.4,
                zorder=3,
            )
        axis.axhline(
            100.0 * float(fixed[metric]),
            color="#e69f00",
            linewidth=1.5,
            linestyle="--",
            label="Fixed sweep",
        )
        random_value = 100.0 * float(random[metric])
        if metric == "false_confirmation_rate":
            axis.text(
                0.02,
                0.96,
                f"Random: {random_value:.1f}% (off scale)",
                transform=axis.transAxes,
                ha="left",
                va="top",
                fontsize=8,
                color="#666666",
            )
            upper_limit = max(10.0, 1.15 * max(max(values) for values in values_by_mode))
            axis.set_ylim(-0.5, upper_limit)
        else:
            axis.axhline(
                random_value,
                color="#777777",
                linewidth=1.1,
                linestyle=":",
                label="Random",
            )
        axis.set_xticks(
            np.arange(len(modes)),
            [MODE_LABELS[mode] for mode in modes],
            rotation=10,
        )
        axis.set_ylabel("Rate (%)")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.22)
        axis.legend(fontsize=8, loc="best")

    axis = axes.flat[3]
    labels = []
    estimates = []
    low_errors = []
    high_errors = []
    for row in paired_summary:
        left = MODE_LABELS.get(str(row["left"]), str(row["left"]))
        right = MODE_LABELS.get(str(row["right"]), "Fixed sweep")
        labels.append(f"{left} - {right}")
        estimate = float(row["difference_percentage_points"])
        estimates.append(estimate)
        low_errors.append(estimate - float(row["ci_low_percentage_points"]))
        high_errors.append(float(row["ci_high_percentage_points"]) - estimate)
    y = np.arange(len(labels))
    axis.errorbar(
        estimates,
        y,
        xerr=np.vstack([low_errors, high_errors]),
        fmt="o",
        color="#333333",
        ecolor="#555555",
        capsize=3,
    )
    axis.axvline(0.0, color="black", linewidth=0.9)
    axis.set_yticks(y, labels, fontsize=8)
    axis.invert_yaxis()
    axis.set_xlabel("Paired success-rate difference (percentage points)")
    axis.set_title("Hierarchical paired bootstrap (95% CI)")
    axis.grid(axis="x", alpha=0.22)

    figure.tight_layout()
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def trace_model(model: PPO, training_seed: int, scene_seed: int) -> dict:
    environment = ReproducibleElectrodeInspectionEnv(
        max_steps=15,
        base_seed=0,
        observation_mode="combined",
    )
    observation, _ = environment.reset(seed=scene_seed)
    trace = {
        "training_seed": training_seed,
        "scene_seed": scene_seed,
        "steps": [],
        "x": [],
        "y": [],
        "true_ratio": [],
        "rgb_ratio": [],
        "height_ratio": [],
        "actions": [],
        "rewards": [],
    }
    for step in range(1, environment.max_steps + 1):
        trace["steps"].append(step)
        trace["x"].append(environment.x)
        trace["y"].append(environment.y)
        trace["true_ratio"].append(environment._true_residue_ratio())
        trace["rgb_ratio"].append(float(observation[2]))
        trace["height_ratio"].append(float(observation[5]))
        action_array, _ = model.predict(observation, deterministic=True)
        action = int(action_array)
        trace["actions"].append(action)
        observation, reward, terminated, truncated, _ = environment.step(action)
        trace["rewards"].append(float(reward))
        if terminated or truncated:
            break
    trace["rgb"] = np.asarray(environment.rgb_image)
    trace["height"] = np.asarray(environment.relative_height_map)
    trace["truth"] = np.asarray(environment.ground_truth_mask)
    trace["stopped"] = bool(trace["actions"][-1] == 3)
    trace["success"] = bool(
        trace["stopped"] and trace["true_ratio"][-1] > CONFIRM_THRESHOLD
    )
    trace["false_confirmation"] = bool(
        trace["stopped"] and trace["true_ratio"][-1] <= CONFIRM_THRESHOLD
    )
    trace["total_reward"] = float(sum(trace["rewards"]))
    environment.close()
    return trace


def plot_failure_case(
    path: Path,
    success_trace: dict,
    failure_trace: dict,
) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(12.0, 7.6))
    traces = [success_trace, failure_trace]
    colours = ["#009e73", "#d62728"]
    titles = [
        f"Successful stop: combined seed {success_trace['training_seed']}",
        f"False confirmation: combined seed {failure_trace['training_seed']}",
    ]
    for axis, trace, colour, title in zip(axes[0, :2], traces, colours, titles):
        axis.imshow(trace["rgb"])
        axis.contour(trace["truth"], levels=[0.5], colors=["#ffd92f"], linewidths=0.8)
        centres_x = np.array(trace["x"]) + WINDOW_SIZE / 2
        centres_y = np.array(trace["y"]) + WINDOW_SIZE / 2
        axis.plot(centres_x, centres_y, color=colour, linewidth=1.8, marker="o", markersize=3)
        axis.add_patch(
            Rectangle(
                (trace["x"][-1], trace["y"][-1]),
                WINDOW_SIZE,
                WINDOW_SIZE,
                fill=False,
                edgecolor=colour,
                linewidth=2.2,
            )
        )
        axis.set_xlim(0, BIG_SIZE)
        axis.set_ylim(BIG_SIZE, 0)
        axis.set_title(title, fontsize=10)
        axis.set_xlabel("x (pixels)")
        axis.set_ylabel("conveyor position y (pixels)")

    axis = axes[0, 2]
    for trace, colour, label in zip(traces, colours, ["Successful stop", "False confirmation"]):
        axis.plot(
            trace["steps"],
            100.0 * np.array(trace["true_ratio"]),
            color=colour,
            marker="o",
            linewidth=1.8,
            label=label,
        )
    axis.axhline(
        100.0 * CONFIRM_THRESHOLD,
        color="black",
        linestyle="--",
        linewidth=1.1,
        label="Confirmation threshold",
    )
    axis.set_xlabel("Decision step")
    axis.set_ylabel("True residue coverage in window (%)")
    axis.set_title("Coverage at each decision")
    axis.grid(alpha=0.22)
    axis.legend(fontsize=8)

    for axis, trace, colour, title in zip(axes[1, :2], traces, colours, titles):
        x = int(trace["x"][-1])
        y = int(trace["y"][-1])
        rgb_crop = trace["rgb"][y : y + WINDOW_SIZE, x : x + WINDOW_SIZE]
        truth_crop = trace["truth"][y : y + WINDOW_SIZE, x : x + WINDOW_SIZE]
        axis.imshow(rgb_crop)
        if truth_crop.any() and not truth_crop.all():
            axis.contour(truth_crop, levels=[0.5], colors=[colour], linewidths=1.5)
        axis.set_title(
            f"Final window: {100.0 * trace['true_ratio'][-1]:.1f}% true coverage",
            fontsize=10,
        )
        axis.axis("off")

    axis = axes[1, 2]
    labels = ["True coverage", "RGB feature ratio", "Height feature ratio"]
    x_positions = np.arange(len(labels))
    width = 0.34
    for offset, trace, colour, label in zip(
        [-width / 2, width / 2],
        traces,
        colours,
        [f"Seed {success_trace['training_seed']}", f"Seed {failure_trace['training_seed']}"],
    ):
        values = 100.0 * np.array(
            [
                trace["true_ratio"][-1],
                trace["rgb_ratio"][-1],
                trace["height_ratio"][-1],
            ]
        )
        axis.bar(x_positions + offset, values, width, color=colour, alpha=0.78, label=label)
    axis.axhline(
        100.0 * CONFIRM_THRESHOLD,
        color="black",
        linestyle="--",
        linewidth=1.0,
    )
    axis.set_xticks(x_positions, labels, rotation=16, ha="right")
    axis.set_ylabel("Window proportion (%)")
    axis.set_title("Final-window observations")
    axis.grid(axis="y", alpha=0.22)
    axis.legend(fontsize=8)

    figure.tight_layout()
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def failure_case_outputs(
    experiment_dir: Path,
    output_dir: Path,
    seeds: list[int],
) -> dict:
    traces: dict[int, dict] = {}
    for seed in seeds:
        model = PPO.load(
            experiment_dir / "combined" / f"seed_{seed}" / "ppo_electrode_agent.zip",
            device="cpu",
        )
        traces[seed] = trace_model(model, seed, FAILURE_SCENE_SEED)
    successful = [trace for trace in traces.values() if trace["success"]]
    failures = [trace for trace in traces.values() if trace["false_confirmation"]]
    if not successful:
        raise RuntimeError(
            f"No combined policy succeeded on scene {FAILURE_SCENE_SEED}."
        )
    if not failures:
        raise RuntimeError(
            f"No combined policy falsely confirmed scene {FAILURE_SCENE_SEED}."
        )
    success_trace = traces.get(101) if traces.get(101, {}).get("success") else successful[0]
    failure_trace = (
        traces.get(202)
        if traces.get(202, {}).get("false_confirmation")
        else failures[0]
    )
    plot_failure_case(
        output_dir / f"failure_case_seed_{FAILURE_SCENE_SEED}.png",
        success_trace,
        failure_trace,
    )
    rows: list[dict] = []
    for outcome, trace in (("successful_stop", success_trace), ("false_confirmation", failure_trace)):
        rows.append(
            {
                "scene_seed": FAILURE_SCENE_SEED,
                "outcome": outcome,
                "observation_mode": "combined",
                "training_seed": trace["training_seed"],
                "steps": len(trace["steps"]),
                "actions": "-".join(str(value) for value in trace["actions"]),
                "final_x": trace["x"][-1],
                "final_y": trace["y"][-1],
                "final_true_ratio": trace["true_ratio"][-1],
                "final_rgb_texture_ratio": trace["rgb_ratio"][-1],
                "final_height_ratio": trace["height_ratio"][-1],
                "total_reward": trace["total_reward"],
            }
        )
    write_csv(output_dir / f"failure_case_seed_{FAILURE_SCENE_SEED}.csv", rows)
    return {
        "scene_seed": FAILURE_SCENE_SEED,
        "successful_training_seed": success_trace["training_seed"],
        "false_confirmation_training_seed": failure_trace["training_seed"],
        "successful_final_true_ratio": success_trace["true_ratio"][-1],
        "false_confirmation_final_true_ratio": failure_trace["true_ratio"][-1],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=Path("experiments_multimodal_10seeds"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("evaluation_outputs_multimodal_10seeds"),
    )
    parser.add_argument("--training-seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--modes", nargs="+", choices=DEFAULT_MODES, default=DEFAULT_MODES)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--calibration-episodes", type=int, default=100)
    parser.add_argument("--modality-scenes", type=int, default=200)
    parser.add_argument("--evaluation-seed", type=int, default=20260827)
    parser.add_argument("--workers", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {args.output_dir.resolve()}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)

    random_generator = np.random.default_rng(args.evaluation_seed)
    calibration_seeds = random_generator.integers(
        1_000_000, 1_999_999, size=args.calibration_episodes
    )
    evaluation_seeds = random_generator.integers(
        2_000_000, 8_999_999, size=args.episodes
    )
    modality_seeds = random_generator.integers(
        9_000_000, 9_999_999, size=args.modality_scenes
    )

    timing_rows: list[dict] = []
    total_start = time.perf_counter()
    started = time.perf_counter()
    print("Calibrating fixed-sweep threshold ...", flush=True)
    sweep_threshold, calibration_rows = calibrate_sweep_threshold(calibration_seeds)
    timing_rows.append(
        {
            "phase": "calibration",
            "label": "fixed_sweep_threshold_search",
            "episodes_or_items": len(calibration_seeds) * len(calibration_rows),
            "wall_seconds": time.perf_counter() - started,
        }
    )

    episode_rows: list[EpisodeResult] = []
    started = time.perf_counter()
    random_rows = evaluate_random(evaluation_seeds)
    timing_rows.append(
        {
            "phase": "evaluation",
            "label": "random",
            "episodes_or_items": len(random_rows),
            "wall_seconds": time.perf_counter() - started,
        }
    )
    episode_rows.extend(random_rows)

    started = time.perf_counter()
    fixed_rows = evaluate_sweep(evaluation_seeds, sweep_threshold)
    timing_rows.append(
        {
            "phase": "evaluation",
            "label": "fixed_sweep",
            "episodes_or_items": len(fixed_rows),
            "wall_seconds": time.perf_counter() - started,
        }
    )
    episode_rows.extend(fixed_rows)

    jobs: list[tuple[str, int, Path]] = []
    for mode in args.modes:
        for seed in args.training_seeds:
            model_path = (
                args.experiment_dir / mode / f"seed_{seed}" / "ppo_electrode_agent.zip"
            )
            if not model_path.exists():
                raise FileNotFoundError(model_path)
            jobs.append((mode, seed, model_path))

    model_metadata: list[dict] = []
    print(
        f"Evaluating {len(jobs)} PPO models with {min(args.workers, len(jobs))} workers ...",
        flush=True,
    )
    with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs))) as executor:
        futures = {
            executor.submit(
                evaluate_model_task,
                mode,
                seed,
                str(model_path),
                evaluation_seeds,
            ): (mode, seed)
            for mode, seed, model_path in jobs
        }
        for future in as_completed(futures):
            mode, seed = futures[future]
            rows, metadata, elapsed = future.result()
            episode_rows.extend(rows)
            model_metadata.append(metadata)
            timing_rows.append(
                {
                    "phase": "evaluation",
                    "label": policy_label(mode, seed),
                    "episodes_or_items": len(rows),
                    "wall_seconds": elapsed,
                }
            )
            print(
                f"Completed {MODE_LABELS[mode]} seed {seed}: "
                f"success={100.0 * np.mean([row.success for row in rows]):.1f}%, "
                f"{elapsed:.1f} s",
                flush=True,
            )

    policy_order = ["random", "fixed_sweep"] + [
        policy_label(mode, seed)
        for mode in args.modes
        for seed in args.training_seeds
    ]
    policy_rank = {label: index for index, label in enumerate(policy_order)}
    episode_rows.sort(key=lambda row: (policy_rank[row.policy], row.scene_seed))
    policy_summary = summarise_results(
        episode_rows,
        policy_order,
        args.evaluation_seed + 10_000,
    )
    across_detailed, condition_summary = across_seed_summaries(
        policy_summary,
        args.modes,
        args.training_seeds,
    )

    matrices = {
        mode: success_matrix(
            episode_rows, mode, args.training_seeds, evaluation_seeds
        )
        for mode in args.modes
    }
    fixed_success = baseline_success_vector(
        episode_rows, "fixed_sweep", evaluation_seeds
    )
    started = time.perf_counter()
    paired_summary = paired_bootstrap(
        matrices,
        fixed_success,
        args.evaluation_seed + 30_000,
    )
    timing_rows.append(
        {
            "phase": "statistics",
            "label": "hierarchical_paired_bootstrap",
            "episodes_or_items": BOOTSTRAP_RESAMPLES,
            "wall_seconds": time.perf_counter() - started,
        }
    )

    print("Evaluating procedural masks ...", flush=True)
    started = time.perf_counter()
    modality_summary, example = evaluate_modalities(
        modality_seeds,
        args.evaluation_seed + 20_000,
    )
    timing_rows.append(
        {
            "phase": "evaluation",
            "label": "procedural_mask_benchmark",
            "episodes_or_items": len(modality_seeds),
            "wall_seconds": time.perf_counter() - started,
        }
    )

    print(f"Generating failure case for scene {FAILURE_SCENE_SEED} ...", flush=True)
    started = time.perf_counter()
    failure_metadata = failure_case_outputs(
        args.experiment_dir,
        args.output_dir,
        args.training_seeds,
    )
    timing_rows.append(
        {
            "phase": "qualitative_analysis",
            "label": f"failure_case_scene_{FAILURE_SCENE_SEED}",
            "episodes_or_items": len(args.training_seeds),
            "wall_seconds": time.perf_counter() - started,
        }
    )

    write_csv(
        args.output_dir / "episode_results.csv",
        [asdict(row) for row in episode_rows],
    )
    write_csv(args.output_dir / "policy_summary.csv", policy_summary)
    write_csv(args.output_dir / "modality_across_seeds.csv", across_detailed)
    write_csv(args.output_dir / "policy_condition_summary.csv", condition_summary)
    write_csv(args.output_dir / "paired_bootstrap_summary.csv", paired_summary)
    write_csv(args.output_dir / "modality_summary.csv", modality_summary)
    write_csv(args.output_dir / "sweep_calibration.csv", calibration_rows)
    timing_rows.append(
        {
            "phase": "overall",
            "label": "complete_evaluation_workflow",
            "episodes_or_items": len(episode_rows),
            "wall_seconds": time.perf_counter() - total_start,
        }
    )
    write_csv(args.output_dir / "evaluation_timing.csv", timing_rows)

    plot_ablation(
        args.output_dir / "policy_ablation_and_paired_bootstrap.png",
        policy_summary,
        paired_summary,
        args.modes,
        args.training_seeds,
    )
    plot_example(
        args.output_dir / "synthetic_rgb_relative_height_example.png",
        example,
    )

    model_metadata.sort(
        key=lambda row: (
            DEFAULT_MODES.index(str(row["observation_mode"])),
            int(row["training_seed"]),
        )
    )
    metadata = {
        "evaluation_type": "three_observation_conditions_by_ten_training_seeds",
        "evaluation_seed": args.evaluation_seed,
        "episodes_per_policy": args.episodes,
        "calibration_episodes": args.calibration_episodes,
        "modality_scenes": args.modality_scenes,
        "scene_bootstrap_resamples": 3000,
        "hierarchical_paired_bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "paired_bootstrap_units": ["training_seed", "shared_scene"],
        "fixed_sweep_threshold": sweep_threshold,
        "training_models": model_metadata,
        "condition_summary": condition_summary,
        "paired_bootstrap_summary": paired_summary,
        "failure_case": failure_metadata,
        "timing_definition": (
            "Wall-clock seconds measured with time.perf_counter. PPO models "
            "were evaluated concurrently with one PyTorch thread per worker."
        ),
        "source_files": {
            "environment": {
                "path": "step3_rl_environment_reproducible.py",
                "sha256": file_sha256(Path("step3_rl_environment_reproducible.py")),
            },
            "training": {
                "path": "train_multimodal_10seeds.py",
                "sha256": file_sha256(Path("train_multimodal_10seeds.py")),
            },
            "evaluation": {
                "path": Path(__file__).name,
                "sha256": file_sha256(Path(__file__)),
            },
        },
    }
    (args.output_dir / "evaluation_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    readme_lines = [
        "# Ten-seed policy ablation and paired evaluation",
        "",
        f"All policies were evaluated on the same {args.episodes:,} scene seeds.",
        f"Fixed-sweep calibration threshold: {sweep_threshold:.3f}",
        "",
        "The hierarchical paired bootstrap resamples both the 10 training seeds and the shared scenes.",
        "",
        "| Condition | Mean success | Sample SD | Mean false confirmation | Mean timeout |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in condition_summary:
        readme_lines.append(
            "| {label} | {success:.2f}% | {sd:.2f} pp | {false:.2f}% | {timeout:.2f}% |".format(
                label=MODE_LABELS[str(row["observation_mode"])],
                success=100.0 * float(row["success_rate_mean"]),
                sd=100.0 * float(row["success_rate_sample_sd"]),
                false=100.0 * float(row["false_confirmation_rate_mean"]),
                timeout=100.0 * float(row["timeout_rate_mean"]),
            )
        )
    (args.output_dir / "README.md").write_text(
        "\n".join(readme_lines) + "\n",
        encoding="utf-8",
    )
    print(f"Outputs: {args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
