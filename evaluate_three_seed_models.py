"""Evaluate three reproducibly trained PPO models on shared synthetic scenes.

This evaluation reports explicit success, false-confirmation and timeout
outcomes.  It also compares RGB-texture and synthetic relative-height masks
with the procedural ground-truth mask.  It does not use real camera or battery
electrode data.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO

from rl_environment import (
    CONFIRM_THRESHOLD,
    HEIGHT_DETECTION_THRESHOLD,
    ReproducibleElectrodeInspectionEnv,
    local_std,
    make_synthetic_electrode,
)


ACTION_LEFT = 0
ACTION_RIGHT = 1
ACTION_STAY = 2
ACTION_STOP = 3
DEFAULT_TRAINING_SEEDS = [101, 202, 303]
BOOTSTRAP_RESAMPLES = 3000


@dataclass
class EpisodeResult:
    policy: str
    scene_seed: int
    success: int
    false_confirmation: int
    timeout: int
    stopped: int
    steps: int
    total_reward: float
    final_true_ratio: float
    final_rgb_texture_ratio: float
    final_height_ratio: float
    actions: str


class FixedSweepPolicy:
    """Pre-programmed lateral sweep with a threshold-triggered stop action."""

    def __init__(self, threshold: float):
        self.threshold = float(threshold)
        self.horizontal_action = ACTION_RIGHT

    def reset(self, observation: np.ndarray) -> None:
        self.horizontal_action = (
            ACTION_RIGHT if float(observation[0]) < 0.5 else ACTION_LEFT
        )

    def predict(self, observation: np.ndarray) -> int:
        detector_score = max(float(observation[2]), float(observation[5]))
        if detector_score >= self.threshold:
            return ACTION_STOP

        x_normalised = float(observation[0])
        if self.horizontal_action == ACTION_RIGHT and x_normalised >= 0.98:
            self.horizontal_action = ACTION_LEFT
        elif self.horizontal_action == ACTION_LEFT and x_normalised <= 0.02:
            self.horizontal_action = ACTION_RIGHT
        return self.horizontal_action


def run_episode(
    environment: ReproducibleElectrodeInspectionEnv,
    scene_seed: int,
    policy_label: str,
    action_selector,
) -> EpisodeResult:
    observation, _ = environment.reset(seed=int(scene_seed))
    actions: list[int] = []
    total_reward = 0.0
    stopped = False
    final_info = {"true_residue_ratio": 0.0}

    if hasattr(action_selector, "reset"):
        action_selector.reset(observation)

    for step_index in range(environment.max_steps):
        if isinstance(action_selector, PPO):
            action_array, _ = action_selector.predict(
                observation,
                deterministic=True,
            )
            action = int(action_array)
        elif callable(action_selector):
            action = int(action_selector(observation))
        else:
            action = int(action_selector.predict(observation))

        actions.append(action)
        observation, reward, terminated, truncated, final_info = environment.step(
            action
        )
        total_reward += float(reward)
        if action == ACTION_STOP:
            stopped = True
        if terminated or truncated:
            break

    true_ratio = float(final_info["true_residue_ratio"])
    success = int(stopped and true_ratio > CONFIRM_THRESHOLD)
    false_confirmation = int(stopped and true_ratio <= CONFIRM_THRESHOLD)
    timeout = int(not stopped)

    return EpisodeResult(
        policy=policy_label,
        scene_seed=int(scene_seed),
        success=success,
        false_confirmation=false_confirmation,
        timeout=timeout,
        stopped=int(stopped),
        steps=step_index + 1,
        total_reward=total_reward,
        final_true_ratio=true_ratio,
        final_rgb_texture_ratio=float(observation[2]),
        final_height_ratio=float(observation[5]),
        actions="-".join(str(action) for action in actions),
    )


def evaluate_random(scene_seeds: np.ndarray) -> list[EpisodeResult]:
    environment = ReproducibleElectrodeInspectionEnv(max_steps=15, base_seed=0)
    rows: list[EpisodeResult] = []
    for scene_seed in scene_seeds:
        action_rng = np.random.default_rng(int(scene_seed) + 71_003)
        rows.append(
            run_episode(
                environment,
                int(scene_seed),
                "random",
                lambda _observation, rng=action_rng: int(rng.integers(0, 4)),
            )
        )
    environment.close()
    return rows


def evaluate_sweep(
    scene_seeds: np.ndarray,
    threshold: float,
    policy_label: str = "fixed_sweep",
) -> list[EpisodeResult]:
    environment = ReproducibleElectrodeInspectionEnv(max_steps=15, base_seed=0)
    policy = FixedSweepPolicy(threshold)
    rows = [
        run_episode(environment, int(seed), policy_label, policy)
        for seed in scene_seeds
    ]
    environment.close()
    return rows


def evaluate_ppo(
    scene_seeds: np.ndarray,
    model: PPO,
    training_seed: int,
) -> list[EpisodeResult]:
    environment = ReproducibleElectrodeInspectionEnv(max_steps=15, base_seed=0)
    policy_label = f"ppo_seed_{training_seed}"
    rows = [
        run_episode(environment, int(seed), policy_label, model)
        for seed in scene_seeds
    ]
    environment.close()
    return rows


def bootstrap_mean_ci(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    sample_count = len(values)
    resampled_means = np.empty(BOOTSTRAP_RESAMPLES, dtype=float)
    for index in range(BOOTSTRAP_RESAMPLES):
        sample_indices = rng.integers(0, sample_count, size=sample_count)
        resampled_means[index] = float(values[sample_indices].mean())
    low, high = np.percentile(resampled_means, [2.5, 97.5])
    return float(low), float(high)


def summarise_results(
    rows: list[EpisodeResult],
    policy_order: list[str],
    bootstrap_seed: int,
) -> list[dict]:
    output: list[dict] = []
    for policy_index, policy in enumerate(policy_order):
        subset = [row for row in rows if row.policy == policy]
        metrics = {
            "success_rate": np.array([row.success for row in subset], dtype=float),
            "false_confirmation_rate": np.array(
                [row.false_confirmation for row in subset], dtype=float
            ),
            "timeout_rate": np.array([row.timeout for row in subset], dtype=float),
            "mean_total_reward": np.array(
                [row.total_reward for row in subset], dtype=float
            ),
            "mean_steps": np.array([row.steps for row in subset], dtype=float),
        }
        row_summary: dict[str, str | int | float] = {
            "policy": policy,
            "episodes": len(subset),
        }
        rng = np.random.default_rng(bootstrap_seed + policy_index)
        for metric_name, values in metrics.items():
            low, high = bootstrap_mean_ci(values, rng)
            row_summary[metric_name] = float(values.mean())
            row_summary[f"{metric_name}_ci_low"] = low
            row_summary[f"{metric_name}_ci_high"] = high
        output.append(row_summary)
    return output


def calibrate_sweep_threshold(
    calibration_seeds: np.ndarray,
) -> tuple[float, list[dict]]:
    candidates = np.linspace(0.04, 0.20, 9)
    best_threshold = float(candidates[0])
    best_rank: tuple[float, float, float] | None = None
    results: list[dict] = []
    for threshold in candidates:
        rows = evaluate_sweep(
            calibration_seeds,
            float(threshold),
            policy_label="calibration_sweep",
        )
        success_rate = float(np.mean([row.success for row in rows]))
        false_rate = float(np.mean([row.false_confirmation for row in rows]))
        mean_steps = float(np.mean([row.steps for row in rows]))
        results.append(
            {
                "threshold": float(threshold),
                "success_rate": success_rate,
                "false_confirmation_rate": false_rate,
                "mean_steps": mean_steps,
            }
        )
        rank = (success_rate, -false_rate, -mean_steps)
        if best_rank is None or rank > best_rank:
            best_rank = rank
            best_threshold = float(threshold)
    return best_threshold, results


def safe_ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def mask_metrics(prediction: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    prediction = np.asarray(prediction, dtype=bool)
    truth = np.asarray(truth, dtype=bool)
    true_positive = int(np.logical_and(prediction, truth).sum())
    false_positive = int(np.logical_and(prediction, ~truth).sum())
    false_negative = int(np.logical_and(~prediction, truth).sum())
    precision = safe_ratio(true_positive, true_positive + false_positive)
    recall = safe_ratio(true_positive, true_positive + false_negative)
    f1 = safe_ratio(2 * precision * recall, precision + recall)
    iou = safe_ratio(
        true_positive,
        true_positive + false_positive + false_negative,
    )
    return {"precision": precision, "recall": recall, "f1": f1, "iou": iou}


def modality_masks(
    rgb_image: np.ndarray,
    relative_height: np.ndarray,
) -> dict[str, np.ndarray]:
    grayscale = np.dot(rgb_image[..., :3], [0.2989, 0.5870, 0.1140])
    texture = local_std(grayscale, size=7)
    texture_threshold = texture.mean() + 1.2 * texture.std()
    texture_mask = texture > texture_threshold
    height_mask = relative_height > HEIGHT_DETECTION_THRESHOLD
    return {
        "rgb_texture": texture_mask,
        "relative_height": height_mask,
        "late_or_fusion": np.logical_or(texture_mask, height_mask),
    }


def evaluate_modalities(
    scene_seeds: np.ndarray,
    bootstrap_seed: int,
) -> tuple[list[dict], dict]:
    per_modality: dict[str, list[dict[str, float]]] = {
        "rgb_texture": [],
        "relative_height": [],
        "late_or_fusion": [],
    }
    example: dict = {}
    for scene_index, scene_seed in enumerate(scene_seeds):
        rgb_image, relative_height, truth = make_synthetic_electrode(
            int(scene_seed)
        )
        masks = modality_masks(rgb_image, relative_height)
        for modality, mask in masks.items():
            per_modality[modality].append(mask_metrics(mask, truth))
        if scene_index == 0:
            example = {
                "rgb": rgb_image,
                "height": relative_height,
                "truth": truth,
                **masks,
            }

    output: list[dict] = []
    for modality_index, (modality, metric_rows) in enumerate(per_modality.items()):
        output_row: dict[str, str | int | float] = {
            "modality": modality,
            "scenes": len(metric_rows),
        }
        rng = np.random.default_rng(bootstrap_seed + modality_index)
        for metric in ("precision", "recall", "f1", "iou"):
            values = np.array(
                [metric_row[metric] for metric_row in metric_rows],
                dtype=float,
            )
            low, high = bootstrap_mean_ci(values, rng)
            output_row[metric] = float(values.mean())
            output_row[f"{metric}_ci_low"] = low
            output_row[f"{metric}_ci_high"] = high
        output.append(output_row)
    return output, example


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def policy_display_name(policy: str) -> str:
    if policy == "random":
        return "Random"
    if policy == "fixed_sweep":
        return "Fixed sweep"
    return policy.replace("ppo_seed_", "PPO ")


def plot_policy_summary(path: Path, summary: list[dict]) -> None:
    labels = [policy_display_name(str(row["policy"])) for row in summary]
    colours = ["#7f8c8d", "#f39c12", "#2e86c1", "#239b56", "#8e44ad"]
    panels = [
        ("success_rate", "Success rate", True),
        ("false_confirmation_rate", "False-confirmation rate", True),
        ("timeout_rate", "Timeout rate", True),
        ("mean_total_reward", "Mean cumulative reward", False),
    ]
    figure, axes = plt.subplots(2, 2, figsize=(11.2, 7.4))
    for axis, (metric, title, percentage) in zip(axes.flat, panels):
        values = np.array([float(row[metric]) for row in summary])
        lows = np.array([float(row[f"{metric}_ci_low"]) for row in summary])
        highs = np.array([float(row[f"{metric}_ci_high"]) for row in summary])
        bars = axis.bar(
            labels,
            values,
            color=colours[: len(labels)],
            edgecolor="black",
            linewidth=0.5,
        )
        axis.errorbar(
            np.arange(len(labels)),
            values,
            yerr=np.vstack([values - lows, highs - values]),
            fmt="none",
            ecolor="black",
            capsize=3,
            linewidth=1,
        )
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
        axis.tick_params(axis="x", labelrotation=18, labelsize=8.5)
        if percentage:
            axis.set_ylim(0, 1.05)
            axis.set_ylabel("Proportion")
        for bar, value in zip(bars, values):
            label = f"{100 * value:.1f}%" if percentage else f"{value:.2f}"
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (0.02 if percentage else 0.07),
                label,
                ha="center",
                va="bottom",
                fontsize=8,
            )
    figure.suptitle("Shared-scene policy evaluation in the pure simulator")
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_modality_summary(path: Path, summary: list[dict]) -> None:
    labels = ["RGB texture", "Relative height", "Late OR fusion"]
    metric_names = ["precision", "recall", "f1", "iou"]
    x_positions = np.arange(len(labels))
    width = 0.19
    colours = ["#566573", "#2e86c1", "#17a589", "#884ea0"]
    figure, axis = plt.subplots(figsize=(9.4, 5.0))
    for metric_index, metric in enumerate(metric_names):
        values = np.array([float(row[metric]) for row in summary])
        axis.bar(
            x_positions + (metric_index - 1.5) * width,
            values,
            width,
            label=metric.upper() if metric != "precision" else "Precision",
            color=colours[metric_index],
        )
    axis.set_xticks(x_positions, labels)
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("Mean per-scene score")
    axis.set_title("Procedural mask benchmark")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(ncol=4, loc="lower center")
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_example(path: Path, example: dict) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(12, 7.5))
    axes[0, 0].imshow(example["rgb"])
    axes[0, 0].set_title("Synthetic RGB")
    height_image = axes[0, 1].imshow(example["height"], cmap="viridis")
    axes[0, 1].set_title("Synthetic relative height (SRHU)")
    figure.colorbar(height_image, ax=axes[0, 1], fraction=0.046, pad=0.04)
    axes[0, 2].imshow(example["truth"], cmap="gray")
    axes[0, 2].set_title("Procedural ground-truth mask")
    axes[1, 0].imshow(example["rgb_texture"], cmap="gray")
    axes[1, 0].set_title("RGB-texture mask")
    axes[1, 1].imshow(example["relative_height"], cmap="gray")
    axes[1, 1].set_title("Relative-height mask")
    axes[1, 2].imshow(example["late_or_fusion"], cmap="gray")
    axes[1, 2].set_title("Late OR fusion mask")
    for axis in axes.flat:
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=Path("experiments_reproducible_v3"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("evaluation_outputs_reproducible_v3"),
    )
    parser.add_argument(
        "--training-seeds",
        type=int,
        nargs="+",
        default=DEFAULT_TRAINING_SEEDS,
    )
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--calibration-episodes", type=int, default=100)
    parser.add_argument("--modality-scenes", type=int, default=200)
    parser.add_argument("--evaluation-seed", type=int, default=20260827)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {args.output_dir.resolve()}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    random_generator = np.random.default_rng(args.evaluation_seed)
    calibration_seeds = random_generator.integers(
        1_000_000,
        1_999_999,
        size=args.calibration_episodes,
    )
    evaluation_seeds = random_generator.integers(
        2_000_000,
        8_999_999,
        size=args.episodes,
    )
    modality_seeds = random_generator.integers(
        9_000_000,
        9_999_999,
        size=args.modality_scenes,
    )

    print("Calibrating fixed-sweep threshold ...", flush=True)
    sweep_threshold, calibration_rows = calibrate_sweep_threshold(
        calibration_seeds
    )
    episode_rows: list[EpisodeResult] = []
    print("Evaluating random and fixed-sweep baselines ...", flush=True)
    episode_rows.extend(evaluate_random(evaluation_seeds))
    episode_rows.extend(evaluate_sweep(evaluation_seeds, sweep_threshold))

    model_metadata: list[dict] = []
    for training_seed in args.training_seeds:
        model_path = (
            args.experiment_dir
            / f"seed_{training_seed}"
            / "ppo_electrode_agent.zip"
        )
        model = PPO.load(model_path)
        if tuple(model.observation_space.shape) != (7,):
            raise ValueError(f"Unexpected observation space in {model_path}")
        if int(model.action_space.n) != 4:
            raise ValueError(f"Unexpected action space in {model_path}")
        print(f"Evaluating PPO training seed {training_seed} ...", flush=True)
        episode_rows.extend(
            evaluate_ppo(evaluation_seeds, model, training_seed)
        )
        model_metadata.append(
            {
                "training_seed": training_seed,
                "path": str(model_path.resolve()),
                "sha256": file_sha256(model_path),
                "stored_timesteps": int(model.num_timesteps),
            }
        )

    policy_order = ["random", "fixed_sweep"] + [
        f"ppo_seed_{seed}" for seed in args.training_seeds
    ]
    policy_summary = summarise_results(
        episode_rows,
        policy_order,
        args.evaluation_seed + 10_000,
    )

    ppo_rows = [
        row for row in policy_summary if str(row["policy"]).startswith("ppo_seed_")
    ]
    ppo_across_seeds: list[dict] = []
    for metric in (
        "success_rate",
        "false_confirmation_rate",
        "timeout_rate",
        "mean_total_reward",
        "mean_steps",
    ):
        values = np.array([float(row[metric]) for row in ppo_rows])
        ppo_across_seeds.append(
            {
                "metric": metric,
                "training_seed_count": len(values),
                "mean_across_training_seeds": float(values.mean()),
                "sample_sd_across_training_seeds": float(values.std(ddof=1)),
                "minimum_training_seed_result": float(values.min()),
                "maximum_training_seed_result": float(values.max()),
            }
        )

    print("Evaluating RGB and relative-height masks ...", flush=True)
    modality_summary, example = evaluate_modalities(
        modality_seeds,
        args.evaluation_seed + 20_000,
    )

    write_csv(
        args.output_dir / "episode_results.csv",
        [asdict(row) for row in episode_rows],
    )
    write_csv(args.output_dir / "policy_summary.csv", policy_summary)
    write_csv(args.output_dir / "ppo_across_training_seeds.csv", ppo_across_seeds)
    write_csv(args.output_dir / "modality_summary.csv", modality_summary)
    write_csv(args.output_dir / "sweep_calibration.csv", calibration_rows)

    plot_policy_summary(
        args.output_dir / "policy_comparison_three_seeds.png",
        policy_summary,
    )
    plot_modality_summary(
        args.output_dir / "modality_comparison_reproducible.png",
        modality_summary,
    )
    plot_example(
        args.output_dir / "synthetic_rgb_relative_height_example.png",
        example,
    )

    metadata = {
        "evaluation_type": "pure_simulation_three_training_seed_evaluation",
        "evaluation_seed": args.evaluation_seed,
        "episodes_per_policy": args.episodes,
        "calibration_episodes": args.calibration_episodes,
        "modality_scenes": args.modality_scenes,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "height_units": "dimensionless synthetic relative-height units (SRHU)",
        "fixed_sweep_threshold": sweep_threshold,
        "training_models": model_metadata,
        "policy_summary": policy_summary,
        "ppo_across_training_seeds": ppo_across_seeds,
        "modality_summary": modality_summary,
        "limitations": [
            "Procedural synthetic scenes only.",
            "No real camera, electrode sample or physical depth measurement was used.",
            "The relative-height signal and ground truth originate from the same latent ellipse geometry.",
            "Every scene contains residue; sample-level clean-foil false alarms were not evaluated.",
            "The three training seeds quantify limited training variation, not real-world uncertainty.",
        ],
        "source_files": {
            "environment": {
                "path": str(Path("rl_environment.py").resolve()),
                "sha256": file_sha256(
                    Path("rl_environment.py")
                ),
            },
            "evaluation": {
                "path": str(Path(__file__).resolve()),
                "sha256": file_sha256(Path(__file__)),
            },
        },
    }
    (args.output_dir / "evaluation_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    readme_lines = [
        "# Reproducible three-seed pure-simulation evaluation",
        "",
        "No real RGB-D frames or electrode samples were used.",
        "The relative-height values are dimensionless synthetic relative-height units (SRHU), not millimetres.",
        "",
        f"Fixed-sweep calibration threshold: {sweep_threshold:.3f}",
        "",
        "| Policy | Success | False confirmation | Timeout | Mean reward | Mean steps |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in policy_summary:
        readme_lines.append(
            "| {policy} | {success:.1f}% | {false:.1f}% | {timeout:.1f}% | {reward:.3f} | {steps:.2f} |".format(
                policy=policy_display_name(str(row["policy"])),
                success=100 * float(row["success_rate"]),
                false=100 * float(row["false_confirmation_rate"]),
                timeout=100 * float(row["timeout_rate"]),
                reward=float(row["mean_total_reward"]),
                steps=float(row["mean_steps"]),
            )
        )
    (args.output_dir / "README.md").write_text(
        "\n".join(readme_lines) + "\n",
        encoding="utf-8",
    )

    print(f"Fixed-sweep threshold: {sweep_threshold:.3f}")
    for row in policy_summary:
        print(
            f"{policy_display_name(str(row['policy'])):<18} "
            f"success={100 * float(row['success_rate']):.1f}% "
            f"false={100 * float(row['false_confirmation_rate']):.1f}% "
            f"timeout={100 * float(row['timeout_rate']):.1f}% "
            f"reward={float(row['mean_total_reward']):.3f} "
            f"steps={float(row['mean_steps']):.2f}"
        )
    print(f"Outputs: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
