"""Redraw the policy comparison figures from the archived summary CSV files.

This script changes presentation only.  It does not load a PPO model, rerun an
episode, recalculate a confidence interval or alter any formal result.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


MODES = ["combined", "rgb_only", "height_only"]
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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def mode_seed_rows(policy_rows: list[dict[str, str]], mode: str) -> list[dict[str, str]]:
    prefix = f"ppo_{mode}_seed_"
    rows = [row for row in policy_rows if row["policy"].startswith(prefix)]
    return sorted(rows, key=lambda row: int(row["policy"].removeprefix(prefix)))


def add_seed_points_and_mean_ci(
    axis: plt.Axes,
    policy_rows: list[dict[str, str]],
    condition_rows: dict[str, dict[str, str]],
    metric: str,
    rng: np.random.Generator,
) -> None:
    for index, mode in enumerate(MODES):
        values = np.array(
            [100.0 * float(row[metric]) for row in mode_seed_rows(policy_rows, mode)]
        )
        jitter = rng.uniform(-0.08, 0.08, size=len(values))
        axis.scatter(
            index + jitter,
            values,
            s=30,
            color=MODE_COLOURS[mode],
            alpha=0.72,
            edgecolor="white",
            linewidth=0.45,
            zorder=3,
        )

        summary = condition_rows[mode]
        mean = 100.0 * float(summary[f"{metric}_mean"])
        low = 100.0 * float(summary[f"{metric}_t_ci_low"])
        high = 100.0 * float(summary[f"{metric}_t_ci_high"])
        axis.errorbar(
            index,
            mean,
            yerr=[[mean - low], [high - mean]],
            fmt="D",
            markersize=6,
            color="black",
            markerfacecolor=MODE_COLOURS[mode],
            markeredgewidth=0.9,
            capsize=4,
            linewidth=1.5,
            zorder=4,
        )


def plot_success_and_paired_differences(
    output_path: Path,
    policy_rows: list[dict[str, str]],
    condition_rows: dict[str, dict[str, str]],
    paired_rows: list[dict[str, str]],
) -> None:
    by_policy = {row["policy"]: row for row in policy_rows}
    figure, axes = plt.subplots(1, 2, figsize=(11.4, 4.7))
    rng = np.random.default_rng(7129)

    axis = axes[0]
    add_seed_points_and_mean_ci(
        axis, policy_rows, condition_rows, "success_rate", rng
    )
    fixed_value = 100.0 * float(by_policy["fixed_sweep"]["success_rate"])
    random_value = 100.0 * float(by_policy["random"]["success_rate"])
    axis.axhline(
        fixed_value,
        color="#e69f00",
        linewidth=1.6,
        linestyle="--",
        label=f"Fixed sweep ({fixed_value:.1f}%)",
    )
    axis.text(
        0.02,
        0.04,
        f"Random: {random_value:.1f}% (off scale)",
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        color="#666666",
    )
    for index, mode in enumerate(MODES):
        mean = 100.0 * float(condition_rows[mode]["success_rate_mean"])
        axis.text(
            index,
            97.35,
            f"{mean:.2f}%",
            ha="center",
            va="center",
            fontsize=9,
            color=MODE_COLOURS[mode],
            fontweight="bold",
        )
    axis.set_xticks(np.arange(len(MODES)), [MODE_LABELS[mode] for mode in MODES])
    axis.tick_params(axis="x", labelrotation=8)
    axis.set_ylim(88.0, 98.0)
    axis.set_yticks(np.arange(88.0, 98.1, 2.0))
    axis.set_ylabel("Success rate (%)")
    axis.set_title("Across-seed success (zoomed scale)")
    axis.grid(axis="y", alpha=0.22)
    axis.legend(fontsize=8, loc="lower right")

    axis = axes[1]
    labels: list[str] = []
    estimates: list[float] = []
    low_errors: list[float] = []
    high_errors: list[float] = []
    colours: list[str] = []
    for row in paired_rows:
        left = MODE_LABELS.get(row["left"], row["left"])
        right = MODE_LABELS.get(row["right"], "Fixed sweep")
        labels.append(f"{left} - {right}")
        estimate = float(row["difference_percentage_points"])
        estimates.append(estimate)
        low_errors.append(estimate - float(row["ci_low_percentage_points"]))
        high_errors.append(float(row["ci_high_percentage_points"]) - estimate)
        colours.append("#4c78a8" if row["right"] == "fixed_sweep" else "#666666")

    y_positions = np.arange(len(labels))
    for y, estimate, low_error, high_error, colour in zip(
        y_positions, estimates, low_errors, high_errors, colours
    ):
        axis.errorbar(
            estimate,
            y,
            xerr=[[low_error], [high_error]],
            fmt="o",
            color=colour,
            ecolor=colour,
            capsize=3,
            markersize=6,
        )
    axis.axvline(0.0, color="black", linewidth=1.0)
    axis.axhline(2.5, color="#cccccc", linewidth=0.9)
    axis.set_yticks(y_positions, labels, fontsize=8)
    axis.invert_yaxis()
    axis.set_xlabel("Paired success-rate difference (percentage points)")
    axis.set_title("Hierarchical paired bootstrap (95% CI)")
    axis.grid(axis="x", alpha=0.22)

    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_error_profiles(
    output_path: Path,
    policy_rows: list[dict[str, str]],
    condition_rows: dict[str, dict[str, str]],
) -> None:
    by_policy = {row["policy"]: row for row in policy_rows}
    figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.4))
    rng = np.random.default_rng(9143)
    panels = [
        (
            "false_confirmation_rate",
            "False confirmation (premature stop)",
            (-0.5, 10.5),
        ),
        ("timeout_rate", "Timeout", (-0.5, 11.5)),
    ]

    for axis, (metric, title, limits) in zip(axes, panels):
        add_seed_points_and_mean_ci(axis, policy_rows, condition_rows, metric, rng)
        fixed_value = 100.0 * float(by_policy["fixed_sweep"][metric])
        random_value = 100.0 * float(by_policy["random"][metric])
        axis.axhline(
            fixed_value,
            color="#e69f00",
            linewidth=1.6,
            linestyle="--",
            label=f"Fixed sweep ({fixed_value:.1f}%)",
        )
        if limits[0] <= random_value <= limits[1]:
            axis.axhline(
                random_value,
                color="#777777",
                linewidth=1.2,
                linestyle=":",
                label=f"Random ({random_value:.1f}%)",
            )
        else:
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
        axis.set_xticks(np.arange(len(MODES)), [MODE_LABELS[mode] for mode in MODES])
        axis.tick_params(axis="x", labelrotation=8)
        axis.set_ylim(*limits)
        axis.set_ylabel("Rate (%)")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.22)
        axis.legend(fontsize=8, loc="best")

    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("evaluation_outputs_multimodal_10seeds"),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir or input_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    policy_rows = read_csv(input_dir / "policy_summary.csv")
    condition_rows = {
        row["observation_mode"]: row
        for row in read_csv(input_dir / "policy_condition_summary.csv")
    }
    paired_rows = read_csv(input_dir / "paired_bootstrap_summary.csv")

    plot_success_and_paired_differences(
        output_dir / "policy_success_and_paired_differences.png",
        policy_rows,
        condition_rows,
        paired_rows,
    )
    plot_error_profiles(
        output_dir / "policy_error_profiles.png", policy_rows, condition_rows
    )


if __name__ == "__main__":
    main()
