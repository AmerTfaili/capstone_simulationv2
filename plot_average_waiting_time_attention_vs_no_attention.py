from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent

DEFAULT_50_ATTENTION = ROOT / "runnerMain_outputs_300_waiting_50" / "training_metrics.json"
DEFAULT_50_NO_ATTENTION = ROOT / "runnerMain_outputs_no_attention_reward" / "training_metrics.json"
DEFAULT_20_ATTENTION = ROOT / "runnerMain_outputs_attention_reward_20_run1" / "training_metrics.json"
DEFAULT_20_NO_ATTENTION = ROOT / "runnerMain_outputs_no_attention_reward_20" / "training_metrics.json"
DEFAULT_OUTPUT = ROOT / "average_waiting_time_attention_vs_no_attention_maps.png"


def load_episode_series(metrics_path: Path) -> tuple[list[int], list[float]]:
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    episode_metrics = payload.get("episode_metrics", [])
    episodes = [int(item["episode"]) for item in episode_metrics]
    values = [float(item["average_waiting_time"]) for item in episode_metrics]
    return episodes, values


def plot_map_comparison(
    ax: plt.Axes,
    title: str,
    attention_path: Path,
    no_attention_path: Path,
) -> None:
    attention_episodes, attention_values = load_episode_series(attention_path)
    no_attention_episodes, no_attention_values = load_episode_series(no_attention_path)

    ax.plot(
        attention_episodes,
        attention_values,
        color="black",
        linewidth=2.0,
        label="Attention Reward",
    )
    ax.plot(
        no_attention_episodes,
        no_attention_values,
        color="dimgray",
        linewidth=2.0,
        label="No-Attention Reward",
    )
    ax.set_title(title)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Average Waiting Time (s)")
    ax.grid(True, axis="y", alpha=0.25)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot average waiting time for attention vs no-attention runnerMain on 50-node and 20-node maps."
    )
    parser.add_argument("--attention-50", type=Path, default=DEFAULT_50_ATTENTION)
    parser.add_argument("--no-attention-50", type=Path, default=DEFAULT_50_NO_ATTENTION)
    parser.add_argument("--attention-20", type=Path, default=DEFAULT_20_ATTENTION)
    parser.add_argument("--no-attention-20", type=Path, default=DEFAULT_20_NO_ATTENTION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)

    plot_map_comparison(
        axes[0],
        "50 Intersections Map",
        args.attention_50,
        args.no_attention_50,
    )
    plot_map_comparison(
        axes[1],
        "20 Intersections Map",
        args.attention_20,
        args.no_attention_20,
    )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.suptitle("Average Waiting Time: Attention vs No-Attention Reward")
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))

    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(output_path)


if __name__ == "__main__":
    main()
