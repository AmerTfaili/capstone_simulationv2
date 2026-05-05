from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_average_waiting_time_attention_vs_no_attention import (
    DEFAULT_20_ATTENTION,
    DEFAULT_20_NO_ATTENTION,
    DEFAULT_50_ATTENTION,
    DEFAULT_50_NO_ATTENTION,
    load_episode_series,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_20 = ROOT / "average_waiting_time_attention_vs_no_attention_20_map.png"
DEFAULT_OUTPUT_50 = ROOT / "average_waiting_time_attention_vs_no_attention_50_map.png"


def plot_single_map(
    title: str,
    attention_path: Path,
    no_attention_path: Path,
    output_path: Path,
) -> None:
    attention_episodes, attention_values = load_episode_series(attention_path)
    no_attention_episodes, no_attention_values = load_episode_series(no_attention_path)

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    ax.plot(
        attention_episodes,
        attention_values,
        color="#000000",
        linewidth=2.2,
        label="Attention Reward",
    )
    ax.plot(
        no_attention_episodes,
        no_attention_values,
        color="#616161",
        linewidth=2.2,
        label="No-Attention Reward",
    )
    ax.set_title(title)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Average Waiting Time (s)")
    ax.set_xlim(1, 300)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()

    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create separate average waiting time plots for attention vs no-attention rewards."
    )
    parser.add_argument("--attention-20", type=Path, default=DEFAULT_20_ATTENTION)
    parser.add_argument("--no-attention-20", type=Path, default=DEFAULT_20_NO_ATTENTION)
    parser.add_argument("--attention-50", type=Path, default=DEFAULT_50_ATTENTION)
    parser.add_argument("--no-attention-50", type=Path, default=DEFAULT_50_NO_ATTENTION)
    parser.add_argument("--output-20", type=Path, default=DEFAULT_OUTPUT_20)
    parser.add_argument("--output-50", type=Path, default=DEFAULT_OUTPUT_50)
    args = parser.parse_args()

    plot_single_map(
        "20-Intersection Map: Average Waiting Time",
        args.attention_20,
        args.no_attention_20,
        args.output_20,
    )
    plot_single_map(
        "50-Intersection Map: Average Waiting Time",
        args.attention_50,
        args.no_attention_50,
        args.output_50,
    )


if __name__ == "__main__":
    main()
