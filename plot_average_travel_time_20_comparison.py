from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent

DEFAULT_MY_APPROACH = ROOT / "runnerMain_outputs_attention_reward_20_run1" / "training_metrics.json"
DEFAULT_FIXED_TIME = ROOT / "runnerFixed20_outputs_fixed30" / "fixed_time_episode_metrics.json"
DEFAULT_DRQN = ROOT / "runnerDRQNModerate20_outputs" / "training_metrics.json"
DEFAULT_DDQN = ROOT / "runnerDQN20_outputs_run2" / "training_metrics.json"
DEFAULT_OUTPUT = ROOT / "average_travel_time_20_map_comparison.png"


def load_runner_main_series(path: Path) -> tuple[list[int], list[float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    series = payload["episode_metrics"]
    return [int(item["episode"]) for item in series], [float(item["average_travel_time"]) for item in series]


def load_training_series(path: Path) -> tuple[list[int], list[float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    series = payload["training"]
    return [int(item["episode"]) for item in series], [float(item["average_travel_time"]) for item in series]


def load_fixed_time_series(path: Path) -> tuple[list[int], list[float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    series = payload["episodes"]
    return [int(item["episode"]) for item in series], [float(item["average_travel_time"]) for item in series]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot average travel time vs episodes for 20-node map: My Approach, Fixed Time, DRQN, DDQN."
    )
    parser.add_argument("--my-approach", type=Path, default=DEFAULT_MY_APPROACH)
    parser.add_argument("--fixed-time", type=Path, default=DEFAULT_FIXED_TIME)
    parser.add_argument("--drqn", type=Path, default=DEFAULT_DRQN)
    parser.add_argument("--ddqn", type=Path, default=DEFAULT_DDQN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    my_x, my_y = load_runner_main_series(args.my_approach)
    fixed_x, fixed_y = load_fixed_time_series(args.fixed_time)
    drqn_x, drqn_y = load_training_series(args.drqn)
    ddqn_x, ddqn_y = load_training_series(args.ddqn)

    plt.figure(figsize=(11.5, 6))
    plt.plot(my_x, my_y, color="#C62828", linewidth=2.2, label="My Approach")
    plt.plot(fixed_x, fixed_y, color="#9E9E9E", linewidth=2.2, label="FT")
    plt.plot(drqn_x, drqn_y, color="#000000", linewidth=2.2, label="DRQN")
    plt.plot(ddqn_x, ddqn_y, color="#616161", linewidth=2.2, label="DDQN")

    plt.title("20-Intersection Map: Average Travel Time vs Episodes")
    plt.xlabel("Episode")
    plt.ylabel("Average Travel Time (s)")
    plt.xlim(1, 300)
    plt.grid(True, axis="y", alpha=0.25)
    plt.legend(loc="upper right", frameon=False)
    plt.tight_layout()

    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(output_path)


if __name__ == "__main__":
    main()
