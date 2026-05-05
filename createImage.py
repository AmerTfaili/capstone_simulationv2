from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
DRQN_METRICS = ROOT / "training_outputs" / "training_metrics.json"
MAIN_METRICS = ROOT / "runnerMain_outputs" / "training_metrics.json"
DQN_METRICS = ROOT / "runnerDQN_outputs" / "training_metrics.json"
OUTPUT_IMAGE = ROOT / "methodology_comparison_average_travel_time.png"


def load_curve(path: Path, label: str, mode: str) -> tuple[list[int], list[float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))

    if "episode_metrics" in payload:
        entries = payload["episode_metrics"]
    elif mode == "eval" and "evaluation" in payload:
        entries = payload["evaluation"]
    elif "training" in payload:
        entries = payload["training"]
    else:
        raise RuntimeError(f"Unsupported metrics format for {label}: {path}")

    xs: list[int] = []
    ys: list[float] = []
    for item in entries:
        avg_tt = item.get("average_travel_time")
        episode = item.get("episode")
        if avg_tt is None or episode is None:
            continue
        if isinstance(avg_tt, float) and avg_tt != avg_tt:
            continue
        xs.append(int(episode))
        ys.append(float(avg_tt))

    if not xs:
        raise RuntimeError(f"No usable points found for {label}: {path}")
    return xs, ys


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot average travel time vs episodes for the three methodologies.")
    parser.add_argument(
        "--mode",
        choices=["train", "eval"],
        default="train",
        help="Use training curves or evaluation curves where available.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_IMAGE,
        help="Output image path.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    curves = [
        ("DRQN", DRQN_METRICS, "tab:blue"),
        ("My Approach", MAIN_METRICS, "tab:green"),
        ("DQN", DQN_METRICS, "tab:red"),
    ]

    fig, ax = plt.subplots(figsize=(11, 6))

    for label, path, color in curves:
        xs, ys = load_curve(path, label, args.mode)
        ax.plot(xs, ys, label=label, color=color, linewidth=2.2)

    ax.set_title(f"Average Travel Time vs Episodes ({args.mode.title()})")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Average Travel Time (s)")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150)
    print(f"Saved comparison plot: {args.output}")


if __name__ == "__main__":
    main()
