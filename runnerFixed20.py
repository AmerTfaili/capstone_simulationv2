from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import runnerFixed as base


ROOT = Path(__file__).resolve().parent
DEFAULT_SCENARIO_DIR = ROOT / "sumo_benchmark"
DEFAULT_OUTPUT_DIR = ROOT / "runnerFixed20_outputs"
DEFAULT_FIXED_GREEN_DURATION = base.FIXED_GREEN_DURATION
DEFAULT_YELLOW_DURATION = base.YELLOW_DURATION


def resolve_path_from_root(raw_path: str) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the 20-node benchmark under custom fixed-time control using runnerFixed travel-time measurement."
    )
    parser.add_argument(
        "--scenario-dir",
        default="sumo_benchmark",
        help="Scenario directory containing benchmark.sumocfg.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory used for fixed-time outputs, metrics, and plot.",
    )
    parser.add_argument("--sumo-binary", default="sumo", help="SUMO binary to execute.")
    parser.add_argument("--seed", type=int, default=11, help="Base simulation seed.")
    parser.add_argument("--episodes", type=int, default=1, help="Number of fixed-time episodes to run.")
    parser.add_argument("--seed-stride", type=int, default=1, help="Seed increment applied between episodes.")
    parser.add_argument("--max-steps", type=int, default=3600, help="Maximum SUMO steps per episode.")
    parser.add_argument(
        "--fixed-green-duration",
        type=int,
        default=DEFAULT_FIXED_GREEN_DURATION,
        help="Green duration in seconds for each major fixed-time phase.",
    )
    parser.add_argument(
        "--yellow-duration",
        type=int,
        default=DEFAULT_YELLOW_DURATION,
        help="Yellow transition duration in seconds between the two major fixed-time phases.",
    )
    parser.add_argument(
        "--metric-source",
        choices=["auto", "libsumo", "xml"],
        default="auto",
        help="Travel-time measurement source. 'auto' prefers libsumo and falls back to XML tripinfo parsing.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    scenario_dir = base.resolve_scenario_dir(args.scenario_dir)
    sumocfg_path = scenario_dir / "benchmark.sumocfg"
    output_dir = resolve_path_from_root(args.output_dir)

    if not sumocfg_path.exists():
        raise FileNotFoundError(f"SUMO config not found: {sumocfg_path}")

    base.SCENARIO_DIR = scenario_dir
    base.SUMOCFG_PATH = sumocfg_path
    base.OUTPUT_DIR = output_dir
    base.SUMMARY_PATH = output_dir / "fixed_time_summary.xml"
    base.TRIPINFO_PATH = output_dir / "fixed_time_tripinfo.xml"
    base.METRICS_PATH = output_dir / "fixed_time_metrics.json"
    base.EPISODE_METRICS_PATH = output_dir / "fixed_time_episode_metrics.json"
    base.PLOT_PATH = output_dir / "fixed_time_average_travel_time.png"
    base.FIXED_TIMING_PATH = output_dir / f"fixed{args.fixed_green_duration}_all.add.xml"

    route_path = base.resolve_sumocfg_input(base.SUMOCFG_PATH, "route-files")
    net_path = base.resolve_sumocfg_input(base.SUMOCFG_PATH, "net-file")
    base.write_fixed_timing_additional(
        net_path,
        base.FIXED_TIMING_PATH,
        args.fixed_green_duration,
        args.yellow_duration,
    )
    total_demand = base.count_total_demand(route_path)

    episode_metrics: list[dict[str, object]] = []
    for episode_index in range(1, args.episodes + 1):
        seed = args.seed + (episode_index - 1) * args.seed_stride
        summary_path, tripinfo_path = base.episode_output_paths(episode_index)
        stats = base.run_episode(
            sumo_binary=args.sumo_binary,
            seed=seed,
            total_demand=total_demand,
            max_steps=args.max_steps,
            summary_path=summary_path,
            tripinfo_path=tripinfo_path,
            metric_source=args.metric_source,
        )
        episode_record = {
            "episode": episode_index,
            "seed": seed,
            **stats,
        }
        episode_metrics.append(episode_record)
        print(
            f"runnerFixed20 ep {episode_index:>3}/{args.episodes} | "
            f"avg_tt={float(stats['average_travel_time']):.2f}s | "
            f"inserted={float(stats['inserted_ratio']):.3f} | "
            f"arrived={int(stats['arrived_vehicle_count'])}"
        )

    aggregate_metrics = base.aggregate_episode_metrics(base.SCENARIO_DIR, base.SUMOCFG_PATH, episode_metrics)
    base.write_episode_metrics(episode_metrics)
    base.write_metrics(aggregate_metrics)
    base.plot_episode_average_travel_time(episode_metrics)
    base.refresh_last_episode_xml(episode_metrics)

    print("runnerFixed20 complete.")
    print("control_mode: all_fixed_time")
    print(f"fixed_green_duration: {args.fixed_green_duration}")
    print(f"yellow_duration: {args.yellow_duration}")
    print(
        f"metric_source: {args.metric_source if args.metric_source != 'auto' else ('libsumo' if base.libsumo is not None else 'xml')}"
    )
    for key, value in aggregate_metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.3f}")
        else:
            print(f"{key}: {value}")
    print(f"episode_metrics_file: {base.EPISODE_METRICS_PATH}")
    print(f"metrics_file: {base.METRICS_PATH}")
    print(f"plot_file: {base.PLOT_PATH}")
    print(f"summary_file: {base.SUMMARY_PATH}")
    print(f"tripinfo_file: {base.TRIPINFO_PATH}")


if __name__ == "__main__":
    main()
