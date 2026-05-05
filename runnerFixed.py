from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from sumo_fixed_timing import write_fixed_timing_additional


ROOT = Path(__file__).resolve().parent
SCENARIO_DIR = ROOT / "sumo_benchmark_50"
SUMOCFG_PATH = SCENARIO_DIR / "benchmark.sumocfg"
OUTPUT_DIR = ROOT / "runnerFixed_outputs"
SUMMARY_PATH = OUTPUT_DIR / "fixed_time_summary.xml"
TRIPINFO_PATH = OUTPUT_DIR / "fixed_time_tripinfo.xml"
METRICS_PATH = OUTPUT_DIR / "fixed_time_metrics.json"
EPISODE_METRICS_PATH = OUTPUT_DIR / "fixed_time_episode_metrics.json"
PLOT_PATH = OUTPUT_DIR / "fixed_time_average_travel_time.png"
FIXED_TIMING_PATH = OUTPUT_DIR / "fixed60_all.add.xml"
FIXED_GREEN_DURATION = 60
YELLOW_DURATION = 3
DLL_DIR_HANDLES: list[Any] = []
LIBSUMO_IMPORT_ERROR: Exception | None = None
libsumo: Any | None = None


def resolve_scenario_dir(raw_path: str) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def ensure_sumo_python_tools() -> None:
    sumo_home = os.environ.get("SUMO_HOME")
    if not sumo_home:
        raise RuntimeError("SUMO_HOME is not set.")
    sumo_root = Path(sumo_home)
    tools_dir = sumo_root / "tools"
    bin_dir = sumo_root / "bin"
    if str(tools_dir) not in sys.path:
        sys.path.append(str(tools_dir))
    path_entries = os.environ.get("PATH", "")
    if str(bin_dir) not in path_entries:
        os.environ["PATH"] = f"{bin_dir};{path_entries}"
    if hasattr(os, "add_dll_directory") and bin_dir.exists():
        DLL_DIR_HANDLES.append(os.add_dll_directory(str(bin_dir.resolve())))


ensure_sumo_python_tools()

try:
    import libsumo  # type: ignore
except Exception as exc:  # pragma: no cover - environment-specific import failure
    LIBSUMO_IMPORT_ERROR = exc
    libsumo = None


def resolve_sumocfg_input(sumocfg_path: Path, xml_key: str) -> Path:
    root = ET.parse(sumocfg_path).getroot()
    input_node = root.find("input")
    if input_node is None:
        raise RuntimeError(f"Invalid SUMO config: missing input block in {sumocfg_path}")
    for child in input_node:
        if child.tag == xml_key:
            return (sumocfg_path.parent / child.attrib["value"]).resolve()
    raise RuntimeError(f"Could not resolve {xml_key} from {sumocfg_path}")


def count_total_demand(route_path: Path) -> int:
    root = ET.parse(route_path).getroot()
    return sum(1 for child in root if child.tag == "vehicle")


def episode_output_paths(episode_index: int) -> tuple[Path, Path]:
    return (
        OUTPUT_DIR / f"fixed_time_summary_ep{episode_index:03d}.xml",
        OUTPUT_DIR / f"fixed_time_tripinfo_ep{episode_index:03d}.xml",
    )


def sumo_cmd(sumo_binary: str, seed: int, summary_path: Path, tripinfo_path: Path) -> list[str]:
    return [
        sumo_binary,
        "-c",
        str(SUMOCFG_PATH),
        "--seed",
        str(seed),
        "--additional-files",
        str(FIXED_TIMING_PATH),
        "--summary-output",
        str(summary_path),
        "--tripinfo-output",
        str(tripinfo_path),
        "--no-step-log",
        "true",
        "--no-warnings",
        "true",
    ]


def summarize_xml_outputs(
    summary_path: Path,
    tripinfo_path: Path,
    total_demand: int,
    seed: int,
) -> dict[str, float | int | str]:
    summary_root = ET.parse(summary_path).getroot()
    last_step = summary_root.findall("step")[-1]
    tripinfo_root = ET.parse(tripinfo_path).getroot()
    trips = tripinfo_root.findall("tripinfo")
    durations = [float(trip.attrib["duration"]) for trip in trips]
    return {
        "scenario_dir": str(SCENARIO_DIR),
        "sumocfg_path": str(SUMOCFG_PATH),
        "summary_file": str(summary_path),
        "tripinfo_file": str(tripinfo_path),
        "average_travel_time": np_mean(durations),
        "max_travel_time": max(durations) if durations else math.nan,
        "arrived_vehicle_count": len(trips),
        "demand_vehicle_count": total_demand,
        "loaded_vehicle_count": int(last_step.attrib["loaded"]),
        "inserted_vehicle_count": int(last_step.attrib["inserted"]),
        "pending_vehicle_count": int(last_step.attrib["waiting"]),
        "inserted_ratio": float(int(last_step.attrib["inserted"]) / total_demand) if total_demand else 0.0,
        "arrived_ratio": float(int(last_step.attrib["arrived"]) / total_demand) if total_demand else 0.0,
        "sumo_seed": seed,
    }


def track_vehicle_progress(
    step: int,
    vehicle_departs: dict[str, int],
    vehicle_travel_times: list[float],
) -> tuple[int, int, int]:
    loaded_ids = libsumo.simulation.getLoadedIDList()
    departed_ids = libsumo.simulation.getDepartedIDList()
    arrived_ids = libsumo.simulation.getArrivedIDList()
    for veh_id in departed_ids:
        vehicle_departs[veh_id] = step
    for veh_id in arrived_ids:
        depart_step = vehicle_departs.pop(veh_id, None)
        if depart_step is not None:
            vehicle_travel_times.append(step - depart_step)
    return len(loaded_ids), len(departed_ids), len(arrived_ids)


def run_episode(
    sumo_binary: str,
    seed: int,
    total_demand: int,
    max_steps: int,
    summary_path: Path,
    tripinfo_path: Path,
    metric_source: str,
) -> dict[str, float | int | str]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if summary_path.exists():
        summary_path.unlink()
    if tripinfo_path.exists():
        tripinfo_path.unlink()

    effective_metric_source = metric_source
    if effective_metric_source == "auto":
        effective_metric_source = "libsumo" if libsumo is not None else "xml"

    if effective_metric_source == "xml":
        cmd = sumo_cmd(sumo_binary, seed, summary_path, tripinfo_path)
        subprocess.run(cmd, check=True, cwd=ROOT)
        return summarize_xml_outputs(summary_path, tripinfo_path, total_demand, seed)

    if libsumo is None:
        raise RuntimeError(
            "runnerFixed was asked to use libsumo, but libsumo could not be imported."
        ) from LIBSUMO_IMPORT_ERROR

    libsumo.start(sumo_cmd(sumo_binary, seed, summary_path, tripinfo_path))
    try:
        vehicle_departs: dict[str, int] = {}
        vehicle_travel_times: list[float] = []
        loaded_total = 0
        inserted_total = 0
        arrived_total = 0
        step = 0

        while step < max_steps:
            libsumo.simulationStep()
            step += 1
            loaded_inc, inserted_inc, arrived_inc = track_vehicle_progress(
                step,
                vehicle_departs,
                vehicle_travel_times,
            )
            loaded_total += loaded_inc
            inserted_total += inserted_inc
            arrived_total += arrived_inc
            if step >= max_steps or libsumo.simulation.getMinExpectedNumber() <= 0:
                break

        average_travel_time = float(np_mean(vehicle_travel_times)) if vehicle_travel_times else math.nan
        max_travel_time = float(max(vehicle_travel_times)) if vehicle_travel_times else math.nan
        pending_vehicle_count = int(libsumo.simulation.getMinExpectedNumber())
        return {
            "scenario_dir": str(SCENARIO_DIR),
            "sumocfg_path": str(SUMOCFG_PATH),
            "summary_file": str(summary_path),
            "tripinfo_file": str(tripinfo_path),
            "average_travel_time": average_travel_time,
            "max_travel_time": max_travel_time,
            "arrived_vehicle_count": len(vehicle_travel_times),
            "demand_vehicle_count": total_demand,
            "loaded_vehicle_count": loaded_total,
            "inserted_vehicle_count": inserted_total,
            "pending_vehicle_count": pending_vehicle_count,
            "inserted_ratio": float(inserted_total / total_demand) if total_demand else 0.0,
            "arrived_ratio": float(arrived_total / total_demand) if total_demand else 0.0,
            "sumo_seed": seed,
        }
    finally:
        libsumo.close()


def np_mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def write_metrics(stats: dict[str, Any]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(stats, indent=2), encoding="utf-8")


def write_episode_metrics(episode_metrics: list[dict[str, Any]]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    EPISODE_METRICS_PATH.write_text(json.dumps({"episodes": episode_metrics}, indent=2), encoding="utf-8")


def plot_episode_average_travel_time(episode_metrics: list[dict[str, Any]]) -> None:
    episodes = [int(item["episode"]) for item in episode_metrics]
    average_travel_times = [float(item["average_travel_time"]) for item in episode_metrics]

    plt.figure(figsize=(10, 5.5))
    plt.plot(episodes, average_travel_times, color="tab:blue", linewidth=2.0)
    plt.title("runnerFixed Average Travel Time")
    plt.xlabel("Episode")
    plt.ylabel("Average Travel Time (s)")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150)
    plt.close()


def aggregate_episode_metrics(
    scenario_dir: Path,
    sumocfg_path: Path,
    episode_metrics: list[dict[str, Any]],
) -> dict[str, Any]:
    average_travel_times = [float(item["average_travel_time"]) for item in episode_metrics]
    max_travel_times = [float(item["max_travel_time"]) for item in episode_metrics]
    pending_vehicle_counts = [int(item["pending_vehicle_count"]) for item in episode_metrics]
    arrived_vehicle_counts = [int(item["arrived_vehicle_count"]) for item in episode_metrics]

    return {
        "scenario_dir": str(scenario_dir),
        "sumocfg_path": str(sumocfg_path),
        "episode_count": len(episode_metrics),
        "average_of_average_travel_time": np_mean(average_travel_times),
        "average_of_max_travel_time": np_mean(max_travel_times),
        "average_pending_vehicle_count": np_mean([float(value) for value in pending_vehicle_counts]),
        "average_arrived_vehicle_count": np_mean([float(value) for value in arrived_vehicle_counts]),
        "best_episode_by_average_travel_time": min(
            episode_metrics,
            key=lambda item: float(item["average_travel_time"]),
        ) if episode_metrics else None,
        "worst_episode_by_average_travel_time": max(
            episode_metrics,
            key=lambda item: float(item["average_travel_time"]),
        ) if episode_metrics else None,
    }


def refresh_last_episode_xml(episode_metrics: list[dict[str, Any]]) -> None:
    if not episode_metrics:
        return
    last_summary = Path(str(episode_metrics[-1]["summary_file"]))
    last_tripinfo = Path(str(episode_metrics[-1]["tripinfo_file"]))
    if last_summary.exists():
        SUMMARY_PATH.write_text(last_summary.read_text(encoding="utf-8"), encoding="utf-8")
    if last_tripinfo.exists():
        TRIPINFO_PATH.write_text(last_tripinfo.read_text(encoding="utf-8"), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the 50-node benchmark under custom fixed-time control using runnerMain-style travel-time measurement."
    )
    parser.add_argument(
        "--scenario-dir",
        default="sumo_benchmark_50",
        help="Scenario directory containing benchmark.sumocfg.",
    )
    parser.add_argument("--sumo-binary", default="sumo", help="SUMO binary to execute.")
    parser.add_argument("--seed", type=int, default=11, help="Base simulation seed.")
    parser.add_argument("--episodes", type=int, default=1, help="Number of fixed-time episodes to run.")
    parser.add_argument("--seed-stride", type=int, default=1, help="Seed increment applied between episodes.")
    parser.add_argument("--max-steps", type=int, default=3600, help="Maximum SUMO steps per episode.")
    parser.add_argument(
        "--fixed-green-duration",
        type=int,
        default=FIXED_GREEN_DURATION,
        help="Green duration in seconds for each major fixed-time phase.",
    )
    parser.add_argument(
        "--yellow-duration",
        type=int,
        default=YELLOW_DURATION,
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
    global SCENARIO_DIR, SUMOCFG_PATH, FIXED_TIMING_PATH
    SCENARIO_DIR = resolve_scenario_dir(args.scenario_dir)
    SUMOCFG_PATH = SCENARIO_DIR / "benchmark.sumocfg"
    if not SUMOCFG_PATH.exists():
        raise FileNotFoundError(f"SUMO config not found: {SUMOCFG_PATH}")
    FIXED_TIMING_PATH = OUTPUT_DIR / f"fixed{args.fixed_green_duration}_all.add.xml"

    route_path = resolve_sumocfg_input(SUMOCFG_PATH, "route-files")
    net_path = resolve_sumocfg_input(SUMOCFG_PATH, "net-file")
    write_fixed_timing_additional(
        net_path,
        FIXED_TIMING_PATH,
        args.fixed_green_duration,
        args.yellow_duration,
    )
    total_demand = count_total_demand(route_path)

    episode_metrics: list[dict[str, Any]] = []
    for episode_index in range(1, args.episodes + 1):
        seed = args.seed + (episode_index - 1) * args.seed_stride
        summary_path, tripinfo_path = episode_output_paths(episode_index)
        stats = run_episode(
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
            f"runnerFixed ep {episode_index:>3}/{args.episodes} | "
            f"avg_tt={float(stats['average_travel_time']):.2f}s | "
            f"inserted={float(stats['inserted_ratio']):.3f} | "
            f"arrived={int(stats['arrived_vehicle_count'])}"
        )

    aggregate_metrics = aggregate_episode_metrics(SCENARIO_DIR, SUMOCFG_PATH, episode_metrics)
    write_episode_metrics(episode_metrics)
    write_metrics(aggregate_metrics)
    plot_episode_average_travel_time(episode_metrics)
    refresh_last_episode_xml(episode_metrics)

    print("runnerFixed complete.")
    print("control_mode: all_fixed_time")
    print(f"fixed_green_duration: {args.fixed_green_duration}")
    print(f"yellow_duration: {args.yellow_duration}")
    print(f"metric_source: {args.metric_source if args.metric_source != 'auto' else ('libsumo' if libsumo is not None else 'xml')}")
    for key, value in aggregate_metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.3f}")
        else:
            print(f"{key}: {value}")
    print(f"episode_metrics_file: {EPISODE_METRICS_PATH}")
    print(f"metrics_file: {METRICS_PATH}")
    print(f"plot_file: {PLOT_PATH}")
    print(f"summary_file: {SUMMARY_PATH}")
    print(f"tripinfo_file: {TRIPINFO_PATH}")


if __name__ == "__main__":
    main()
