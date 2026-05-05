from __future__ import annotations

import argparse
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SCENARIO_DIR = ROOT / "sumo_benchmark"
SUMOCFG_PATH = SCENARIO_DIR / "benchmark.sumocfg"
OUTPUT_DIR = ROOT / "training_outputs"
SUMMARY_PATH = OUTPUT_DIR / "fixed_time_summary.xml"
TRIPINFO_PATH = OUTPUT_DIR / "fixed_time_tripinfo.xml"


def resolve_scenario_dir(raw_path: str) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def run_sumo(sumo_binary: str, seed: int) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [
        sumo_binary,
        "-c",
        str(SUMOCFG_PATH),
        "--seed",
        str(seed),
        "--summary-output",
        str(SUMMARY_PATH),
        "--tripinfo-output",
        str(TRIPINFO_PATH),
        "--no-step-log",
        "true",
    ]
    subprocess.run(cmd, check=True, cwd=ROOT)


def summarize_outputs() -> dict[str, float | int]:
    summary_root = ET.parse(SUMMARY_PATH).getroot()
    last_step = summary_root.findall("step")[-1]

    tripinfo_root = ET.parse(TRIPINFO_PATH).getroot()
    trips = tripinfo_root.findall("tripinfo")
    durations = [float(trip.attrib["duration"]) for trip in trips]
    waits = [float(trip.attrib["waitingTime"]) for trip in trips]

    return {
        "loaded": int(last_step.attrib["loaded"]),
        "inserted": int(last_step.attrib["inserted"]),
        "arrived": int(last_step.attrib["arrived"]),
        "waiting_to_insert": int(last_step.attrib["waiting"]),
        "teleports": int(last_step.attrib["teleports"]),
        "mean_summary_travel_time": float(last_step.attrib["meanTravelTime"]),
        "mean_summary_waiting_time": float(last_step.attrib["meanWaitingTime"]),
        "arrived_trip_count": len(trips),
        "average_trip_duration": sum(durations) / len(durations) if durations else float("nan"),
        "average_trip_waiting_time": sum(waits) / len(waits) if waits else float("nan"),
        "max_trip_waiting_time": max(waits) if waits else float("nan"),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the benchmark as an all-fixed-time baseline and print key metrics.")
    parser.add_argument(
        "--scenario-dir",
        default="sumo_benchmark",
        help="Scenario directory containing benchmark.sumocfg.",
    )
    parser.add_argument("--sumo-binary", default="sumo", help="SUMO binary to execute.")
    parser.add_argument("--seed", type=int, default=11, help="Simulation seed.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    global SCENARIO_DIR, SUMOCFG_PATH
    SCENARIO_DIR = resolve_scenario_dir(args.scenario_dir)
    SUMOCFG_PATH = SCENARIO_DIR / "benchmark.sumocfg"
    if not SUMOCFG_PATH.exists():
        raise FileNotFoundError(f"SUMO config not found: {SUMOCFG_PATH}")
    try:
        run_sumo(args.sumo_binary, args.seed)
    except subprocess.CalledProcessError as exc:
        print(exc, file=sys.stderr)
        sys.exit(exc.returncode)

    stats = summarize_outputs()
    print("Fixed-time baseline complete.")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"{key}: {value:.3f}")
        else:
            print(f"{key}: {value}")
    print(f"summary_file: {SUMMARY_PATH}")
    print(f"tripinfo_file: {TRIPINFO_PATH}")


if __name__ == "__main__":
    main()
