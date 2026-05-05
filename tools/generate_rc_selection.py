from __future__ import annotations

import argparse
import json
import random
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def resolve_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def infer_default_budget(scenario_dir: Path) -> int:
    metadata_path = scenario_dir / "scenario_metadata.json"
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            metadata = None
        if isinstance(metadata, dict):
            budget = metadata.get("control_budget_target")
            if isinstance(budget, int) and budget > 0:
                return budget
    return 10


def load_signalized_nodes(net_path: Path) -> list[str]:
    root = ET.parse(net_path).getroot()
    node_ids = [
        junction.get("id")
        for junction in root.findall("junction")
        if junction.get("type") == "traffic_light" and junction.get("id")
    ]
    return sorted(node_ids)


def compute_rc_selection(scenario_dir: Path, budget: int, seed: int) -> dict[str, object]:
    net_path = scenario_dir / "generated" / "benchmark.net.xml"
    if not net_path.exists():
        raise FileNotFoundError(f"Network file not found: {net_path}")

    signalized_nodes = load_signalized_nodes(net_path)
    if not signalized_nodes:
        raise RuntimeError(f"No signalized nodes found in network: {net_path}")

    ordered_nodes = list(signalized_nodes)
    rng = random.Random(seed)
    rng.shuffle(ordered_nodes)

    ranking = [
        {
            "node_id": node_id,
            "random_rank": idx + 1,
        }
        for idx, node_id in enumerate(ordered_nodes)
    ]
    drqn_nodes = ordered_nodes[:budget]
    fixed_time_nodes = ordered_nodes[budget:]

    return {
        "method": "RC",
        "metric": "uniform random permutation over signalized intersections",
        "scenario_dir": str(scenario_dir),
        "network_file": str(net_path),
        "budget": budget,
        "seed": seed,
        "node_count": len(signalized_nodes),
        "ranking": ranking,
        "drqn_nodes": drqn_nodes,
        "fixed_time_nodes": fixed_time_nodes,
        "notes": [
            "RC baseline uses a seeded random ordering of all signalized intersections.",
            "Budget-k evaluation should use the first k nodes from this fixed random order.",
        ],
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate random-choice (RC) node selection for a SUMO benchmark.")
    parser.add_argument(
        "--scenario-dir",
        type=str,
        default="sumo_benchmark_50",
        help="Scenario directory containing generated/benchmark.net.xml.",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=None,
        help="Number of top-ranked nodes to place in drqn_nodes. Defaults to scenario_metadata control_budget_target or 10.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed used to generate the RC ordering.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON path. Defaults to <scenario-dir>/control_roles_rc_seed<seed>.json.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    scenario_dir = resolve_path(args.scenario_dir)
    budget = args.budget if args.budget is not None else infer_default_budget(scenario_dir)
    if budget < 1:
        raise RuntimeError("--budget must be at least 1.")
    output_path = (
        resolve_path(args.output)
        if args.output
        else scenario_dir / f"control_roles_rc_seed{args.seed}.json"
    )

    selection = compute_rc_selection(scenario_dir, budget, args.seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(selection, indent=2), encoding="utf-8")
    print(f"RC selection written to: {output_path}")
    print(f"seed: {args.seed}")
    print("top nodes:", ", ".join(selection["drqn_nodes"]))


if __name__ == "__main__":
    main()
