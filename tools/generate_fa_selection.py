from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def resolve_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def load_signalized_nodes(net_path: Path) -> list[str]:
    root = ET.parse(net_path).getroot()
    node_ids = [
        junction.get("id")
        for junction in root.findall("junction")
        if junction.get("type") == "traffic_light" and junction.get("id")
    ]
    return sorted(node_ids)


def iter_vehicle_routes(route_path: Path) -> list[list[str]]:
    routes: list[list[str]] = []
    context = ET.iterparse(route_path, events=("end",))
    for _event, elem in context:
        if elem.tag != "vehicle":
            continue
        route_elem = elem.find("route")
        edges_attr = route_elem.get("edges") if route_elem is not None else None
        if edges_attr:
            routes.append(edges_attr.split())
        elem.clear()
    return routes


def to_edge_target(edge_id: str) -> str:
    sep = "_to_"
    if sep not in edge_id:
        raise RuntimeError(f"Unsupported edge id format for FA selection: {edge_id}")
    return edge_id.split(sep, 1)[1]


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


def compute_fa_selection(scenario_dir: Path, budget: int) -> dict[str, object]:
    net_path = scenario_dir / "generated" / "benchmark.net.xml"
    route_path = scenario_dir / "generated" / "benchmark.rou.xml"
    if not net_path.exists():
        raise FileNotFoundError(f"Network file not found: {net_path}")
    if not route_path.exists():
        raise FileNotFoundError(f"Route file not found: {route_path}")

    signalized_nodes = set(load_signalized_nodes(net_path))
    if not signalized_nodes:
        raise RuntimeError(f"No signalized nodes found in network: {net_path}")

    flow_counter: Counter[str] = Counter({node_id: 0 for node_id in signalized_nodes})
    vehicle_routes = iter_vehicle_routes(route_path)
    for edges in vehicle_routes:
        for edge_id in edges:
            target_node = to_edge_target(edge_id)
            if target_node in signalized_nodes:
                flow_counter[target_node] += 1

    total_vehicle_count = len(vehicle_routes)
    ranked = sorted(flow_counter.items(), key=lambda item: (-item[1], item[0]))
    ranking = [
        {
            "node_id": node_id,
            "vehicle_count": int(vehicle_count),
            "score": round(vehicle_count / total_vehicle_count, 8) if total_vehicle_count else 0.0,
        }
        for node_id, vehicle_count in ranked
    ]
    drqn_nodes = [item["node_id"] for item in ranking[:budget]]
    fixed_time_nodes = [item["node_id"] for item in ranking[budget:]]

    return {
        "method": "FA",
        "metric": "traffic-flow-only node ranking from routed vehicle trajectories",
        "scenario_dir": str(scenario_dir),
        "network_file": str(net_path),
        "route_file": str(route_path),
        "budget": budget,
        "total_vehicle_count": total_vehicle_count,
        "ranking": ranking,
        "drqn_nodes": drqn_nodes,
        "fixed_time_nodes": fixed_time_nodes,
        "notes": [
            "FA baseline ranks signalized intersections only by how many routed vehicles traverse them.",
            "Scores are computed from benchmark.rou.xml, which is the routed vehicle set used by the scenario.",
        ],
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate flow-based (FA) node selection for a SUMO benchmark.")
    parser.add_argument(
        "--scenario-dir",
        type=str,
        default="sumo_benchmark_50",
        help="Scenario directory containing generated/benchmark.net.xml and generated/benchmark.rou.xml.",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=None,
        help="Number of top-ranked nodes to place in drqn_nodes. Defaults to scenario_metadata control_budget_target or 10.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON path. Defaults to <scenario-dir>/control_roles_fa.json.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    scenario_dir = resolve_path(args.scenario_dir)
    budget = args.budget if args.budget is not None else infer_default_budget(scenario_dir)
    if budget < 1:
        raise RuntimeError("--budget must be at least 1.")
    output_path = resolve_path(args.output) if args.output else scenario_dir / "control_roles_fa.json"

    selection = compute_fa_selection(scenario_dir, budget)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(selection, indent=2), encoding="utf-8")
    print(f"FA selection written to: {output_path}")
    print("top nodes:", ", ".join(selection["drqn_nodes"]))


if __name__ == "__main__":
    main()
