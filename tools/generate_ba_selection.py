from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import networkx as nx


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


def parse_network(net_path: Path) -> tuple[set[str], nx.Graph]:
    root = ET.parse(net_path).getroot()
    signalized_nodes = {
        junction.get("id")
        for junction in root.findall("junction")
        if junction.get("type") == "traffic_light" and junction.get("id")
    }

    graph = nx.Graph()
    for edge in root.findall("edge"):
        if edge.get("function") == "internal":
            continue
        from_node = edge.get("from")
        to_node = edge.get("to")
        if not from_node or not to_node:
            continue
        lane_lengths = []
        for lane in edge.findall("lane"):
            raw_length = lane.get("length")
            if raw_length is None:
                continue
            lane_lengths.append(float(raw_length))
        edge_length = min(lane_lengths) if lane_lengths else 1.0
        if graph.has_edge(from_node, to_node):
            graph[from_node][to_node]["weight"] = min(graph[from_node][to_node]["weight"], edge_length)
        else:
            graph.add_edge(from_node, to_node, weight=edge_length)

    return signalized_nodes, graph


def compute_ba_selection(scenario_dir: Path, budget: int) -> dict[str, object]:
    net_path = scenario_dir / "generated" / "benchmark.net.xml"
    if not net_path.exists():
        raise FileNotFoundError(f"Network file not found: {net_path}")

    signalized_nodes, graph = parse_network(net_path)
    if not signalized_nodes:
        raise RuntimeError(f"No signalized nodes found in network: {net_path}")
    if graph.number_of_nodes() == 0 or graph.number_of_edges() == 0:
        raise RuntimeError(f"Network graph is empty: {net_path}")

    centrality = nx.betweenness_centrality(graph, weight="weight", normalized=True)
    ranked = sorted(
        ((node_id, float(centrality.get(node_id, 0.0))) for node_id in signalized_nodes),
        key=lambda item: (-item[1], item[0]),
    )

    ranking = [
        {
            "node_id": node_id,
            "betweenness_centrality": round(score, 8),
            "score": round(score, 8),
        }
        for node_id, score in ranked
    ]
    drqn_nodes = [item["node_id"] for item in ranking[:budget]]
    fixed_time_nodes = [item["node_id"] for item in ranking[budget:]]

    return {
        "method": "BA",
        "metric": "betweenness centrality on the physical road-network graph",
        "graph_type": "undirected",
        "edge_weight": "lane length",
        "scenario_dir": str(scenario_dir),
        "network_file": str(net_path),
        "budget": budget,
        "node_count": graph.number_of_nodes(),
        "edge_count": graph.number_of_edges(),
        "ranking": ranking,
        "drqn_nodes": drqn_nodes,
        "fixed_time_nodes": fixed_time_nodes,
        "notes": [
            "BA ranks signalized intersections only by shortest-path betweenness centrality.",
            "Shortest paths are computed on the generated SUMO network with lane length as the edge weight.",
        ],
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate betweenness-based (BA) node selection for a SUMO benchmark.")
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
        "--output",
        type=str,
        default=None,
        help="Output JSON path. Defaults to <scenario-dir>/control_roles_ba.json.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    scenario_dir = resolve_path(args.scenario_dir)
    budget = args.budget if args.budget is not None else infer_default_budget(scenario_dir)
    if budget < 1:
        raise RuntimeError("--budget must be at least 1.")
    output_path = resolve_path(args.output) if args.output else scenario_dir / "control_roles_ba.json"

    selection = compute_ba_selection(scenario_dir, budget)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(selection, indent=2), encoding="utf-8")
    print(f"BA selection written to: {output_path}")
    print("top nodes:", ", ".join(selection["drqn_nodes"]))


if __name__ == "__main__":
    main()
