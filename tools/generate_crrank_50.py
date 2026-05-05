from __future__ import annotations

import json
from collections import defaultdict
from heapq import heappop, heappush
from pathlib import Path

from benchmark50_spec import get_demand_flows, get_edges, get_internal_node_ids

ROOT = Path(__file__).resolve().parent.parent
SCENARIO_DIR = ROOT / "sumo_benchmark_50"
OUTPUT_PATH = SCENARIO_DIR / "control_roles_crrank.json"


def road_weight(road_class: str) -> float:
    if road_class == "main":
        return 1.0
    if road_class == "spine":
        return 1.15
    if road_class == "feeder":
        return 1.45
    if road_class == "bottleneck":
        return 1.8
    return 1.95


def road_score(road_class: str) -> float:
    if road_class == "main":
        return 1.10
    if road_class == "spine":
        return 1.05
    if road_class == "feeder":
        return 1.00
    if road_class == "bottleneck":
        return 0.95
    return 0.90


def build_graph() -> dict[str, list[tuple[str, str, float]]]:
    graph: dict[str, list[tuple[str, str, float]]] = defaultdict(list)
    for edge in get_edges():
        graph[edge.from_node].append((edge.to_node, edge.road_class, road_weight(edge.road_class)))
    return graph


def shortest_path(graph: dict[str, list[tuple[str, str, float]]], source: str, target: str) -> list[str]:
    queue: list[tuple[float, str]] = [(0.0, source)]
    dist = {source: 0.0}
    prev: dict[str, str | None] = {source: None}
    while queue:
        cost, node = heappop(queue)
        if node == target:
            break
        if cost > dist[node]:
            continue
        for nxt, _road_class, edge_cost in graph[node]:
            new_cost = cost + edge_cost
            if new_cost < dist.get(nxt, float("inf")):
                dist[nxt] = new_cost
                prev[nxt] = node
                heappush(queue, (new_cost, nxt))
    if target not in prev:
        raise RuntimeError(f"No path found from {source} to {target}")
    path: list[str] = []
    cursor: str | None = target
    while cursor is not None:
        path.append(cursor)
        cursor = prev[cursor]
    path.reverse()
    return path


def candidate_paths(graph: dict[str, list[tuple[str, str, float]]], source: str, target: str) -> list[list[str]]:
    primary = shortest_path(graph, source, target)
    alternatives = [primary]
    for pivot in primary[1:-1]:
        penalized_graph: dict[str, list[tuple[str, str, float]]] = defaultdict(list)
        for node, neighbors in graph.items():
            for nxt, road_class, edge_cost in neighbors:
                penalty = 0.65 if node == pivot or nxt == pivot else 0.0
                penalized_graph[node].append((nxt, road_class, edge_cost + penalty))
        alt = shortest_path(penalized_graph, source, target)
        if alt not in alternatives:
            alternatives.append(alt)
        if len(alternatives) == 3:
            break
    return alternatives


def route_edge_classes(path: list[str]) -> list[str]:
    edge_lookup = {(edge.from_node, edge.to_node): edge.road_class for edge in get_edges()}
    return [edge_lookup[(a, b)] for a, b in zip(path[:-1], path[1:])]


def compute_crrank() -> dict[str, object]:
    graph = build_graph()
    internal_ids = get_internal_node_ids()

    od_items = []
    for flow in get_demand_flows():
        source = flow.from_edge.split("_to_")[0]
        target = flow.to_edge.split("_to_")[1]
        od_items.append({"id": flow.flow_id, "source": source, "target": target, "weight": flow.vehs_per_hour})

    od_total = sum(item["weight"] for item in od_items)
    l0 = {item["id"]: item["weight"] / od_total for item in od_items}

    path_items = []
    path_weight_sum = 0.0
    for item in od_items:
        paths = candidate_paths(graph, item["source"], item["target"])
        base_weights = [1.0 / (idx + 1) for idx in range(len(paths))]
        base_total = sum(base_weights)
        for idx, path in enumerate(paths):
            share = item["weight"] * (base_weights[idx] / base_total)
            path_id = f"{item['id']}__p{idx + 1}"
            path_items.append({"id": path_id, "od_id": item["id"], "nodes": path, "weight": share})
            path_weight_sum += share

    h0 = {item["id"]: item["weight"] / path_weight_sum for item in path_items}

    c0_raw = {}
    for node_id in internal_ids:
        incident_scores = []
        for edge in get_edges():
            if edge.from_node == node_id or edge.to_node == node_id:
                incident_scores.append(road_score(edge.road_class))
        c0_raw[node_id] = sum(incident_scores)
    c0_total = sum(c0_raw.values())
    c0 = {node_id: score / c0_total for node_id, score in c0_raw.items()}

    x_ep = defaultdict(float)
    x_pe = defaultdict(float)
    y_pv = defaultdict(float)
    y_vp = defaultdict(float)

    od_to_weight = {item["id"]: item["weight"] for item in od_items}
    for item in path_items:
        od_id = item["od_id"]
        x_ep[(od_id, item["id"])] = item["weight"] / od_to_weight[od_id]
        x_pe[(item["id"], od_id)] = 1.0

        path_nodes = [node for node in item["nodes"] if node in internal_ids]
        classes = route_edge_classes(item["nodes"])
        upstream_bonus = 0.0
        for path_pos, node_id in enumerate(path_nodes):
            if path_pos > 0:
                upstream_bonus = 0.01 + 0.03 * sum(
                    1.0 for road_class in classes[: path_pos + 1] if road_class in {"main", "spine", "feeder"}
                )
            y_pv[(item["id"], node_id)] = 1.0 + upstream_bonus * c0[node_id]
            y_vp[(node_id, item["id"])] = h0[item["id"]]

    alpha = 0.85
    l = dict(l0)
    h = dict(h0)
    c = dict(c0)

    for _ in range(40):
        new_h = {}
        for path in path_items:
            acc = sum(x_ep[(od["id"], path["id"])] * l[od["id"]] for od in od_items if (od["id"], path["id"]) in x_ep)
            new_h[path["id"]] = alpha * acc + (1.0 - alpha) * h0[path["id"]]
        norm = sum(new_h.values())
        new_h = {k: v / norm for k, v in new_h.items()}

        new_c = {}
        for node_id in internal_ids:
            acc = sum(y_pv[(path["id"], node_id)] * new_h[path["id"]] for path in path_items if (path["id"], node_id) in y_pv)
            new_c[node_id] = alpha * acc + (1.0 - alpha) * c0[node_id]
        norm = sum(new_c.values())
        new_c = {k: v / norm for k, v in new_c.items()}

        new_h_back = {}
        for path in path_items:
            acc = sum(y_vp[(node_id, path["id"])] * new_c[node_id] for node_id in internal_ids if (node_id, path["id"]) in y_vp)
            new_h_back[path["id"]] = alpha * acc + (1.0 - alpha) * h0[path["id"]]
        norm = sum(new_h_back.values())
        new_h_back = {k: v / norm for k, v in new_h_back.items()}

        new_l = {}
        for od in od_items:
            acc = sum(x_pe[(path["id"], od["id"])] * new_h_back[path["id"]] for path in path_items if (path["id"], od["id"]) in x_pe)
            new_l[od["id"]] = alpha * acc + (1.0 - alpha) * l0[od["id"]]
        norm = sum(new_l.values())
        new_l = {k: v / norm for k, v in new_l.items()}

        l = new_l
        h = new_h_back
        c = new_c

    ranked = sorted(c.items(), key=lambda item: item[1], reverse=True)
    drqn_nodes = [node_id for node_id, _score in ranked[:10]]
    fixed_nodes = [node_id for node_id, _score in ranked[10:]]
    return {
        "ranking": [{"node_id": node_id, "score": round(score, 8)} for node_id, score in ranked],
        "drqn_nodes": drqn_nodes,
        "fixed_time_nodes": fixed_nodes,
        "method": "trajectory-inspired CRRank over weighted OD pairs and candidate paths",
        "notes": [
            "Generated for the 50-intersection benchmark using the same CRRank-style propagation logic as the 20-node builder.",
            "This output is kept separate from any manually chosen control_roles file.",
        ],
    }


def main() -> None:
    selection = compute_crrank()
    OUTPUT_PATH.write_text(json.dumps(selection, indent=2), encoding="utf-8")
    print(f"roles: {OUTPUT_PATH}")
    print("top10:", ", ".join(selection["drqn_nodes"]))


if __name__ == "__main__":
    main()
