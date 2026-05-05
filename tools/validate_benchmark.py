from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SCENARIO_DIR = ROOT / "sumo_benchmark"
NET_PATH = SCENARIO_DIR / "generated" / "benchmark.net.xml"
ROLES_PATH = SCENARIO_DIR / "control_roles.json"
SUMMARY_PATH = SCENARIO_DIR / "generated" / "summary.xml"
TRIPINFO_PATH = SCENARIO_DIR / "generated" / "tripinfo.xml"


def parse_net() -> tuple[list[str], list[str]]:
    root = ET.parse(NET_PATH).getroot()
    tls_nodes = []
    all_nodes = []
    for junction in root.findall("junction"):
        node_id = junction.attrib["id"]
        if node_id.startswith(":"):
            continue
        all_nodes.append(node_id)
        if junction.attrib.get("type", "").startswith("traffic_light"):
            tls_nodes.append(node_id)
    return tls_nodes, all_nodes


def parse_summary() -> dict[str, object]:
    if not SUMMARY_PATH.exists():
        return {"summary_present": False}
    steps = ET.parse(SUMMARY_PATH).getroot().findall("step")
    last = steps[-1]
    loaded = int(last.attrib["loaded"])
    inserted = int(last.attrib["inserted"])
    arrived = int(last.attrib["arrived"])
    return {
        "summary_present": True,
        "loaded": loaded,
        "inserted": inserted,
        "arrived": arrived,
        "waiting_to_insert": int(last.attrib["waiting"]),
        "teleports": int(last.attrib["teleports"]),
        "inserted_ratio": round(inserted / loaded, 4) if loaded else 0.0,
        "arrived_ratio": round(arrived / loaded, 4) if loaded else 0.0,
        "mean_travel_time": float(last.attrib["meanTravelTime"]),
        "mean_waiting_time": float(last.attrib["meanWaitingTime"]),
    }


def parse_tripinfo() -> dict[str, object]:
    if not TRIPINFO_PATH.exists():
        return {"tripinfo_present": False}
    trips = ET.parse(TRIPINFO_PATH).getroot().findall("tripinfo")
    waits = [float(trip.attrib["waitingTime"]) for trip in trips]
    durations = [float(trip.attrib["duration"]) for trip in trips]
    return {
        "tripinfo_present": True,
        "arrived_trip_count": len(trips),
        "avg_duration": round(sum(durations) / len(durations), 2) if durations else 0.0,
        "avg_wait": round(sum(waits) / len(waits), 2) if waits else 0.0,
        "max_wait": round(max(waits), 2) if waits else 0.0,
    }


def main() -> None:
    tls_nodes, all_nodes = parse_net()
    roles = json.loads(ROLES_PATH.read_text(encoding="utf-8"))
    drqn_nodes = roles["drqn_nodes"]
    fixed_nodes = roles["fixed_time_nodes"]
    report = {
        "total_non_internal_nodes_in_net": len(all_nodes),
        "signalized_intersections": len(tls_nodes),
        "signalized_ids": tls_nodes,
        "drqn_count": len(drqn_nodes),
        "fixed_count": len(fixed_nodes),
        "drqn_nodes": drqn_nodes,
        "fixed_nodes": fixed_nodes,
        "all_selected_are_signalized": all(node_id in set(tls_nodes) for node_id in drqn_nodes + fixed_nodes),
        "selection_covers_all_signalized_nodes": sorted(drqn_nodes + fixed_nodes) == sorted(tls_nodes),
        "summary": parse_summary(),
        "tripinfo": parse_tripinfo(),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
