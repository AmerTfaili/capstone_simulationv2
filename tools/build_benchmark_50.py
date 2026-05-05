from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from benchmark50_spec import ROAD_TYPES, get_demand_flows, get_edges, get_internal_node_ids, get_nodes


ROOT = Path(__file__).resolve().parent.parent
SCENARIO_DIR = ROOT / "sumo_benchmark_50"
INPUT_DIR = SCENARIO_DIR / "inputs"
OUTPUT_DIR = SCENARIO_DIR / "generated"


def ensure_dirs() -> None:
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def write_nodes() -> Path:
    path = INPUT_DIR / "benchmark.nod.xml"
    lines = ["<nodes>"]
    for node in get_nodes():
        lines.append(
            f'    <node id="{node.node_id}" x="{node.x:.1f}" y="{node.y:.1f}" type="{node.node_type}"/>'
        )
    lines.append("</nodes>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_types() -> Path:
    path = INPUT_DIR / "benchmark.typ.xml"
    lines = ["<types>"]
    for type_id, attrs in ROAD_TYPES.items():
        lines.append(
            f'    <type id="{type_id}" priority="{attrs["priority"]}" numLanes="{attrs["numLanes"]}" speed="{attrs["speed"]}"/>'
        )
    lines.append("</types>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_edges() -> Path:
    path = INPUT_DIR / "benchmark.edg.xml"
    lines = ["<edges>"]
    for edge in get_edges():
        lines.append(
            f'    <edge id="{edge.edge_id}" from="{edge.from_node}" to="{edge.to_node}" type="{edge.road_class}"/>'
        )
    lines.append("</edges>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_trips() -> Path:
    path = INPUT_DIR / "benchmark.trips.xml"
    lines = ["<routes>"]
    for flow in sorted(get_demand_flows(), key=lambda item: (item.begin, item.end, item.flow_id)):
        lines.append(
            "    "
            + (
                f'<flow id="{flow.flow_id}" begin="{flow.begin}" end="{flow.end}" '
                f'vehsPerHour="{flow.vehs_per_hour}" from="{flow.from_edge}" to="{flow.to_edge}" '
                'departLane="best" departSpeed="max" departPos="base"/>'
            )
        )
    lines.append("</routes>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_sumocfg(net_path: Path, route_path: Path) -> Path:
    path = SCENARIO_DIR / "benchmark.sumocfg"
    content = f"""<configuration>
    <input>
        <net-file value="{net_path.relative_to(SCENARIO_DIR).as_posix()}"/>
        <route-files value="{route_path.relative_to(SCENARIO_DIR).as_posix()}"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="3600"/>
        <step-length value="1"/>
    </time>
    <processing>
        <ignore-route-errors value="false"/>
        <time-to-teleport value="-1"/>
    </processing>
    <report>
        <verbose value="false"/>
        <no-step-log value="true"/>
    </report>
</configuration>
"""
    path.write_text(content, encoding="utf-8")
    return path


def write_metadata() -> Path:
    path = SCENARIO_DIR / "scenario_metadata.json"
    payload = {
        "scenario_id": "benchmark_50",
        "signalized_intersection_count": len(get_internal_node_ids()),
        "signalized_intersection_ids": get_internal_node_ids(),
        "control_budget_target": 10,
        "preselected_control_nodes": [],
        "design_notes": [
            "The network is intentionally asymmetric and not a uniform open grid.",
            "The main corridor is the strongest west-east route through the center of the network.",
            "A stronger upper distributor and a central north-south spine create layered corridor structure.",
            "Additional feeder columns, bottlenecks, and oblique connectors create multiple plausible high-impact control candidates without preselecting them.",
        ],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def write_readme() -> Path:
    path = SCENARIO_DIR / "README.md"
    content = """# 50-Intersection SUMO Benchmark

This benchmark extends the existing paper-inspired style into a larger scenario with:
- exactly 50 signalized internal intersections
- one dominant west-east central corridor
- one stronger upper distributor and one central north-south spine
- selective feeder columns, bottlenecks, and oblique shortcuts
- asymmetric time-varying demand with two strong peak periods
- no preselected controlled intersections

## Files
- `generated/benchmark.net.xml`: compiled SUMO network
- `generated/benchmark.rou.xml`: routed demand
- `benchmark.sumocfg`: runnable SUMO configuration
- `scenario_metadata.json`: topology metadata and the intended 10-node control budget without a chosen subset

## Notes
- The network is engineered to support later budgeted control selection, but it does not rank or hardcode the 10 controlled intersections.
- The geometry is staggered and only partially cross-connected, so it behaves like an asymmetric corridor network rather than a regular grid.
- The strongest movement is the central corridor, followed by the upper distributor and the central spine. Lower bands and diagonal connectors serve as feeders and alternate pressure paths.
"""
    path.write_text(content, encoding="utf-8")
    return path


def run_command(args: list[str]) -> None:
    subprocess.run(args, check=True, cwd=ROOT)


def build() -> None:
    ensure_dirs()
    node_path = write_nodes()
    type_path = write_types()
    edge_path = write_edges()
    trip_path = write_trips()

    net_path = OUTPUT_DIR / "benchmark.net.xml"
    route_path = OUTPUT_DIR / "benchmark.rou.xml"

    run_command(
        [
            "netconvert",
            "--node-files",
            str(node_path),
            "--edge-files",
            str(edge_path),
            "--type-files",
            str(type_path),
            "--output-file",
            str(net_path),
            "--tls.guess",
            "true",
            "--tls.default-type",
            "static",
            "--tls.green.time",
            "21",
            "--tls.yellow.time",
            "3",
            "--tls.red.time",
            "2",
        ]
    )

    run_command(
        [
            "duarouter",
            "--net-file",
            str(net_path),
            "--route-files",
            str(trip_path),
            "--output-file",
            str(route_path),
            "--ignore-errors",
            "false",
        ]
    )

    sumocfg_path = write_sumocfg(net_path, route_path)
    metadata_path = write_metadata()
    readme_path = write_readme()

    print(f"network: {net_path}")
    print(f"routes: {route_path}")
    print(f"config: {sumocfg_path}")
    print(f"metadata: {metadata_path}")
    print(f"readme: {readme_path}")


if __name__ == "__main__":
    try:
        build()
    except subprocess.CalledProcessError as exc:
        print(exc, file=sys.stderr)
        sys.exit(exc.returncode)

