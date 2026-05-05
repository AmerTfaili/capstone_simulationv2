from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Node:
    node_id: str
    x: float
    y: float
    node_type: str


@dataclass(frozen=True)
class Edge:
    edge_id: str
    from_node: str
    to_node: str
    road_class: str


@dataclass(frozen=True)
class DemandFlow:
    flow_id: str
    begin: int
    end: int
    vehs_per_hour: int
    from_edge: str
    to_edge: str


ROAD_TYPES = {
    "main": {"numLanes": 3, "speed": 16.7, "priority": 90},
    "spine": {"numLanes": 2, "speed": 15.0, "priority": 80},
    "feeder": {"numLanes": 2, "speed": 13.0, "priority": 60},
    "peripheral": {"numLanes": 1, "speed": 11.0, "priority": 40},
    "bottleneck": {"numLanes": 1, "speed": 10.0, "priority": 55},
}


INTERNAL_NODES = [
    Node("J01", 0.0, 310.0, "traffic_light"),
    Node("J02", 135.0, 330.0, "traffic_light"),
    Node("J03", 270.0, 315.0, "traffic_light"),
    Node("J04", 410.0, 335.0, "traffic_light"),
    Node("J05", 560.0, 310.0, "traffic_light"),
    Node("J06", -20.0, 170.0, "traffic_light"),
    Node("J07", 125.0, 180.0, "traffic_light"),
    Node("J08", 270.0, 175.0, "traffic_light"),
    Node("J09", 420.0, 185.0, "traffic_light"),
    Node("J10", 585.0, 170.0, "traffic_light"),
    Node("J11", 5.0, 35.0, "traffic_light"),
    Node("J12", 145.0, 45.0, "traffic_light"),
    Node("J13", 280.0, 35.0, "traffic_light"),
    Node("J14", 425.0, 50.0, "traffic_light"),
    Node("J15", 565.0, 25.0, "traffic_light"),
    Node("J16", 25.0, -120.0, "traffic_light"),
    Node("J17", 165.0, -135.0, "traffic_light"),
    Node("J18", 295.0, -120.0, "traffic_light"),
    Node("J19", 440.0, -140.0, "traffic_light"),
    Node("J20", 585.0, -120.0, "traffic_light"),
]


BOUNDARY_NODES = [
    Node("W_TOP", -180.0, 330.0, "priority"),
    Node("W_MAIN", -210.0, 170.0, "priority"),
    Node("W_LOWER", -185.0, 35.0, "priority"),
    Node("E_TOP", 740.0, 310.0, "priority"),
    Node("E_MAIN", 780.0, 170.0, "priority"),
    Node("E_LOWER", 730.0, -10.0, "priority"),
    Node("N_SPINE", 270.0, 500.0, "priority"),
    Node("N_EAST", 560.0, 500.0, "priority"),
    Node("S_SPINE", 295.0, -285.0, "priority"),
    Node("S_EAST", 585.0, -285.0, "priority"),
]


def _add_bidirectional(edges: list[Edge], a: str, b: str, road_class: str) -> None:
    edges.append(Edge(f"{a}_to_{b}", a, b, road_class))
    edges.append(Edge(f"{b}_to_{a}", b, a, road_class))


def build_edges() -> list[Edge]:
    edges: list[Edge] = []

    # Dominant west-east corridor.
    for a, b in [("W_MAIN", "J06"), ("J06", "J07"), ("J07", "J08"), ("J08", "J09"), ("J09", "J10"), ("J10", "E_MAIN")]:
        _add_bidirectional(edges, a, b, "main")

    # Secondary north-south spine.
    for a, b in [("N_SPINE", "J03"), ("J03", "J08"), ("J08", "J13"), ("J13", "J18"), ("J18", "S_SPINE")]:
        _add_bidirectional(edges, a, b, "spine")

    # Upper belt and access roads.
    for a, b in [("W_TOP", "J02"), ("J01", "J02"), ("J02", "J03"), ("J03", "J04"), ("J04", "J05"), ("J05", "E_TOP")]:
        _add_bidirectional(edges, a, b, "peripheral")

    # Lower distributor, intentionally weaker than the core.
    for a, b in [("W_LOWER", "J11"), ("J11", "J12"), ("J12", "J13"), ("J13", "J14"), ("J14", "J15"), ("J14", "E_LOWER")]:
        _add_bidirectional(edges, a, b, "feeder")

    # Far south edge, weakest family.
    for a, b in [("J16", "J17"), ("J17", "J18"), ("J18", "J19"), ("J19", "J20"), ("J20", "S_EAST")]:
        _add_bidirectional(edges, a, b, "peripheral")

    # Feeder columns into the corridor and lower distributor.
    for a, b, road_class in [
        ("J01", "J06", "peripheral"),
        ("J02", "J07", "feeder"),
        ("J04", "J09", "feeder"),
        ("J05", "J10", "peripheral"),
        ("J07", "J12", "bottleneck"),
        ("J09", "J14", "bottleneck"),
        ("J11", "J16", "peripheral"),
        ("J12", "J17", "feeder"),
        ("J14", "J19", "feeder"),
        ("J15", "J20", "peripheral"),
    ]:
        _add_bidirectional(edges, a, b, road_class)

    # A small number of off-axis connectors to prevent brittle dead zones without creating many bypasses.
    for a, b, road_class in [
        ("J03", "N_EAST", "peripheral"),
        ("J10", "J15", "bottleneck"),
        ("J07", "J11", "peripheral"),
        ("J09", "J15", "peripheral"),
    ]:
        _add_bidirectional(edges, a, b, road_class)

    return edges


DEMAND_FLOWS = [
    DemandFlow("warmup_main_through", 0, 600, 760, "W_MAIN_to_J06", "J10_to_E_MAIN"),
    DemandFlow("warmup_north_to_east", 0, 600, 430, "N_SPINE_to_J03", "J10_to_E_MAIN"),
    DemandFlow("warmup_main_to_south", 0, 600, 180, "W_MAIN_to_J06", "J18_to_S_SPINE"),
    DemandFlow("warmup_wtop_to_east", 0, 600, 150, "W_TOP_to_J02", "J10_to_E_MAIN"),
    DemandFlow("warmup_wlower_to_elower", 0, 600, 90, "W_LOWER_to_J11", "J14_to_E_LOWER"),
    DemandFlow("peak1_main_through", 600, 1400, 1100, "W_MAIN_to_J06", "J10_to_E_MAIN"),
    DemandFlow("peak1_north_to_east", 600, 1400, 680, "N_SPINE_to_J03", "J10_to_E_MAIN"),
    DemandFlow("peak1_main_to_south", 600, 1400, 320, "W_MAIN_to_J06", "J18_to_S_SPINE"),
    DemandFlow("peak1_wtop_to_east", 600, 1400, 230, "W_TOP_to_J02", "J10_to_E_MAIN"),
    DemandFlow("peak1_north_to_lower", 600, 1400, 200, "N_SPINE_to_J03", "J14_to_E_LOWER"),
    DemandFlow("peak1_wlower_to_south", 600, 1400, 140, "W_LOWER_to_J11", "J18_to_S_SPINE"),
    DemandFlow("recovery_main_through", 1400, 2200, 520, "W_MAIN_to_J06", "J10_to_E_MAIN"),
    DemandFlow("recovery_north_to_east", 1400, 2200, 320, "N_SPINE_to_J03", "J10_to_E_MAIN"),
    DemandFlow("recovery_main_to_south", 1400, 2200, 150, "W_MAIN_to_J06", "J18_to_S_SPINE"),
    DemandFlow("recovery_wlower_to_elower", 1400, 2200, 110, "W_LOWER_to_J11", "J14_to_E_LOWER"),
    DemandFlow("recovery_background_east_to_wlower", 1400, 2200, 90, "N_EAST_to_J03", "J11_to_W_LOWER"),
    DemandFlow("peak2_main_through", 2200, 3000, 900, "W_MAIN_to_J06", "J10_to_E_MAIN"),
    DemandFlow("peak2_spine_surge", 2200, 3000, 760, "N_SPINE_to_J03", "J18_to_S_SPINE"),
    DemandFlow("peak2_main_to_lower", 2200, 3000, 560, "W_MAIN_to_J06", "J14_to_E_LOWER"),
    DemandFlow("peak2_wlower_to_seast", 2200, 3000, 300, "W_LOWER_to_J11", "J20_to_S_EAST"),
    DemandFlow("peak2_wmain_to_seast", 2200, 3000, 240, "W_MAIN_to_J06", "J20_to_S_EAST"),
    DemandFlow("peak2_wtop_to_east", 2200, 3000, 180, "W_TOP_to_J02", "J10_to_E_MAIN"),
    DemandFlow("cooldown_main_through", 3000, 3600, 420, "W_MAIN_to_J06", "J10_to_E_MAIN"),
    DemandFlow("cooldown_north_to_east", 3000, 3600, 260, "N_SPINE_to_J03", "J10_to_E_MAIN"),
    DemandFlow("cooldown_wmain_to_seast", 3000, 3600, 130, "W_MAIN_to_J06", "J20_to_S_EAST"),
]


def get_nodes() -> list[Node]:
    return INTERNAL_NODES + BOUNDARY_NODES


def get_internal_node_ids() -> list[str]:
    return [node.node_id for node in INTERNAL_NODES]


def get_edges() -> list[Edge]:
    return build_edges()


def get_demand_flows() -> list[DemandFlow]:
    return DEMAND_FLOWS
