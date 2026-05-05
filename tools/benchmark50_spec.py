from __future__ import annotations

from benchmark_spec import DemandFlow, Edge, Node, ROAD_TYPES


ROW_XS = {
    "A": [0.0, 130.0, 255.0, 390.0, 530.0, 675.0, 820.0, 970.0, 1125.0, 1280.0],
    "B": [-35.0, 115.0, 250.0, 400.0, 545.0, 690.0, 835.0, 980.0, 1120.0, 1265.0],
    "C": [-55.0, 95.0, 240.0, 395.0, 555.0, 710.0, 870.0, 1030.0, 1185.0, 1335.0],
    "D": [-20.0, 135.0, 285.0, 440.0, 595.0, 745.0, 900.0, 1060.0, 1215.0, 1370.0],
    "E": [35.0, 185.0, 335.0, 490.0, 650.0, 805.0, 965.0, 1125.0, 1295.0, 1450.0],
}

ROW_YS = {
    "A": 470.0,
    "B": 320.0,
    "C": 155.0,
    "D": -20.0,
    "E": -190.0,
}


def _build_internal_nodes() -> list[Node]:
    nodes: list[Node] = []
    node_index = 1
    for row_name in ["A", "B", "C", "D", "E"]:
        for x in ROW_XS[row_name]:
            nodes.append(Node(f"J{node_index:02d}", x, ROW_YS[row_name], "traffic_light"))
            node_index += 1
    return nodes


INTERNAL_NODES = _build_internal_nodes()

A_ROW = [f"J{idx:02d}" for idx in range(1, 11)]
B_ROW = [f"J{idx:02d}" for idx in range(11, 21)]
C_ROW = [f"J{idx:02d}" for idx in range(21, 31)]
D_ROW = [f"J{idx:02d}" for idx in range(31, 41)]
E_ROW = [f"J{idx:02d}" for idx in range(41, 51)]


BOUNDARY_NODES = [
    Node("W_TOP", -210.0, 470.0, "priority"),
    Node("W_UPPER", -240.0, 320.0, "priority"),
    Node("W_MAIN", -260.0, 155.0, "priority"),
    Node("W_LOWER", -220.0, -20.0, "priority"),
    Node("W_SOUTH", -180.0, -190.0, "priority"),
    Node("E_TOP", 1440.0, 470.0, "priority"),
    Node("E_UPPER", 1485.0, 320.0, "priority"),
    Node("E_MAIN", 1540.0, 155.0, "priority"),
    Node("E_LOWER", 1505.0, -20.0, "priority"),
    Node("E_SOUTH", 1575.0, -190.0, "priority"),
    Node("N_WEST", 255.0, 665.0, "priority"),
    Node("N_CENTRAL", 705.0, 705.0, "priority"),
    Node("N_EAST", 1135.0, 670.0, "priority"),
    Node("S_WEST", 335.0, -365.0, "priority"),
    Node("S_CENTRAL", 810.0, -390.0, "priority"),
    Node("S_EAST", 1305.0, -370.0, "priority"),
]


def _add_bidirectional(edges: list[Edge], a: str, b: str, road_class: str) -> None:
    edges.append(Edge(f"{a}_to_{b}", a, b, road_class))
    edges.append(Edge(f"{b}_to_{a}", b, a, road_class))


def _add_chain(edges: list[Edge], node_ids: list[str], road_class: str) -> None:
    for a, b in zip(node_ids[:-1], node_ids[1:]):
        _add_bidirectional(edges, a, b, road_class)


def build_edges() -> list[Edge]:
    edges: list[Edge] = []

    # Five horizontal bands, with the main corridor centered in row C.
    _add_chain(edges, ["W_TOP", *A_ROW, "E_TOP"], "peripheral")
    _add_chain(edges, ["W_UPPER", *B_ROW, "E_UPPER"], "spine")
    _add_chain(edges, ["W_MAIN", *C_ROW, "E_MAIN"], "main")
    _add_chain(edges, ["W_LOWER", *D_ROW, "E_LOWER"], "feeder")
    _add_chain(edges, ["W_SOUTH", *E_ROW, "E_SOUTH"], "peripheral")

    # Three north-south trunks of different strength.
    _add_chain(edges, ["N_WEST", "J03", "J13", "J23", "J33", "J43", "S_WEST"], "feeder")
    _add_chain(edges, ["N_CENTRAL", "J06", "J16", "J26", "J36", "J46", "S_CENTRAL"], "spine")
    _add_chain(edges, ["N_EAST", "J09", "J19", "J29", "J39", "J49", "S_EAST"], "feeder")

    # Selected feeder columns only, to keep the layout asymmetric rather than fully gridded.
    for a, b, road_class in [
        ("J01", "J11", "peripheral"),
        ("J11", "J21", "feeder"),
        ("J21", "J31", "peripheral"),
        ("J12", "J22", "feeder"),
        ("J22", "J32", "bottleneck"),
        ("J32", "J42", "peripheral"),
        ("J04", "J14", "peripheral"),
        ("J14", "J24", "feeder"),
        ("J24", "J34", "feeder"),
        ("J34", "J44", "peripheral"),
        ("J15", "J25", "bottleneck"),
        ("J25", "J35", "feeder"),
        ("J35", "J45", "feeder"),
        ("J07", "J17", "peripheral"),
        ("J17", "J27", "feeder"),
        ("J27", "J37", "bottleneck"),
        ("J37", "J47", "feeder"),
        ("J18", "J28", "feeder"),
        ("J28", "J38", "feeder"),
        ("J38", "J48", "peripheral"),
        ("J10", "J20", "peripheral"),
        ("J20", "J30", "feeder"),
        ("J30", "J40", "peripheral"),
        ("J40", "J50", "peripheral"),
    ]:
        _add_bidirectional(edges, a, b, road_class)

    # Sparse oblique shortcuts and bottlenecks to create nonuniform pressure points.
    for a, b, road_class in [
        ("J04", "J15", "peripheral"),
        ("J08", "J19", "peripheral"),
        ("J15", "J26", "bottleneck"),
        ("J18", "J29", "bottleneck"),
        ("J23", "J34", "peripheral"),
        ("J26", "J37", "bottleneck"),
        ("J33", "J44", "peripheral"),
        ("J38", "J49", "peripheral"),
    ]:
        _add_bidirectional(edges, a, b, road_class)

    return edges


DEMAND_FLOWS = [
    DemandFlow("warmup_main_through", 0, 600, 880, "W_MAIN_to_J21", "J30_to_E_MAIN"),
    DemandFlow("warmup_upper_through", 0, 600, 320, "W_UPPER_to_J11", "J20_to_E_UPPER"),
    DemandFlow("warmup_ncentral_to_east", 0, 600, 460, "N_CENTRAL_to_J06", "J30_to_E_MAIN"),
    DemandFlow("warmup_nwest_to_swest", 0, 600, 150, "N_WEST_to_J03", "J43_to_S_WEST"),
    DemandFlow("warmup_wlower_to_scentral", 0, 600, 140, "W_LOWER_to_J31", "J46_to_S_CENTRAL"),
    DemandFlow("warmup_wtop_to_eupper", 0, 600, 110, "W_TOP_to_J01", "J20_to_E_UPPER"),
    DemandFlow("peak1_main_through", 600, 1500, 1300, "W_MAIN_to_J21", "J30_to_E_MAIN"),
    DemandFlow("peak1_upper_to_emain", 600, 1500, 560, "W_UPPER_to_J11", "J30_to_E_MAIN"),
    DemandFlow("peak1_ncentral_to_emain", 600, 1500, 720, "N_CENTRAL_to_J06", "J30_to_E_MAIN"),
    DemandFlow("peak1_wmain_to_scentral", 600, 1500, 320, "W_MAIN_to_J21", "J46_to_S_CENTRAL"),
    DemandFlow("peak1_wlower_to_elower", 600, 1500, 280, "W_LOWER_to_J31", "J40_to_E_LOWER"),
    DemandFlow("peak1_nwest_to_swest", 600, 1500, 220, "N_WEST_to_J03", "J43_to_S_WEST"),
    DemandFlow("peak1_wtop_to_eupper", 600, 1500, 170, "W_TOP_to_J01", "J20_to_E_UPPER"),
    DemandFlow("peak1_eupper_to_wlower", 600, 1500, 130, "E_UPPER_to_J20", "J31_to_W_LOWER"),
    DemandFlow("recovery_main_through", 1500, 2400, 680, "W_MAIN_to_J21", "J30_to_E_MAIN"),
    DemandFlow("recovery_ncentral_to_emain", 1500, 2400, 400, "N_CENTRAL_to_J06", "J30_to_E_MAIN"),
    DemandFlow("recovery_upper_through", 1500, 2400, 270, "W_UPPER_to_J11", "J20_to_E_UPPER"),
    DemandFlow("recovery_lower_through", 1500, 2400, 180, "W_LOWER_to_J31", "J40_to_E_LOWER"),
    DemandFlow("recovery_emain_to_wlower", 1500, 2400, 140, "E_MAIN_to_J30", "J31_to_W_LOWER"),
    DemandFlow("recovery_neast_to_seast", 1500, 2400, 120, "N_EAST_to_J09", "J49_to_S_EAST"),
    DemandFlow("peak2_main_through", 2400, 3200, 1160, "W_MAIN_to_J21", "J30_to_E_MAIN"),
    DemandFlow("peak2_ncentral_to_scentral", 2400, 3200, 700, "N_CENTRAL_to_J06", "J46_to_S_CENTRAL"),
    DemandFlow("peak2_wupper_to_seast", 2400, 3200, 560, "W_UPPER_to_J11", "J49_to_S_EAST"),
    DemandFlow("peak2_wmain_to_elower", 2400, 3200, 450, "W_MAIN_to_J21", "J40_to_E_LOWER"),
    DemandFlow("peak2_wlower_to_elower", 2400, 3200, 390, "W_LOWER_to_J31", "J40_to_E_LOWER"),
    DemandFlow("peak2_neast_to_emain", 2400, 3200, 340, "N_EAST_to_J09", "J30_to_E_MAIN"),
    DemandFlow("peak2_wtop_to_etop", 2400, 3200, 150, "W_TOP_to_J01", "J10_to_E_TOP"),
    DemandFlow("peak2_eupper_to_swest", 2400, 3200, 190, "E_UPPER_to_J20", "J43_to_S_WEST"),
    DemandFlow("cooldown_main_through", 3200, 3600, 560, "W_MAIN_to_J21", "J30_to_E_MAIN"),
    DemandFlow("cooldown_upper_through", 3200, 3600, 200, "W_UPPER_to_J11", "J20_to_E_UPPER"),
    DemandFlow("cooldown_ncentral_to_scentral", 3200, 3600, 250, "N_CENTRAL_to_J06", "J46_to_S_CENTRAL"),
    DemandFlow("cooldown_wlower_to_elower", 3200, 3600, 130, "W_LOWER_to_J31", "J40_to_E_LOWER"),
    DemandFlow("cooldown_emain_to_wmain", 3200, 3600, 110, "E_MAIN_to_J30", "J21_to_W_MAIN"),
]


def get_nodes() -> list[Node]:
    return INTERNAL_NODES + BOUNDARY_NODES


def get_internal_node_ids() -> list[str]:
    return [node.node_id for node in INTERNAL_NODES]


def get_edges() -> list[Edge]:
    return build_edges()


def get_demand_flows() -> list[DemandFlow]:
    return DEMAND_FLOWS
