from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path


def _phase_weight(state: str) -> float:
    return sum(1.0 if char == "G" else 0.65 for char in state if char in {"G", "g"})


def _phase_overlap(lhs: str, rhs: str) -> float:
    lhs_positions = {idx for idx, char in enumerate(lhs) if char in {"G", "g"}}
    rhs_positions = {idx for idx, char in enumerate(rhs) if char in {"G", "g"}}
    if not lhs_positions or not rhs_positions:
        return 0.0
    return len(lhs_positions & rhs_positions) / len(lhs_positions | rhs_positions)


def _select_major_green_states(phase_states: list[str]) -> list[str]:
    green_states = [
        state
        for state in phase_states
        if "y" not in state.lower() and any(char in {"G", "g"} for char in state)
    ]
    if len(green_states) <= 2:
        return green_states
    first = max(green_states, key=_phase_weight)
    remainder = [state for state in green_states if state != first]
    if not remainder:
        return [first, first]
    second = max(remainder, key=lambda state: _phase_weight(state) - 1.2 * _phase_overlap(first, state))
    return [first, second]


def _build_transition_state(from_state: str, to_state: str) -> str:
    chars: list[str] = []
    for from_char, to_char in zip(from_state, to_state):
        if from_char in {"G", "g"} and to_char in {"r", "R"}:
            chars.append("y")
        elif from_char in {"G", "g"} and to_char in {"G", "g"}:
            chars.append("G" if from_char == "G" or to_char == "G" else "g")
        else:
            chars.append("r")
    return "".join(chars)


def write_fixed_timing_additional(
    net_path: Path,
    output_path: Path,
    green_duration: int,
    yellow_duration: int,
) -> Path:
    root = ET.parse(net_path).getroot()
    additional = ET.Element("additional")

    for tl_logic in root.findall("tlLogic"):
        tls_id = tl_logic.attrib.get("id")
        if not tls_id:
            continue
        phase_states = [phase.attrib["state"] for phase in tl_logic.findall("phase") if "state" in phase.attrib]
        green_states = _select_major_green_states(phase_states)
        if not green_states:
            continue
        if len(green_states) == 1:
            green_states = [green_states[0], green_states[0]]
        green_a, green_b = green_states
        logic = ET.SubElement(
            additional,
            "tlLogic",
            {
                "id": tls_id,
                "type": "static",
                "programID": f"fixed{green_duration}_{tls_id}",
                "offset": "0",
            },
        )
        ET.SubElement(logic, "phase", {"duration": str(green_duration), "state": green_a})
        ET.SubElement(
            logic,
            "phase",
            {"duration": str(yellow_duration), "state": _build_transition_state(green_a, green_b)},
        )
        ET.SubElement(logic, "phase", {"duration": str(green_duration), "state": green_b})
        ET.SubElement(
            logic,
            "phase",
            {"duration": str(yellow_duration), "state": _build_transition_state(green_b, green_a)},
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(additional).write(output_path, encoding="utf-8", xml_declaration=True)
    return output_path
