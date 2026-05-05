from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "plot3.json"
DEFAULT_OUTPUT = ROOT / "runnerMain_selection_grouped_bar.png"

METHOD_ORDER = ["My Approach", "FA", "BA", "RC", "CRRank + DRQN"]
METHOD_COLORS = {
    "My Approach": "#000000",
    "FA": "#3a3a3a",
    "BA": "#7a7a7a",
    "RC": "#d9d9d9",
    "CRRank + DRQN": "#f4f4f4",
}


def resolve_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def infer_selection_method(record: dict[str, Any]) -> str:
    label = str(record.get("label", "")).lower()
    raw = str(record.get("selection_method", "")).strip().upper()
    methodology = str(record.get("methodology", "")).strip().upper()

    if methodology == "DRQN":
        return "CRRank + DRQN"

    if "_fa_" in label or raw == "FA":
        return "FA"
    if "_ba_" in label or raw == "BA":
        return "BA"
    if "_rc_" in label or raw == "RC":
        return "RC"
    return "My Approach"


def load_records(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        records = payload.get("records")
        if isinstance(records, list):
            return [item for item in records if isinstance(item, dict)]
        return [payload]
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    raise RuntimeError(f"Unsupported JSON structure in {path}")


def collect_grouped_data(paths: list[Path], methodology: str) -> dict[str, dict[int, float]]:
    grouped: dict[str, dict[int, float]] = {method: {} for method in METHOD_ORDER}
    for path in paths:
        for record in load_records(path):
            if methodology.lower() != "all" and str(record.get("methodology", "")) != methodology:
                continue
            budget = record.get("budget")
            avg_tt = record.get("average_travel_time")
            if budget is None or avg_tt is None:
                continue
            method = infer_selection_method(record)
            grouped.setdefault(method, {})[int(budget)] = float(avg_tt)
    return grouped


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a grouped bar chart of average travel time by controlled-node budget and selection method."
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=[str(DEFAULT_INPUT)],
        help="One or more plot3-style JSON files. Later files override earlier duplicates.",
    )
    parser.add_argument(
        "--methodology",
        type=str,
        default="all",
        help="Methodology name to filter records by, or 'all' to include every supported series.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT),
        help="Output image path.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Average Travel Time by Budget and Selection Method",
        help="Plot title.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    input_paths = [resolve_path(raw_path) for raw_path in args.inputs]
    for path in input_paths:
        if not path.exists():
            raise FileNotFoundError(f"Input JSON not found: {path}")
    output_path = resolve_path(args.output)

    grouped = collect_grouped_data(input_paths, args.methodology)
    available_methods = [method for method in METHOD_ORDER if grouped.get(method)]
    if not available_methods:
        raise RuntimeError(
            f"No usable records found for methodology '{args.methodology}' in: "
            + ", ".join(str(path) for path in input_paths)
        )

    budgets = sorted({budget for by_budget in grouped.values() for budget in by_budget})
    if not budgets:
        raise RuntimeError("No budgets found in the selected records.")

    x = np.arange(len(budgets), dtype=float)
    group_width = 0.84
    bar_width = group_width / max(1, len(available_methods))

    fig, ax = plt.subplots(figsize=(12, 6.5))
    for idx, method in enumerate(available_methods):
        offset = (idx - (len(available_methods) - 1) / 2.0) * bar_width
        heights = np.array([grouped[method].get(budget, np.nan) for budget in budgets], dtype=float)
        valid = np.isfinite(heights)
        bars = ax.bar(
            x[valid] + offset,
            heights[valid],
            width=bar_width * 0.92,
            label=method,
            color=METHOD_COLORS.get(method, "tab:gray"),
            edgecolor="black",
            linewidth=0.7,
        )

    ax.set_title(args.title)
    ax.set_xlabel("Number of Controlled Nodes")
    ax.set_ylabel("Average Travel Time (s)")
    ax.set_xticks(x)
    ax.set_xticklabels([str(budget) for budget in budgets])
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    ax.legend(
        title="Selection Method",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=len(available_methods),
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)

    print(f"Saved grouped bar chart: {output_path}")
    print("Budgets:", ", ".join(str(budget) for budget in budgets))
    for method in available_methods:
        present = sorted(grouped[method])
        print(f"{method}: budgets {present}")


if __name__ == "__main__":
    main()
