from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import runnerMain as base
import networkx as nx
import numpy as np
import torch

from sumo_fixed_timing import write_fixed_timing_additional


OUTPUT_DIR_V2 = base.ROOT / "runnerMainV2_outputs"
CHECKPOINT_DIR_V2 = base.ROOT / "runnerMainV2_checkpoints"

if getattr(base, "SUMO_BACKEND", "libsumo") != "libsumo":
    raise RuntimeError(
        "runnerMainV2 requires libsumo only. "
        "The current environment fell back to traci because libsumo could not be loaded. "
        "Fix the SUMO/libsumo installation mismatch before running runnerMainV2."
    )


@dataclass
class MainConfigV2(base.MainConfig):
    output_dir: Path = OUTPUT_DIR_V2
    checkpoint_dir: Path = CHECKPOINT_DIR_V2
    selector_score_ema_alpha: float = 0.65
    selector_history_decay: float = 0.85
    selector_pair_decay: float = 0.88
    selector_history_bonus_weight: float = 0.12
    selector_pair_bonus_weight: float = 0.10
    selector_cohesion_weight: float = 0.14
    selector_corridor_bonus_weight: float = 0.10
    selector_set_improvement_margin: float = 0.025
    selector_candidate_pool_size: int = 14
    selector_pair_memory_scale: float = 0.50


class MethodologyRunnerV2(base.MethodologyRunner):
    def __init__(self, config: MainConfigV2) -> None:
        self.config = config
        self.device = torch.device(base.resolve_torch_device(config.device))
        self.config.device = str(self.device)
        base.set_global_seed(config.seed)

        self.output_dir = config.output_dir
        self.checkpoint_dir = config.checkpoint_dir
        self.metrics_dir = self.output_dir / "metrics"
        self.plots_dir = self.output_dir / "plots"
        self.selectors_dir = self.output_dir / "selectors"
        self.artifacts_dir = self.output_dir / "artifacts"
        self.phase_checkpoint_dir = self.checkpoint_dir / "phase"
        self.best_checkpoint_dir = self.checkpoint_dir / "best"
        self.final_checkpoint_dir = self.checkpoint_dir / "final"
        for path in (
            self.output_dir,
            self.metrics_dir,
            self.plots_dir,
            self.selectors_dir,
            self.artifacts_dir,
            self.checkpoint_dir,
            self.phase_checkpoint_dir,
            self.best_checkpoint_dir,
            self.final_checkpoint_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

        self.net_path = self._resolve_sumocfg_path("net-file")
        self.route_path = self._resolve_sumocfg_path("route-files")
        self.fixed_timing_additional_path = (
            self.artifacts_dir / f"fixed{self.config.fixed_green_duration}_uncontrolled.add.xml"
        )
        write_fixed_timing_additional(
            self.net_path,
            self.fixed_timing_additional_path,
            self.config.fixed_green_duration,
            self.config.yellow_duration,
        )
        self.total_demand = self._count_total_demand(self.route_path)
        self.node_ids, self.graph, self.static_scores, self.static_context = self._build_static_graph_and_scores()
        self.node_to_idx = {node_id: idx for idx, node_id in enumerate(self.node_ids)}
        self.node_context_matrix = self._build_node_context_matrix()
        self.adjacency = self._build_adjacency_tensor()
        self.control_specs = self._build_control_specs()
        self.selected_nodes = self._initial_selected_subset()
        self.shared_controller_id = "shared_policy"

        sample_state_dim = self._state_dim()
        self.controllers: dict[str, base.ControllerRuntime] = {
            self.shared_controller_id: base.ControllerRuntime(
                sample_state_dim,
                config.shared_hidden_size,
                config,
                num_nodes=len(self.node_ids),
                node_context_matrix=self.node_context_matrix,
            )
        }
        self.node_controller_assignment: dict[str, str] = {
            node_id: self.shared_controller_id for node_id in self.selected_nodes
        }

        self.gat_input_dim = self._raw_feature_dim() * self.config.history_length
        self.gat_model = base.GraphAttentionForecaster(
            input_dim=self.gat_input_dim,
            hidden_dim=self.config.gat_hidden_size,
            horizon=self.config.prediction_horizon,
            num_heads=self.config.gat_heads,
        ).to(self.device)
        self.gat_optimizer = base.optim.Adam(self.gat_model.parameters(), lr=config.gat_lr)
        self.frozen_gat_stats: dict[str, Any] | None = None

        self.episode_metrics: list[dict[str, Any]] = []
        self.eval_metrics: list[dict[str, Any]] = []
        self.phase_history: list[dict[str, Any]] = []
        self.selector_quality_history: list[dict[str, Any]] = []
        self.best_eval_travel_time: float | None = None
        self.best_eval_score: float | None = None
        self.resume_episode: int = 0

        self.selector_score_ema: dict[str, float] = {}
        self.node_success_memory: dict[str, float] = {node_id: 0.0 for node_id in self.node_ids}
        self.pair_success_memory: dict[tuple[str, str], float] = {}
        self.graph_affinity_matrix = self._build_graph_affinity_matrix()

        self.plot = base.MainLivePlot(self.plots_dir / "average_travel_time.png")
        self.plot.ax.set_title("runnerMainV2 Phase-wise Training")

    def _build_graph_affinity_matrix(self) -> np.ndarray:
        node_count = len(self.node_ids)
        affinity = np.eye(node_count, dtype=np.float32)
        undirected_graph = self.graph.to_undirected()
        lengths = dict(nx.all_pairs_shortest_path_length(undirected_graph, cutoff=6))
        for src_node, targets in lengths.items():
            src_idx = self.node_to_idx[src_node]
            for dst_node, distance in targets.items():
                dst_idx = self.node_to_idx[dst_node]
                affinity[src_idx, dst_idx] = 1.0 / (1.0 + float(distance))
        return affinity

    @staticmethod
    def _pair_key(lhs: str, rhs: str) -> tuple[str, str]:
        return (lhs, rhs) if lhs <= rhs else (rhs, lhs)

    def _graph_affinity(self, lhs: str, rhs: str) -> float:
        return float(self.graph_affinity_matrix[self.node_to_idx[lhs], self.node_to_idx[rhs]])

    def _pair_memory_value(self, lhs: str, rhs: str) -> float:
        return float(self.pair_success_memory.get(self._pair_key(lhs, rhs), 0.0))

    def _selector_quality_signal(self) -> float:
        if not self.eval_metrics:
            return 0.0
        scores = np.array([item["composite_score"] for item in self.eval_metrics], dtype=np.float32)
        current_score = float(scores[-1])
        mean_score = float(np.mean(scores))
        std_score = float(np.std(scores) + 1e-6)
        signal = math.tanh((mean_score - current_score) / std_score)
        if self.best_eval_score is not None and current_score <= self.best_eval_score * (
            1.0 + self.config.selector_good_set_tolerance
        ):
            signal += 0.20
        return float(np.clip(signal, -1.0, 1.0))

    def _record_phase_selector_history(self, phase_index: int, selected_nodes: list[str]) -> float:
        quality_signal = self._selector_quality_signal()

        for node_id in self.node_ids:
            self.node_success_memory[node_id] *= self.config.selector_history_decay
        for pair_key in list(self.pair_success_memory.keys()):
            self.pair_success_memory[pair_key] *= self.config.selector_pair_decay
            if abs(self.pair_success_memory[pair_key]) < 1.0e-4:
                del self.pair_success_memory[pair_key]

        for node_id in selected_nodes:
            self.node_success_memory[node_id] += quality_signal
        for lhs, rhs in combinations(sorted(selected_nodes), 2):
            key = self._pair_key(lhs, rhs)
            self.pair_success_memory[key] = (
                self.pair_success_memory.get(key, 0.0)
                + quality_signal * self.config.selector_pair_memory_scale
            )

        history_entry = {
            "phase": phase_index,
            "selected_nodes": list(selected_nodes),
            "quality_signal": float(quality_signal),
            "latest_eval_score": (
                float(self.eval_metrics[-1]["composite_score"]) if self.eval_metrics else None
            ),
        }
        self.selector_quality_history.append(history_entry)
        return quality_signal

    def _smooth_selector_scores(self, raw_scores: dict[str, float]) -> dict[str, float]:
        smoothed_scores: dict[str, float] = {}
        alpha = self.config.selector_score_ema_alpha
        for node_id, current_value in raw_scores.items():
            previous_value = self.selector_score_ema.get(node_id, current_value)
            smoothed_scores[node_id] = float(alpha * current_value + (1.0 - alpha) * previous_value)
        self.selector_score_ema = dict(smoothed_scores)
        return smoothed_scores

    def _node_fit_score(
        self,
        node_id: str,
        reference_set: list[str],
        base_scores: dict[str, float],
        embedding: np.ndarray,
    ) -> float:
        others = [other for other in reference_set if other != node_id]
        score = float(base_scores[node_id]) + (
            self.config.selector_history_bonus_weight * self.node_success_memory.get(node_id, 0.0)
        )
        if not others:
            return score
        pair_bonus = float(np.mean([self._pair_memory_value(node_id, other) for other in others]))
        cohesion = float(np.mean([self._graph_affinity(node_id, other) for other in others]))
        corridor_bonus = float(self.static_scores[node_id]) * cohesion
        redundancy = float(
            np.mean(
                [
                    max(
                        0.0,
                        self._cosine_similarity(
                            embedding[self.node_to_idx[node_id]],
                            embedding[self.node_to_idx[other]],
                        )
                        - 0.90,
                    )
                    for other in others
                ]
            )
        )
        return (
            score
            + self.config.selector_pair_bonus_weight * pair_bonus
            + self.config.selector_cohesion_weight * cohesion
            + self.config.selector_corridor_bonus_weight * corridor_bonus
            - self.config.selector_redundancy_penalty * redundancy
        )

    def _set_score(
        self,
        selected_nodes: list[str],
        base_scores: dict[str, float],
        embedding: np.ndarray,
    ) -> float:
        if not selected_nodes:
            return float("-inf")
        ordered_nodes = sorted(selected_nodes)
        member_value = float(np.mean([base_scores[node_id] for node_id in ordered_nodes]))
        history_bonus = float(np.mean([self.node_success_memory.get(node_id, 0.0) for node_id in ordered_nodes]))

        pair_values = list(combinations(ordered_nodes, 2))
        if pair_values:
            pair_bonus = float(np.mean([self._pair_memory_value(lhs, rhs) for lhs, rhs in pair_values]))
            cohesion = float(np.mean([self._graph_affinity(lhs, rhs) for lhs, rhs in pair_values]))
            redundancy = float(
                np.mean(
                    [
                        max(
                            0.0,
                            self._cosine_similarity(
                                embedding[self.node_to_idx[lhs]],
                                embedding[self.node_to_idx[rhs]],
                            )
                            - 0.90,
                        )
                        for lhs, rhs in pair_values
                    ]
                )
            )
        else:
            pair_bonus = 0.0
            cohesion = 0.0
            redundancy = 0.0

        corridor_bonus = float(np.mean([self.static_scores[node_id] for node_id in ordered_nodes])) * cohesion
        return (
            member_value
            + self.config.selector_history_bonus_weight * history_bonus
            + self.config.selector_pair_bonus_weight * pair_bonus
            + self.config.selector_cohesion_weight * cohesion
            + self.config.selector_corridor_bonus_weight * corridor_bonus
            - self.config.selector_redundancy_penalty * redundancy
        )

    def _serialize_pair_memory(self, limit: int = 40) -> list[dict[str, Any]]:
        ranked_pairs = sorted(
            self.pair_success_memory.items(),
            key=lambda item: (-abs(item[1]), item[0][0], item[0][1]),
        )[:limit]
        return [
            {"lhs": lhs, "rhs": rhs, "value": float(value)}
            for (lhs, rhs), value in ranked_pairs
        ]

    def load_checkpoint(self, checkpoint_path: Path) -> None:
        super().load_checkpoint(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        saved_score_ema = checkpoint.get("selector_score_ema", {})
        self.selector_score_ema = {
            node_id: float(saved_score_ema.get(node_id, 0.0))
            for node_id in self.node_ids
            if node_id in saved_score_ema
        }

        saved_node_memory = checkpoint.get("node_success_memory", {})
        self.node_success_memory = {
            node_id: float(saved_node_memory.get(node_id, 0.0))
            for node_id in self.node_ids
        }

        pair_payload = checkpoint.get("pair_success_memory", {})
        restored_pairs: dict[tuple[str, str], float] = {}
        for serialized_key, value in pair_payload.items():
            if "|" not in serialized_key:
                continue
            lhs, rhs = serialized_key.split("|", 1)
            if lhs in self.node_to_idx and rhs in self.node_to_idx:
                restored_pairs[self._pair_key(lhs, rhs)] = float(value)
        self.pair_success_memory = restored_pairs
        self.selector_quality_history = list(checkpoint.get("selector_quality_history", []))

    def _reselect_nodes(
        self,
        phase_logs: list[dict[str, list[np.ndarray]]],
        attention: np.ndarray | None,
        embedding: np.ndarray | None,
        phase_index: int,
    ) -> tuple[list[str], dict[str, Any]]:
        raw_base_scores, component_scores = self._selector_scores(phase_logs, attention)
        base_scores = self._smooth_selector_scores(raw_base_scores)
        stability = self._selection_stability_context()
        if embedding is None:
            embedding = np.stack([self._static_feature_vector(node_id) for node_id in self.node_ids], axis=0)

        current_selected = list(self.selected_nodes)
        if len(current_selected) != self.config.budget:
            ranked = sorted(self.node_ids, key=lambda node_id: (-base_scores[node_id], node_id))
            current_selected = ranked[: self.config.budget]

        anchor_count = min(
            self.config.budget,
            max(1, int(math.ceil(self.config.budget * self.config.selector_anchor_ratio))),
        )
        protected_anchors = sorted(
            current_selected,
            key=lambda node_id: (
                -self._node_fit_score(node_id, current_selected, base_scores, embedding),
                node_id,
            ),
        )[:anchor_count]
        protected_anchor_set = set(protected_anchors)

        selected = list(current_selected)
        current_set_score = self._set_score(selected, base_scores, embedding)
        swap_budget = min(
            self.config.budget,
            (
                self.config.selector_good_set_max_swaps
                if stability["good_current_subset"]
                else self.config.selector_max_swaps_per_phase
            ),
        )
        improvement_margin = max(
            self.config.selector_set_improvement_margin,
            self.config.selector_replacement_margin
            * (1.5 if stability["good_current_subset"] else 1.0),
        )
        replacement_events: list[dict[str, Any]] = []

        for _ in range(swap_budget):
            replaceable_nodes = [node_id for node_id in selected if node_id not in protected_anchor_set]
            remaining_nodes = [node_id for node_id in self.node_ids if node_id not in selected]
            if not replaceable_nodes or not remaining_nodes:
                break

            candidate_pool = sorted(
                remaining_nodes,
                key=lambda node_id: (
                    -self._node_fit_score(node_id, selected, base_scores, embedding),
                    node_id,
                ),
            )[: self.config.selector_candidate_pool_size]

            best_swap: dict[str, Any] | None = None
            for candidate_node in candidate_pool:
                for incumbent_node in replaceable_nodes:
                    proposed = [node_id for node_id in selected if node_id != incumbent_node]
                    proposed.append(candidate_node)
                    proposed_score = self._set_score(proposed, base_scores, embedding)
                    score_gain = proposed_score - current_set_score
                    if best_swap is None or score_gain > best_swap["score_gain"]:
                        best_swap = {
                            "added": candidate_node,
                            "removed": incumbent_node,
                            "set_score_before": float(current_set_score),
                            "set_score_after": float(proposed_score),
                            "score_gain": float(score_gain),
                            "candidate_fit": float(
                                self._node_fit_score(candidate_node, selected, base_scores, embedding)
                            ),
                            "removed_fit": float(
                                self._node_fit_score(incumbent_node, selected, base_scores, embedding)
                            ),
                        }

            if best_swap is None or best_swap["score_gain"] <= improvement_margin:
                break

            selected = [node_id for node_id in selected if node_id != best_swap["removed"]]
            selected.append(best_swap["added"])
            current_set_score = float(best_swap["set_score_after"])
            replacement_events.append(best_swap)

        selected = sorted(
            selected,
            key=lambda node_id: (
                -self._node_fit_score(node_id, selected, base_scores, embedding),
                node_id,
            ),
        )[: self.config.budget]
        phase_change = self._phase_change_details(current_selected, selected)
        details = {
            "phase": phase_index,
            "selection_version": "runnerMainV2",
            "raw_base_scores": {node_id: float(raw_base_scores[node_id]) for node_id in self.node_ids},
            "smoothed_base_scores": {node_id: float(base_scores[node_id]) for node_id in self.node_ids},
            "component_scores": component_scores,
            "selected_nodes": selected,
            "current_selected": current_selected,
            "anchor_nodes": protected_anchors,
            "replacement_events": replacement_events,
            "swap_budget": int(swap_budget),
            "set_improvement_margin": float(improvement_margin),
            "stability": stability,
            "node_success_memory": {
                node_id: float(self.node_success_memory.get(node_id, 0.0))
                for node_id in self.node_ids
            },
            "top_pair_memory": self._serialize_pair_memory(),
            "final_set_score": float(self._set_score(selected, base_scores, embedding)),
            **phase_change,
        }
        return selected, details

    def _save_phase_selector_outputs(
        self,
        phase_index: int,
        gat_stats: dict[str, Any] | None,
        selection_details: dict[str, Any] | None,
        selected_before: list[str],
        selected_after: list[str] | None,
    ) -> None:
        payload: dict[str, Any] = {
            "phase": phase_index,
            "runner": "runnerMainV2",
            "selected_before": selected_before,
            "selected_after": selected_after if selected_after is not None else selected_before,
            "selector_quality_history": self.selector_quality_history[-1] if self.selector_quality_history else None,
            "gat_sample_count": 0,
        }
        payload.update(self._phase_change_details(selected_before, payload["selected_after"]))
        if gat_stats is not None:
            payload["gat_sample_count"] = int(gat_stats["sample_count"])
            payload["pressure_mean"] = float(gat_stats["pressure_mean"])
            payload["pressure_std"] = float(gat_stats["pressure_std"])
            payload["occupancy_mean"] = float(gat_stats.get("occupancy_mean", 0.0))
            payload["occupancy_std"] = float(gat_stats.get("occupancy_std", 0.0))
            payload["phase_competition_mean"] = float(gat_stats.get("phase_competition_mean", 0.0))
            payload["phase_competition_std"] = float(gat_stats.get("phase_competition_std", 0.0))
            payload["mean_pressure_prediction"] = gat_stats["mean_pressure_prediction"].tolist()
            payload["mean_occupancy_prediction"] = gat_stats["mean_occupancy_prediction"].tolist()
            payload["mean_phase_competition_prediction"] = gat_stats["mean_phase_competition_prediction"].tolist()
            payload["mean_attention"] = gat_stats["mean_attention"].tolist()
            payload["mean_embedding"] = gat_stats["mean_embedding"].tolist()
            payload["training_loss"] = gat_stats.get("training_loss")
        if selection_details is not None:
            payload["selection"] = selection_details
        out_path = self.selectors_dir / f"selector_phase_{phase_index:02d}.json"
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _write_metrics(self) -> None:
        payload = {
            "config": {
                key: (str(value) if isinstance(value, Path) else value)
                for key, value in asdict(self.config).items()
            },
            "current_selected_nodes": self.selected_nodes,
            "current_node_controller_assignment": self.node_controller_assignment,
            "episode_metrics": self.episode_metrics,
            "evaluation_metrics": self.eval_metrics,
            "phase_history": self.phase_history,
            "phase_change_report": self._phase_change_report(),
            "best_eval_travel_time": self.best_eval_travel_time,
            "best_eval_score": self.best_eval_score,
            "selector_quality_history": self.selector_quality_history,
            "selector_score_ema": self.selector_score_ema,
            "node_success_memory": self.node_success_memory,
            "top_pair_success_memory": self._serialize_pair_memory(),
        }
        metrics_path = self.metrics_dir / "training_metrics.json"
        metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _save_checkpoint(self, episode: int, phase_index: int, is_final: bool = False, is_best: bool = False) -> Path:
        if is_best:
            checkpoint_path = self.best_checkpoint_dir / "best_checkpoint.pt"
        elif is_final:
            checkpoint_path = self.final_checkpoint_dir / "final_checkpoint.pt"
        else:
            checkpoint_path = self.phase_checkpoint_dir / f"phase_{phase_index:02d}_ep{episode:03d}.pt"

        payload = {
            "episode": episode,
            "phase": phase_index,
            "is_final": is_final,
            "is_best": is_best,
            "config": {
                key: (str(value) if isinstance(value, Path) else value)
                for key, value in asdict(self.config).items()
            },
            "selected_nodes": self.selected_nodes,
            "node_controller_assignment": self.node_controller_assignment,
            "node_ids": self.node_ids,
            "static_scores": self.static_scores,
            "controllers": {
                controller_id: {
                    "online_state_dict": controller.online_net.state_dict(),
                    "target_state_dict": controller.target_net.state_dict(),
                    "optimizer_state_dict": controller.optimizer.state_dict(),
                    "latest_loss": controller.latest_loss,
                }
                for controller_id, controller in self.controllers.items()
            },
            "gat_state_dict": self.gat_model.state_dict(),
            "gat_optimizer_state_dict": self.gat_optimizer.state_dict(),
            "frozen_gat_stats": {
                key: (value.tolist() if isinstance(value, np.ndarray) else value)
                for key, value in (self.frozen_gat_stats or {}).items()
            },
            "episode_metrics": self.episode_metrics,
            "evaluation_metrics": self.eval_metrics,
            "best_eval_travel_time": self.best_eval_travel_time,
            "best_eval_score": self.best_eval_score,
            "phase_history": self.phase_history,
            "selector_quality_history": self.selector_quality_history,
            "selector_score_ema": self.selector_score_ema,
            "node_success_memory": self.node_success_memory,
            "pair_success_memory": {
                f"{lhs}|{rhs}": float(value)
                for (lhs, rhs), value in self.pair_success_memory.items()
            },
        }
        torch.save(payload, checkpoint_path)
        return checkpoint_path

    def _print_episode_progress(self, metrics: dict[str, Any], mean_loss: float | None) -> None:
        avg_text = f"{metrics['average_travel_time']:.2f}" if not math.isnan(metrics["average_travel_time"]) else "nan"
        loss_text = f"{mean_loss:.4f}" if mean_loss is not None else "n/a"
        print(
            f"runnerMainV2 phase {metrics['phase']} | ep {metrics['episode']:>3}/{self.config.total_episodes} | "
            f"avg_tt={avg_text}s | inserted={metrics['inserted_ratio']:.3f} | "
            f"arrived={metrics['arrived_vehicle_count']} | epsilon={metrics['epsilon']:.3f} | loss={loss_text}"
        )

    def _print_eval_progress(self, metrics: dict[str, Any]) -> None:
        avg_text = f"{metrics['average_travel_time']:.2f}" if not math.isnan(metrics["average_travel_time"]) else "nan"
        print(
            f"runnerMainV2 eval  phase {metrics['phase']} | ep {metrics['episode']:>3}/{self.config.total_episodes} | "
            f"avg_tt={avg_text}s | inserted={metrics['inserted_ratio']:.3f} | "
            f"arrived={metrics['arrived_vehicle_count']} | score={metrics['composite_score']:.2f}"
        )

    def run(self) -> dict[str, Any]:
        total_phases = math.ceil(self.config.total_episodes / self.config.phase_length)
        phase_logs: list[dict[str, list[np.ndarray]]] = []

        start_episode = self.resume_episode + 1
        for episode_index in range(start_episode, self.config.total_episodes + 1):
            phase_index = math.ceil(episode_index / self.config.phase_length)
            metrics, episode_phase_log = self._rollout_episode(
                episode_index,
                phase_index,
                sumo_seed=self._training_sumo_seed(episode_index),
                epsilon=self._epsilon(episode_index - 1),
                collect_experience=True,
            )
            update_losses = []
            for _ in range(self.config.updates_per_episode):
                for controller in self.controllers.values():
                    loss = self._update_policy(controller)
                    if loss is not None:
                        update_losses.append(loss)
            if episode_index % self.config.target_sync_interval == 0:
                for controller in self.controllers.values():
                    controller.target_net.load_state_dict(controller.online_net.state_dict())
            metrics["mean_loss"] = float(np.mean(update_losses)) if update_losses else None
            self.episode_metrics.append(metrics)
            phase_logs.append(episode_phase_log)
            self._print_episode_progress(metrics, metrics["mean_loss"])

            if episode_index % self.config.eval_interval == 0:
                eval_rollouts = []
                for eval_index in range(self.config.eval_episodes):
                    eval_rollout, _ = self._rollout_episode(
                        episode_index,
                        phase_index,
                        sumo_seed=self._evaluation_sumo_seed(episode_index, eval_index),
                        epsilon=0.0,
                        collect_experience=False,
                    )
                    eval_rollouts.append(eval_rollout)
                eval_metrics = {
                    "episode": episode_index,
                    "phase": phase_index,
                    "average_travel_time": float(np.mean([item["average_travel_time"] for item in eval_rollouts])),
                    "max_travel_time": float(np.mean([item["max_travel_time"] for item in eval_rollouts])),
                    "arrived_vehicle_count": int(np.mean([item["arrived_vehicle_count"] for item in eval_rollouts])),
                    "demand_vehicle_count": self.total_demand,
                    "loaded_vehicle_count": int(np.mean([item["loaded_vehicle_count"] for item in eval_rollouts])),
                    "inserted_vehicle_count": int(np.mean([item["inserted_vehicle_count"] for item in eval_rollouts])),
                    "pending_vehicle_count": int(np.mean([item["pending_vehicle_count"] for item in eval_rollouts])),
                    "inserted_ratio": float(np.mean([item["inserted_ratio"] for item in eval_rollouts])),
                    "arrived_ratio": float(np.mean([item["arrived_ratio"] for item in eval_rollouts])),
                    "epsilon": 0.0,
                }
                eval_metrics["composite_score"] = self._composite_eval_score(eval_metrics)
                self.eval_metrics.append(eval_metrics)
                self._print_eval_progress(eval_metrics)
                if self.best_eval_score is None or eval_metrics["composite_score"] < self.best_eval_score:
                    self.best_eval_score = eval_metrics["composite_score"]
                    self.best_eval_travel_time = eval_metrics["average_travel_time"]
                    best_checkpoint = self._save_checkpoint(episode_index, phase_index, is_best=True)
                    print(f"Saved runnerMainV2 best checkpoint: {best_checkpoint}")

            self.plot.update(self.episode_metrics, self.eval_metrics, self.config.phase_length)
            self._write_metrics()

            phase_complete = (
                episode_index % self.config.phase_length == 0
                or episode_index == self.config.total_episodes
            )
            if not phase_complete:
                continue

            selected_before = list(self.selected_nodes)
            print(
                f"runnerMainV2 phase {phase_index}: phase complete at episode {episode_index}, starting selector pipeline"
            )
            gat_stats = self._train_gat_offline(phase_logs, phase_index)
            if gat_stats is not None:
                self.frozen_gat_stats = gat_stats

            quality_signal = self._record_phase_selector_history(phase_index, selected_before)
            selection_details = None
            selected_after = None
            if phase_index < total_phases:
                print(
                    f"runnerMainV2 phase {phase_index}: computing reselection under budget {self.config.budget}"
                )
                selected_after, selection_details = self._reselect_nodes(
                    phase_logs,
                    gat_stats["mean_attention"] if gat_stats is not None else None,
                    gat_stats["mean_embedding"] if gat_stats is not None else None,
                    phase_index,
                )
                if selection_details is not None:
                    selection_details["quality_signal"] = float(quality_signal)
                self.selected_nodes = selected_after
                self.node_controller_assignment = self._remap_controllers(selected_after)
                print(f"runnerMainV2 phase {phase_index}: next selected nodes = {self.selected_nodes}")
                if selection_details is not None:
                    print(
                        f"runnerMainV2 phase {phase_index}: changed {selection_details['changed_node_count']} nodes "
                        f"(added={selection_details['added_nodes']}, removed={selection_details['removed_nodes']})"
                    )

            phase_change = self._phase_change_details(
                selected_before,
                selected_after if selected_after is not None else selected_before,
            )
            phase_summary = {
                "phase": phase_index,
                "episode_end": episode_index,
                "selected_before": selected_before,
                "selected_after": selected_after if selected_after is not None else selected_before,
                "gat_trained": gat_stats is not None,
                "gat_sample_count": int(gat_stats["sample_count"]) if gat_stats is not None else 0,
                "node_controller_assignment": dict(self.node_controller_assignment),
                "selector_quality_signal": float(quality_signal),
                **phase_change,
            }
            self.phase_history.append(phase_summary)
            self._save_phase_selector_outputs(
                phase_index=phase_index,
                gat_stats=gat_stats,
                selection_details=selection_details,
                selected_before=selected_before,
                selected_after=selected_after,
            )
            checkpoint_path = self._save_checkpoint(
                episode_index,
                phase_index,
                is_final=episode_index == self.config.total_episodes,
            )
            print(f"Saved runnerMainV2 checkpoint: {checkpoint_path}")
            self._write_metrics()
            phase_logs = []

        if self.phase_history and self.phase_history[-1]["episode_end"] != self.config.total_episodes:
            final_phase = math.ceil(self.config.total_episodes / self.config.phase_length)
            checkpoint_path = self._save_checkpoint(self.config.total_episodes, final_phase, is_final=True)
            print(f"Saved runnerMainV2 final checkpoint: {checkpoint_path}")
        return {
            "episode_metrics": self.episode_metrics,
            "evaluation_metrics": self.eval_metrics,
            "phase_history": self.phase_history,
            "selected_nodes": self.selected_nodes,
            "node_controller_assignment": self.node_controller_assignment,
        }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="runnerMainV2: phase-wise shared DQN + GAT selector with set-aware reselection."
    )
    parser.add_argument("--scenario-dir", type=str, default="sumo_benchmark", help="Scenario directory containing benchmark.sumocfg.")
    parser.add_argument("--episodes", type=int, default=250, help="Total training episodes.")
    parser.add_argument("--phase-length", type=int, default=50, help="Episodes per phase before GAT reselection.")
    parser.add_argument("--budget", type=int, default=10, help="Number of intersections controlled each phase.")
    parser.add_argument("--max-steps", type=int, default=3600, help="Maximum SUMO steps per episode.")
    parser.add_argument("--decision-interval", type=int, default=10, help="Seconds between control decisions.")
    parser.add_argument("--updates-per-episode", type=int, default=8, help="Shared-policy gradient steps per episode.")
    parser.add_argument("--gat-epochs", type=int, default=18, help="Offline GAT epochs after each phase.")
    parser.add_argument("--eval-interval", type=int, default=10, help="Run evaluation every N training episodes.")
    parser.add_argument("--eval-episodes", type=int, default=3, help="Number of epsilon=0 evaluation rollouts.")
    parser.add_argument("--policy-lr", type=float, default=3.0e-4, help="Shared-policy learning rate.")
    parser.add_argument("--fixed-green-duration", type=int, default=30, help="Fixed-time green duration for uncontrolled intersections.")
    parser.add_argument("--yellow-duration", type=int, default=3, help="Yellow duration in seconds.")
    parser.add_argument("--selector-anchor-ratio", type=float, default=0.6, help="Anchor fraction retained during reselection.")
    parser.add_argument("--selector-score-ema-alpha", type=float, default=0.65, help="EMA weight for current phase selector scores.")
    parser.add_argument("--selector-history-decay", type=float, default=0.85, help="Decay applied to per-node success memory after each phase.")
    parser.add_argument("--selector-pair-decay", type=float, default=0.88, help="Decay applied to pair synergy memory after each phase.")
    parser.add_argument("--selector-history-bonus-weight", type=float, default=0.12, help="Weight of proven-node history inside set scoring.")
    parser.add_argument("--selector-pair-bonus-weight", type=float, default=0.10, help="Weight of pair synergy memory inside set scoring.")
    parser.add_argument("--selector-cohesion-weight", type=float, default=0.14, help="Weight of graph cohesion inside set scoring.")
    parser.add_argument("--selector-corridor-bonus-weight", type=float, default=0.10, help="Weight of corridor-core preservation inside set scoring.")
    parser.add_argument("--selector-set-improvement-margin", type=float, default=0.025, help="Minimum whole-set score gain required to accept a swap.")
    parser.add_argument("--selector-candidate-pool-size", type=int, default=14, help="How many outsider candidates to consider per swap round.")
    parser.add_argument("--selector-pair-memory-scale", type=float, default=0.50, help="Per-phase scale applied to pair-memory updates.")
    parser.add_argument("--train-seed-stride", type=int, default=37, help="Seed delta applied between training episodes.")
    parser.add_argument("--eval-seed-offset", type=int, default=1000, help="Offset added to evaluation seeds.")
    parser.add_argument("--eval-seed-stride", type=int, default=53, help="Seed delta applied between evaluation checkpoints.")
    parser.add_argument("--resume-from", type=str, default=None, help="Checkpoint path to resume from.")
    parser.add_argument("--device", type=str, default="auto", help="Torch device: auto, cpu, or cuda.")
    parser.add_argument("--seed", type=int, default=7, help="Global random seed.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    base.SCENARIO_DIR = base.resolve_scenario_dir(args.scenario_dir)
    base.SUMOCFG_PATH = base.SCENARIO_DIR / "benchmark.sumocfg"
    if not base.SUMOCFG_PATH.exists():
        raise FileNotFoundError(f"SUMO config not found: {base.SUMOCFG_PATH}")

    config = MainConfigV2(
        total_episodes=args.episodes,
        phase_length=args.phase_length,
        budget=args.budget,
        max_steps=args.max_steps,
        decision_interval=args.decision_interval,
        updates_per_episode=args.updates_per_episode,
        gat_epochs=args.gat_epochs,
        eval_interval=args.eval_interval,
        eval_episodes=args.eval_episodes,
        policy_lr=args.policy_lr,
        fixed_green_duration=args.fixed_green_duration,
        yellow_duration=args.yellow_duration,
        selector_anchor_ratio=args.selector_anchor_ratio,
        selector_score_ema_alpha=args.selector_score_ema_alpha,
        selector_history_decay=args.selector_history_decay,
        selector_pair_decay=args.selector_pair_decay,
        selector_history_bonus_weight=args.selector_history_bonus_weight,
        selector_pair_bonus_weight=args.selector_pair_bonus_weight,
        selector_cohesion_weight=args.selector_cohesion_weight,
        selector_corridor_bonus_weight=args.selector_corridor_bonus_weight,
        selector_set_improvement_margin=args.selector_set_improvement_margin,
        selector_candidate_pool_size=args.selector_candidate_pool_size,
        selector_pair_memory_scale=args.selector_pair_memory_scale,
        train_seed_stride=args.train_seed_stride,
        eval_seed_offset=args.eval_seed_offset,
        eval_seed_stride=args.eval_seed_stride,
        device=args.device,
        seed=args.seed,
    )
    trainer = MethodologyRunnerV2(config)
    print(f"Using torch device: {trainer.device}")
    if args.resume_from:
        resume_path = Path(args.resume_from)
        if not resume_path.is_absolute():
            resume_path = (base.ROOT / resume_path).resolve()
        if not resume_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {resume_path}")
        print(f"Resuming runnerMainV2 from checkpoint: {resume_path}")
        trainer.load_checkpoint(resume_path)
    trainer.run()


if __name__ == "__main__":
    main()
