from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import xml.etree.ElementTree as ET
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parent
SCENARIO_DIR = ROOT / "sumo_benchmark"
SUMOCFG_PATH = SCENARIO_DIR / "benchmark.sumocfg"
OUTPUT_DIR = ROOT / "runnerMain_outputs_no_attention_reward"
CHECKPOINT_DIR = ROOT / "runnerMain_checkpoints_no_attention_reward"
DLL_DIR_HANDLES: list[Any] = []


def resolve_scenario_dir(raw_path: str) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def resolve_path_from_root(raw_path: str) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def load_ordered_nodes_from_selection_json(selection_path: Path) -> list[str]:
    try:
        payload = json.loads(selection_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Selection file is not valid JSON: {selection_path}") from exc

    ordered_nodes: list[str] = []
    if isinstance(payload, dict):
        for key in ("drqn_nodes", "selected_nodes"):
            value = payload.get(key)
            if isinstance(value, list):
                ordered_nodes = [str(item) for item in value]
                break
        if not ordered_nodes:
            ranking = payload.get("ranking")
            if isinstance(ranking, list):
                for item in ranking:
                    if isinstance(item, dict) and "node_id" in item:
                        ordered_nodes.append(str(item["node_id"]))
                    elif isinstance(item, str):
                        ordered_nodes.append(item)
    elif isinstance(payload, list):
        ordered_nodes = [str(item) for item in payload]

    unique_nodes: list[str] = []
    seen: set[str] = set()
    for node_id in ordered_nodes:
        if node_id not in seen:
            unique_nodes.append(node_id)
            seen.add(node_id)
    if not unique_nodes:
        raise RuntimeError(
            "Selection file must contain a non-empty ordered node list in 'drqn_nodes', "
            "'selected_nodes', 'ranking', or as a top-level JSON array."
        )
    return unique_nodes


def ensure_sumo_python_tools() -> None:
    sumo_home = os.environ.get("SUMO_HOME")
    if not sumo_home:
        raise RuntimeError("SUMO_HOME is not set.")
    sumo_root = Path(sumo_home)
    tools_dir = sumo_root / "tools"
    bin_dir = sumo_root / "bin"
    if str(tools_dir) not in sys.path:
        sys.path.append(str(tools_dir))
    path_entries = os.environ.get("PATH", "")
    if str(bin_dir) not in path_entries:
        os.environ["PATH"] = f"{bin_dir};{path_entries}"
    if hasattr(os, "add_dll_directory") and bin_dir.exists():
        DLL_DIR_HANDLES.append(os.add_dll_directory(str(bin_dir.resolve())))


ensure_sumo_python_tools()

SUMO_BACKEND = "libsumo"
try:
    import libsumo  # type: ignore
except Exception:
    try:
        import traci as libsumo  # type: ignore

        SUMO_BACKEND = "traci"
    except Exception as exc:
        raise RuntimeError(
            "Neither libsumo nor traci is usable in the current environment."
        ) from exc

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from sumo_fixed_timing import write_fixed_timing_additional


@dataclass
class MainConfig:
    total_episodes: int = 250
    phase_length: int = 50
    budget: int = 10
    max_steps: int = 3600
    decision_interval: int = 10
    min_green_duration: int = 12
    fixed_green_duration: int = 30
    yellow_duration: int = 3
    history_length: int = 4
    prediction_horizon: int = 3
    shared_hidden_size: int = 160
    controller_node_embed_dim: int = 16
    gat_hidden_size: int = 96
    gat_epochs: int = 18
    gat_heads: int = 4
    policy_lr: float = 3.0e-4
    gat_lr: float = 1.0e-3
    gamma: float = 0.99
    batch_size: int = 16
    replay_capacity: int = 3500
    sequence_length: int = 4
    updates_per_episode: int = 8
    target_sync_interval: int = 10
    eval_interval: int = 10
    eval_episodes: int = 3
    epsilon_start: float = 1.0
    epsilon_end: float = 0.08
    epsilon_decay_episodes: int = 160
    shaping_beta: float = 0.25
    reward_neighbor_weight: float = 0.45
    reward_local_improvement_weight: float = 1.20
    reward_switch_penalty: float = 0.05
    reward_abs_pressure_weight: float = 0.10
    reward_improvement_scale: float = 12.0
    selector_static_weight: float = 0.30
    selector_phase_competition_weight: float = 0.25
    selector_spillback_weight: float = 0.15
    selector_neighbor_weight: float = 0.15
    selector_control_sensitivity_weight: float = 0.15
    selector_redundancy_penalty: float = 0.08
    selector_anchor_ratio: float = 0.6
    selector_exploration_start: float = 0.08
    selector_exploration_end: float = 0.005
    selector_max_swaps_per_phase: int = 2
    selector_good_set_max_swaps: int = 1
    selector_incumbent_bonus: float = 0.05
    selector_replacement_margin: float = 0.04
    selector_good_set_tolerance: float = 0.08
    gat_aux_task_weight: float = 0.35
    seed: int = 7
    sumo_seed: int = 11
    train_seed_stride: int = 37
    eval_seed_offset: int = 1000
    eval_seed_stride: int = 53
    eval_pending_weight: float = 0.10
    eval_unarrived_weight: float = 0.25
    device: str = "auto"
    sumo_binary: str = "sumo"
    output_dir: Path = OUTPUT_DIR
    checkpoint_dir: Path = CHECKPOINT_DIR


@dataclass
class PhaseGroup:
    incoming_lanes: list[str]
    outgoing_lanes: list[str]


@dataclass
class NodeTransition:
    state: np.ndarray
    node_index: int
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class SharedQNetwork(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_size: int,
        num_nodes: int,
        node_context_matrix: np.ndarray,
        node_embedding_dim: int = 16,
        action_dim: int = 2,
    ) -> None:
        super().__init__()
        node_context_dim = int(node_context_matrix.shape[1]) if node_context_matrix.ndim == 2 else 0
        context_hidden_size = max(32, hidden_size // 2)
        head_hidden_size = max(32, hidden_size // 2)

        self.node_embedding = nn.Embedding(num_nodes, node_embedding_dim)
        self.register_buffer(
            "node_context_matrix",
            torch.tensor(node_context_matrix, dtype=torch.float32),
        )
        self.state_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.context_encoder = nn.Sequential(
            nn.Linear(node_embedding_dim + node_context_dim, context_hidden_size),
            nn.ReLU(),
            nn.LayerNorm(context_hidden_size),
            nn.Linear(context_hidden_size, context_hidden_size),
            nn.ReLU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size + context_hidden_size, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, head_hidden_size),
            nn.ReLU(),
            nn.Linear(head_hidden_size, 1),
        )
        self.advantage_head = nn.Sequential(
            nn.Linear(hidden_size, head_hidden_size),
            nn.ReLU(),
            nn.Linear(head_hidden_size, action_dim),
        )

    def forward(self, x: torch.Tensor, node_indices: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x[:, -1, :]
        if node_indices.dim() > 1:
            node_indices = node_indices.view(-1)
        node_embedding = self.node_embedding(node_indices)
        node_context = self.node_context_matrix[node_indices]
        encoded_state = self.state_encoder(x)
        encoded_context = self.context_encoder(torch.cat([node_embedding, node_context], dim=-1))
        fused = self.fusion(torch.cat([encoded_state, encoded_context], dim=-1))
        value = self.value_head(fused)
        advantage = self.advantage_head(fused)
        return value + advantage - advantage.mean(dim=1, keepdim=True)


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.episodes: deque[list[NodeTransition]] = deque(maxlen=capacity)

    def add_episode(self, transitions: list[NodeTransition]) -> None:
        if transitions:
            self.episodes.append(transitions)

    def can_sample(self, batch_size: int, sequence_length: int) -> bool:
        transition_count = sum(len(episode) for episode in self.episodes)
        return transition_count >= batch_size

    def sample(self, batch_size: int, sequence_length: int, device: torch.device) -> dict[str, torch.Tensor]:
        transitions = [transition for episode in self.episodes for transition in episode]
        states = []
        next_states = []
        node_indices = []
        actions = []
        rewards = []
        dones = []
        sampled = random.sample(transitions, batch_size)
        for item in sampled:
            states.append(item.state)
            next_states.append(item.next_state)
            node_indices.append(int(item.node_index))
            actions.append(int(item.action))
            rewards.append(float(item.reward))
            dones.append(float(item.done))
        return {
            "states": torch.tensor(np.stack(states), dtype=torch.float32, device=device),
            "next_states": torch.tensor(np.stack(next_states), dtype=torch.float32, device=device),
            "node_indices": torch.tensor(node_indices, dtype=torch.long, device=device),
            "actions": torch.tensor(actions, dtype=torch.long, device=device),
            "rewards": torch.tensor(rewards, dtype=torch.float32, device=device),
            "dones": torch.tensor(dones, dtype=torch.float32, device=device),
        }


class ControllerRuntime:
    def __init__(
        self,
        input_dim: int,
        hidden_size: int,
        config: MainConfig,
        num_nodes: int,
        node_context_matrix: np.ndarray,
    ) -> None:
        self.online_net = SharedQNetwork(
            input_dim,
            hidden_size,
            num_nodes,
            node_context_matrix,
            node_embedding_dim=config.controller_node_embed_dim,
        ).to(config.device)
        self.target_net = SharedQNetwork(
            input_dim,
            hidden_size,
            num_nodes,
            node_context_matrix,
            node_embedding_dim=config.controller_node_embed_dim,
        ).to(config.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=config.policy_lr)
        self.replay = ReplayBuffer(config.replay_capacity)
        self.latest_loss: float | None = None


class GraphAttentionLayer(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.query_proj = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_heads)])
        self.key_proj = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_heads)])
        self.value_proj = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_heads)])
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, h: torch.Tensor, adjacency: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        head_embeddings = []
        head_attentions = []
        mask = adjacency.unsqueeze(0)
        neg_inf = torch.full_like(mask, -1e9)
        for q_proj, k_proj, v_proj in zip(self.query_proj, self.key_proj, self.value_proj):
            q = q_proj(h)
            k = k_proj(h)
            v = v_proj(h)
            scores = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(q.shape[-1])
            scores = torch.where(mask > 0, scores, neg_inf)
            alpha = torch.softmax(scores, dim=-1)
            z = torch.matmul(alpha, v)
            head_embeddings.append(z)
            head_attentions.append(alpha)
        attention = torch.stack(head_attentions, dim=0).mean(dim=0)
        merged = torch.stack(head_embeddings, dim=0).mean(dim=0)
        updated = self.norm(h + self.activation(self.out_proj(merged)))
        return updated, attention


class GraphAttentionForecaster(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, horizon: int, num_heads: int) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.input_activation = nn.LeakyReLU(0.2)
        self.layers = nn.ModuleList(
            [
                GraphAttentionLayer(hidden_dim, num_heads),
                GraphAttentionLayer(hidden_dim, num_heads),
            ]
        )
        self.pressure_head = self._make_prediction_head(hidden_dim, horizon, positive_only=False)
        self.occupancy_head = self._make_prediction_head(hidden_dim, horizon, positive_only=True)
        self.competition_head = self._make_prediction_head(hidden_dim, horizon, positive_only=True)

    @staticmethod
    def _make_prediction_head(hidden_dim: int, horizon: int, positive_only: bool) -> nn.Sequential:
        layers: list[nn.Module] = [
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, horizon),
        ]
        if positive_only:
            layers.append(nn.Softplus())
        return nn.Sequential(*layers)

    def forward(
        self,
        x: torch.Tensor,
        adjacency: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.input_norm(self.input_activation(self.input_proj(x)))
        attention = None
        for layer in self.layers:
            h, attention = layer(h, adjacency)
        if attention is None:
            raise RuntimeError("GraphAttentionForecaster produced no attention weights.")
        embedding = h
        pressure_pred = self.pressure_head(embedding)
        occupancy_pred = self.occupancy_head(embedding)
        competition_pred = self.competition_head(embedding)
        return pressure_pred, occupancy_pred, competition_pred, attention, embedding


class EpisodeMetricPlot:
    def __init__(
        self,
        output_path: Path,
        metric_key: str,
        line_label: str,
        y_label: str,
        title: str,
        color: str,
    ) -> None:
        self.output_path = output_path
        self.metric_key = metric_key
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.fig, self.ax = plt.subplots(figsize=(10, 5.5))
        self.train_line, = self.ax.plot([], [], color=color, linewidth=2.0, label=line_label)
        self.phase_lines: list[Any] = []
        self.ax.set_title(title)
        self.ax.set_xlabel("Episode")
        self.ax.set_ylabel(y_label)
        self.ax.grid(True, linestyle="--", alpha=0.35)
        self.ax.legend()
        self.fig.tight_layout()

    def update(
        self,
        train_metrics: list[dict[str, Any]],
        eval_metrics: list[dict[str, Any]],
        phase_length: int,
    ) -> None:
        del eval_metrics
        xs = [int(item["episode"]) for item in train_metrics]
        ys = [float(item.get(self.metric_key, math.nan)) for item in train_metrics]
        self.train_line.set_data(xs, ys)
        self.ax.relim()
        self.ax.autoscale_view()
        for line in self.phase_lines:
            line.remove()
        self.phase_lines = []
        for boundary in range(phase_length, (xs[-1] if xs else 0) + phase_length, phase_length):
            self.phase_lines.append(self.ax.axvline(boundary, color="gray", alpha=0.25, linestyle=":"))
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        self.fig.savefig(self.output_path, dpi=150)


class MethodologyRunner:
    def __init__(self, config: MainConfig) -> None:
        self.config = config
        self.device = torch.device(resolve_torch_device(config.device))
        self.config.device = str(self.device)
        set_global_seed(config.seed)
        self.output_dir = config.output_dir
        self.checkpoint_dir = config.checkpoint_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.net_path = self._resolve_sumocfg_path("net-file")
        self.route_path = self._resolve_sumocfg_path("route-files")
        self.fixed_timing_additional_path = self.output_dir / "fixed30_uncontrolled.add.xml"
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
        self.controllers: dict[str, ControllerRuntime] = {
            self.shared_controller_id: ControllerRuntime(
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
        self.gat_model = GraphAttentionForecaster(
            input_dim=self.gat_input_dim,
            hidden_dim=self.config.gat_hidden_size,
            horizon=self.config.prediction_horizon,
            num_heads=self.config.gat_heads,
        ).to(self.device)
        self.gat_optimizer = optim.Adam(self.gat_model.parameters(), lr=config.gat_lr)
        self.frozen_gat_stats: dict[str, Any] | None = None

        self.episode_metrics: list[dict[str, Any]] = []
        self.eval_metrics: list[dict[str, Any]] = []
        self.phase_history: list[dict[str, Any]] = []
        self.best_eval_travel_time: float | None = None
        self.best_eval_score: float | None = None
        self.resume_episode: int = 0
        self.plots = [
            EpisodeMetricPlot(
                self.output_dir / "average_travel_time.png",
                metric_key="average_travel_time",
                line_label="Average Travel Time",
                y_label="Average Travel Time (s)",
                title="runnerMain Phase-wise Training",
                color="tab:green",
            ),
            EpisodeMetricPlot(
                self.output_dir / "average_waiting_time.png",
                metric_key="average_waiting_time",
                line_label="Average Waiting Time",
                y_label="Average Waiting Time (s)",
                title="runnerMain Average Waiting Time vs Episodes",
                color="tab:blue",
            ),
            EpisodeMetricPlot(
                self.output_dir / "max_waiting_time.png",
                metric_key="max_waiting_time",
                line_label="Maximum Waiting Time",
                y_label="Maximum Waiting Time (s)",
                title="runnerMain Maximum Waiting Time vs Episodes",
                color="tab:red",
            ),
        ]

    @staticmethod
    def _restore_checkpoint_arrays(payload: dict[str, Any] | None) -> dict[str, Any] | None:
        if payload is None:
            return None
        restored: dict[str, Any] = {}
        for key, value in payload.items():
            if isinstance(value, list):
                restored[key] = np.array(value, dtype=np.float32)
            else:
                restored[key] = value
        return restored

    def load_checkpoint(self, checkpoint_path: Path, allow_completed: bool = False) -> None:
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        resume_episode = int(checkpoint["episode"])
        if resume_episode % self.config.phase_length != 0 and resume_episode != self.config.total_episodes:
            raise RuntimeError(
                "Resuming is only supported from phase-boundary checkpoints with the current implementation."
            )
        if resume_episode >= self.config.total_episodes and not allow_completed:
            raise RuntimeError(
                f"Checkpoint episode {resume_episode} is already at or beyond total episodes {self.config.total_episodes}."
            )
        if checkpoint["node_ids"] != self.node_ids:
            raise RuntimeError("Checkpoint node ordering does not match the current scenario.")

        controller_key = self.shared_controller_id
        if controller_key not in checkpoint["controllers"]:
            raise RuntimeError(
                f"Checkpoint does not contain shared controller '{controller_key}'."
            )
        controller_payload = checkpoint["controllers"][controller_key]
        shared_controller = self.controllers[self.shared_controller_id]
        shared_controller.online_net.load_state_dict(controller_payload["online_state_dict"])
        shared_controller.target_net.load_state_dict(controller_payload["target_state_dict"])
        shared_controller.optimizer.load_state_dict(controller_payload["optimizer_state_dict"])
        shared_controller.latest_loss = controller_payload.get("latest_loss")

        self.gat_model.load_state_dict(checkpoint["gat_state_dict"])
        self.gat_optimizer.load_state_dict(checkpoint["gat_optimizer_state_dict"])
        self.frozen_gat_stats = self._restore_checkpoint_arrays(checkpoint.get("frozen_gat_stats"))

        self.selected_nodes = list(checkpoint["selected_nodes"])
        self.node_controller_assignment = {
            node_id: self.shared_controller_id for node_id in self.selected_nodes
        }
        self.episode_metrics = list(checkpoint.get("episode_metrics", []))
        self.eval_metrics = list(checkpoint.get("evaluation_metrics", []))
        self.phase_history = list(checkpoint.get("phase_history", []))
        self.best_eval_travel_time = checkpoint.get("best_eval_travel_time")
        self.best_eval_score = checkpoint.get("best_eval_score")
        self.resume_episode = resume_episode
        self._update_training_plots()

    def override_selected_nodes(self, selected_nodes: list[str]) -> None:
        if not selected_nodes:
            raise RuntimeError("Selected node override cannot be empty.")
        invalid_nodes = [node_id for node_id in selected_nodes if node_id not in self.node_ids]
        if invalid_nodes:
            raise RuntimeError(f"Selected node override contains unknown nodes: {invalid_nodes}")
        self.selected_nodes = list(selected_nodes)
        self.node_controller_assignment = {
            node_id: self.shared_controller_id for node_id in self.selected_nodes
        }

    def _update_training_plots(self) -> None:
        for plot in self.plots:
            plot.update(self.episode_metrics, self.eval_metrics, self.config.phase_length)

    @staticmethod
    def _append_plot_record(output_path: Path, record: dict[str, Any]) -> None:
        payload: dict[str, Any] = {"plot": "plot3", "records": []}
        if output_path.exists():
            try:
                loaded = json.loads(output_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                loaded = None
            if isinstance(loaded, dict):
                records = loaded.get("records")
                if isinstance(records, list):
                    payload = loaded
                else:
                    payload = {"plot": "plot3", "records": [loaded]}
        payload.setdefault("plot", "plot3")
        payload.setdefault("records", [])
        payload["records"].append(record)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def run_single_evaluation(
        self,
        output_json_path: Path,
        checkpoint_path: Path | None = None,
        methodology: str = "runnerMain",
        selection_method: str | None = None,
        sumo_seed: int | None = None,
        label: str | None = None,
    ) -> dict[str, Any]:
        eval_episode = self.resume_episode if self.resume_episode > 0 else 1
        phase_index = math.ceil(max(1, eval_episode) / self.config.phase_length)
        eval_seed = int(
            sumo_seed if sumo_seed is not None else self.config.sumo_seed + self.config.eval_seed_offset
        )
        metrics, _phase_log = self._rollout_episode(
            episode_index=eval_episode,
            phase_index=phase_index,
            sumo_seed=eval_seed,
            epsilon=0.0,
            collect_experience=False,
        )
        record = {
            "plot": "plot3",
            "methodology": methodology,
            "label": label or methodology,
            "selection_method": selection_method or methodology,
            "budget": len(self.selected_nodes),
            "scenario_dir": str(SCENARIO_DIR),
            "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else None,
            "checkpoint_episode": self.resume_episode if self.resume_episode > 0 else None,
            "controlled_node_count": len(self.selected_nodes),
            "selected_nodes": list(self.selected_nodes),
            "evaluation_mode": "single_episode_eval_only",
            "sumo_seed": eval_seed,
            "average_travel_time": metrics["average_travel_time"],
            "max_travel_time": metrics["max_travel_time"],
            "average_waiting_time": metrics.get("average_waiting_time"),
            "max_waiting_time": metrics.get("max_waiting_time"),
            "arrived_vehicle_count": metrics["arrived_vehicle_count"],
            "pending_vehicle_count": metrics["pending_vehicle_count"],
            "inserted_ratio": metrics["inserted_ratio"],
            "arrived_ratio": metrics["arrived_ratio"],
            "metrics": metrics,
        }
        self._append_plot_record(output_json_path, record)
        avg_text = (
            f"{metrics['average_travel_time']:.2f}"
            if not math.isnan(metrics["average_travel_time"])
            else "nan"
        )
        print(
            f"runnerMain eval-only | nodes={len(self.selected_nodes)} | "
            f"avg_tt={avg_text}s | seed={eval_seed} | saved={output_json_path}"
        )
        return record

    def _resolve_sumocfg_path(self, xml_key: str) -> Path:
        root = ET.parse(SUMOCFG_PATH).getroot()
        input_node = root.find("input")
        if input_node is None:
            raise RuntimeError(f"Invalid SUMO config: missing input block in {SUMOCFG_PATH}")
        for child in input_node:
            if child.tag == xml_key:
                return (SUMOCFG_PATH.parent / child.attrib["value"]).resolve()
        raise RuntimeError(f"Could not resolve {xml_key}")

    @staticmethod
    def _count_total_demand(route_path: Path) -> int:
        root = ET.parse(route_path).getroot()
        return sum(1 for child in root if child.tag == "vehicle")

    def _build_static_graph_and_scores(self) -> tuple[list[str], nx.DiGraph, dict[str, float], dict[str, dict[str, float]]]:
        root = ET.parse(self.net_path).getroot()
        node_ids = []
        graph = nx.DiGraph()
        for junction in root.findall("junction"):
            node_id = junction.attrib["id"]
            if node_id.startswith(":"):
                continue
            if junction.attrib.get("type", "").startswith("traffic_light"):
                node_ids.append(node_id)
                graph.add_node(node_id)
        lane_counts = {node_id: 0 for node_id in node_ids}
        for edge in root.findall("edge"):
            from_node = edge.attrib.get("from")
            to_node = edge.attrib.get("to")
            if from_node in graph and to_node in graph:
                graph.add_edge(from_node, to_node, weight=len(edge.findall("lane")))
            if from_node in lane_counts:
                lane_counts[from_node] += len(edge.findall("lane"))
            if to_node in lane_counts:
                lane_counts[to_node] += len(edge.findall("lane"))
        bet = nx.betweenness_centrality(graph.to_undirected(), normalized=True)
        degree = nx.degree_centrality(graph.to_undirected())
        close = nx.closeness_centrality(graph.to_undirected())
        max_lanes = max(lane_counts.values()) if lane_counts else 1
        static_scores = {
            node_id: 0.55 * bet[node_id] + 0.30 * degree[node_id] + 0.15 * (lane_counts[node_id] / max_lanes)
            for node_id in node_ids
        }
        static_context = {
            node_id: {
                "betweenness": float(bet[node_id]),
                "degree": float(degree[node_id]),
                "closeness": float(close[node_id]),
                "lane_norm": float(lane_counts[node_id] / max_lanes),
            }
            for node_id in node_ids
        }
        return sorted(node_ids), graph, static_scores, static_context

    def _build_node_context_matrix(self) -> np.ndarray:
        feature_matrix = np.array(
            [
                [
                    self.static_scores[node_id],
                    self.static_context[node_id]["betweenness"],
                    self.static_context[node_id]["degree"],
                    self.static_context[node_id]["closeness"],
                    self.static_context[node_id]["lane_norm"],
                ]
                for node_id in self.node_ids
            ],
            dtype=np.float32,
        )
        normalized_columns = [
            self._normalize_vector(feature_matrix[:, column_idx])
            for column_idx in range(feature_matrix.shape[1])
        ]
        return np.stack(normalized_columns, axis=1).astype(np.float32)

    def _build_adjacency_tensor(self) -> torch.Tensor:
        n = len(self.node_ids)
        adjacency = np.zeros((n, n), dtype=np.float32)
        for i, node_i in enumerate(self.node_ids):
            adjacency[i, i] = 1.0
            for node_j in self.graph.successors(node_i):
                adjacency[i, self.node_to_idx[node_j]] = 1.0
            for node_j in self.graph.predecessors(node_i):
                adjacency[i, self.node_to_idx[node_j]] = 1.0
        return torch.tensor(adjacency, dtype=torch.float32, device=self.device)

    def _initial_selected_subset(self) -> list[str]:
        ranked = sorted(self.static_scores.items(), key=lambda item: (-item[1], item[0]))
        return [node_id for node_id, _score in ranked[: self.config.budget]]

    def _sumo_cmd(self, seed: int) -> list[str]:
        return [
            self.config.sumo_binary,
            "-c",
            str(SUMOCFG_PATH),
            "--seed",
            str(seed),
            "--additional-files",
            str(self.fixed_timing_additional_path),
            "--no-step-log",
            "true",
            "--no-warnings",
            "true",
        ]

    def _training_sumo_seed(self, episode_index: int) -> int:
        return self.config.sumo_seed + max(0, episode_index - 1) * self.config.train_seed_stride

    def _evaluation_sumo_seed(self, episode_index: int, rollout_index: int) -> int:
        return (
            self.config.sumo_seed
            + self.config.eval_seed_offset
            + max(0, episode_index - 1) * self.config.eval_seed_stride
            + rollout_index
        )

    def _composite_eval_score(self, metrics: dict[str, Any]) -> float:
        avg_travel_time = float(metrics["average_travel_time"])
        if math.isnan(avg_travel_time):
            avg_travel_time = float(self.config.max_steps)
        demand = int(metrics["demand_vehicle_count"])
        arrived = int(metrics["arrived_vehicle_count"])
        pending = int(metrics["pending_vehicle_count"])
        unarrived = max(0, demand - arrived)
        return (
            avg_travel_time
            + self.config.eval_pending_weight * pending
            + self.config.eval_unarrived_weight * unarrived
        )

    @staticmethod
    def _unique_preserve_order(values: list[str]) -> list[str]:
        return list(dict.fromkeys(values))

    @staticmethod
    def _phase_weight(state: str) -> float:
        return sum(1.0 if char == "G" else 0.65 for char in state if char in {"G", "g"})

    @staticmethod
    def _phase_overlap(lhs: str, rhs: str) -> float:
        lhs_positions = {idx for idx, char in enumerate(lhs) if char in {"G", "g"}}
        rhs_positions = {idx for idx, char in enumerate(rhs) if char in {"G", "g"}}
        if not lhs_positions or not rhs_positions:
            return 0.0
        return len(lhs_positions & rhs_positions) / len(lhs_positions | rhs_positions)

    def _select_major_green_states(self, phase_states: list[str]) -> list[str]:
        green_states = [state for state in phase_states if "y" not in state.lower() and any(char in {"G", "g"} for char in state)]
        if len(green_states) <= 2:
            return green_states
        green_states = sorted(green_states, key=self._phase_weight, reverse=True)
        return [green_states[0], green_states[1]]

    @staticmethod
    def _build_transition_state(from_state: str, to_state: str) -> str:
        chars = []
        for from_char, to_char in zip(from_state, to_state):
            if from_char in {"G", "g"} and to_char in {"r", "R"}:
                chars.append("y")
            elif from_char in {"G", "g"} and to_char in {"G", "g"}:
                chars.append("G" if from_char == "G" or to_char == "G" else "g")
            else:
                chars.append("r")
        return "".join(chars)

    def _build_control_specs(self) -> dict[str, dict[str, Any]]:
        specs: dict[str, dict[str, Any]] = {}
        libsumo.start(self._sumo_cmd(self.config.sumo_seed))
        try:
            for node_id in self.node_ids:
                logic = libsumo.trafficlight.getAllProgramLogics(node_id)[0]
                green_states = self._select_major_green_states([phase.state for phase in logic.getPhases()])
                if not green_states:
                    raise RuntimeError(f"No usable green phases found for {node_id}")
                if len(green_states) == 1:
                    green_states = [green_states[0], green_states[0]]
                controlled_links = libsumo.trafficlight.getControlledLinks(node_id)
                phase_groups = []
                all_incoming = []
                all_outgoing = []
                for state in green_states:
                    incoming = []
                    outgoing = []
                    for link_idx, link_set in enumerate(controlled_links):
                        if state[link_idx] not in {"G", "g"}:
                            continue
                        for link in link_set:
                            incoming.append(link[0])
                            outgoing.append(link[1])
                    incoming = self._unique_preserve_order(incoming)
                    outgoing = self._unique_preserve_order(outgoing)
                    all_incoming.extend(incoming)
                    all_outgoing.extend(outgoing)
                    phase_groups.append(PhaseGroup(incoming_lanes=incoming, outgoing_lanes=outgoing))
                specs[node_id] = {
                    "green_states": green_states,
                    "phase_groups": phase_groups,
                    "all_incoming_lanes": self._unique_preserve_order(all_incoming),
                    "all_outgoing_lanes": self._unique_preserve_order(all_outgoing),
                }
        finally:
            libsumo.close()
        return specs

    def _raw_feature_dim(self) -> int:
        return 8

    def _state_dim(self) -> int:
        return self._raw_feature_dim() * 2 + 3

    def _configure_selected_programs(self) -> None:
        for node_id in self.selected_nodes:
            spec = self.control_specs[node_id]
            green_a, green_b = spec["green_states"]
            logic = libsumo.trafficlight.Logic(
                f"main_{node_id}",
                0,
                0,
                [
                    libsumo.trafficlight.Phase(self.config.min_green_duration, green_a),
                    libsumo.trafficlight.Phase(self.config.yellow_duration, self._build_transition_state(green_a, green_b)),
                    libsumo.trafficlight.Phase(self.config.min_green_duration, green_b),
                    libsumo.trafficlight.Phase(self.config.yellow_duration, self._build_transition_state(green_b, green_a)),
                ],
            )
            libsumo.trafficlight.setProgramLogic(node_id, logic)
            libsumo.trafficlight.setProgram(node_id, logic.programID)
            libsumo.trafficlight.setPhase(node_id, 0)
            libsumo.trafficlight.setPhaseDuration(node_id, self.config.min_green_duration)

    def _capture_raw_node_features(self) -> tuple[np.ndarray, np.ndarray]:
        features = []
        queues = []
        max_static = max(self.static_scores.values()) if self.static_scores else 1.0
        for node_id in self.node_ids:
            spec = self.control_specs[node_id]
            incoming = spec["all_incoming_lanes"]
            outgoing = spec["all_outgoing_lanes"]
            queue_total = sum(float(libsumo.lane.getLastStepHaltingNumber(lane)) for lane in incoming)
            volume_total = sum(float(libsumo.lane.getLastStepVehicleNumber(lane)) for lane in incoming)
            out_queue = sum(float(libsumo.lane.getLastStepHaltingNumber(lane)) for lane in outgoing)
            speed_terms = []
            for lane in incoming:
                mean_speed = float(libsumo.lane.getLastStepMeanSpeed(lane))
                max_speed = max(1.0, float(libsumo.lane.getMaxSpeed(lane)))
                speed_terms.append(mean_speed / max_speed)
            occupancy_terms = [float(libsumo.lane.getLastStepOccupancy(lane)) / 100.0 for lane in outgoing] or [0.0]
            mean_speed_ratio = float(np.mean(speed_terms)) if speed_terms else 1.0
            mean_occupancy = float(np.mean(occupancy_terms))
            features.append(
                [
                    self.static_scores[node_id] / max_static,
                    self.static_context[node_id]["degree"],
                    self.static_context[node_id]["lane_norm"],
                    min(2.0, queue_total / max(1.0, len(incoming) * 5.0)),
                    min(2.0, volume_total / max(1.0, len(incoming) * 7.0)),
                    min(2.0, out_queue / max(1.0, len(outgoing) * 5.0)),
                    mean_speed_ratio,
                    mean_occupancy,
                ]
            )
            queues.append(queue_total)
        return np.array(features, dtype=np.float32), np.array(queues, dtype=np.float32)

    def _build_rl_state(self, node_id: str, raw_features: np.ndarray) -> np.ndarray:
        idx = self.node_to_idx[node_id]
        own = raw_features[idx]
        neighbors = list(self.graph.predecessors(node_id)) + list(self.graph.successors(node_id))
        neighbor_feats = [raw_features[self.node_to_idx[n]] for n in self._unique_preserve_order(neighbors)]
        neighbor_mean = np.mean(neighbor_feats, axis=0) if neighbor_feats else np.zeros_like(own)
        current_slot = self._current_slot_for_node(node_id)
        phase_age = min(
            1.0,
            float(libsumo.trafficlight.getSpentDuration(node_id))
            / max(1.0, float(libsumo.trafficlight.getPhaseDuration(node_id))),
        )
        return np.concatenate([own, neighbor_mean, np.array([float(current_slot), 1.0 - float(current_slot), phase_age], dtype=np.float32)])

    def _local_pressure(self, node_id: str) -> float:
        spec = self.control_specs[node_id]
        incoming = spec["all_incoming_lanes"]
        outgoing = spec["all_outgoing_lanes"]
        in_queue = sum(float(libsumo.lane.getLastStepHaltingNumber(lane)) for lane in incoming)
        out_queue = sum(float(libsumo.lane.getLastStepHaltingNumber(lane)) for lane in outgoing)
        return in_queue - 0.5 * out_queue

    @staticmethod
    def _normalize_vector(values: np.ndarray) -> np.ndarray:
        if values.size == 0:
            return values.astype(np.float32)
        values = values.astype(np.float32)
        minimum = float(np.min(values))
        maximum = float(np.max(values))
        if maximum - minimum <= 1e-6:
            return np.zeros_like(values, dtype=np.float32)
        return (values - minimum) / (maximum - minimum)

    def _attention_matrix(self, attention: np.ndarray | None) -> np.ndarray:
        if attention is not None:
            matrix = np.maximum(np.array(attention, dtype=np.float32), 0.0)
        else:
            matrix = self.adjacency.detach().cpu().numpy().astype(np.float32)
        np.fill_diagonal(matrix, 0.0)
        row_sums = matrix.sum(axis=1, keepdims=True)
        return np.divide(
            matrix,
            row_sums,
            out=np.zeros_like(matrix, dtype=np.float32),
            where=row_sums > 1e-6,
        )

    @staticmethod
    def _phase_group_pressure(group: PhaseGroup) -> float:
        in_queue = sum(float(libsumo.lane.getLastStepHaltingNumber(lane_id)) for lane_id in group.incoming_lanes)
        out_queue = sum(float(libsumo.lane.getLastStepHaltingNumber(lane_id)) for lane_id in group.outgoing_lanes)
        return in_queue - 0.5 * out_queue

    def _current_slot_for_node(self, node_id: str) -> int:
        current_state = libsumo.trafficlight.getRedYellowGreenState(node_id)
        green_a, green_b = self.control_specs[node_id]["green_states"]
        overlap_a = self._phase_overlap(current_state, green_a)
        overlap_b = self._phase_overlap(current_state, green_b)
        return 0 if overlap_a >= overlap_b else 1

    def _capture_control_leverage_snapshot(self) -> dict[str, np.ndarray]:
        local_pressures = []
        phase_competition = []
        spillback_ratio = []
        outgoing_occupancy = []
        phase_mismatch = []
        current_slot = []

        for node_id in self.node_ids:
            spec = self.control_specs[node_id]
            phase_groups = spec["phase_groups"]
            phase_pressures = [self._phase_group_pressure(group) for group in phase_groups[:2]]
            if len(phase_pressures) == 1:
                phase_pressures.append(phase_pressures[0])
            elif not phase_pressures:
                phase_pressures = [0.0, 0.0]

            outgoing_lanes = spec["all_outgoing_lanes"]
            occupancies = [float(libsumo.lane.getLastStepOccupancy(lane_id)) / 100.0 for lane_id in outgoing_lanes] or [0.0]
            spillbacks = sum(1.0 for value in occupancies if value >= 0.78)
            node_pressure = self._local_pressure(node_id)
            slot = self._current_slot_for_node(node_id)
            preferred_slot = 0 if phase_pressures[0] >= phase_pressures[1] else 1

            local_pressures.append(node_pressure)
            phase_competition.append(abs(phase_pressures[0] - phase_pressures[1]))
            spillback_ratio.append(spillbacks / max(1.0, float(len(outgoing_lanes))))
            outgoing_occupancy.append(float(np.mean(occupancies)))
            phase_mismatch.append(1.0 if preferred_slot != slot else 0.0)
            current_slot.append(float(slot))

        return {
            "local_pressure": np.array(local_pressures, dtype=np.float32),
            "phase_competition": np.array(phase_competition, dtype=np.float32),
            "spillback_ratio": np.array(spillback_ratio, dtype=np.float32),
            "outgoing_occupancy": np.array(outgoing_occupancy, dtype=np.float32),
            "phase_mismatch": np.array(phase_mismatch, dtype=np.float32),
            "current_slot": np.array(current_slot, dtype=np.float32),
        }

    def _capture_pressure_vector(self) -> np.ndarray:
        return np.array([self._local_pressure(node_id) for node_id in self.node_ids], dtype=np.float32)

    def _graph_inference(self, history_window: list[np.ndarray]) -> dict[str, np.ndarray]:
        stacked = np.concatenate(history_window, axis=1)
        tensor = torch.tensor(stacked, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            pressure_pred, occupancy_pred, competition_pred, attention, embedding = self.gat_model(tensor, self.adjacency)
        return {
            "pressure": pressure_pred.squeeze(0).cpu().numpy(),
            "occupancy": occupancy_pred.squeeze(0).cpu().numpy(),
            "phase_competition": competition_pred.squeeze(0).cpu().numpy(),
            "attention": attention.squeeze(0).cpu().numpy(),
            "embedding": embedding.squeeze(0).cpu().numpy(),
        }

    @staticmethod
    def _positive_pressure(pressure: float) -> float:
        return max(0.0, float(pressure))

    def _uniform_neighbor_value(
        self,
        node_index: int,
        values: np.ndarray,
    ) -> float:
        node_id = self.node_ids[node_index]
        neighbor_ids = self._unique_preserve_order(
            list(self.graph.predecessors(node_id)) + list(self.graph.successors(node_id))
        )
        if not neighbor_ids:
            return 0.0
        neighbor_values = [float(values[self.node_to_idx[neighbor_id]]) for neighbor_id in neighbor_ids]
        return float(np.mean(neighbor_values))

    def _pressure_improvement_reward(
        self,
        node_index: int,
        action: int,
        previous_pressures: np.ndarray,
        current_pressures: np.ndarray,
    ) -> float:
        local_delta = float(previous_pressures[node_index] - current_pressures[node_index])
        local_term = math.tanh(local_delta / self.config.reward_improvement_scale)
        reward = self.config.reward_local_improvement_weight * local_term

        previous_neighbor = self._uniform_neighbor_value(node_index, previous_pressures)
        current_neighbor = self._uniform_neighbor_value(node_index, current_pressures)
        neighbor_delta = previous_neighbor - current_neighbor
        reward += self.config.reward_neighbor_weight * math.tanh(
            neighbor_delta / self.config.reward_improvement_scale
        )

        absolute_penalty = math.tanh(
            self._positive_pressure(float(current_pressures[node_index])) / self.config.reward_improvement_scale
        )
        reward -= self.config.reward_abs_pressure_weight * absolute_penalty
        if action == 1:
            reward -= self.config.reward_switch_penalty
        return float(reward)

    def _select_action(
        self,
        controller: ControllerRuntime,
        state: np.ndarray,
        node_index: int,
        epsilon: float,
    ) -> int:
        if random.random() < epsilon:
            return random.randint(0, 1)
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        node_tensor = torch.tensor([node_index], dtype=torch.long, device=self.device)
        with torch.no_grad():
            q_values = controller.online_net(state_tensor, node_tensor)
        return int(torch.argmax(q_values, dim=1).item())

    def _apply_action(self, node_id: str, action: int) -> None:
        current_phase = int(libsumo.trafficlight.getPhase(node_id))
        if current_phase not in {0, 2}:
            return
        spent = float(libsumo.trafficlight.getSpentDuration(node_id))
        if action == 0:
            remaining = max(
                1.0,
                float(self.config.decision_interval),
                float(self.config.min_green_duration) - spent + 1.0,
            )
            libsumo.trafficlight.setPhaseDuration(node_id, remaining)
            return
        if spent < self.config.min_green_duration:
            remaining = max(1.0, float(self.config.min_green_duration) - spent)
            libsumo.trafficlight.setPhaseDuration(node_id, remaining)
            return
        if current_phase == 0:
            libsumo.trafficlight.setPhase(node_id, 1)
        elif current_phase == 2:
            libsumo.trafficlight.setPhase(node_id, 3)

    def _epsilon(self, episode_index: int) -> float:
        if episode_index >= self.config.epsilon_decay_episodes:
            return self.config.epsilon_end
        progress = episode_index / max(1, self.config.epsilon_decay_episodes)
        return self.config.epsilon_start + progress * (self.config.epsilon_end - self.config.epsilon_start)

    @staticmethod
    def _cosine_similarity(lhs: np.ndarray, rhs: np.ndarray) -> float:
        lhs_norm = float(np.linalg.norm(lhs))
        rhs_norm = float(np.linalg.norm(rhs))
        if lhs_norm <= 1e-8 or rhs_norm <= 1e-8:
            return 0.0
        return float(np.dot(lhs, rhs) / (lhs_norm * rhs_norm))

    def _static_feature_vector(self, node_id: str) -> np.ndarray:
        return np.array(
            [
                self.static_scores[node_id],
                self.static_context[node_id]["betweenness"],
                self.static_context[node_id]["degree"],
                self.static_context[node_id]["closeness"],
                self.static_context[node_id]["lane_norm"],
            ],
            dtype=np.float32,
        )

    def _update_policy(self, controller: ControllerRuntime) -> float | None:
        if not controller.replay.can_sample(self.config.batch_size, self.config.sequence_length):
            return None
        batch = controller.replay.sample(self.config.batch_size, self.config.sequence_length, self.device)
        q_values = controller.online_net(batch["states"], batch["node_indices"])
        q_selected = q_values.gather(1, batch["actions"].unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_online_q = controller.online_net(batch["next_states"], batch["node_indices"])
            next_actions = torch.argmax(next_online_q, dim=1, keepdim=True)
            target_next_q = controller.target_net(batch["next_states"], batch["node_indices"])
            target_q = target_next_q.gather(1, next_actions).squeeze(1)
            td_target = batch["rewards"] + (1.0 - batch["dones"]) * self.config.gamma * target_q
        loss = nn.functional.smooth_l1_loss(q_selected, td_target)
        controller.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(controller.online_net.parameters(), max_norm=5.0)
        controller.optimizer.step()
        controller.latest_loss = float(loss.item())
        return controller.latest_loss

    def _track_vehicle_progress(
        self,
        step: int,
        vehicle_departs: dict[str, int],
        vehicle_travel_times: list[float],
        vehicle_waiting_times: dict[str, float],
    ) -> tuple[int, int, int]:
        loaded_ids = libsumo.simulation.getLoadedIDList()
        departed_ids = libsumo.simulation.getDepartedIDList()
        arrived_ids = libsumo.simulation.getArrivedIDList()
        for veh_id in libsumo.vehicle.getIDList():
            vehicle_waiting_times[veh_id] = max(
                vehicle_waiting_times.get(veh_id, 0.0),
                float(libsumo.vehicle.getAccumulatedWaitingTime(veh_id)),
            )
        for veh_id in departed_ids:
            vehicle_departs[veh_id] = step
            vehicle_waiting_times.setdefault(veh_id, 0.0)
        for veh_id in arrived_ids:
            depart_step = vehicle_departs.pop(veh_id, None)
            if depart_step is not None:
                vehicle_travel_times.append(step - depart_step)
        return len(loaded_ids), len(departed_ids), len(arrived_ids)

    def _build_prediction_dataset(
        self,
        phase_logs: list[dict[str, list[np.ndarray]]],
    ) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        samples: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        history_length = self.config.history_length
        horizon = self.config.prediction_horizon
        for episode_log in phase_logs:
            raw_series = episode_log["raw_features"]
            pressure_series = episode_log["pressures"]
            occupancy_series = episode_log["selector_outgoing_occupancy"]
            competition_series = episode_log["selector_phase_competition"]
            if min(
                len(raw_series),
                len(pressure_series),
                len(occupancy_series),
                len(competition_series),
            ) < history_length + horizon:
                continue
            last_index = len(raw_series) - horizon
            for current in range(history_length - 1, last_index):
                history = raw_series[current - history_length + 1 : current + 1]
                x_sample = np.concatenate(history, axis=1).astype(np.float32)
                pressure_target = np.stack(
                    pressure_series[current + 1 : current + 1 + horizon],
                    axis=1,
                ).astype(np.float32)
                occupancy_target = np.stack(
                    occupancy_series[current + 1 : current + 1 + horizon],
                    axis=1,
                ).astype(np.float32)
                competition_target = np.stack(
                    competition_series[current + 1 : current + 1 + horizon],
                    axis=1,
                ).astype(np.float32)
                samples.append((x_sample, pressure_target, occupancy_target, competition_target))
        return samples

    def _train_gat_offline(self, phase_logs: list[dict[str, list[np.ndarray]]], phase_index: int) -> dict[str, Any] | None:
        samples = self._build_prediction_dataset(phase_logs)
        if not samples:
            print(f"runnerMain phase {phase_index}: no GAT samples collected, skipping forecaster training")
            return None

        print(f"runnerMain phase {phase_index}: training GAT on {len(samples)} samples for {self.config.gat_epochs} epochs")
        self.gat_model.train()
        last_epoch_losses: dict[str, float] = {
            "total": 0.0,
            "pressure": 0.0,
            "occupancy": 0.0,
            "competition": 0.0,
        }
        for epoch_index in range(self.config.gat_epochs):
            random.shuffle(samples)
            epoch_totals = {
                "total": 0.0,
                "pressure": 0.0,
                "occupancy": 0.0,
                "competition": 0.0,
            }
            batch_count = 0
            for batch_start in range(0, len(samples), self.config.batch_size):
                batch_samples = samples[batch_start : batch_start + self.config.batch_size]
                x_batch = torch.tensor(
                    np.stack([item[0] for item in batch_samples]),
                    dtype=torch.float32,
                    device=self.device,
                )
                pressure_batch = torch.tensor(
                    np.stack([item[1] for item in batch_samples]),
                    dtype=torch.float32,
                    device=self.device,
                )
                occupancy_batch = torch.tensor(
                    np.stack([item[2] for item in batch_samples]),
                    dtype=torch.float32,
                    device=self.device,
                )
                competition_batch = torch.tensor(
                    np.stack([item[3] for item in batch_samples]),
                    dtype=torch.float32,
                    device=self.device,
                )
                pressure_pred, occupancy_pred, competition_pred, _attention, _embedding = self.gat_model(
                    x_batch,
                    self.adjacency,
                )
                pressure_loss = nn.functional.mse_loss(pressure_pred, pressure_batch)
                occupancy_loss = nn.functional.mse_loss(occupancy_pred, occupancy_batch)
                competition_loss = nn.functional.mse_loss(competition_pred, competition_batch)
                loss = pressure_loss + self.config.gat_aux_task_weight * (occupancy_loss + competition_loss)
                self.gat_optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.gat_model.parameters(), max_norm=5.0)
                self.gat_optimizer.step()
                epoch_totals["total"] += float(loss.item())
                epoch_totals["pressure"] += float(pressure_loss.item())
                epoch_totals["occupancy"] += float(occupancy_loss.item())
                epoch_totals["competition"] += float(competition_loss.item())
                batch_count += 1
            last_epoch_losses = {
                key: value / max(1, batch_count)
                for key, value in epoch_totals.items()
            }
            if (epoch_index + 1) == 1 or (epoch_index + 1) == self.config.gat_epochs or (epoch_index + 1) % 5 == 0:
                print(
                    f"runnerMain phase {phase_index}: GAT epoch {epoch_index + 1}/{self.config.gat_epochs} "
                    f"loss={last_epoch_losses['total']:.4f} "
                    f"(pressure={last_epoch_losses['pressure']:.4f}, "
                    f"occupancy={last_epoch_losses['occupancy']:.4f}, "
                    f"competition={last_epoch_losses['competition']:.4f})"
                )

        self.gat_model.eval()
        pressure_prediction_list = []
        occupancy_prediction_list = []
        competition_prediction_list = []
        attention_list = []
        embedding_list = []
        print(f"runnerMain phase {phase_index}: running batched GAT inference for selector outputs")
        with torch.no_grad():
            for batch_start in range(0, len(samples), self.config.batch_size):
                batch_samples = samples[batch_start : batch_start + self.config.batch_size]
                x_batch = torch.tensor(
                    np.stack([item[0] for item in batch_samples]),
                    dtype=torch.float32,
                    device=self.device,
                )
                pressure_pred, occupancy_pred, competition_pred, attention, embedding = self.gat_model(
                    x_batch,
                    self.adjacency,
                )
                pressure_prediction_list.append(pressure_pred.cpu().numpy())
                occupancy_prediction_list.append(occupancy_pred.cpu().numpy())
                competition_prediction_list.append(competition_pred.cpu().numpy())
                attention_list.append(attention.cpu().numpy())
                embedding_list.append(embedding.cpu().numpy())

        mean_pressure_prediction = np.mean(np.concatenate(pressure_prediction_list, axis=0), axis=0)
        mean_occupancy_prediction = np.mean(np.concatenate(occupancy_prediction_list, axis=0), axis=0)
        mean_competition_prediction = np.mean(np.concatenate(competition_prediction_list, axis=0), axis=0)
        mean_attention = np.mean(np.concatenate(attention_list, axis=0), axis=0)
        mean_embedding = np.mean(np.concatenate(embedding_list, axis=0), axis=0)
        pressure_target_values = np.concatenate([item[1].reshape(-1) for item in samples], axis=0)
        occupancy_target_values = np.concatenate([item[2].reshape(-1) for item in samples], axis=0)
        competition_target_values = np.concatenate([item[3].reshape(-1) for item in samples], axis=0)
        stats = {
            "phase": phase_index,
            "pressure_mean": float(np.mean(pressure_target_values)),
            "pressure_std": float(np.std(pressure_target_values) + 1e-6),
            "occupancy_mean": float(np.mean(occupancy_target_values)),
            "occupancy_std": float(np.std(occupancy_target_values) + 1e-6),
            "phase_competition_mean": float(np.mean(competition_target_values)),
            "phase_competition_std": float(np.std(competition_target_values) + 1e-6),
            "mean_pressure_prediction": mean_pressure_prediction,
            "mean_occupancy_prediction": mean_occupancy_prediction,
            "mean_phase_competition_prediction": mean_competition_prediction,
            "mean_attention": mean_attention,
            "mean_embedding": mean_embedding,
            "training_loss": last_epoch_losses,
            "sample_count": len(samples),
        }
        print(f"runnerMain phase {phase_index}: GAT training and inference complete")
        return stats

    def _selector_scores(
        self,
        phase_logs: list[dict[str, list[np.ndarray]]],
        attention: np.ndarray | None,
    ) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
        node_count = len(self.node_ids)
        competition_sum = np.zeros(node_count, dtype=np.float32)
        spillback_sum = np.zeros(node_count, dtype=np.float32)
        occupancy_sum = np.zeros(node_count, dtype=np.float32)
        mismatch_sum = np.zeros(node_count, dtype=np.float32)
        snapshot_count = np.zeros(node_count, dtype=np.float32)
        delta_sum = np.zeros(node_count, dtype=np.float32)
        delta_count = np.zeros(node_count, dtype=np.float32)
        switch_response_sum = np.zeros(node_count, dtype=np.float32)
        switch_count = np.zeros(node_count, dtype=np.float32)

        for episode_log in phase_logs:
            if not episode_log["selector_local_pressure"]:
                continue
            pressure_series = np.stack(episode_log["selector_local_pressure"], axis=0)
            competition_series = np.stack(episode_log["selector_phase_competition"], axis=0)
            spillback_series = np.stack(episode_log["selector_spillback_ratio"], axis=0)
            occupancy_series = np.stack(episode_log["selector_outgoing_occupancy"], axis=0)
            mismatch_series = np.stack(episode_log["selector_phase_mismatch"], axis=0)
            slot_series = np.stack(episode_log["selector_current_slot"], axis=0)

            competition_sum += np.sum(competition_series, axis=0)
            spillback_sum += np.sum(spillback_series, axis=0)
            occupancy_sum += np.sum(occupancy_series, axis=0)
            mismatch_sum += np.sum(mismatch_series, axis=0)
            snapshot_count += float(competition_series.shape[0])

            if pressure_series.shape[0] > 1:
                pressure_delta = np.abs(np.diff(pressure_series, axis=0))
                delta_sum += np.sum(pressure_delta, axis=0)
                delta_count += float(pressure_delta.shape[0])

                slot_delta = (np.diff(slot_series, axis=0) != 0.0).astype(np.float32)
                switch_response_sum += np.sum(pressure_delta * slot_delta, axis=0)
                switch_count += np.sum(slot_delta, axis=0)

        snapshot_count = np.maximum(snapshot_count, 1.0)
        delta_count = np.maximum(delta_count, 1.0)

        mean_competition = competition_sum / snapshot_count
        mean_spillback = spillback_sum / snapshot_count
        mean_occupancy = occupancy_sum / snapshot_count
        mismatch_ratio = mismatch_sum / snapshot_count
        pressure_variability = delta_sum / delta_count
        phase_response = np.divide(
            switch_response_sum,
            switch_count,
            out=np.zeros_like(switch_response_sum),
            where=switch_count > 0.0,
        )

        static_raw = np.array(
            [
                0.65 * self.static_scores[node_id]
                + 0.20 * self.static_context[node_id]["lane_norm"]
                + 0.15 * self.static_context[node_id]["closeness"]
                for node_id in self.node_ids
            ],
            dtype=np.float32,
        )
        spillback_raw = 0.55 * mean_spillback + 0.45 * mean_occupancy
        control_sensitivity_raw = (
            0.45 * (mismatch_ratio * mean_competition)
            + 0.35 * phase_response
            + 0.20 * pressure_variability
        )
        attention_matrix = self._attention_matrix(attention)
        neighbor_seed = 0.60 * mean_competition + 0.40 * spillback_raw
        neighborhood_raw = attention_matrix @ neighbor_seed

        static_component = self._normalize_vector(static_raw)
        phase_competition_component = self._normalize_vector(mean_competition)
        spillback_component = self._normalize_vector(spillback_raw)
        neighborhood_component = self._normalize_vector(neighborhood_raw)
        control_sensitivity_component = self._normalize_vector(control_sensitivity_raw)

        base_scores: dict[str, float] = {}
        component_scores: dict[str, dict[str, float]] = {}
        for idx, node_id in enumerate(self.node_ids):
            score = (
                self.config.selector_static_weight * static_component[idx]
                + self.config.selector_phase_competition_weight * phase_competition_component[idx]
                + self.config.selector_spillback_weight * spillback_component[idx]
                + self.config.selector_neighbor_weight * neighborhood_component[idx]
                + self.config.selector_control_sensitivity_weight * control_sensitivity_component[idx]
            )
            base_scores[node_id] = float(score)
            component_scores[node_id] = {
                "static_importance": float(static_component[idx]),
                "phase_competition": float(phase_competition_component[idx]),
                "spillback_exposure": float(spillback_component[idx]),
                "neighborhood_influence": float(neighborhood_component[idx]),
                "control_sensitivity": float(control_sensitivity_component[idx]),
                "mean_phase_competition": float(mean_competition[idx]),
                "mean_spillback_ratio": float(mean_spillback[idx]),
                "mean_outgoing_occupancy": float(mean_occupancy[idx]),
                "phase_mismatch_ratio": float(mismatch_ratio[idx]),
                "phase_response": float(phase_response[idx]),
                "pressure_variability": float(pressure_variability[idx]),
            }
        return base_scores, component_scores

    def _selection_stability_context(self) -> dict[str, Any]:
        latest_eval_score = None
        if self.eval_metrics:
            latest_eval_score = float(self.eval_metrics[-1]["composite_score"])
        best_eval_score = float(self.best_eval_score) if self.best_eval_score is not None else None
        good_current_subset = False
        if latest_eval_score is not None and best_eval_score is not None and best_eval_score > 1e-6:
            good_current_subset = latest_eval_score <= best_eval_score * (1.0 + self.config.selector_good_set_tolerance)
        return {
            "latest_eval_score": latest_eval_score,
            "best_eval_score": best_eval_score,
            "good_current_subset": good_current_subset,
        }

    @staticmethod
    def _phase_change_details(selected_before: list[str], selected_after: list[str]) -> dict[str, Any]:
        before_set = set(selected_before)
        after_set = set(selected_after)
        removed_nodes = sorted(before_set - after_set)
        added_nodes = sorted(after_set - before_set)
        return {
            "changed_node_count": len(added_nodes),
            "added_nodes": added_nodes,
            "removed_nodes": removed_nodes,
            "changed": bool(added_nodes or removed_nodes),
        }

    def _phase_change_report(self) -> dict[str, Any]:
        phases_with_changes: list[dict[str, Any]] = []
        total_changed_nodes = 0
        for phase_entry in self.phase_history:
            changed_node_count = int(phase_entry.get("changed_node_count", 0))
            if changed_node_count <= 0:
                continue
            phases_with_changes.append(
                {
                    "phase": int(phase_entry["phase"]),
                    "episode_end": int(phase_entry["episode_end"]),
                    "changed_node_count": changed_node_count,
                    "added_nodes": list(phase_entry.get("added_nodes", [])),
                    "removed_nodes": list(phase_entry.get("removed_nodes", [])),
                }
            )
            total_changed_nodes += changed_node_count
        return {
            "phase_change_event_count": len(phases_with_changes),
            "total_changed_nodes_across_phases": total_changed_nodes,
            "phases_with_changes": phases_with_changes,
        }

    def _reselect_nodes(
        self,
        phase_logs: list[dict[str, list[np.ndarray]]],
        attention: np.ndarray | None,
        embedding: np.ndarray | None,
        phase_index: int,
    ) -> tuple[list[str], dict[str, Any]]:
        base_scores, component_scores = self._selector_scores(phase_logs, attention)
        ranked = sorted(self.node_ids, key=lambda node_id: (-base_scores[node_id], node_id))
        stability = self._selection_stability_context()
        anchor_count = min(
            self.config.budget,
            max(1, int(math.ceil(self.config.budget * self.config.selector_anchor_ratio))),
        )
        if embedding is None:
            embedding = np.stack([self._static_feature_vector(node_id) for node_id in self.node_ids], axis=0)
        current_selected = list(self.selected_nodes)
        if len(current_selected) != self.config.budget:
            current_selected = ranked[: self.config.budget]

        incumbent_adjusted_scores = {
            node_id: base_scores[node_id] + self.config.selector_incumbent_bonus
            for node_id in current_selected
        }
        protected_anchors = sorted(
            current_selected,
            key=lambda node_id: (-incumbent_adjusted_scores[node_id], node_id),
        )[:anchor_count]
        protected_anchor_set = set(protected_anchors)
        selected = list(current_selected)
        incumbent_replacement_pool = [node_id for node_id in current_selected if node_id not in protected_anchor_set]
        swap_budget = min(
            self.config.budget,
            self.config.selector_good_set_max_swaps if stability["good_current_subset"] else self.config.selector_max_swaps_per_phase,
        )
        replacement_margin = self.config.selector_replacement_margin * (
            1.5 if stability["good_current_subset"] else 1.0
        )
        replacement_events: list[dict[str, Any]] = []

        for _swap_index in range(swap_budget):
            if not incumbent_replacement_pool:
                break
            remaining = [node_id for node_id in self.node_ids if node_id not in selected]
            if not remaining:
                break
            rescored: list[tuple[str, float, float]] = []
            for node_id in remaining:
                idx = self.node_to_idx[node_id]
                redundancy = 0.0
                if selected:
                    redundancy = max(
                        self._cosine_similarity(embedding[idx], embedding[self.node_to_idx[picked]])
                        for picked in selected
                    )
                score = base_scores[node_id] - self.config.selector_redundancy_penalty * redundancy
                rescored.append((node_id, score, redundancy))
            rescored.sort(key=lambda item: (-item[1], item[0]))
            chosen_node, chosen_score, chosen_redundancy = rescored[0]

            weakest_incumbent = min(
                incumbent_replacement_pool,
                key=lambda node_id: (incumbent_adjusted_scores[node_id], node_id),
            )
            weakest_threshold = incumbent_adjusted_scores[weakest_incumbent] + replacement_margin
            if chosen_score <= weakest_threshold:
                break

            selected = [node_id for node_id in selected if node_id != weakest_incumbent]
            selected.append(chosen_node)
            incumbent_replacement_pool = [node_id for node_id in incumbent_replacement_pool if node_id != weakest_incumbent]
            replacement_events.append(
                {
                    "added": chosen_node,
                    "removed": weakest_incumbent,
                    "candidate_score": float(chosen_score),
                    "candidate_redundancy": float(chosen_redundancy),
                    "replaced_threshold": float(weakest_threshold),
                }
            )

        selected = sorted(selected, key=lambda node_id: (-base_scores[node_id], node_id))[: self.config.budget]
        phase_change = self._phase_change_details(current_selected, selected)

        details = {
            "phase": phase_index,
            "base_scores": {node_id: float(base_scores[node_id]) for node_id in self.node_ids},
            "component_scores": component_scores,
            "selected_nodes": selected,
            "anchor_nodes": protected_anchors,
            "selection_mode": "deterministic",
            "current_selected": current_selected,
            "replacement_events": replacement_events,
            "swap_budget": swap_budget,
            "replacement_margin": float(replacement_margin),
            "stability": stability,
            **phase_change,
        }
        return selected, details

    def _remap_controllers(self, next_selected_nodes: list[str]) -> dict[str, str]:
        return {node_id: self.shared_controller_id for node_id in next_selected_nodes}

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
            "selected_before": selected_before,
            "selected_after": selected_after if selected_after is not None else selected_before,
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
        out_path = self.output_dir / f"selector_phase_{phase_index:02d}.json"
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
        }
        metrics_path = self.output_dir / "training_metrics.json"
        metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _save_checkpoint(self, episode: int, phase_index: int, is_final: bool = False, is_best: bool = False) -> Path:
        if is_best:
            filename = "best_checkpoint.pt"
        else:
            filename = "final_checkpoint.pt" if is_final else f"phase_{phase_index:02d}_ep{episode:03d}.pt"
        checkpoint_path = self.checkpoint_dir / filename
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
        }
        torch.save(payload, checkpoint_path)
        return checkpoint_path

    def _rollout_episode(
        self,
        episode_index: int,
        phase_index: int,
        sumo_seed: int,
        epsilon: float,
        collect_experience: bool,
    ) -> tuple[dict[str, Any], dict[str, list[np.ndarray]]]:
        libsumo.start(self._sumo_cmd(sumo_seed))
        try:
            self._configure_selected_programs()
            vehicle_departs: dict[str, int] = {}
            vehicle_travel_times: list[float] = []
            vehicle_waiting_times: dict[str, float] = {}
            loaded_total = 0
            inserted_total = 0
            arrived_total = 0
            step = 0
            node_transitions: dict[str, list[NodeTransition]] = {node_id: [] for node_id in self.selected_nodes}
            phase_log = {
                "raw_features": [],
                "pressures": [],
                "selector_local_pressure": [],
                "selector_phase_competition": [],
                "selector_spillback_ratio": [],
                "selector_outgoing_occupancy": [],
                "selector_phase_mismatch": [],
                "selector_current_slot": [],
            }
            shared_controller = self.controllers[self.shared_controller_id]

            current_raw, _current_queues = self._capture_raw_node_features()
            current_pressures = self._capture_pressure_vector()
            current_selector = self._capture_control_leverage_snapshot()
            history_window: deque[np.ndarray] = deque([current_raw], maxlen=self.config.history_length)
            if collect_experience:
                phase_log["raw_features"].append(current_raw.copy())
                phase_log["pressures"].append(current_pressures.copy())
                for key, value in current_selector.items():
                    phase_log[f"selector_{key}"].append(value.copy())

            while step < self.config.max_steps:
                states = {
                    node_id: self._build_rl_state(node_id, current_raw).astype(np.float32)
                    for node_id in self.selected_nodes
                }
                node_indices = {node_id: self.node_to_idx[node_id] for node_id in self.selected_nodes}
                actions = {}
                for node_id, state in states.items():
                    actions[node_id] = self._select_action(
                        shared_controller,
                        state,
                        node_indices[node_id],
                        epsilon,
                    )

                interval_done = False
                for _ in range(self.config.decision_interval):
                    for node_id, action in actions.items():
                        self._apply_action(node_id, action)
                    libsumo.simulationStep()
                    step += 1
                    loaded_inc, inserted_inc, arrived_inc = self._track_vehicle_progress(
                        step,
                        vehicle_departs,
                        vehicle_travel_times,
                        vehicle_waiting_times,
                    )
                    loaded_total += loaded_inc
                    inserted_total += inserted_inc
                    arrived_total += arrived_inc
                    if step >= self.config.max_steps or libsumo.simulation.getMinExpectedNumber() <= 0:
                        interval_done = True
                        break

                next_raw, _next_queues = self._capture_raw_node_features()
                next_pressures = self._capture_pressure_vector()
                next_selector = self._capture_control_leverage_snapshot()
                next_history = deque(history_window, maxlen=self.config.history_length)
                next_history.append(next_raw)

                done = interval_done
                if collect_experience:
                    for node_id in self.selected_nodes:
                        node_index = node_indices[node_id]
                        reward = self._pressure_improvement_reward(
                            node_index=node_index,
                            action=actions[node_id],
                            previous_pressures=current_pressures,
                            current_pressures=next_pressures,
                        )
                        node_transitions[node_id].append(
                            NodeTransition(
                                state=states[node_id],
                                node_index=node_index,
                                action=actions[node_id],
                                reward=float(reward),
                                next_state=self._build_rl_state(node_id, next_raw).astype(np.float32),
                                done=done,
                            )
                        )

                current_raw = next_raw
                current_pressures = next_pressures
                current_selector = next_selector
                history_window = next_history
                if collect_experience:
                    phase_log["raw_features"].append(current_raw.copy())
                    phase_log["pressures"].append(current_pressures.copy())
                    for key, value in current_selector.items():
                        phase_log[f"selector_{key}"].append(value.copy())
                if done:
                    break

            if collect_experience:
                for node_id in self.selected_nodes:
                    shared_controller.replay.add_episode(node_transitions[node_id])

            avg_travel_time = float(np.mean(vehicle_travel_times)) if vehicle_travel_times else math.nan
            waiting_time_values = list(vehicle_waiting_times.values())
            avg_waiting_time = float(np.mean(waiting_time_values)) if waiting_time_values else math.nan
            max_waiting_time = float(np.max(waiting_time_values)) if waiting_time_values else math.nan
            metrics = {
                "episode": episode_index,
                "phase": phase_index,
                "average_travel_time": avg_travel_time,
                "max_travel_time": float(np.max(vehicle_travel_times)) if vehicle_travel_times else math.nan,
                "average_waiting_time": avg_waiting_time,
                "max_waiting_time": max_waiting_time,
                "arrived_vehicle_count": len(vehicle_travel_times),
                "demand_vehicle_count": self.total_demand,
                "loaded_vehicle_count": loaded_total,
                "inserted_vehicle_count": inserted_total,
                "pending_vehicle_count": int(libsumo.simulation.getMinExpectedNumber()),
                "inserted_ratio": float(inserted_total / self.total_demand) if self.total_demand else 0.0,
                "arrived_ratio": float(arrived_total / self.total_demand) if self.total_demand else 0.0,
                "epsilon": epsilon,
                "sumo_seed": sumo_seed,
                "selected_nodes": list(self.selected_nodes),
            }
            metrics["composite_score"] = self._composite_eval_score(metrics)
            return metrics, phase_log
        finally:
            libsumo.close()

    def _print_episode_progress(self, metrics: dict[str, Any], mean_loss: float | None) -> None:
        avg_text = f"{metrics['average_travel_time']:.2f}" if not math.isnan(metrics["average_travel_time"]) else "nan"
        avg_wait_text = f"{metrics['average_waiting_time']:.2f}" if not math.isnan(metrics["average_waiting_time"]) else "nan"
        max_wait_text = f"{metrics['max_waiting_time']:.2f}" if not math.isnan(metrics["max_waiting_time"]) else "nan"
        loss_text = f"{mean_loss:.4f}" if mean_loss is not None else "n/a"
        print(
            f"runnerMain phase {metrics['phase']} | ep {metrics['episode']:>3}/{self.config.total_episodes} | "
            f"avg_tt={avg_text}s | avg_wait={avg_wait_text}s | max_wait={max_wait_text}s | inserted={metrics['inserted_ratio']:.3f} | "
            f"arrived={metrics['arrived_vehicle_count']} | epsilon={metrics['epsilon']:.3f} | loss={loss_text}"
        )

    def _print_eval_progress(self, metrics: dict[str, Any]) -> None:
        avg_text = f"{metrics['average_travel_time']:.2f}" if not math.isnan(metrics["average_travel_time"]) else "nan"
        avg_wait_text = f"{metrics['average_waiting_time']:.2f}" if not math.isnan(metrics["average_waiting_time"]) else "nan"
        max_wait_text = f"{metrics['max_waiting_time']:.2f}" if not math.isnan(metrics["max_waiting_time"]) else "nan"
        print(
            f"runnerMain eval  phase {metrics['phase']} | ep {metrics['episode']:>3}/{self.config.total_episodes} | "
            f"avg_tt={avg_text}s | avg_wait={avg_wait_text}s | max_wait={max_wait_text}s | inserted={metrics['inserted_ratio']:.3f} | "
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
                    eval_rollout, _eval_phase_log = self._rollout_episode(
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
                    "average_waiting_time": float(np.mean([item["average_waiting_time"] for item in eval_rollouts])),
                    "max_waiting_time": float(np.mean([item["max_waiting_time"] for item in eval_rollouts])),
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
                    print(f"Saved runnerMain best checkpoint: {best_checkpoint}")
            self._update_training_plots()
            self._write_metrics()

            phase_complete = (
                episode_index % self.config.phase_length == 0
                or episode_index == self.config.total_episodes
            )
            if not phase_complete:
                continue

            selected_before = list(self.selected_nodes)
            print(f"runnerMain phase {phase_index}: phase complete at episode {episode_index}, starting selector pipeline")
            gat_stats = self._train_gat_offline(phase_logs, phase_index)
            selection_details = None
            selected_after = None
            if gat_stats is not None:
                self.frozen_gat_stats = gat_stats
            if phase_index < total_phases:
                print(f"runnerMain phase {phase_index}: computing reselection under budget {self.config.budget}")
                selected_after, selection_details = self._reselect_nodes(
                    phase_logs,
                    gat_stats["mean_attention"] if gat_stats is not None else None,
                    gat_stats["mean_embedding"] if gat_stats is not None else None,
                    phase_index,
                )
                new_assignment = self._remap_controllers(selected_after)
                self.selected_nodes = selected_after
                self.node_controller_assignment = new_assignment
                print(f"runnerMain phase {phase_index}: next selected nodes = {self.selected_nodes}")
                if selection_details is not None:
                    print(
                        f"runnerMain phase {phase_index}: changed {selection_details['changed_node_count']} nodes "
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
            checkpoint_path = self._save_checkpoint(episode_index, phase_index, is_final=episode_index == self.config.total_episodes)
            print(f"Saved runnerMain checkpoint: {checkpoint_path}")
            self._write_metrics()
            phase_logs = []

        if self.phase_history and self.phase_history[-1]["episode_end"] != self.config.total_episodes:
            final_phase = math.ceil(self.config.total_episodes / self.config.phase_length)
            checkpoint_path = self._save_checkpoint(self.config.total_episodes, final_phase, is_final=True)
            print(f"Saved runnerMain final checkpoint: {checkpoint_path}")
        return {
            "episode_metrics": self.episode_metrics,
            "evaluation_metrics": self.eval_metrics,
            "phase_history": self.phase_history,
            "selected_nodes": self.selected_nodes,
            "node_controller_assignment": self.node_controller_assignment,
        }


def resolve_torch_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_arg


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase-wise shared DQN + GAT selector training on the SUMO benchmark.")
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
    parser.add_argument("--shaping-beta", type=float, default=0.25, help="Potential-based reward shaping strength.")
    parser.add_argument("--selector-anchor-ratio", type=float, default=0.6, help="Anchor fraction retained during reselection.")
    parser.add_argument("--selector-exploration-start", type=float, default=0.08, help="Deprecated selector exploration argument; selection is deterministic.")
    parser.add_argument("--selector-exploration-end", type=float, default=0.005, help="Deprecated selector exploration argument; selection is deterministic.")
    parser.add_argument("--train-seed-stride", type=int, default=37, help="Seed delta applied between training episodes.")
    parser.add_argument("--eval-seed-offset", type=int, default=1000, help="Offset added to evaluation seeds.")
    parser.add_argument("--eval-seed-stride", type=int, default=53, help="Seed delta applied between evaluation checkpoints.")
    parser.add_argument("--resume-from", type=str, default=None, help="Checkpoint path to resume from.")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory used for plots and training_metrics.json.")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Directory used for new runnerMain checkpoints.")
    parser.add_argument("--eval-only", action="store_true", help="Load a checkpoint and run one epsilon=0 rollout without training.")
    parser.add_argument(
        "--eval-output-json",
        type=str,
        default="plot3.json",
        help="JSON file used by --eval-only to append a single evaluation result.",
    )
    parser.add_argument(
        "--eval-sumo-seed",
        type=int,
        default=None,
        help="SUMO seed used by --eval-only. Defaults to sumo_seed + eval_seed_offset.",
    )
    parser.add_argument(
        "--eval-label",
        type=str,
        default="runnerMain",
        help="Label stored in the eval-only output JSON.",
    )
    parser.add_argument(
        "--eval-selection-method",
        type=str,
        default=None,
        help="Optional selection-method label stored in the eval-only output JSON.",
    )
    parser.add_argument(
        "--eval-top-k",
        type=int,
        default=None,
        help="For --eval-only, keep only the top-k nodes from the checkpoint selected_nodes order.",
    )
    parser.add_argument(
        "--eval-selected-nodes-file",
        type=str,
        default=None,
        help="Optional JSON file providing an external ordered node selection for --eval-only.",
    )
    parser.add_argument("--device", type=str, default="auto", help="Torch device: auto, cpu, or cuda.")
    parser.add_argument("--seed", type=int, default=7, help="Global random seed.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    global SCENARIO_DIR, SUMOCFG_PATH
    SCENARIO_DIR = resolve_scenario_dir(args.scenario_dir)
    SUMOCFG_PATH = SCENARIO_DIR / "benchmark.sumocfg"
    if not SUMOCFG_PATH.exists():
        raise FileNotFoundError(f"SUMO config not found: {SUMOCFG_PATH}")
    config = MainConfig(
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
        shaping_beta=args.shaping_beta,
        selector_anchor_ratio=args.selector_anchor_ratio,
        selector_exploration_start=args.selector_exploration_start,
        selector_exploration_end=args.selector_exploration_end,
        train_seed_stride=args.train_seed_stride,
        eval_seed_offset=args.eval_seed_offset,
        eval_seed_stride=args.eval_seed_stride,
        device=args.device,
        seed=args.seed,
        output_dir=resolve_path_from_root(args.output_dir) if args.output_dir else OUTPUT_DIR,
        checkpoint_dir=resolve_path_from_root(args.checkpoint_dir) if args.checkpoint_dir else CHECKPOINT_DIR,
    )
    print(f"Using torch device: {resolve_torch_device(config.device)}")
    runner = MethodologyRunner(config)
    resume_path: Path | None = None
    if args.resume_from:
        resume_path = resolve_path_from_root(args.resume_from)
        print(f"Resuming runnerMain from checkpoint: {resume_path}")
        runner.load_checkpoint(resume_path, allow_completed=args.eval_only)
    if args.eval_only:
        if resume_path is None:
            raise RuntimeError("--eval-only requires --resume-from so the trained model can be loaded.")
        eval_output_path = resolve_path_from_root(args.eval_output_json)
        selected_node_order = list(runner.selected_nodes)
        selection_source = "checkpoint selected_nodes order"
        if args.eval_selected_nodes_file:
            selection_path = resolve_path_from_root(args.eval_selected_nodes_file)
            selected_node_order = load_ordered_nodes_from_selection_json(selection_path)
            selection_source = f"selection file {selection_path}"
        if args.eval_top_k is not None:
            if args.eval_top_k < 1:
                raise RuntimeError("--eval-top-k must be at least 1.")
            if args.eval_top_k > len(selected_node_order):
                raise RuntimeError(
                    f"--eval-top-k={args.eval_top_k} exceeds available selection size {len(selected_node_order)}."
                )
            selected_node_order = selected_node_order[: args.eval_top_k]
        if args.eval_selected_nodes_file or args.eval_top_k is not None:
            runner.override_selected_nodes(selected_node_order)
            print(
                f"runnerMain eval-only: using {len(runner.selected_nodes)} nodes from {selection_source} = "
                f"{runner.selected_nodes}"
            )
        runner.run_single_evaluation(
            output_json_path=eval_output_path,
            checkpoint_path=resume_path,
            methodology="runnerMain",
            selection_method=args.eval_selection_method,
            sumo_seed=args.eval_sumo_seed,
            label=args.eval_label,
        )
        return
    runner.run()


if __name__ == "__main__":
    main()
