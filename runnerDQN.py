from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")


ROOT = Path(__file__).resolve().parent
SCENARIO_DIR = ROOT / "sumo_benchmark"
SUMOCFG_PATH = SCENARIO_DIR / "benchmark.sumocfg"
CONTROL_ROLES_PATH = SCENARIO_DIR / "control_roles.json"
OUTPUT_DIR = ROOT / "runnerDQN_outputs"
CHECKPOINT_DIR = ROOT / "runnerDQN_checkpoints"
DLL_DIR_HANDLES: list[Any] = []


def resolve_scenario_dir(raw_path: str) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def ensure_sumo_python_tools() -> None:
    sumo_home = os.environ.get("SUMO_HOME")
    if not sumo_home:
        raise RuntimeError("SUMO_HOME is not set. SUMO is installed, but its Python tools path is required.")
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

try:
    import libsumo  # type: ignore
except ModuleNotFoundError as exc:
    raise RuntimeError(
        "The Python binding for libsumo is not available in the current environment. "
        "Install a libsumo version compatible with your local SUMO binaries."
    ) from exc

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from sumo_fixed_timing import write_fixed_timing_additional


@dataclass
class DqnConfig:
    episodes: int = 500
    max_steps: int = 3600
    decision_interval: int = 10
    min_green_duration: int = 12
    fixed_green_duration: int = 30
    yellow_duration: int = 3
    gamma: float = 0.99
    learning_rate: float = 3.0e-4
    batch_size: int = 32
    replay_capacity: int = 6000
    hidden_size: int = 128
    epsilon_start: float = 1.0
    epsilon_end: float = 0.04
    epsilon_decay_episodes: int = 120
    updates_per_episode: int = 12
    target_sync_interval: int = 10
    eval_interval: int = 5
    eval_episodes: int = 3
    checkpoint_interval: int = 50
    seed: int = 7
    sumo_seed: int = 11
    train_seed_stride: int = 37
    eval_seed_offset: int = 1000
    eval_seed_stride: int = 53
    device: str = "auto"
    sumo_binary: str = "sumo"
    plot_path: Path = OUTPUT_DIR / "average_travel_time.png"
    metrics_path: Path = OUTPUT_DIR / "training_metrics.json"
    checkpoint_dir: Path = CHECKPOINT_DIR


@dataclass
class PhaseGroup:
    incoming_lanes: list[str]
    outgoing_lanes: list[str]


@dataclass
class DecisionSnapshot:
    state: np.ndarray
    total_queue: float
    spillback_lanes: float
    imbalance: float
    mean_speed: float
    score: float


@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class DQN(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, action_dim: int = 2) -> None:
        super().__init__()
        head_hidden_size = max(32, hidden_size // 2)
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(x)
        value = self.value_head(features)
        advantage = self.advantage_head(features)
        return value + advantage - advantage.mean(dim=1, keepdim=True)


class TransitionReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.memory: deque[Transition] = deque(maxlen=capacity)

    def add(self, transition: Transition) -> None:
        self.memory.append(transition)

    def can_sample(self, batch_size: int) -> bool:
        return len(self.memory) >= batch_size

    def sample(self, batch_size: int, device: torch.device) -> dict[str, torch.Tensor]:
        batch = random.sample(self.memory, batch_size)
        return {
            "states": torch.tensor(np.stack([item.state for item in batch]), dtype=torch.float32, device=device),
            "next_states": torch.tensor(np.stack([item.next_state for item in batch]), dtype=torch.float32, device=device),
            "actions": torch.tensor([item.action for item in batch], dtype=torch.long, device=device),
            "rewards": torch.tensor([item.reward for item in batch], dtype=torch.float32, device=device),
            "dones": torch.tensor([float(item.done) for item in batch], dtype=torch.float32, device=device),
        }


class LivePlot:
    def __init__(self, output_path: Path) -> None:
        self.output_path = output_path
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.fig, self.ax = plt.subplots(figsize=(10, 5.5))
        self.train_line, = self.ax.plot([], [], color="tab:blue", linewidth=1.8, label="Train")
        self.eval_line, = self.ax.plot([], [], color="tab:orange", linewidth=2.0, marker="o", label="Eval")
        self.ax.set_title("runnerDQN Average Travel Time vs Episode")
        self.ax.set_xlabel("Episode")
        self.ax.set_ylabel("Average Travel Time (s)")
        self.ax.grid(True, linestyle="--", alpha=0.35)
        self.ax.legend()
        self.fig.tight_layout()

    def update(self, train_metrics: list[dict[str, Any]], eval_metrics: list[dict[str, Any]]) -> None:
        train_x = [item["episode"] for item in train_metrics]
        train_y = [item["average_travel_time"] for item in train_metrics]
        eval_x = [item["episode"] for item in eval_metrics]
        eval_y = [item["average_travel_time"] for item in eval_metrics]
        self.train_line.set_data(train_x, train_y)
        self.eval_line.set_data(eval_x, eval_y)
        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        self.fig.savefig(self.output_path, dpi=150)


class AgentRuntime:
    def __init__(
        self,
        tls_id: str,
        state_dim: int,
        phase_groups: list[PhaseGroup],
        all_incoming_lanes: list[str],
        all_outgoing_lanes: list[str],
        green_states: list[str],
        config: DqnConfig,
    ) -> None:
        self.tls_id = tls_id
        self.phase_groups = phase_groups
        self.all_incoming_lanes = all_incoming_lanes
        self.all_outgoing_lanes = all_outgoing_lanes
        self.green_states = green_states
        self.online_net = DQN(state_dim, config.hidden_size).to(config.device)
        self.target_net = DQN(state_dim, config.hidden_size).to(config.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=config.learning_rate)
        self.replay = TransitionReplayBuffer(config.replay_capacity)
        self.pending_snapshot: DecisionSnapshot | None = None
        self.pending_action: int | None = None
        self.latest_loss: float | None = None
        self.device = torch.device(config.device)

    def reset_episode(self) -> None:
        self.pending_snapshot = None
        self.pending_action = None

    def select_action(self, state: np.ndarray, epsilon: float) -> int:
        if random.random() < epsilon:
            return random.randint(0, 1)
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.online_net(state_tensor)
        return int(torch.argmax(q_values, dim=1).item())


class SumoDqnTrainer:
    def __init__(self, config: DqnConfig, resume_path: Path | None = None) -> None:
        self.config = config
        self.device = torch.device(config.device)
        self.roles = json.loads(CONTROL_ROLES_PATH.read_text(encoding="utf-8"))
        self.controlled_tls = self.roles["drqn_nodes"]
        self.fixed_tls = self.roles["fixed_time_nodes"]
        self.net_path = self._resolve_sumocfg_path("net-file")
        self.route_path = self._resolve_sumocfg_path("route-files")
        self.fixed_timing_additional_path = (
            self.config.plot_path.parent / f"fixed{self.config.fixed_green_duration}_uncontrolled.add.xml"
        )
        write_fixed_timing_additional(
            self.net_path,
            self.fixed_timing_additional_path,
            self.config.fixed_green_duration,
            self.config.yellow_duration,
        )
        self.total_demand = self._count_total_demand(self.route_path)
        self.agent_specs = self._build_agent_specs()
        self.agents = {
            tls_id: AgentRuntime(
                tls_id=tls_id,
                state_dim=spec["state_dim"],
                phase_groups=spec["phase_groups"],
                all_incoming_lanes=spec["all_incoming_lanes"],
                all_outgoing_lanes=spec["all_outgoing_lanes"],
                green_states=spec["green_states"],
                config=self.config,
            )
            for tls_id, spec in self.agent_specs.items()
        }
        self.train_metrics: list[dict[str, Any]] = []
        self.eval_metrics: list[dict[str, Any]] = []
        self.live_plot = LivePlot(self.config.plot_path)
        self.start_episode = 0
        if resume_path is not None:
            self.start_episode = self._load_checkpoint(resume_path)

    def _resolve_sumocfg_path(self, xml_key: str) -> Path:
        import xml.etree.ElementTree as ET

        root = ET.parse(SUMOCFG_PATH).getroot()
        input_node = root.find("input")
        if input_node is None:
            raise RuntimeError(f"Invalid SUMO config: missing input block in {SUMOCFG_PATH}")
        for child in input_node:
            if child.tag == xml_key:
                return (SUMOCFG_PATH.parent / child.attrib["value"]).resolve()
        raise RuntimeError(f"Could not resolve {xml_key} from {SUMOCFG_PATH}")

    @staticmethod
    def _unique_preserve_order(values: list[str]) -> list[str]:
        return list(dict.fromkeys(values))

    @staticmethod
    def _phase_weight(state: str) -> float:
        return sum(1.0 if char == "G" else 0.65 for char in state if char in {"G", "g"})

    @staticmethod
    def _phase_overlap(a: str, b: str) -> float:
        a_positions = {idx for idx, char in enumerate(a) if char in {"G", "g"}}
        b_positions = {idx for idx, char in enumerate(b) if char in {"G", "g"}}
        if not a_positions or not b_positions:
            return 0.0
        return len(a_positions & b_positions) / len(a_positions | b_positions)

    def _select_major_green_states(self, phase_states: list[str]) -> list[str]:
        green_states = [state for state in phase_states if "y" not in state.lower() and any(char in {"G", "g"} for char in state)]
        if len(green_states) <= 2:
            return green_states
        first = max(green_states, key=self._phase_weight)
        remainder = [state for state in green_states if state != first]
        second = max(remainder, key=lambda state: self._phase_weight(state) - 1.2 * self._phase_overlap(first, state))
        return [first, second]

    @staticmethod
    def _build_transition_state(from_state: str, to_state: str) -> str:
        chars: list[str] = []
        for from_char, to_char in zip(from_state, to_state):
            if from_char in {"G", "g"} and to_char in {"r", "R"}:
                chars.append("y")
            elif from_char in {"G", "g"} and to_char in {"G", "g"}:
                chars.append("G" if from_char == "G" or to_char == "G" else "g")
            elif from_char in {"r", "R"} and to_char in {"G", "g"}:
                chars.append("r")
            else:
                chars.append("r")
        return "".join(chars)

    def _build_phase_groups(self, tls_id: str, green_states: list[str]) -> tuple[list[PhaseGroup], list[str], list[str]]:
        controlled_links = libsumo.trafficlight.getControlledLinks(tls_id)
        phase_groups: list[PhaseGroup] = []
        all_incoming: list[str] = []
        all_outgoing: list[str] = []

        for state in green_states:
            incoming_lanes: list[str] = []
            outgoing_lanes: list[str] = []
            for link_index, link_set in enumerate(controlled_links):
                if state[link_index] not in {"G", "g"}:
                    continue
                for link in link_set:
                    incoming_lanes.append(link[0])
                    outgoing_lanes.append(link[1])
            incoming_lanes = self._unique_preserve_order(incoming_lanes)
            outgoing_lanes = self._unique_preserve_order(outgoing_lanes)
            all_incoming.extend(incoming_lanes)
            all_outgoing.extend(outgoing_lanes)
            phase_groups.append(PhaseGroup(incoming_lanes=incoming_lanes, outgoing_lanes=outgoing_lanes))

        return phase_groups, self._unique_preserve_order(all_incoming), self._unique_preserve_order(all_outgoing)

    def _build_agent_specs(self) -> dict[str, dict[str, Any]]:
        specs: dict[str, dict[str, Any]] = {}
        libsumo.start(self._sumo_cmd(self.config.sumo_seed))
        try:
            for tls_id in self.controlled_tls:
                logic = libsumo.trafficlight.getAllProgramLogics(tls_id)[0]
                phase_states = [phase.state for phase in logic.getPhases()]
                green_states = self._select_major_green_states(phase_states)
                phase_groups, all_incoming, all_outgoing = self._build_phase_groups(tls_id, green_states)
                state_dim = len(phase_groups) * 5 + len(phase_groups) + 3
                specs[tls_id] = {
                    "green_states": green_states,
                    "phase_groups": phase_groups,
                    "all_incoming_lanes": all_incoming,
                    "all_outgoing_lanes": all_outgoing,
                    "state_dim": state_dim,
                }
        finally:
            libsumo.close()
        return specs

    @staticmethod
    def _count_total_demand(route_path: Path) -> int:
        import xml.etree.ElementTree as ET

        root = ET.parse(route_path).getroot()
        return sum(1 for child in root if child.tag == "vehicle")

    def _make_simplified_logic(self, tls_id: str, green_states: list[str]) -> Any:
        green_a, green_b = green_states
        yellow_ab = self._build_transition_state(green_a, green_b)
        yellow_ba = self._build_transition_state(green_b, green_a)
        phases = [
            libsumo.trafficlight.Phase(self.config.min_green_duration, green_a),
            libsumo.trafficlight.Phase(self.config.yellow_duration, yellow_ab),
            libsumo.trafficlight.Phase(self.config.min_green_duration, green_b),
            libsumo.trafficlight.Phase(self.config.yellow_duration, yellow_ba),
        ]
        return libsumo.trafficlight.Logic(f"dqn_{tls_id}", 0, 0, phases)

    def _configure_controlled_programs(self) -> None:
        for tls_id, spec in self.agent_specs.items():
            logic = self._make_simplified_logic(tls_id, spec["green_states"])
            libsumo.trafficlight.setProgramLogic(tls_id, logic)
            libsumo.trafficlight.setProgram(tls_id, logic.programID)
            libsumo.trafficlight.setPhase(tls_id, 0)
            libsumo.trafficlight.setPhaseDuration(tls_id, self.config.min_green_duration)

    @staticmethod
    def _current_green_slot(current_phase: int) -> int:
        return 0 if current_phase in {0, 1} else 1

    def _capture_snapshot(self, agent: AgentRuntime) -> DecisionSnapshot:
        phase_features: list[float] = []
        phase_pressures: list[float] = []
        total_queue = 0.0
        spillback_lanes = 0.0
        speed_terms: list[float] = []

        for phase_group in agent.phase_groups:
            in_queue = 0.0
            in_volume = 0.0
            in_speed_ratios: list[float] = []
            out_queue = 0.0
            out_occupancies: list[float] = []

            for lane_id in phase_group.incoming_lanes:
                in_queue += float(libsumo.lane.getLastStepHaltingNumber(lane_id))
                in_volume += float(libsumo.lane.getLastStepVehicleNumber(lane_id))
                mean_speed = float(libsumo.lane.getLastStepMeanSpeed(lane_id))
                max_speed = max(1.0, float(libsumo.lane.getMaxSpeed(lane_id)))
                in_speed_ratios.append(mean_speed / max_speed)

            for lane_id in phase_group.outgoing_lanes:
                out_queue += float(libsumo.lane.getLastStepHaltingNumber(lane_id))
                occupancy = float(libsumo.lane.getLastStepOccupancy(lane_id)) / 100.0
                out_occupancies.append(occupancy)
                if occupancy >= 0.78:
                    spillback_lanes += 1.0

            mean_speed_ratio = float(np.mean(in_speed_ratios)) if in_speed_ratios else 1.0
            out_occ_mean = float(np.mean(out_occupancies)) if out_occupancies else 0.0
            pressure = in_queue - 0.55 * out_queue

            total_queue += in_queue
            speed_terms.append(mean_speed_ratio)
            phase_pressures.append(pressure)
            phase_features.extend(
                [
                    min(2.0, in_queue / max(1.0, len(phase_group.incoming_lanes) * 5.0)),
                    min(2.0, in_volume / max(1.0, len(phase_group.incoming_lanes) * 7.0)),
                    mean_speed_ratio,
                    out_occ_mean,
                    math.tanh(pressure / 12.0),
                ]
            )

        current_phase = int(libsumo.trafficlight.getPhase(agent.tls_id))
        green_slot = self._current_green_slot(current_phase)
        phase_one_hot = [0.0, 0.0]
        phase_one_hot[green_slot] = 1.0
        phase_age = min(
            1.0,
            float(libsumo.trafficlight.getSpentDuration(agent.tls_id))
            / max(1.0, float(libsumo.trafficlight.getPhaseDuration(agent.tls_id))),
        )
        mean_speed = float(np.mean(speed_terms)) if speed_terms else 1.0
        imbalance = abs(phase_pressures[0] - phase_pressures[1]) if len(phase_pressures) == 2 else 0.0
        total_queue_norm = min(2.0, total_queue / max(1.0, len(agent.all_incoming_lanes) * 5.0))
        spillback_ratio = spillback_lanes / max(1.0, len(agent.all_outgoing_lanes))
        state = np.array(phase_features + phase_one_hot + [phase_age, total_queue_norm, spillback_ratio], dtype=np.float32)
        score = 1.9 * total_queue + 18.0 * spillback_lanes + 4.5 * imbalance - 5.0 * mean_speed
        return DecisionSnapshot(
            state=state,
            total_queue=total_queue,
            spillback_lanes=spillback_lanes,
            imbalance=imbalance,
            mean_speed=mean_speed,
            score=score,
        )

    def _reward_from_snapshots(self, previous: DecisionSnapshot, current: DecisionSnapshot, action: int) -> float:
        score_improvement = previous.score - current.score
        queue_bonus = 0.75 * (previous.total_queue - current.total_queue)
        speed_bonus = 1.25 * (current.mean_speed - previous.mean_speed)
        switch_penalty = 0.20 if action == 1 else 0.0
        reward = score_improvement + queue_bonus + speed_bonus - switch_penalty
        return float(np.clip(reward, -15.0, 15.0))

    def _heuristic_action(self, snapshot: DecisionSnapshot) -> int:
        phase_pressures = [float(snapshot.state[4]), float(snapshot.state[9])]
        current_slot = 0 if snapshot.state[10] > 0.5 else 1
        other_slot = 1 - current_slot
        phase_age = float(snapshot.state[12])
        if phase_pressures[other_slot] > phase_pressures[current_slot] + 0.18 and phase_age > 0.35:
            return 1
        return 0

    def _choose_action(self, agent: AgentRuntime, snapshot: DecisionSnapshot, epsilon: float, policy_mode: str) -> int:
        if policy_mode == "random":
            return random.randint(0, 1)
        return agent.select_action(snapshot.state, epsilon)

    def _apply_action(self, tls_id: str, action: int) -> None:
        current_phase = int(libsumo.trafficlight.getPhase(tls_id))
        if current_phase not in {0, 2}:
            return
        spent = float(libsumo.trafficlight.getSpentDuration(tls_id))
        if action == 0:
            remaining = max(
                1.0,
                float(self.config.decision_interval),
                float(self.config.min_green_duration) - spent + 1.0,
            )
            libsumo.trafficlight.setPhaseDuration(tls_id, remaining)
            return
        if spent < self.config.min_green_duration:
            remaining = max(1.0, float(self.config.min_green_duration) - spent)
            libsumo.trafficlight.setPhaseDuration(tls_id, remaining)
            return
        if current_phase == 0:
            libsumo.trafficlight.setPhase(tls_id, 1)
        elif current_phase == 2:
            libsumo.trafficlight.setPhase(tls_id, 3)

    def _train_agent(self, agent: AgentRuntime) -> float | None:
        if not agent.replay.can_sample(self.config.batch_size):
            return None
        batch = agent.replay.sample(self.config.batch_size, self.device)
        q_values = agent.online_net(batch["states"])
        chosen_q = q_values.gather(1, batch["actions"].unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_online_q = agent.online_net(batch["next_states"])
            next_actions = torch.argmax(next_online_q, dim=1, keepdim=True)
            next_target_q = agent.target_net(batch["next_states"])
            target_q = next_target_q.gather(1, next_actions).squeeze(1)
            td_target = batch["rewards"] + self.config.gamma * (1.0 - batch["dones"]) * target_q
        loss = nn.functional.smooth_l1_loss(chosen_q, td_target)
        agent.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(agent.online_net.parameters(), max_norm=5.0)
        agent.optimizer.step()
        agent.latest_loss = float(loss.item())
        return agent.latest_loss

    def _sync_target_networks(self) -> None:
        for agent in self.agents.values():
            agent.target_net.load_state_dict(agent.online_net.state_dict())

    def _epsilon(self, episode_index: int) -> float:
        if episode_index >= self.config.epsilon_decay_episodes:
            return self.config.epsilon_end
        progress = episode_index / max(1, self.config.epsilon_decay_episodes)
        return self.config.epsilon_start + progress * (self.config.epsilon_end - self.config.epsilon_start)

    def _training_sumo_seed(self, episode_index: int) -> int:
        return self.config.sumo_seed + max(0, episode_index - 1) * self.config.train_seed_stride

    def _evaluation_sumo_seed(self, episode_index: int, rollout_index: int) -> int:
        return (
            self.config.sumo_seed
            + self.config.eval_seed_offset
            + max(0, episode_index - 1) * self.config.eval_seed_stride
            + rollout_index
        )

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

    def _rollout(
        self,
        epsilon: float,
        policy_mode: str,
        collect_experience: bool,
        episode_index: int,
        sumo_seed: int,
    ) -> dict[str, Any]:
        libsumo.start(self._sumo_cmd(sumo_seed))
        try:
            self._configure_controlled_programs()
            for agent in self.agents.values():
                agent.reset_episode()

            vehicle_departs: dict[str, int] = {}
            vehicle_travel_times: list[float] = []
            loaded_total = 0
            inserted_total = 0
            arrived_total = 0
            step = 0

            while step < self.config.max_steps:
                if step % self.config.decision_interval == 0:
                    for agent in self.agents.values():
                        snapshot = self._capture_snapshot(agent)
                        if collect_experience and agent.pending_snapshot is not None and agent.pending_action is not None:
                            reward = self._reward_from_snapshots(agent.pending_snapshot, snapshot, agent.pending_action)
                            agent.replay.add(
                                Transition(
                                    state=agent.pending_snapshot.state.astype(np.float32),
                                    action=int(agent.pending_action),
                                    reward=reward,
                                    next_state=snapshot.state.astype(np.float32),
                                    done=False,
                                )
                            )
                        action = self._choose_action(agent, snapshot, epsilon, policy_mode)
                        agent.pending_snapshot = snapshot
                        agent.pending_action = action

                interval_done = False
                for _ in range(self.config.decision_interval):
                    for agent in self.agents.values():
                        if agent.pending_action is not None:
                            self._apply_action(agent.tls_id, agent.pending_action)
                    libsumo.simulationStep()
                    step += 1

                    loaded_ids = libsumo.simulation.getLoadedIDList()
                    departed_ids = libsumo.simulation.getDepartedIDList()
                    arrived_ids = libsumo.simulation.getArrivedIDList()
                    loaded_total += len(loaded_ids)
                    inserted_total += len(departed_ids)
                    arrived_total += len(arrived_ids)

                    for veh_id in departed_ids:
                        vehicle_departs[veh_id] = step
                    for veh_id in arrived_ids:
                        depart_step = vehicle_departs.pop(veh_id, None)
                        if depart_step is not None:
                            vehicle_travel_times.append(step - depart_step)

                    if step >= self.config.max_steps or libsumo.simulation.getMinExpectedNumber() <= 0:
                        interval_done = True
                        break

                if interval_done:
                    break

            if collect_experience:
                for agent in self.agents.values():
                    if agent.pending_snapshot is None or agent.pending_action is None:
                        continue
                    final_snapshot = self._capture_snapshot(agent)
                    reward = self._reward_from_snapshots(agent.pending_snapshot, final_snapshot, agent.pending_action)
                    agent.replay.add(
                        Transition(
                            state=agent.pending_snapshot.state.astype(np.float32),
                            action=int(agent.pending_action),
                            reward=reward,
                            next_state=final_snapshot.state.astype(np.float32),
                            done=True,
                        )
                    )

            average_travel_time = float(np.mean(vehicle_travel_times)) if vehicle_travel_times else math.nan
            return {
                "episode": episode_index,
            "policy_mode": policy_mode,
                "average_travel_time": average_travel_time,
                "max_travel_time": float(np.max(vehicle_travel_times)) if vehicle_travel_times else math.nan,
                "arrived_vehicle_count": len(vehicle_travel_times),
                "demand_vehicle_count": self.total_demand,
                "loaded_vehicle_count": loaded_total,
                "inserted_vehicle_count": inserted_total,
                "pending_vehicle_count": int(libsumo.simulation.getMinExpectedNumber()),
                "inserted_ratio": float(inserted_total / self.total_demand) if self.total_demand else 0.0,
                "arrived_ratio": float(arrived_total / self.total_demand) if self.total_demand else 0.0,
                "epsilon": epsilon,
                "sumo_seed": sumo_seed,
            }
        finally:
            libsumo.close()

    def _write_metrics(self) -> None:
        payload = {
            "config": {
                key: (str(value) if isinstance(value, Path) else value)
                for key, value in asdict(self.config).items()
            },
            "training": self.train_metrics,
            "evaluation": self.eval_metrics,
        }
        self.config.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        self.config.metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _save_checkpoint(self, episode: int, is_final: bool = False) -> Path:
        self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        filename = "final_checkpoint.pt" if is_final else f"checkpoint_ep{episode:03d}.pt"
        checkpoint_path = self.config.checkpoint_dir / filename
        payload = {
            "episode": episode,
            "is_final": is_final,
            "config": {
                key: (str(value) if isinstance(value, Path) else value)
                for key, value in asdict(self.config).items()
            },
            "controlled_tls": self.controlled_tls,
            "agent_specs": {
                tls_id: {
                    "green_states": spec["green_states"],
                    "all_incoming_lanes": spec["all_incoming_lanes"],
                    "all_outgoing_lanes": spec["all_outgoing_lanes"],
                    "state_dim": spec["state_dim"],
                }
                for tls_id, spec in self.agent_specs.items()
            },
            "agents": {
                tls_id: {
                    "online_state_dict": agent.online_net.state_dict(),
                    "target_state_dict": agent.target_net.state_dict(),
                    "optimizer_state_dict": agent.optimizer.state_dict(),
                    "latest_loss": agent.latest_loss,
                }
                for tls_id, agent in self.agents.items()
            },
            "training_metrics": self.train_metrics,
            "evaluation_metrics": self.eval_metrics,
        }
        torch.save(payload, checkpoint_path)
        return checkpoint_path

    def _load_checkpoint(self, checkpoint_path: Path) -> int:
        payload = torch.load(checkpoint_path, map_location=self.device)
        for tls_id, agent_payload in payload["agents"].items():
            if tls_id not in self.agents:
                continue
            agent = self.agents[tls_id]
            agent.online_net.load_state_dict(agent_payload["online_state_dict"])
            agent.target_net.load_state_dict(agent_payload["target_state_dict"])
            agent.optimizer.load_state_dict(agent_payload["optimizer_state_dict"])
            agent.latest_loss = agent_payload.get("latest_loss")
        self.train_metrics = payload.get("training_metrics", [])
        self.eval_metrics = payload.get("evaluation_metrics", [])
        self.live_plot.update(self.train_metrics, self.eval_metrics)
        return int(payload.get("episode", 0))

    def _print_train_progress(self, metrics: dict[str, Any], mean_loss: float | None) -> None:
        avg_text = f"{metrics['average_travel_time']:.2f}" if not math.isnan(metrics["average_travel_time"]) else "nan"
        loss_text = f"{mean_loss:.4f}" if mean_loss is not None else "n/a"
        print(
            f"DQN Train {metrics['episode']:>3}/{self.config.episodes} | "
            f"avg_tt={avg_text}s | inserted={metrics['inserted_ratio']:.3f} | "
            f"arrived={metrics['arrived_vehicle_count']} | epsilon={metrics['epsilon']:.3f} | loss={loss_text}"
        )

    def _print_eval_progress(self, metrics: dict[str, Any]) -> None:
        avg_text = f"{metrics['average_travel_time']:.2f}" if not math.isnan(metrics["average_travel_time"]) else "nan"
        print(
            f"DQN Eval  {metrics['episode']:>3}/{self.config.episodes} | "
            f"avg_tt={avg_text}s | inserted={metrics['inserted_ratio']:.3f} | "
            f"arrived={metrics['arrived_vehicle_count']}"
        )

    def train(self) -> dict[str, list[dict[str, Any]]]:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        if self.start_episode == 0:
            self._sync_target_networks()

        for episode_index in range(self.start_episode, self.config.episodes):
            epsilon = self._epsilon(episode_index)
            training_sumo_seed = self._training_sumo_seed(episode_index + 1)
            train_metrics = self._rollout(
                epsilon=epsilon,
                policy_mode="dqn",
                collect_experience=True,
                episode_index=episode_index + 1,
                sumo_seed=training_sumo_seed,
            )

            losses: list[float] = []
            for _ in range(self.config.updates_per_episode):
                for agent in self.agents.values():
                    loss = self._train_agent(agent)
                    if loss is not None:
                        losses.append(loss)
            if (episode_index + 1) % self.config.target_sync_interval == 0:
                self._sync_target_networks()

            train_metrics["mean_loss"] = float(np.mean(losses)) if losses else None
            self.train_metrics.append(train_metrics)
            self._print_train_progress(train_metrics, train_metrics["mean_loss"])

            if (episode_index + 1) % self.config.eval_interval == 0:
                eval_seeds = [
                    self._evaluation_sumo_seed(episode_index + 1, rollout_index)
                    for rollout_index in range(self.config.eval_episodes)
                ]
                eval_rollouts = [
                    self._rollout(
                        epsilon=0.0,
                        policy_mode="dqn",
                        collect_experience=False,
                        episode_index=episode_index + 1,
                        sumo_seed=eval_seed,
                    )
                    for eval_seed in eval_seeds
                ]
                eval_metrics = {
                    "episode": episode_index + 1,
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
                    "eval_seeds": eval_seeds,
                    "average_travel_time_std": float(np.std([item["average_travel_time"] for item in eval_rollouts])),
                }
                self.eval_metrics.append(eval_metrics)
                self._print_eval_progress(eval_metrics)

            self.live_plot.update(self.train_metrics, self.eval_metrics)
            self._write_metrics()
            if (episode_index + 1) % self.config.checkpoint_interval == 0:
                checkpoint_path = self._save_checkpoint(episode_index + 1)
                print(f"Saved DQN checkpoint: {checkpoint_path}")

        final_checkpoint_path = self._save_checkpoint(self.config.episodes, is_final=True)
        print(f"Saved DQN final checkpoint: {final_checkpoint_path}")
        return {"training": self.train_metrics, "evaluation": self.eval_metrics}


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
    defaults = DqnConfig()
    parser = argparse.ArgumentParser(description="Train independent DQNs on the 10 critical SUMO intersections using libsumo.")
    parser.add_argument("--scenario-dir", type=str, default="sumo_benchmark", help="Scenario directory containing benchmark.sumocfg.")
    parser.add_argument("--control-roles", type=str, default=None, help="Optional path to a control_roles.json file for the chosen scenario.")
    parser.add_argument("--episodes", type=int, default=defaults.episodes, help="Number of training episodes.")
    parser.add_argument("--max-steps", type=int, default=defaults.max_steps, help="Maximum simulation steps per episode.")
    parser.add_argument("--decision-interval", type=int, default=defaults.decision_interval, help="Seconds between DQN decisions.")
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size, help="Replay batch size per update.")
    parser.add_argument("--updates-per-episode", type=int, default=defaults.updates_per_episode, help="Gradient updates per episode.")
    parser.add_argument("--eval-interval", type=int, default=defaults.eval_interval, help="Run one evaluation every N training episodes.")
    parser.add_argument("--eval-episodes", type=int, default=defaults.eval_episodes, help="Evaluation rollouts per checkpoint.")
    parser.add_argument("--checkpoint-interval", type=int, default=defaults.checkpoint_interval, help="Save a .pt checkpoint every N episodes.")
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate, help="Optimizer learning rate.")
    parser.add_argument("--epsilon-end", type=float, default=defaults.epsilon_end, help="Final epsilon after exploration decay.")
    parser.add_argument("--epsilon-decay-episodes", type=int, default=defaults.epsilon_decay_episodes, help="Episodes used to decay epsilon from start to end.")
    parser.add_argument("--target-sync-interval", type=int, default=defaults.target_sync_interval, help="Episodes between target-network syncs.")
    parser.add_argument("--train-seed-stride", type=int, default=defaults.train_seed_stride, help="Seed delta applied between training episodes.")
    parser.add_argument("--eval-seed-offset", type=int, default=defaults.eval_seed_offset, help="Offset added to evaluation seeds.")
    parser.add_argument("--eval-seed-stride", type=int, default=defaults.eval_seed_stride, help="Seed delta applied between evaluation checkpoints.")
    parser.add_argument("--device", type=str, default="auto", help="Torch device: auto, cpu, or cuda.")
    parser.add_argument("--seed", type=int, default=defaults.seed, help="Random seed.")
    parser.add_argument("--resume-from", type=Path, default=None, help="Resume training from a DQN checkpoint path.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    global SCENARIO_DIR, SUMOCFG_PATH, CONTROL_ROLES_PATH
    SCENARIO_DIR = resolve_scenario_dir(args.scenario_dir)
    SUMOCFG_PATH = SCENARIO_DIR / "benchmark.sumocfg"
    CONTROL_ROLES_PATH = Path(args.control_roles).resolve() if args.control_roles else SCENARIO_DIR / "control_roles.json"
    if not SUMOCFG_PATH.exists():
        raise FileNotFoundError(f"SUMO config not found: {SUMOCFG_PATH}")
    if not CONTROL_ROLES_PATH.exists():
        raise FileNotFoundError(f"Control roles file not found: {CONTROL_ROLES_PATH}")
    config = DqnConfig(
        episodes=args.episodes,
        max_steps=args.max_steps,
        decision_interval=args.decision_interval,
        batch_size=args.batch_size,
        updates_per_episode=args.updates_per_episode,
        eval_interval=args.eval_interval,
        eval_episodes=args.eval_episodes,
        checkpoint_interval=args.checkpoint_interval,
        learning_rate=args.learning_rate,
        epsilon_end=args.epsilon_end,
        epsilon_decay_episodes=args.epsilon_decay_episodes,
        target_sync_interval=args.target_sync_interval,
        train_seed_stride=args.train_seed_stride,
        eval_seed_offset=args.eval_seed_offset,
        eval_seed_stride=args.eval_seed_stride,
        device=resolve_torch_device(args.device),
        seed=args.seed,
    )
    set_global_seed(config.seed)
    print(f"Using torch device: {config.device}")
    trainer = SumoDqnTrainer(config, resume_path=args.resume_from)
    trainer.train()


if __name__ == "__main__":
    main()
