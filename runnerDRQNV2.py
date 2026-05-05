from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import runner as base


ROOT = Path(__file__).resolve().parent
DEFAULT_SCENARIO_DIR = ROOT / "sumo_benchmark"
DEFAULT_OUTPUT_DIR = ROOT / "runnerDRQNV2_outputs"
DEFAULT_CHECKPOINT_DIR = ROOT / "runnerDRQNV2_checkpoints"


def resolve_path_from_root(raw_path: str) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


class DrqnV2Trainer(base.SumoDrqnTrainer):
    def _reward_from_snapshots(
        self,
        previous: base.DecisionSnapshot,
        current: base.DecisionSnapshot,
        action: int,
    ) -> float:
        # Intentionally rewards congestion growth and unnecessary switching.
        worse_score = current.score - previous.score
        queue_regression = 0.35 * (current.total_queue - previous.total_queue)
        speed_regression = 0.75 * (previous.mean_speed - current.mean_speed)
        switch_bonus = 0.25 if action == 1 else 0.0
        reward = worse_score + queue_regression + speed_regression + switch_bonus
        return float(np.clip(reward, -15.0, 15.0))


def build_arg_parser() -> argparse.ArgumentParser:
    defaults = base.TrainingConfig(
        episodes=300,
        learning_rate=1.5e-3,
        batch_size=8,
        replay_capacity=800,
        sequence_length=2,
        hidden_size=32,
        epsilon_end=0.35,
        epsilon_decay_episodes=300,
        target_tau=0.05,
        plot_path=DEFAULT_OUTPUT_DIR / "average_travel_time.png",
        metrics_path=DEFAULT_OUTPUT_DIR / "training_metrics.json",
        checkpoint_dir=DEFAULT_CHECKPOINT_DIR,
    )
    parser = argparse.ArgumentParser(
        description="Train the DRQN v2 baseline for SUMO traffic signal control."
    )
    parser.add_argument("--scenario-dir", type=str, default="sumo_benchmark", help="Scenario directory containing benchmark.sumocfg.")
    parser.add_argument("--control-roles", type=str, default=None, help="Optional path to control_roles.json for the chosen scenario.")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Directory used for plots, metrics, and generated fixed-time XML.")
    parser.add_argument("--checkpoint-dir", type=str, default=str(DEFAULT_CHECKPOINT_DIR), help="Directory used for checkpoints.")
    parser.add_argument("--episodes", type=int, default=defaults.episodes, help="Number of training episodes.")
    parser.add_argument("--max-steps", type=int, default=defaults.max_steps, help="Maximum simulation steps per episode.")
    parser.add_argument("--decision-interval", type=int, default=defaults.decision_interval, help="Seconds between DRQN decisions.")
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size, help="Sequence batch size per episode update.")
    parser.add_argument("--sequence-length", type=int, default=defaults.sequence_length, help="Replay sequence length.")
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate, help="Optimizer learning rate.")
    parser.add_argument("--target-tau", type=float, default=defaults.target_tau, help="Soft-update rate for target networks.")
    parser.add_argument("--device", type=str, default="auto", help="Torch device: auto, cpu, or cuda.")
    parser.add_argument("--seed", type=int, default=defaults.seed, help="Global random seed.")
    parser.add_argument("--gamma", type=float, default=defaults.gamma, help="Discount factor.")
    parser.add_argument("--hidden-size", type=int, default=defaults.hidden_size, help="DRQN hidden width.")
    parser.add_argument("--epsilon-start", type=float, default=defaults.epsilon_start, help="Initial epsilon.")
    parser.add_argument("--epsilon-end", type=float, default=defaults.epsilon_end, help="Final epsilon.")
    parser.add_argument("--epsilon-decay-episodes", type=int, default=defaults.epsilon_decay_episodes, help="Episodes used for epsilon decay.")
    parser.add_argument("--eval-interval", type=int, default=defaults.eval_interval, help="Evaluate every N episodes.")
    parser.add_argument("--eval-episodes", type=int, default=defaults.eval_episodes, help="Evaluation rollouts per evaluation point.")
    parser.add_argument("--checkpoint-interval", type=int, default=defaults.checkpoint_interval, help="Save checkpoint every N episodes.")
    parser.add_argument("--sumo-seed", type=int, default=defaults.sumo_seed, help="Base SUMO seed.")
    parser.add_argument("--min-green-duration", type=int, default=defaults.min_green_duration, help="Minimum green duration for controlled nodes.")
    parser.add_argument("--fixed-green-duration", type=int, default=defaults.fixed_green_duration, help="Fixed-time green duration for uncontrolled nodes.")
    parser.add_argument("--yellow-duration", type=int, default=defaults.yellow_duration, help="Yellow duration for controlled and fixed programs.")
    parser.add_argument("--sumo-binary", type=str, default=defaults.sumo_binary, help="SUMO binary to use.")
    parser.add_argument("--resume-from", type=str, default=None, help="Resume training from a DRQN checkpoint path.")
    parser.add_argument("--eval-only", action="store_true", help="Load a DRQN checkpoint and run one epsilon=0 rollout without training.")
    parser.add_argument("--eval-output-json", type=str, default="plot3.json", help="JSON file used by --eval-only to append a single evaluation result.")
    parser.add_argument("--eval-sumo-seed", type=int, default=None, help="SUMO seed used by --eval-only. Defaults to sumo_seed + 1000.")
    parser.add_argument("--eval-label", type=str, default="DRQN v2", help="Label stored in the eval-only output JSON.")
    parser.add_argument("--eval-selection-method", type=str, default="CRRank + DRQN v2", help="Selection-method label stored in the eval-only output JSON.")
    parser.add_argument("--eval-top-k", type=int, default=None, help="For --eval-only, keep only the top-k nodes from the current control-roles / checkpoint order.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    scenario_dir = base.resolve_scenario_dir(args.scenario_dir)
    sumocfg_path = scenario_dir / "benchmark.sumocfg"
    control_roles_path = resolve_path_from_root(args.control_roles) if args.control_roles else scenario_dir / "control_roles.json"
    output_dir = resolve_path_from_root(args.output_dir)
    checkpoint_dir = resolve_path_from_root(args.checkpoint_dir)
    resume_path = resolve_path_from_root(args.resume_from) if args.resume_from is not None else None

    if not sumocfg_path.exists():
        raise FileNotFoundError(f"SUMO config not found: {sumocfg_path}")
    if not control_roles_path.exists():
        raise FileNotFoundError(f"Control roles file not found: {control_roles_path}")
    if resume_path is not None and not resume_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {resume_path}")

    base.SCENARIO_DIR = scenario_dir
    base.SUMOCFG_PATH = sumocfg_path
    base.CONTROL_ROLES_PATH = control_roles_path
    base.OUTPUT_DIR = output_dir
    base.CHECKPOINT_DIR = checkpoint_dir

    checkpoint_config: dict[str, object] | None = None
    if resume_path is not None:
        checkpoint_config = base.load_checkpoint_config(resume_path)

    config = base.TrainingConfig(
        episodes=args.episodes,
        max_steps=args.max_steps,
        decision_interval=args.decision_interval,
        min_green_duration=args.min_green_duration,
        fixed_green_duration=args.fixed_green_duration,
        yellow_duration=args.yellow_duration,
        gamma=args.gamma,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        replay_capacity=800,
        sequence_length=args.sequence_length,
        hidden_size=int(checkpoint_config.get("hidden_size", args.hidden_size)) if checkpoint_config else args.hidden_size,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay_episodes=args.epsilon_decay_episodes,
        target_tau=args.target_tau,
        eval_interval=args.eval_interval,
        eval_episodes=args.eval_episodes,
        checkpoint_interval=args.checkpoint_interval,
        seed=args.seed,
        sumo_seed=args.sumo_seed,
        device=base.resolve_torch_device(args.device),
        sumo_binary=args.sumo_binary,
        plot_path=output_dir / "average_travel_time.png",
        metrics_path=output_dir / "training_metrics.json",
        checkpoint_dir=checkpoint_dir,
    )

    base.set_global_seed(config.seed)
    print(f"Using torch device: {config.device}")
    print(f"runnerDRQNV2 scenario: {scenario_dir}")
    print(f"runnerDRQNV2 outputs: {output_dir}")
    print(f"runnerDRQNV2 checkpoints: {checkpoint_dir}")

    trainer = DrqnV2Trainer(config, resume_path=resume_path)
    if args.eval_only:
        if resume_path is None:
            raise RuntimeError("--eval-only requires --resume-from so the trained DRQN weights can be loaded.")
        eval_output_path = resolve_path_from_root(args.eval_output_json)
        if args.eval_top_k is not None:
            if args.eval_top_k < 1:
                raise RuntimeError("--eval-top-k must be at least 1.")
            if args.eval_top_k > len(trainer.controlled_tls):
                raise RuntimeError(
                    f"--eval-top-k={args.eval_top_k} exceeds available controlled TLS count {len(trainer.controlled_tls)}."
                )
            trainer.override_controlled_tls(trainer.controlled_tls[: args.eval_top_k])
            print(f"DRQN v2 eval-only: using top-{args.eval_top_k} nodes = {trainer.controlled_tls}")
        trainer.run_single_evaluation(
            output_json_path=eval_output_path,
            checkpoint_path=resume_path,
            methodology="DRQN v2",
            selection_method=args.eval_selection_method,
            sumo_seed=args.eval_sumo_seed,
            label=args.eval_label,
        )
        return
    trainer.train()


if __name__ == "__main__":
    main()
