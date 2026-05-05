from __future__ import annotations

import argparse
from pathlib import Path

import runnerDQN as base


ROOT = Path(__file__).resolve().parent
DEFAULT_SCENARIO_DIR = ROOT / "sumo_benchmark"
DEFAULT_OUTPUT_DIR = ROOT / "runnerDQN20_outputs"
DEFAULT_CHECKPOINT_DIR = ROOT / "runnerDQN20_checkpoints"


def resolve_path_from_root(raw_path: str) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def build_arg_parser() -> argparse.ArgumentParser:
    defaults = base.DqnConfig(
        episodes=300,
        plot_path=DEFAULT_OUTPUT_DIR / "average_travel_time.png",
        metrics_path=DEFAULT_OUTPUT_DIR / "training_metrics.json",
        checkpoint_dir=DEFAULT_CHECKPOINT_DIR,
    )
    parser = argparse.ArgumentParser(
        description="Train runnerDQN on the 20-intersection SUMO benchmark with isolated outputs."
    )
    parser.add_argument(
        "--scenario-dir",
        type=str,
        default="sumo_benchmark",
        help="Scenario directory containing benchmark.sumocfg.",
    )
    parser.add_argument(
        "--control-roles",
        type=str,
        default=None,
        help="Optional path to a control_roles.json file for the chosen scenario.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory used for plots, metrics, and generated fixed-time XML.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=str(DEFAULT_CHECKPOINT_DIR),
        help="Directory used for DQN checkpoints.",
    )
    parser.add_argument("--episodes", type=int, default=defaults.episodes, help="Number of training episodes.")
    parser.add_argument("--max-steps", type=int, default=defaults.max_steps, help="Maximum simulation steps per episode.")
    parser.add_argument("--decision-interval", type=int, default=defaults.decision_interval, help="Seconds between DQN decisions.")
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size, help="Replay batch size per update.")
    parser.add_argument("--updates-per-episode", type=int, default=defaults.updates_per_episode, help="Gradient updates per episode.")
    parser.add_argument("--eval-interval", type=int, default=defaults.eval_interval, help="Run evaluation every N training episodes.")
    parser.add_argument("--eval-episodes", type=int, default=defaults.eval_episodes, help="Evaluation rollouts per checkpoint.")
    parser.add_argument("--checkpoint-interval", type=int, default=defaults.checkpoint_interval, help="Save a .pt checkpoint every N episodes.")
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate, help="Optimizer learning rate.")
    parser.add_argument("--epsilon-end", type=float, default=defaults.epsilon_end, help="Final epsilon after exploration decay.")
    parser.add_argument(
        "--epsilon-decay-episodes",
        type=int,
        default=defaults.epsilon_decay_episodes,
        help="Episodes used to decay epsilon from start to end.",
    )
    parser.add_argument(
        "--target-sync-interval",
        type=int,
        default=defaults.target_sync_interval,
        help="Episodes between target-network syncs.",
    )
    parser.add_argument("--train-seed-stride", type=int, default=defaults.train_seed_stride, help="Seed delta applied between training episodes.")
    parser.add_argument("--eval-seed-offset", type=int, default=defaults.eval_seed_offset, help="Offset added to evaluation seeds.")
    parser.add_argument("--eval-seed-stride", type=int, default=defaults.eval_seed_stride, help="Seed delta applied between evaluation checkpoints.")
    parser.add_argument("--device", type=str, default="auto", help="Torch device: auto, cpu, or cuda.")
    parser.add_argument("--seed", type=int, default=defaults.seed, help="Random seed.")
    parser.add_argument("--resume-from", type=str, default=None, help="Resume training from a DQN checkpoint path.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    scenario_dir = base.resolve_scenario_dir(args.scenario_dir)
    sumocfg_path = scenario_dir / "benchmark.sumocfg"
    control_roles_path = resolve_path_from_root(args.control_roles) if args.control_roles else scenario_dir / "control_roles.json"
    output_dir = resolve_path_from_root(args.output_dir)
    checkpoint_dir = resolve_path_from_root(args.checkpoint_dir)
    resume_path = resolve_path_from_root(args.resume_from) if args.resume_from else None

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

    config = base.DqnConfig(
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
        device=base.resolve_torch_device(args.device),
        seed=args.seed,
        plot_path=output_dir / "average_travel_time.png",
        metrics_path=output_dir / "training_metrics.json",
        checkpoint_dir=checkpoint_dir,
    )

    base.set_global_seed(config.seed)
    print(f"Using torch device: {config.device}")
    print(f"runnerDQN20 scenario: {scenario_dir}")
    print(f"runnerDQN20 outputs: {output_dir}")
    print(f"runnerDQN20 checkpoints: {checkpoint_dir}")
    trainer = base.SumoDqnTrainer(config, resume_path=resume_path)
    trainer.train()


if __name__ == "__main__":
    main()
