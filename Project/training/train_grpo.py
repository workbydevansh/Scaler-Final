"""Minimal TRL/Unsloth GRPO training entrypoint for SupportOps-X.

Run this in Colab or a GPU runtime. Local CPU smoke testing is not recommended.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from training.supportops_tool_env import SupportOpsToolEnv, reward_func


def build_dataset(repeats: int):
    from datasets import Dataset

    prompts = []
    for task_id in ["damaged_refund", "account_takeover", "enterprise_outage"]:
        content = (
            "You are the Support Lead Agent in SupportOps-X. Resolve the support "
            "case by using tools in a safe order: inspect evidence, search policy, "
            "coordinate with specialists, assign ownership, schedule follow-up, "
            "send a compliant customer update, and finalize with the right status."
        )
        prompts.extend(
            {
                "prompt": [{"role": "user", "content": content}],
                "task_id": task_id,
            }
            for _ in range(repeats)
        )
    return Dataset.from_list(prompts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--repeats", type=int, default=16)
    parser.add_argument("--output-dir", default="outputs/supportops-x-grpo")
    parser.add_argument("--max-steps", type=int, default=80)
    args = parser.parse_args()

    from trl import GRPOConfig, GRPOTrainer

    train_dataset = build_dataset(args.repeats)
    config = GRPOConfig(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        per_device_train_batch_size=2,
        generation_batch_size=2,
        num_generations=2,
        max_completion_length=256,
        log_completions=False,
        save_steps=max(20, args.max_steps // 2),
        logging_steps=1,
        report_to=[],
    )

    trainer = GRPOTrainer(
        model=args.model,
        args=config,
        train_dataset=train_dataset,
        reward_funcs=reward_func,
        environment_factory=SupportOpsToolEnv,
    )
    trainer.train()
    trainer.save_model(args.output_dir)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_history = trainer.state.log_history
    (output_dir / "trainer_log_history.json").write_text(json.dumps(log_history, indent=2))
    _write_training_plot(log_history, output_dir / "trl_training_curve.png")


def _write_training_plot(log_history: list[dict], output_path: Path) -> None:
    """Write a simple reward/loss plot when matplotlib is available."""

    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    steps = []
    rewards = []
    losses = []
    for row in log_history:
        step = row.get("step")
        if step is None:
            continue
        reward = _first_present(row, "reward", "rewards/mean", "train/reward")
        loss = _first_present(row, "loss", "train_loss")
        if reward is not None:
            steps.append(step)
            rewards.append(float(reward))
            losses.append(float(loss) if loss is not None else None)

    if not steps:
        return

    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.plot(steps, rewards, color="#2e7d32", label="reward")
    ax1.set_xlabel("training step")
    ax1.set_ylabel("reward", color="#2e7d32")
    ax1.tick_params(axis="y", labelcolor="#2e7d32")

    if any(value is not None for value in losses):
        ax2 = ax1.twinx()
        loss_steps = [step for step, value in zip(steps, losses) if value is not None]
        loss_values = [value for value in losses if value is not None]
        ax2.plot(loss_steps, loss_values, color="#1f5f8b", label="loss")
        ax2.set_ylabel("loss", color="#1f5f8b")
        ax2.tick_params(axis="y", labelcolor="#1f5f8b")

    fig.suptitle("SupportOps-X TRL/GRPO training")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _first_present(row: dict, *keys: str):
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return None


if __name__ == "__main__":
    main()
