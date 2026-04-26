"""Train a lightweight task-conditioned policy against SupportOps-X.

This local run uses a cross-entropy RL optimizer over discrete action sequences.
It proves the environment rewards are learnable and produces submission plots.
The official hackathon LLM run should still use training/train_grpo.py with TRL.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from supportops_x_env.models import SupportopsXAction
from supportops_x_env.server.environment import SupportopsXEnvironment


TASKS = ["damaged_refund", "account_takeover", "enterprise_outage"]
MAX_STEPS = 8


ACTION_CATALOG = [
    SupportopsXAction(action_type="inspect_ticket"),
    SupportopsXAction(action_type="search_kb", target="damaged_goods_refund"),
    SupportopsXAction(action_type="search_kb", target="account_takeover_policy"),
    SupportopsXAction(action_type="search_kb", target="enterprise_incident_protocol"),
    SupportopsXAction(action_type="escalate", target="billing"),
    SupportopsXAction(action_type="escalate", target="security"),
    SupportopsXAction(action_type="escalate", target="sre"),
    SupportopsXAction(action_type="escalate", target="account_management"),
    SupportopsXAction(action_type="assign_queue", target="billing"),
    SupportopsXAction(action_type="assign_queue", target="security"),
    SupportopsXAction(action_type="assign_queue", target="sre"),
    SupportopsXAction(action_type="assign_queue", target="account_management"),
    SupportopsXAction(action_type="assign_queue", target="general"),
    SupportopsXAction(action_type="set_followup", hours=1),
    SupportopsXAction(action_type="set_followup", hours=2),
    SupportopsXAction(action_type="set_followup", hours=24),
    SupportopsXAction(
        action_type="send_customer_reply",
        message=(
            "Billing reviewed the damaged item report. Please attach a photo if available; "
            "we can proceed with a refund or replacement under policy."
        ),
    ),
    SupportopsXAction(
        action_type="send_customer_reply",
        message=(
            "Security is protecting your account and reviewing verification. This case is "
            "pending and we will send a follow-up after checks complete."
        ),
    ),
    SupportopsXAction(
        action_type="send_customer_reply",
        message=(
            "We have opened an incident with SRE and looped in your account manager. "
            "The case remains pending and we will provide updates while recovery is monitored."
        ),
    ),
    SupportopsXAction(
        action_type="send_customer_reply",
        message="We are checking this and will get back later.",
    ),
    SupportopsXAction(action_type="finalize", target="solved"),
    SupportopsXAction(action_type="finalize", target="pending"),
    SupportopsXAction(action_type="finalize", target="escalated"),
]


def sample_index(probs: list[float], rng: random.Random) -> int:
    draw = rng.random()
    running = 0.0
    for index, value in enumerate(probs):
        running += value
        if draw <= running:
            return index
    return len(probs) - 1


def run_sequence(task_id: str, action_indices: list[int]) -> dict:
    env = SupportopsXEnvironment()
    obs = env.reset(task_id=task_id)
    total_reward = 0.0
    trajectory = []
    for action_index in action_indices:
        action = ACTION_CATALOG[action_index]
        obs = env.step(action)
        total_reward += float(obs.reward or 0.0)
        trajectory.append(action.model_dump(exclude_none=True))
        if obs.done:
            break
    return {
        "task_id": task_id,
        "score": obs.score,
        "total_reward": round(total_reward, 4),
        "steps": obs.step,
        "trajectory": trajectory,
        "scorecard": obs.scorecard,
    }


def greedy_sequence(policy: list[list[float]]) -> list[int]:
    return [max(range(len(step)), key=lambda index: step[index]) for step in policy]


def evaluate(policies: dict[str, list[list[float]]]) -> list[dict]:
    rows = []
    for task_id in TASKS:
        rows.append(run_sequence(task_id, greedy_sequence(policies[task_id])))
    return rows


def uniform_policies() -> dict[str, list[list[float]]]:
    n_actions = len(ACTION_CATALOG)
    return {
        task_id: [[1.0 / n_actions for _ in ACTION_CATALOG] for _ in range(MAX_STEPS)]
        for task_id in TASKS
    }


def train() -> tuple[dict[str, list[list[float]]], list[float], list[float], list[dict]]:
    rng = random.Random(2026)
    policies = uniform_policies()
    best_rows = []
    iteration_scores = []
    greedy_scores = []
    iterations = 90
    population = 120
    elite_fraction = 0.18
    smoothing = 0.72
    floor = 0.002

    for iteration in range(iterations):
        task_scores = []
        for task_id in TASKS:
            samples = []
            for _ in range(population):
                indices = [
                    sample_index(policies[task_id][step], rng)
                    for step in range(MAX_STEPS)
                ]
                row = run_sequence(task_id, indices)
                samples.append((row["score"], indices, row))
                task_scores.append(row["score"])

            samples.sort(key=lambda item: item[0], reverse=True)
            best_rows.append(samples[0][2])
            elite_count = max(1, int(population * elite_fraction))
            elites = samples[:elite_count]

            for step in range(MAX_STEPS):
                counts = [floor for _ in ACTION_CATALOG]
                for _, indices, _ in elites:
                    counts[indices[step]] += 1.0
                total = sum(counts)
                learned = [count / total for count in counts]
                old = policies[task_id][step]
                policies[task_id][step] = [
                    smoothing * old_value + (1.0 - smoothing) * new_value
                    for old_value, new_value in zip(old, learned)
                ]
                renorm = sum(policies[task_id][step])
                policies[task_id][step] = [
                    value / renorm for value in policies[task_id][step]
                ]

        iteration_scores.append(mean(task_scores))
        greedy_rows = evaluate(policies)
        greedy_scores.append(mean(row["score"] for row in greedy_rows))

        if iteration > 20 and greedy_scores[-1] >= 0.96:
            break

    return policies, iteration_scores, greedy_scores, best_rows


def plot_training(iteration_scores, greedy_scores, before_score, after_score, output_dir):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(iteration_scores, color="#6a8caf", label="sampled policy mean")
    ax.plot(greedy_scores, color="#2e7d32", linewidth=2.2, label="greedy policy eval")
    ax.axhline(before_score, color="#8a8a8a", linestyle="--", label=f"before {before_score:.2f}")
    ax.axhline(after_score, color="#194d33", linestyle="--", label=f"after {after_score:.2f}")
    ax.set_title("SupportOps-X local RL reward improvement")
    ax.set_xlabel("Training iteration")
    ax.set_ylabel("Average normalized task score")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_dir / "tabular_training_curve.png", dpi=180)
    plt.close(fig)


def main() -> None:
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    before_rows = evaluate(uniform_policies())
    before_score = mean(row["score"] for row in before_rows)
    policies, iteration_scores, greedy_scores, best_rows = train()
    after_rows = evaluate(policies)
    after_score = mean(row["score"] for row in after_rows)

    plot_training(iteration_scores, greedy_scores, before_score, after_score, output_dir)

    report = {
        "training_type": "task_conditioned_cross_entropy_rl_smoke_run",
        "note": "Local RL smoke-run; official LLM training uses training/train_grpo.py.",
        "before_average_score": round(before_score, 4),
        "after_average_score": round(after_score, 4),
        "best_sampled_average_score": round(
            mean(max(row["score"] for row in best_rows if row["task_id"] == task_id) for task_id in TASKS),
            4,
        ),
        "iterations": len(iteration_scores),
        "population_per_task": 120,
        "tasks": TASKS,
        "action_count": len(ACTION_CATALOG),
        "trained_policy_examples": {row["task_id"]: row for row in after_rows},
    }
    (output_dir / "tabular_training_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"Wrote {output_dir / 'tabular_training_curve.png'}")
    print(f"Wrote {output_dir / 'tabular_training_report.json'}")


if __name__ == "__main__":
    main()

