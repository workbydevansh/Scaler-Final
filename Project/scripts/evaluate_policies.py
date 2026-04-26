"""Evaluate baseline and expert policies for SupportOps-X."""

from __future__ import annotations

import json
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


def baseline_policy(task_id: str):
    return [
        SupportopsXAction(action_type="inspect_ticket"),
        SupportopsXAction(action_type="assign_queue", target="general"),
        SupportopsXAction(
            action_type="send_customer_reply",
            message="We are checking this and will get back later.",
        ),
        SupportopsXAction(action_type="finalize", target="solved"),
    ]


def expert_policy(task_id: str):
    policies = {
        "damaged_refund": [
            SupportopsXAction(action_type="inspect_ticket"),
            SupportopsXAction(action_type="search_kb", target="damaged_goods_refund"),
            SupportopsXAction(action_type="escalate", target="billing"),
            SupportopsXAction(action_type="assign_queue", target="billing"),
            SupportopsXAction(action_type="set_followup", hours=24),
            SupportopsXAction(
                action_type="send_customer_reply",
                message=(
                    "Billing reviewed the damaged item report. Please attach a photo if "
                    "available; we can proceed with a refund or replacement under policy."
                ),
            ),
            SupportopsXAction(action_type="finalize", target="solved"),
        ],
        "account_takeover": [
            SupportopsXAction(action_type="inspect_ticket"),
            SupportopsXAction(action_type="search_kb", target="account_takeover_policy"),
            SupportopsXAction(action_type="escalate", target="security"),
            SupportopsXAction(action_type="assign_queue", target="security"),
            SupportopsXAction(action_type="set_followup", hours=1),
            SupportopsXAction(
                action_type="send_customer_reply",
                message=(
                    "Security is protecting your account and reviewing verification. "
                    "This case is pending and we will send a follow-up after checks complete."
                ),
            ),
            SupportopsXAction(action_type="finalize", target="pending"),
        ],
        "enterprise_outage": [
            SupportopsXAction(action_type="inspect_ticket"),
            SupportopsXAction(action_type="search_kb", target="enterprise_incident_protocol"),
            SupportopsXAction(action_type="escalate", target="sre"),
            SupportopsXAction(action_type="escalate", target="account_management"),
            SupportopsXAction(action_type="assign_queue", target="sre"),
            SupportopsXAction(action_type="set_followup", hours=1),
            SupportopsXAction(
                action_type="send_customer_reply",
                message=(
                    "We have opened an incident with SRE and looped in your account manager. "
                    "The case remains pending and we will provide updates while recovery is monitored."
                ),
            ),
            SupportopsXAction(action_type="finalize", target="pending"),
        ],
    }
    return policies[task_id]


def run_policy(task_id: str, policy_name: str):
    env = SupportopsXEnvironment()
    obs = env.reset(task_id=task_id)
    rewards = []
    actions = expert_policy(task_id) if policy_name == "expert" else baseline_policy(task_id)
    for action in actions:
        obs = env.step(action)
        rewards.append(float(obs.reward or 0.0))
        if obs.done:
            break
    return {
        "task_id": task_id,
        "policy": policy_name,
        "score": obs.score,
        "total_reward": round(sum(rewards), 4),
        "steps": obs.step,
        "scorecard": obs.scorecard,
    }


def plot_summary(rows, output_dir: Path) -> None:
    by_policy = {
        policy: [row["score"] for row in rows if row["policy"] == policy]
        for policy in ["baseline", "expert"]
    }
    labels = list(by_policy)
    values = [mean(by_policy[label]) for label in labels]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, values, color=["#7f8c8d", "#2e7d32"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Average normalized task score")
    ax.set_xlabel("Policy")
    ax.set_title("SupportOps-X reference policy comparison")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.2f}", ha="center")
    fig.tight_layout()
    fig.savefig(output_dir / "baseline_vs_expert.png", dpi=180)
    plt.close(fig)


def plot_component_summary(rows, output_dir: Path) -> None:
    component_names = [
        "opened_ticket",
        "policy_retrieval",
        "specialist_coordination",
        "queue_assignment",
        "sla_followup",
        "customer_communication",
        "final_status",
        "safety",
    ]
    policies = ["baseline", "expert"]
    values = {
        policy: [
            mean(
                row["scorecard"]["rubric_components"][component]
                for row in rows
                if row["policy"] == policy
            )
            for component in component_names
        ]
        for policy in policies
    }

    x_positions = range(len(component_names))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.bar(
        [x - width / 2 for x in x_positions],
        values["baseline"],
        width,
        label="baseline",
        color="#7f8c8d",
    )
    ax.bar(
        [x + width / 2 for x in x_positions],
        values["expert"],
        width,
        label="expert",
        color="#2e7d32",
    )
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Average component score")
    ax.set_title("SupportOps-X reward rubric components")
    ax.set_xticks(list(x_positions))
    ax.set_xticklabels(
        [
            "ticket",
            "policy",
            "specialists",
            "queue",
            "follow-up",
            "reply",
            "final",
            "safety",
        ],
        rotation=30,
        ha="right",
    )
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(output_dir / "reward_components.png", dpi=180)
    plt.close(fig)


def main() -> None:
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    rows = [
        run_policy(task_id, policy)
        for policy in ["baseline", "expert"]
        for task_id in TASKS
    ]
    (output_dir / "policy_eval.json").write_text(json.dumps(rows, indent=2))
    plot_summary(rows, output_dir)
    plot_component_summary(rows, output_dir)
    print(json.dumps(rows, indent=2))
    print(f"Wrote {output_dir / 'policy_eval.json'}")
    print(f"Wrote {output_dir / 'baseline_vs_expert.png'}")
    print(f"Wrote {output_dir / 'reward_components.png'}")


if __name__ == "__main__":
    main()
