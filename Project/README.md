---
title: SupportOps-X
sdk: docker
app_port: 8000
license: mit
---

# SupportOps-X

SupportOps-X is an OpenEnv environment where a trainable Support Lead Agent resolves realistic customer support escalations by coordinating with specialist agents, using policy knowledge, sending customer updates, and deciding when to keep a case pending or close it.

The environment targets three Round 2 themes:

- Multi-Agent Interactions: Billing, Security, SRE, and Account Management agents provide partial or delayed information.
- Long-Horizon Planning and Instruction Following: cases require several ordered actions, follow-up timing, and recovery from new evidence.
- World Modeling for Professional Tasks: hidden ownership, SLA pressure, policy constraints, customer tier, and incident state must be tracked.

## Current Status

This repo is submission-ready for the OpenEnv Round 2 environment requirement:

- OpenEnv-compatible package, manifest, server, client, and Dockerfile
- deterministic multi-step support cases with programmatic rewards
- tests for reset, step, scoring, and expert trajectories
- baseline vs expert evaluation script with aggregate and component reward plots
- local reward-learning run showing `0.08 -> 0.9867` average score
- TRL/GRPO training script and Colab notebook for LLM post-training
- Hugging Face blog draft, pitch script, and demo script
- live Hugging Face Space that passes OpenEnv validation

## Environment

The agent operates inside a support operations world with:

- Ticket Inbox: customer message, tier, SLA, severity, and case metadata
- Knowledge Base: refund, security, and enterprise incident policies
- Specialist Agents: Billing, Security, SRE, and Account Management
- Customer Reply Workspace: customer-facing updates are checked for quality and safety
- Follow-up Scheduler: urgent cases must remain pending with the correct follow-up window

Hidden state includes true owner, required policies, required specialists, final case status, policy-sensitive forbidden claims, and delayed customer updates.

## Tasks

The MVP includes three deterministic tasks:

- `damaged_refund`: validate damaged-goods refund policy, involve Billing, route correctly, and close safely.
- `account_takeover`: identify security urgency, escalate to Security, avoid unsafe customer replies, and keep the case pending.
- `enterprise_outage`: coordinate with SRE and Account Management, keep the enterprise customer updated, and leave the case pending during the incident.

## Action Space

The OpenEnv action model is `SupportopsXAction`:

```python
SupportopsXAction(
    action_type="inspect_ticket",
    target=None,
    message=None,
    hours=None,
)
```

Supported `action_type` values:

- `inspect_ticket`
- `search_kb`
- `escalate`
- `assign_queue`
- `set_followup`
- `send_customer_reply`
- `finalize`

## Reward Design

The reward is process-aware and hard to solve with one lucky final answer. It combines:

- ticket inspection and evidence gathering
- correct KB retrieval
- required specialist coordination
- correct queue assignment
- SLA-appropriate follow-up timing
- safe and useful customer communication
- correct final status
- penalties for invalid, repeated, unsafe, or premature actions

The scoring implementation uses named OpenEnv rubrics in `supportops_x_env/rewards.py`. Final normalized task score is exposed in `observation.score`, and component scores, weights, penalties, and classic checklist fields are exposed in `observation.scorecard`.

## Quick Start

Install locally:

```bash
python -m pip install -e .[dev]
```

Run tests:

```bash
python -m pytest
```

Run the local evaluation smoke test:

```bash
python scripts/evaluate_policies.py
```

Run the local RL reward-improvement smoke test:

```bash
python scripts/train_tabular_policy.py
```

Run the OpenEnv server:

```bash
python -m supportops_x_env.server.app --port 8000
```

Then connect with the client:

```python
from supportops_x_env import SupportopsXAction, SupportopsXEnv

with SupportopsXEnv(base_url="http://localhost:8000").sync() as env:
    obs = env.reset(task_id="enterprise_outage").observation
    print(obs.visible_state)
    result = env.step(SupportopsXAction(action_type="inspect_ticket"))
    print(result.observation.visible_state)
```

## Browser Demo

For a judge-friendly click-through demo, use the stateful browser endpoints in the live docs:

- `POST /demo/reset`
- `POST /demo/step`
- `GET /demo/state`

The standard OpenEnv `/reset` and `/step` endpoints are kept for validator compatibility. The `/demo/*` endpoints reuse one in-memory episode so a human can click through the full workflow and see the final `score: 1.0`.

## Training

The post-training assets are in:

- `training/train_grpo.py`
- `training/supportops_tool_env.py`
- `notebooks/supportops_x_grpo_colab.ipynb`

Use the notebook with Hugging Face credits for a longer LLM run. The notebook installs the project, creates tool-callable environment wrappers, runs GRPO, and saves plots for README submission.

Local reward-learning smoke result:

- before training average score: `0.08`
- after local RL training average score: `0.9867`
- best sampled average score: `0.99`

This local run uses `scripts/train_tabular_policy.py` to prove that the environment reward is learnable. A short TRL/GRPO smoke run was also executed with `Qwen/Qwen3-0.6B`; it completed and produced trainer logs, but the small smoke model did not learn meaningful tool use in six steps. The stronger submission evidence is therefore the validated environment plus local reward-improvement curve, with `training/train_grpo.py` included as the reproducible LLM post-training path.

## Submission Links

- Hugging Face Space: https://huggingface.co/spaces/devanshverma/supportops-x
- Live OpenEnv server: https://devanshverma-supportops-x.hf.space
- Colab training notebook: https://huggingface.co/spaces/devanshverma/supportops-x/blob/main/notebooks/supportops_x_grpo_colab.ipynb
- Hugging Face mini-blog draft: https://huggingface.co/spaces/devanshverma/supportops-x/blob/main/docs/huggingface_blog.md
- Top-level blog article: https://huggingface.co/spaces/devanshverma/supportops-x/blob/main/BLOG.md
- Judge packet: https://huggingface.co/spaces/devanshverma/supportops-x/blob/main/docs/judge_packet.md
- Pitch script: https://huggingface.co/spaces/devanshverma/supportops-x/blob/main/PITCH.md
- Demo script: https://huggingface.co/spaces/devanshverma/supportops-x/blob/main/DEMO_SCRIPT.md
- Judging alignment: https://huggingface.co/spaces/devanshverma/supportops-x/blob/main/docs/judging_alignment.md
- Reference baseline-vs-expert plot: https://huggingface.co/spaces/devanshverma/supportops-x/blob/main/results/baseline_vs_expert.png
- Reward component plot: https://huggingface.co/spaces/devanshverma/supportops-x/blob/main/results/reward_components.png
- Local RL reward-improvement plot: https://huggingface.co/spaces/devanshverma/supportops-x/blob/main/results/tabular_training_curve.png
- Local RL training report: https://huggingface.co/spaces/devanshverma/supportops-x/blob/main/results/tabular_training_report.json
- TRL smoke-run trainer log: https://huggingface.co/spaces/devanshverma/supportops-x/blob/main/results/trl_grpo_smoke/trainer_log_history.json
- TRL smoke-run plot: https://huggingface.co/spaces/devanshverma/supportops-x/blob/main/results/trl_grpo_smoke/trl_training_curve.png

If time remains during judging, run the Colab notebook longer with a stronger tool-calling model and add the final adapter/plot links. The core submission does not depend on that optional extension.

## Why This Should Score

Environment Innovation: SupportOps-X is a realistic enterprise workflow with hidden state, multi-agent coordination, policy constraints, SLA pressure, delayed signals, and customer-safety constraints.

Storytelling: the demo is easy to follow: an enterprise customer has an outage, the baseline mishandles it, and the trained agent escalates and communicates correctly.

Reward Improvement: the repo includes deterministic baseline/expert evaluation, local reward-learning evidence, and readable before/after plots.

Reward and Pipeline: the reward is decomposed into named OpenEnv rubric components, with penalties for repeated, invalid, premature, and unsafe behavior.
