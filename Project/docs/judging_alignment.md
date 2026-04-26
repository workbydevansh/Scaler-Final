# SupportOps-X Judging Alignment

## One-Sentence Problem Statement

SupportOps-X trains an AI Support Lead Agent to resolve realistic customer support escalations by planning over multiple steps, coordinating with specialist agents, using policy knowledge, communicating safely with customers, and choosing the correct case outcome.

## Theme Fit

- Multi-Agent Interactions: the support lead must coordinate with Billing, Security, SRE, and Account Management.
- Long-Horizon Planning and Instruction Following: each case requires an ordered sequence of inspection, policy lookup, escalation, routing, follow-up, reply, and finalization.
- World Modeling for Professional Tasks: hidden owner, policy requirements, SLA pressure, customer tier, and incident state must be inferred and tracked.

## Required Submission Elements

- Problem statement: `docs/problem_statement.md`
- Environment: ticket inbox, KB, specialist queues, customer reply workspace, follow-up scheduler, and hidden case state.
- Agent capabilities: inspect ticket, search KB, escalate, assign queue, schedule follow-up, send reply, finalize.
- Tasks: damaged refund, suspected account takeover, enterprise outage.
- Reward/evaluation logic: named OpenEnv rubric components with evidence, policy, coordination, queue, SLA, communication, final status, and safety penalties.
- Post-training/self-improvement: local reward-learning smoke run plus reproducible TRL/GRPO script and Colab notebook.

## Evidence To Show Judges

- Judge packet: `docs/judge_packet.md`
- Live Space: https://huggingface.co/spaces/devanshverma/supportops-x
- Live server: https://devanshverma-supportops-x.hf.space
- API docs: https://devanshverma-supportops-x.hf.space/docs
- Browser demo: use `POST /demo/reset` and `POST /demo/step` in the API docs for a stateful click-through episode.
- OpenEnv validation: `openenv validate --url https://devanshverma-supportops-x.hf.space` passes 6/6 checks.
- Local tests: `python -m pytest -q` passes 6/6 tests.
- Reward improvement: local RL improves average score from `0.08` to `0.9867`.
- Training curve: `results/tabular_training_curve.png`
- Training report: `results/tabular_training_report.json`
- Reward components: `results/reward_components.png`
- TRL smoke log: `results/trl_grpo_smoke/trainer_log_history.json`
- TRL smoke plot: `results/trl_grpo_smoke/trl_training_curve.png`

## Honest Training Note

The local RL run is the measured reward-improvement result. The TRL/GRPO LLM path is implemented and smoke-tested, but the short six-step run with a small model did not learn robust tool use. For maximum score, run the notebook longer with a stronger tool-calling model and add the final adapter and reward/loss plots.
