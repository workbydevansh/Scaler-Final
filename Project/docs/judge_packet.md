# SupportOps-X Judge Packet

This page maps SupportOps-X directly to the OpenEnv Hackathon judging guide so
reviewers can find the strongest evidence quickly.

## Minimum Requirements

| Requirement | Evidence |
| --- | --- |
| Use OpenEnv latest release | OpenEnv `Environment` subclass in `supportops_x_env/server/environment.py`; FastAPI server from `openenv.core.env_server.http_server.create_app`; validated with `openenv validate`. |
| Hosted on Hugging Face Spaces | Space: https://huggingface.co/spaces/devanshverma/supportops-x; live server: https://devanshverma-supportops-x.hf.space |
| OpenEnv-compliant API | `/reset`, `/step`, `/state`, `/metadata`, `/schema`, `/health`, `/mcp`, and `/ws`; live validation passes 6/6 runtime criteria. |
| Training script using TRL/Unsloth/HF stack | `training/train_grpo.py`, `training/supportops_tool_env.py`, and `notebooks/supportops_x_grpo_colab.ipynb`. |
| Evidence of training | Local reward-learning run improves average score from `0.08` to `0.9867`; TRL smoke run logs and plot are committed. |
| Mini-blog, video, or short presentation | `BLOG.md`, `docs/huggingface_blog.md`, `PITCH.md`, and `DEMO_SCRIPT.md`. |
| README links to all materials | README links the Space, live server, notebook, blog, pitch, demo script, plots, and training reports. |

## Judging Criteria

### Environment Innovation - 40%

SupportOps-X is a partially observable enterprise support world, not a simple
game clone. The agent must coordinate across Billing, Security, SRE, and Account
Management while tracking hidden owner, severity, SLA pressure, delayed customer
updates, policy constraints, and final case state.

Why this is challenging:

- The correct action is not visible from the ticket title alone.
- Some tasks require multiple specialists, not one final classification.
- Urgent cases can be high-scoring only if kept pending with short follow-up.
- Customer replies are scored for usefulness and safety, not just keyword match.
- Delayed updates force state tracking over a multi-step trajectory.

### Storytelling - 30%

The recommended demo story is `enterprise_outage`:

1. Reset to the enterprise checkout outage case.
2. Show that a baseline assigns generically or closes too early.
3. Show the expert/trained policy inspect the ticket, read policy, escalate to
   SRE, loop in Account Management, schedule a 1-hour follow-up, write a safe
   customer update, and keep the case pending.
4. Show the scorecard reaching `1.0`.

Judge-friendly entry points:

- Live docs: https://devanshverma-supportops-x.hf.space/docs
- Browser demo endpoints: `POST /demo/reset`, `POST /demo/step`, `GET /demo/state`
- Demo script: `DEMO_SCRIPT.md`
- Pitch script: `PITCH.md`

### Showing Improvement in Rewards - 20%

Committed evidence:

- `results/tabular_training_curve.png`: local RL curve.
- `results/tabular_training_report.json`: before/after metrics and learned
  trajectories.
- `results/baseline_vs_expert.png`: baseline vs expert score comparison.
- `results/reward_components.png`: baseline vs expert reward component scores.
- `results/trl_grpo_smoke/trainer_log_history.json`: TRL smoke-run trainer log.
- `results/trl_grpo_smoke/trl_training_curve.png`: TRL smoke-run plot.

Measured result:

- Before local reward learning: `0.08`
- After local reward learning: `0.9867`
- Best sampled average score: `0.99`

### Reward and Training Pipeline - 10%

The reward model uses named OpenEnv rubrics in `supportops_x_env/rewards.py`.
The score is a weighted sum of independent checks:

| Component | Weight |
| --- | ---: |
| opened ticket | 0.10 |
| policy retrieval | 0.15 |
| specialist coordination | 0.20 |
| queue assignment | 0.15 |
| SLA follow-up | 0.10 |
| customer communication | 0.15 |
| final status | 0.10 |
| safety | 0.05 |

Anti-hacking safeguards:

- Invalid actions are penalized.
- Repeated actions are penalized.
- Unsafe customer replies trigger policy violations.
- Keyword-stuffed replies without policy and specialist work stay low-scoring.
- Episodes terminate at a fixed step limit.
- Hidden task data is not exposed in observations.

## Final Win-Maximizer

The biggest remaining upside is to run the Colab GRPO notebook longer with a
stronger tool-calling model, then add the final adapter link and LLM reward curve.
The environment, validation, local reward-learning evidence, and TRL pipeline are
already present; a stronger LLM run would mainly improve the 20% reward-improvement
bucket and the 10% pipeline bucket.
