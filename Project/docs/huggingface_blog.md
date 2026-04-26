# SupportOps-X: Training Support Agents for Real Escalation Work

SupportOps-X is an OpenEnv environment for customer support escalation. The goal is to train a support lead agent that can coordinate across policy, specialists, customer communication, and SLA pressure instead of only classifying a ticket.

The environment contains three MVP cases:

- damaged goods refund
- suspected account takeover
- enterprise checkout outage

Each case has hidden state: true owner, required KB policy, required specialist agents, correct final status, policy-sensitive banned language, and delayed customer updates. The agent must inspect the ticket, search the right policy, escalate to the right specialist, assign the right queue, schedule follow-up, send a safe customer reply, and finalize the case.

This targets three OpenEnv Round 2 themes:

- Multi-Agent Interactions
- Long-Horizon Planning and Instruction Following
- World Modeling for Professional Tasks

The reward is decomposed into independent checks: evidence gathering, KB retrieval, coordination, queue correctness, follow-up timing, communication quality, final status, and safety. Repeated actions, invalid actions, unsafe customer replies, and premature closure are penalized.

The scorecard exposes named OpenEnv rubric components for each of those checks, including component weights and anti-hacking penalties. This makes reward progress inspectable instead of hiding it behind one scalar.

In our demo, the baseline agent often assigns to a generic queue or closes urgent cases too early. A local RL smoke-run already shows that the environment reward is learnable: average score improves from `0.08` before training to `0.9867` after training. The trained policy discovers the right policy lookup, specialist escalation, ownership queue, follow-up, customer update, and final status patterns.

A short TRL/GRPO smoke run also completed and saved trainer logs. That run proves the LLM post-training pipeline executes; the local RL curve is the stronger measured improvement result. The included Colab notebook and `training/train_grpo.py` script are ready for a longer credit-backed LLM run with a stronger tool-calling model.

Repository links:

- Environment Space: https://huggingface.co/spaces/devanshverma/supportops-x
- Live OpenEnv server: https://devanshverma-supportops-x.hf.space
- Training notebook: https://huggingface.co/spaces/devanshverma/supportops-x/blob/main/notebooks/supportops_x_grpo_colab.ipynb
- Reference results plot: https://huggingface.co/spaces/devanshverma/supportops-x/blob/main/results/baseline_vs_expert.png
- Reward component plot: https://huggingface.co/spaces/devanshverma/supportops-x/blob/main/results/reward_components.png
- Local RL training curve: https://huggingface.co/spaces/devanshverma/supportops-x/blob/main/results/tabular_training_curve.png
- TRL smoke-run trainer log: https://huggingface.co/spaces/devanshverma/supportops-x/blob/main/results/trl_grpo_smoke/trainer_log_history.json
- TRL smoke-run plot: https://huggingface.co/spaces/devanshverma/supportops-x/blob/main/results/trl_grpo_smoke/trl_training_curve.png
