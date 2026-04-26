# SupportOps-X Pitch

## 30-Second Version

SupportOps-X is an OpenEnv environment for training support agents that can handle real enterprise escalations. The agent must inspect tickets, search policy, coordinate with Billing, Security, SRE, and Account Management, communicate safely with customers, and choose the right final status.

It targets multi-agent interaction, long-horizon planning, and professional world modeling. The live Space passes OpenEnv validation, and a local RL run improves average score from `0.08` to `0.9867`, showing the reward is learnable.

## 3-Minute Version

Most support benchmarks are too shallow. They test whether a model can classify a ticket or write a nice reply, but real support operations require state tracking, policy use, specialist coordination, SLA awareness, and safe communication.

SupportOps-X turns that into an OpenEnv environment. The trainable agent is a Support Lead Agent. It operates across a ticket inbox, knowledge base, specialist queues, customer reply workspace, and follow-up scheduler.

The environment includes three MVP cases:

- damaged goods refund
- suspected account takeover
- enterprise checkout outage

Each case has hidden state. The agent does not initially know the true owner, required policy, required specialist, customer risk, or correct final status. It has to gather evidence and update its plan.

The reward is decomposed into independent checks: ticket inspection, KB retrieval, specialist coordination, correct queue, follow-up timing, communication quality, final status, and safety. Repeated, invalid, unsafe, and premature actions are penalized.

Our live environment is hosted on Hugging Face Spaces and passes OpenEnv runtime validation. We also include a local reward-learning run that improves average score from `0.08` to `0.9867`, plus a TRL/Unsloth GRPO training scaffold for LLM post-training.

The demo story is simple: an enterprise customer reports checkout failure during launch hour. The baseline mishandles it. The trained policy escalates to SRE, loops in Account Management, sends a safe incident update, schedules a one-hour follow-up, and keeps the case pending.

That is the kind of professional, multi-agent, long-horizon behavior OpenEnv is designed to evaluate.

