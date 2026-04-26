# SupportOps-X Problem Statement

## Problem Statement

SupportOps-X is an OpenEnv environment where an AI Support Lead Agent manages complex customer support cases that evolve over time and require coordination with Billing, Security, SRE, and Account Management.

The challenge is partially observable and long-horizon. The agent must gather evidence, retrieve policy, ask the right specialist agents, assign ownership, schedule follow-up, write safe customer communication, and decide whether the case should be solved or kept pending.

## Environment

The agent operates across:

- Ticket Inbox
- Knowledge Base
- Specialist queues
- Internal escalation console
- Customer reply workspace
- Follow-up scheduler

Hidden state includes the true owner, required policy, required specialists, customer tier, SLA pressure, delayed updates, and unsafe reply patterns.

## Capabilities

The Support Lead Agent can:

- inspect tickets
- search KB policy pages
- escalate to specialist agents
- assign an ownership queue
- set a follow-up window
- draft customer-facing replies
- finalize the case as solved, pending, or escalated

## Tasks

The MVP tasks are:

- `damaged_refund`: route a damaged goods refund through policy and Billing.
- `account_takeover`: handle a security-sensitive account takeover without unsafe closure.
- `enterprise_outage`: coordinate SRE and Account Management during a critical enterprise outage.

## Reward Model

Reward is decomposed into:

- ticket inspection
- KB usage
- specialist coordination
- queue correctness
- follow-up timing
- communication quality
- final status correctness
- safety and anti-cheating penalties

The scorecard is exposed in every observation for transparent evaluation.

## Post-Training Strategy

After the base environment works, increase difficulty by adding delayed messages, conflicting specialist responses, policy drift, more customer tiers, and generated cases based on issue type, urgency, ownership, and policy state.

