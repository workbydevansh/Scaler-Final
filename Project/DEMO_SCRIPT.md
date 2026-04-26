# SupportOps-X Demo Script

## Setup

Open:

- Space repo: https://huggingface.co/spaces/devanshverma/supportops-x
- Live server docs: https://devanshverma-supportops-x.hf.space/docs
- Training curve: `results/tabular_training_curve.png`
- Training report: `results/tabular_training_report.json`

## Demo Flow

1. Show OpenEnv validation:

```bash
openenv validate --url https://devanshverma-supportops-x.hf.space
```

Say: "The environment is live and passes all OpenEnv runtime criteria."

2. Show the enterprise outage task.

Say: "This is a critical enterprise customer. The agent needs to coordinate with SRE and Account Management, not just reply politely."

3. Show baseline behavior.

Say: "The baseline often assigns to a generic queue or closes too early. It gets partial credit for opening the ticket, but misses policy, coordination, and final-status constraints."

4. Show trained policy behavior.

Say: "After reward optimization, the policy learns to retrieve incident policy, escalate to SRE, loop in Account Management, assign the right queue, set a short follow-up, send a safe update, and keep the case pending."

5. Show the training curve.

Say: "The local RL smoke run improves average score from 0.08 to 0.9867. The included TRL/Unsloth notebook is the LLM training path."

6. Close with the theme mapping.

Say: "This directly hits multi-agent interactions, long-horizon planning, and world modeling for professional tasks."

## One-Liner For Judges

SupportOps-X trains an AI support lead to behave like a real enterprise teammate: policy-aware, coordinated, safe, and stateful across evolving customer escalations.

