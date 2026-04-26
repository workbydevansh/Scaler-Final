"""Core SupportOps-X environment implementation."""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import EnvironmentMetadata

from ..data import (
    CASE_LIBRARY,
    KB_PAGES,
    VALID_FINAL_STATUSES,
    VALID_QUEUES,
    VALID_SPECIALISTS,
    normalize_target,
)
from ..models import SupportopsXAction, SupportopsXObservation, SupportopsXState
from ..rewards import RewardBreakdown, RewardContext, SupportOpsRewardModel


class SupportopsXEnvironment(
    Environment[SupportopsXAction, SupportopsXObservation, SupportopsXState]
):
    """Multi-agent customer support escalation environment."""

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self) -> None:
        super().__init__()
        self._state = SupportopsXState(episode_id=str(uuid4()))
        self._case: Dict[str, Any] = {}
        self._visible_facts: List[str] = []
        self._events: List[str] = []
        self._actions_seen: set[str] = set()
        self._score = 0.0
        self._last_score = 0.0
        self._reward_breakdown: Optional[RewardBreakdown] = None
        self._reward_model = SupportOpsRewardModel()
        self._done = False
        self._policy_violations = 0
        self._invalid_actions = 0
        self._repeated_actions = 0

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_id: Optional[str] = None,
        **_: Any,
    ) -> SupportopsXObservation:
        """Reset the environment and return the initial ticket summary."""

        rng = random.Random(seed)
        if task_id is None:
            task_id = rng.choice(sorted(CASE_LIBRARY))
        if task_id not in CASE_LIBRARY:
            raise ValueError(f"Unknown task_id: {task_id}")

        self._case = CASE_LIBRARY[task_id]
        self._visible_facts = [
            (
                f"Ticket {self._case['ticket_id']}: {self._case['title']} "
                f"from {self._case['customer']}."
            ),
            (
                f"Tier={self._case['customer_tier']}, severity={self._case['severity']}, "
                f"SLA={self._case['sla_hours']}h."
            ),
        ]
        self._events = ["Episode reset. Inspect the ticket before deciding ownership."]
        self._actions_seen = set()
        self._score = 0.0
        self._last_score = 0.0
        self._reward_breakdown = None
        self._done = False
        self._policy_violations = 0
        self._invalid_actions = 0
        self._repeated_actions = 0
        self._state = SupportopsXState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            task_id=task_id,
            ticket_id=self._case["ticket_id"],
            done=False,
            score=0.0,
        )
        return self._observation(reward=0.0)

    def step(  # type: ignore[override]
        self, action: SupportopsXAction, timeout_s: Optional[float] = None, **_: Any
    ) -> SupportopsXObservation:
        """Apply one support operation and return the next observation."""

        if not self._case:
            self.reset(task_id="enterprise_outage")

        if self._done:
            return self._observation(
                reward=-0.25,
                events=["Episode is already complete. Reset before sending more actions."],
            )

        self._state.step_count += 1
        events: List[str] = []
        penalty = -0.02
        action_key = self._action_key(action)
        if action_key in self._actions_seen:
            self._repeated_actions += 1
            penalty -= 0.05
            events.append("Repeated action penalty applied.")
        self._actions_seen.add(action_key)

        handler = getattr(self, f"_handle_{action.action_type}")
        penalty += handler(action, events)
        self._inject_delayed_update(events)

        previous_score = self._last_score
        self._reward_breakdown = self._compute_reward_breakdown()
        self._score = self._reward_breakdown.normalized_score
        self._last_score = self._score
        reward = round((self._score - previous_score) + penalty, 4)

        if action.action_type == "finalize":
            self._done = True
            self._state.done = True
            reward = round(reward + self._score, 4)
            events.append(f"Episode finalized with normalized score {self._score:.2f}.")
        elif self._state.step_count >= 12:
            self._done = True
            self._state.done = True
            reward = round(reward - 0.25 + self._score, 4)
            events.append("Step limit reached. Episode ended.")

        self._sync_state_score()
        self._events.extend(events)
        return self._observation(reward=reward, events=events)

    @property
    def state(self) -> SupportopsXState:
        """Return a non-secret state summary."""

        self._sync_state_score()
        return self._state

    def get_metadata(self) -> EnvironmentMetadata:
        """Return human-readable metadata for OpenEnv inspectors."""

        return EnvironmentMetadata(
            name="SupportOps-X",
            description=(
                "A multi-agent customer support escalation environment with "
                "policy retrieval, specialist coordination, SLA-aware follow-up, "
                "customer communication, and deterministic rewards."
            ),
            version="0.1.0",
            author="OpenEnv Hackathon Team",
        )

    def _handle_inspect_ticket(
        self, action: SupportopsXAction, events: List[str]
    ) -> float:
        if self._state.opened_ticket:
            events.append("Ticket was already inspected.")
            return -0.03
        self._state.opened_ticket = True
        self._visible_facts.append(f"Customer message: {self._case['initial_message']}")
        events.append("Ticket inspected and customer message revealed.")
        return 0.04

    def _handle_search_kb(self, action: SupportopsXAction, events: List[str]) -> float:
        page = normalize_target(action.target)
        if page not in KB_PAGES:
            self._invalid_actions += 1
            events.append(f"KB page '{action.target}' was not found.")
            return -0.12
        if page not in self._state.kb_pages_viewed:
            self._state.kb_pages_viewed.append(page)
            self._visible_facts.append(f"KB {page}: {KB_PAGES[page]}")
            events.append(f"Knowledge base page viewed: {page}.")
            return 0.05 if page in self._case["required_kb"] else -0.02
        events.append(f"Knowledge base page already viewed: {page}.")
        return -0.03

    def _handle_escalate(self, action: SupportopsXAction, events: List[str]) -> float:
        specialist = normalize_target(action.target)
        if specialist not in VALID_SPECIALISTS:
            self._invalid_actions += 1
            events.append(f"Unknown specialist target: {action.target}.")
            return -0.12
        if specialist not in self._state.escalations:
            self._state.escalations.append(specialist)
            response = self._case["specialist_responses"][specialist]
            self._visible_facts.append(response)
            events.append(f"Escalated to {specialist}.")
            return 0.06 if specialist in self._case["required_specialists"] else -0.04
        events.append(f"Already escalated to {specialist}.")
        return -0.03

    def _handle_assign_queue(
        self, action: SupportopsXAction, events: List[str]
    ) -> float:
        queue = normalize_target(action.target)
        if queue not in VALID_QUEUES:
            self._invalid_actions += 1
            events.append(f"Unknown queue target: {action.target}.")
            return -0.12
        self._state.assigned_queue = queue
        events.append(f"Assigned ticket to {queue}.")
        return 0.05 if queue == self._case["correct_queue"] else -0.08

    def _handle_set_followup(
        self, action: SupportopsXAction, events: List[str]
    ) -> float:
        if action.hours is None:
            self._invalid_actions += 1
            events.append("Follow-up action requires hours.")
            return -0.12
        self._state.followup_hours = action.hours
        max_hours = self._case["followup_max_hours"]
        if action.hours <= max_hours:
            events.append(f"Follow-up scheduled in {action.hours}h.")
            return 0.05
        events.append(f"Follow-up in {action.hours}h misses the {max_hours}h expectation.")
        return -0.08

    def _handle_send_customer_reply(
        self, action: SupportopsXAction, events: List[str]
    ) -> float:
        message = (action.message or "").strip()
        if not message:
            self._invalid_actions += 1
            events.append("Customer reply cannot be empty.")
            return -0.12
        self._state.customer_reply = message
        lowered = message.lower()
        banned_hits = [word for word in self._case["banned_keywords"] if word in lowered]
        if banned_hits:
            self._policy_violations += len(banned_hits)
            events.append("Customer reply contained unsafe or non-compliant language.")
            return -0.15
        events.append("Customer reply drafted.")
        return 0.05

    def _handle_finalize(self, action: SupportopsXAction, events: List[str]) -> float:
        status = normalize_target(action.target)
        if status not in VALID_FINAL_STATUSES:
            self._invalid_actions += 1
            events.append(f"Unknown final status: {action.target}.")
            self._state.final_status = "pending"
            return -0.15
        self._state.final_status = status
        if status != self._case["expected_final_status"]:
            events.append(
                f"Final status '{status}' does not match case expectation "
                f"'{self._case['expected_final_status']}'."
            )
            return -0.12
        events.append(f"Final status set to {status}.")
        return 0.03

    def _compute_reward_breakdown(self) -> RewardBreakdown:
        return self._reward_model.score(
            RewardContext(
                case=self._case,
                state=self._state,
                policy_violations=self._policy_violations,
                invalid_actions=self._invalid_actions,
                repeated_actions=self._repeated_actions,
            )
        )

    def _communication_score(self, reply: str) -> float:
        if not reply:
            return 0.0
        keyword_hits = sum(1 for word in self._case["reply_keywords"] if word in reply)
        return min(1.0, keyword_hits / max(1, len(self._case["reply_keywords"])))

    def _scorecard(self) -> Dict[str, Any]:
        case = self._case
        breakdown = self._reward_breakdown or self._compute_reward_breakdown()
        return {
            "opened_ticket": self._state.opened_ticket,
            "kb_required": case["required_kb"],
            "kb_pages_viewed": self._state.kb_pages_viewed,
            "coordination_required": case["required_specialists"],
            "escalations": self._state.escalations,
            "correct_queue": self._state.assigned_queue == case["correct_queue"],
            "followup_within_sla": (
                self._state.followup_hours is not None
                and self._state.followup_hours <= case["followup_max_hours"]
            ),
            "communication_quality": self._communication_score(
                (self._state.customer_reply or "").lower()
            ),
            "correct_final_status": self._state.final_status
            == case["expected_final_status"],
            "policy_violations": self._policy_violations,
            "invalid_actions": self._invalid_actions,
            "repeated_actions": self._repeated_actions,
            "rubric_components": breakdown.components,
            "rubric_weights": breakdown.weights,
            "rubric_penalty": breakdown.penalty,
            "rubric_weighted_score": breakdown.weighted_score,
            "normalized_score": self._score,
        }

    def _observation(
        self, reward: float, events: Optional[List[str]] = None
    ) -> SupportopsXObservation:
        return SupportopsXObservation(
            task_id=self._state.task_id or "",
            ticket_id=self._state.ticket_id or "",
            visible_state="\n".join(self._visible_facts),
            available_actions=[
                "inspect_ticket",
                "search_kb",
                "escalate",
                "assign_queue",
                "set_followup",
                "send_customer_reply",
                "finalize",
            ],
            step=self._state.step_count,
            opened_ticket=self._state.opened_ticket,
            kb_pages_viewed=list(self._state.kb_pages_viewed),
            escalations=list(self._state.escalations),
            assigned_queue=self._state.assigned_queue,
            followup_hours=self._state.followup_hours,
            customer_reply=self._state.customer_reply,
            final_status=self._state.final_status,
            score=self._score,
            scorecard=self._scorecard() if self._case else {},
            events=events or list(self._events[-3:]),
            done=self._done,
            reward=reward,
            metadata={
                "task_id": self._state.task_id,
                "score": self._score,
                "scorecard": self._scorecard() if self._case else {},
            },
        )

    def _sync_state_score(self) -> None:
        self._state.score = self._score
        self._state.done = self._done
        self._state.policy_violations = self._policy_violations

    def _inject_delayed_update(self, events: List[str]) -> None:
        update_step = self._case.get("delayed_update_step")
        update = self._case.get("delayed_update")
        if not update_step or not update:
            return
        if self._state.step_count == update_step and update not in self._visible_facts:
            self._visible_facts.append(update)
            events.append("A delayed customer update arrived.")

    @staticmethod
    def _action_key(action: SupportopsXAction) -> str:
        return "|".join(
            [
                action.action_type,
                normalize_target(action.target),
                (action.message or "").strip().lower(),
                str(action.hours or ""),
            ]
        )
