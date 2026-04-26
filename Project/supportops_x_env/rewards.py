"""Composable OpenEnv rubrics for SupportOps-X scoring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping

from openenv.core.rubrics.base import Rubric
from openenv.core.rubrics.containers import WeightedSum

from .models import SupportopsXState


COMPONENT_WEIGHTS: Dict[str, float] = {
    "opened_ticket": 0.10,
    "policy_retrieval": 0.15,
    "specialist_coordination": 0.20,
    "queue_assignment": 0.15,
    "sla_followup": 0.10,
    "customer_communication": 0.15,
    "final_status": 0.10,
    "safety": 0.05,
}


@dataclass(frozen=True)
class RewardContext:
    """Inputs needed by the reward rubrics."""

    case: Mapping[str, Any]
    state: SupportopsXState
    policy_violations: int
    invalid_actions: int
    repeated_actions: int


@dataclass(frozen=True)
class RewardBreakdown:
    """Component-level reward report exposed to judges and trainers."""

    components: Dict[str, float]
    weights: Dict[str, float]
    weighted_score: float
    penalty: float
    normalized_score: float


class TicketOpenedRubric(Rubric):
    def forward(self, action: Any, observation: RewardContext) -> float:
        return 1.0 if observation.state.opened_ticket else 0.0


class PolicyRetrievalRubric(Rubric):
    def forward(self, action: Any, observation: RewardContext) -> float:
        required = set(observation.case["required_kb"])
        viewed = set(observation.state.kb_pages_viewed)
        return len(required & viewed) / max(1, len(required))


class SpecialistCoordinationRubric(Rubric):
    def forward(self, action: Any, observation: RewardContext) -> float:
        required = set(observation.case["required_specialists"])
        escalations = set(observation.state.escalations)
        return len(required & escalations) / max(1, len(required))


class QueueAssignmentRubric(Rubric):
    def forward(self, action: Any, observation: RewardContext) -> float:
        return (
            1.0
            if observation.state.assigned_queue == observation.case["correct_queue"]
            else 0.0
        )


class SlaFollowupRubric(Rubric):
    def forward(self, action: Any, observation: RewardContext) -> float:
        return (
            1.0
            if observation.state.followup_hours is not None
            and observation.state.followup_hours <= observation.case["followup_max_hours"]
            else 0.0
        )


class CustomerCommunicationRubric(Rubric):
    def forward(self, action: Any, observation: RewardContext) -> float:
        reply = (observation.state.customer_reply or "").lower()
        if not reply:
            return 0.0
        keywords = observation.case["reply_keywords"]
        keyword_hits = sum(1 for word in keywords if word in reply)
        return min(1.0, keyword_hits / max(1, len(keywords)))


class FinalStatusRubric(Rubric):
    def forward(self, action: Any, observation: RewardContext) -> float:
        return (
            1.0
            if observation.state.final_status == observation.case["expected_final_status"]
            else 0.0
        )


class SafetyRubric(Rubric):
    def forward(self, action: Any, observation: RewardContext) -> float:
        return 1.0 if observation.policy_violations == 0 else 0.0


class SupportOpsRewardModel:
    """Named, composable reward model built from OpenEnv rubrics."""

    def __init__(self) -> None:
        self.components: Dict[str, Rubric] = {
            "opened_ticket": TicketOpenedRubric(),
            "policy_retrieval": PolicyRetrievalRubric(),
            "specialist_coordination": SpecialistCoordinationRubric(),
            "queue_assignment": QueueAssignmentRubric(),
            "sla_followup": SlaFollowupRubric(),
            "customer_communication": CustomerCommunicationRubric(),
            "final_status": FinalStatusRubric(),
            "safety": SafetyRubric(),
        }
        self.weighted_sum = WeightedSum(
            [self.components[name] for name in COMPONENT_WEIGHTS],
            weights=[COMPONENT_WEIGHTS[name] for name in COMPONENT_WEIGHTS],
        )

    def score(self, context: RewardContext) -> RewardBreakdown:
        """Return weighted score plus anti-hacking penalties."""

        weighted_score = float(self.weighted_sum(None, context))
        components = {
            name: round(float(rubric.last_score or 0.0), 4)
            for name, rubric in self.components.items()
        }
        penalty = min(
            0.20,
            0.03 * context.invalid_actions + 0.01 * context.repeated_actions,
        )
        normalized_score = round(max(0.0, min(1.0, weighted_score - penalty)), 4)
        return RewardBreakdown(
            components=components,
            weights=dict(COMPONENT_WEIGHTS),
            weighted_score=round(weighted_score, 4),
            penalty=round(penalty, 4),
            normalized_score=normalized_score,
        )
