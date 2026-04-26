"""Pydantic models for SupportOps-X."""

from typing import Any, Dict, List, Literal, Optional

from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field


SupportActionType = Literal[
    "inspect_ticket",
    "search_kb",
    "escalate",
    "assign_queue",
    "set_followup",
    "send_customer_reply",
    "finalize",
]


class SupportopsXAction(Action):
    """Action sent by the support lead agent."""

    action_type: SupportActionType = Field(
        ..., description="The support operation to perform."
    )
    target: Optional[str] = Field(
        default=None,
        description="Queue, KB page, specialist, or final status depending on action_type.",
    )
    message: Optional[str] = Field(
        default=None,
        description="Customer-facing message when action_type is send_customer_reply.",
    )
    hours: Optional[int] = Field(
        default=None,
        ge=1,
        le=168,
        description="Follow-up window in hours when action_type is set_followup.",
    )


class SupportopsXObservation(Observation):
    """Observation returned by the SupportOps-X environment."""

    task_id: str = Field(default="", description="Current task id.")
    ticket_id: str = Field(default="", description="Current support ticket id.")
    visible_state: str = Field(default="", description="Text visible to the agent.")
    available_actions: List[str] = Field(default_factory=list)
    step: int = 0
    opened_ticket: bool = False
    kb_pages_viewed: List[str] = Field(default_factory=list)
    escalations: List[str] = Field(default_factory=list)
    assigned_queue: Optional[str] = None
    followup_hours: Optional[int] = None
    customer_reply: Optional[str] = None
    final_status: Optional[str] = None
    score: float = 0.0
    scorecard: Dict[str, Any] = Field(default_factory=dict)
    events: List[str] = Field(default_factory=list)


class SupportopsXState(State):
    """Serializable state summary for the current episode."""

    task_id: Optional[str] = None
    ticket_id: Optional[str] = None
    done: bool = False
    score: float = 0.0
    opened_ticket: bool = False
    kb_pages_viewed: List[str] = Field(default_factory=list)
    escalations: List[str] = Field(default_factory=list)
    assigned_queue: Optional[str] = None
    followup_hours: Optional[int] = None
    customer_reply: Optional[str] = None
    final_status: Optional[str] = None
    policy_violations: int = 0
