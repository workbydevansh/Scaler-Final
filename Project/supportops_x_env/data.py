"""Static task data for SupportOps-X."""

from __future__ import annotations

from typing import Any, Dict


KB_PAGES: Dict[str, str] = {
    "damaged_goods_refund": (
        "Damaged goods refunds require a ticket note, photo or customer attestation, "
        "Billing ownership, and a customer reply explaining refund or replacement timing."
    ),
    "account_takeover_policy": (
        "Suspected account takeover must be treated as security-sensitive. Escalate to "
        "Security, do not request passwords, keep the case pending, and schedule a short follow-up."
    ),
    "enterprise_incident_protocol": (
        "Enterprise outage reports require SRE escalation, Account Management notification, "
        "customer updates during the incident, and pending status until recovery is confirmed."
    ),
}


CASE_LIBRARY: Dict[str, Dict[str, Any]] = {
    "damaged_refund": {
        "ticket_id": "TCK-1042",
        "title": "Damaged item refund request",
        "customer": "Maya Patel",
        "customer_tier": "standard",
        "severity": "medium",
        "sla_hours": 24,
        "initial_message": (
            "My replacement coffee grinder arrived with a cracked housing. I need a refund "
            "or replacement before this weekend."
        ),
        "delayed_update_step": None,
        "delayed_update": "",
        "required_kb": ["damaged_goods_refund"],
        "required_specialists": ["billing"],
        "correct_queue": "billing",
        "expected_final_status": "solved",
        "followup_max_hours": 24,
        "reply_keywords": ["refund", "replacement", "damaged", "photo", "billing"],
        "banned_keywords": ["guaranteed without review", "ignore policy"],
        "specialist_responses": {
            "billing": (
                "Billing: customer is eligible if the damaged item is documented. "
                "Offer refund or replacement and note the evidence request."
            ),
            "security": "Security: no account risk is visible. This is not a Security-owned issue.",
            "sre": "SRE: no production incident is tied to this ticket.",
            "account_management": "Account Management: standard account, no AM intervention needed.",
        },
    },
    "account_takeover": {
        "ticket_id": "TCK-2198",
        "title": "Possible account takeover",
        "customer": "Daniel Cho",
        "customer_tier": "pro",
        "severity": "urgent",
        "sla_hours": 2,
        "initial_message": (
            "I am locked out and I see orders I did not place. Someone changed my recovery email."
        ),
        "delayed_update_step": 3,
        "delayed_update": (
            "New customer update: the attacker just attempted another password reset."
        ),
        "required_kb": ["account_takeover_policy"],
        "required_specialists": ["security"],
        "correct_queue": "security",
        "expected_final_status": "pending",
        "followup_max_hours": 2,
        "reply_keywords": ["security", "protected", "verification", "follow-up", "pending"],
        "banned_keywords": ["send your password", "closed", "solved", "ignore verification"],
        "specialist_responses": {
            "security": (
                "Security: suspicious login and recovery-email change confirmed. "
                "Lock risky sessions, keep ticket pending, and update the customer after verification."
            ),
            "billing": "Billing: unauthorized orders may need review later, but Security owns the case first.",
            "sre": "SRE: no service outage is active.",
            "account_management": "Account Management: no enterprise contract is attached.",
        },
    },
    "enterprise_outage": {
        "ticket_id": "TCK-3307",
        "title": "Enterprise checkout outage",
        "customer": "Northstar Retail",
        "customer_tier": "enterprise",
        "severity": "critical",
        "sla_hours": 1,
        "initial_message": (
            "Our checkout is failing across stores during launch hour. We need an incident owner now."
        ),
        "delayed_update_step": 4,
        "delayed_update": (
            "New customer update: failures are concentrated in EU checkout and revenue impact is rising."
        ),
        "required_kb": ["enterprise_incident_protocol"],
        "required_specialists": ["sre", "account_management"],
        "correct_queue": "sre",
        "expected_final_status": "pending",
        "followup_max_hours": 1,
        "reply_keywords": ["incident", "sre", "account", "updates", "pending"],
        "banned_keywords": ["resolved", "no outage", "wait a few days", "closed"],
        "specialist_responses": {
            "sre": (
                "SRE: active incident INC-224. Error spike started after the checkout deploy. "
                "Rollback is in progress; keep support case pending."
            ),
            "account_management": (
                "Account Management: Northstar is high-touch enterprise. Send frequent updates "
                "and note that the AM has been looped in."
            ),
            "billing": "Billing: no invoice or payment collection issue is visible.",
            "security": "Security: no compromise indicators are attached to this case.",
        },
    },
}


VALID_QUEUES = {"billing", "security", "sre", "account_management", "general"}
VALID_SPECIALISTS = {"billing", "security", "sre", "account_management"}
VALID_FINAL_STATUSES = {"solved", "pending", "escalated"}


def normalize_target(value: str | None) -> str:
    """Normalize free-form target names into environment identifiers."""

    if not value:
        return ""
    lowered = value.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "account_manager": "account_management",
        "account_management_agent": "account_management",
        "am": "account_management",
        "site_reliability": "sre",
        "site_reliability_engineering": "sre",
        "support": "general",
    }
    return aliases.get(lowered, lowered)

