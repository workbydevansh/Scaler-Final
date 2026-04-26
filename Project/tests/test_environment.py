from fastapi.testclient import TestClient

from supportops_x_env.models import SupportopsXAction
from supportops_x_env.server.app import app
from supportops_x_env.server.environment import SupportopsXEnvironment


def expert_actions(task_id: str):
    if task_id == "damaged_refund":
        return [
            SupportopsXAction(action_type="inspect_ticket"),
            SupportopsXAction(action_type="search_kb", target="damaged_goods_refund"),
            SupportopsXAction(action_type="escalate", target="billing"),
            SupportopsXAction(action_type="assign_queue", target="billing"),
            SupportopsXAction(action_type="set_followup", hours=24),
            SupportopsXAction(
                action_type="send_customer_reply",
                message=(
                    "We reviewed the damaged item report with Billing. Please attach a photo "
                    "if available; we can proceed with a refund or replacement under policy."
                ),
            ),
            SupportopsXAction(action_type="finalize", target="solved"),
        ]
    if task_id == "account_takeover":
        return [
            SupportopsXAction(action_type="inspect_ticket"),
            SupportopsXAction(action_type="search_kb", target="account_takeover_policy"),
            SupportopsXAction(action_type="escalate", target="security"),
            SupportopsXAction(action_type="assign_queue", target="security"),
            SupportopsXAction(action_type="set_followup", hours=1),
            SupportopsXAction(
                action_type="send_customer_reply",
                message=(
                    "Security is protecting your account and reviewing verification. "
                    "We will keep this pending and send a follow-up after checks complete."
                ),
            ),
            SupportopsXAction(action_type="finalize", target="pending"),
        ]
    return [
        SupportopsXAction(action_type="inspect_ticket"),
        SupportopsXAction(action_type="search_kb", target="enterprise_incident_protocol"),
        SupportopsXAction(action_type="escalate", target="sre"),
        SupportopsXAction(action_type="escalate", target="account_management"),
        SupportopsXAction(action_type="assign_queue", target="sre"),
        SupportopsXAction(action_type="set_followup", hours=1),
        SupportopsXAction(
            action_type="send_customer_reply",
            message=(
                "We have opened an incident with SRE and looped in your account manager. "
                "The case remains pending and we will provide updates while recovery is monitored."
            ),
        ),
        SupportopsXAction(action_type="finalize", target="pending"),
    ]


def test_reset_returns_initial_observation():
    env = SupportopsXEnvironment()
    obs = env.reset(task_id="enterprise_outage")

    assert obs.task_id == "enterprise_outage"
    assert obs.ticket_id == "TCK-3307"
    assert obs.done is False
    assert "Enterprise checkout outage" in obs.visible_state


def test_expert_paths_score_high():
    for task_id in ["damaged_refund", "account_takeover", "enterprise_outage"]:
        env = SupportopsXEnvironment()
        obs = env.reset(task_id=task_id)
        for action in expert_actions(task_id):
            obs = env.step(action)

        assert obs.done is True
        assert obs.score >= 0.92
        assert obs.scorecard["policy_violations"] == 0
        assert obs.scorecard["rubric_components"]["policy_retrieval"] == 1.0
        assert obs.scorecard["rubric_components"]["specialist_coordination"] == 1.0


def test_wrong_queue_and_unsafe_reply_are_penalized():
    env = SupportopsXEnvironment()
    env.reset(task_id="account_takeover")
    env.step(SupportopsXAction(action_type="inspect_ticket"))
    env.step(SupportopsXAction(action_type="assign_queue", target="billing"))
    obs = env.step(
        SupportopsXAction(
            action_type="send_customer_reply",
            message="This is solved. Send your password and we will close the case.",
        )
    )

    assert obs.scorecard["correct_queue"] is False
    assert obs.scorecard["policy_violations"] > 0
    assert obs.score < 0.4


def test_keyword_stuffing_without_process_does_not_score_high():
    env = SupportopsXEnvironment()
    env.reset(task_id="enterprise_outage")
    obs = env.step(
        SupportopsXAction(
            action_type="send_customer_reply",
            message=(
                "incident sre account updates pending incident sre account updates pending"
            ),
        )
    )
    obs = env.step(SupportopsXAction(action_type="finalize", target="pending"))

    assert obs.scorecard["rubric_components"]["customer_communication"] == 1.0
    assert obs.scorecard["rubric_components"]["policy_retrieval"] == 0.0
    assert obs.scorecard["rubric_components"]["specialist_coordination"] == 0.0
    assert obs.score < 0.35


def test_repeated_invalid_actions_and_step_limit_are_penalized():
    env = SupportopsXEnvironment()
    obs = env.reset(task_id="damaged_refund")
    for _ in range(12):
        obs = env.step(SupportopsXAction(action_type="search_kb", target="missing"))
        if obs.done:
            break

    assert obs.done is True
    assert obs.scorecard["invalid_actions"] >= 1
    assert obs.scorecard["repeated_actions"] >= 1
    assert obs.scorecard["rubric_penalty"] == 0.2
    assert obs.score <= 0.05


def test_demo_endpoints_are_task_selectable_and_stateful():
    client = TestClient(app)

    reset_response = client.post("/demo/reset", json={"task_id": "account_takeover"})
    assert reset_response.status_code == 200
    reset_payload = reset_response.json()
    assert reset_payload["observation"]["task_id"] == "account_takeover"
    assert reset_payload["observation"]["step"] == 0

    step_response = client.post(
        "/demo/step",
        json={"action": {"action_type": "inspect_ticket"}},
    )
    assert step_response.status_code == 200
    step_payload = step_response.json()
    assert step_payload["observation"]["task_id"] == "account_takeover"
    assert step_payload["observation"]["step"] == 1
    assert "Customer message" in step_payload["observation"]["visible_state"]

    invalid_response = client.post("/demo/reset", json={"task_id": "not_a_task"})
    assert invalid_response.status_code == 400
