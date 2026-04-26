"""Tool-callable environment wrapper for TRL GRPO."""

from __future__ import annotations

from supportops_x_env.models import SupportopsXAction
from supportops_x_env.server.environment import SupportopsXEnvironment


class SupportOpsToolEnv:
    """TRL environment_factory wrapper with descriptive tools."""

    def __init__(self) -> None:
        self.env = SupportopsXEnvironment()
        self.reward = 0.0
        self.done = False

    def reset(self, **kwargs) -> str:
        task_id = kwargs.get("task_id")
        obs = self.env.reset(task_id=task_id)
        self.reward = 0.0
        self.done = False
        return self._format_observation(obs)

    def inspect_ticket(self) -> str:
        """
        Open the current support ticket and reveal the customer message.

        Returns:
            Updated environment observation.
        """

        return self._step(SupportopsXAction(action_type="inspect_ticket"))

    def search_kb(self, page: str) -> str:
        """
        Search a knowledge-base page by id.

        Args:
            page: One of damaged_goods_refund, account_takeover_policy, or enterprise_incident_protocol.

        Returns:
            The policy result and current score feedback.
        """

        return self._step(SupportopsXAction(action_type="search_kb", target=page))

    def escalate(self, specialist: str) -> str:
        """
        Ask a specialist agent for help.

        Args:
            specialist: One of billing, security, sre, or account_management.

        Returns:
            The specialist response and current score feedback.
        """

        return self._step(SupportopsXAction(action_type="escalate", target=specialist))

    def assign_queue(self, queue: str) -> str:
        """
        Assign the ticket to an ownership queue.

        Args:
            queue: One of billing, security, sre, account_management, or general.

        Returns:
            Current assignment feedback.
        """

        return self._step(SupportopsXAction(action_type="assign_queue", target=queue))

    def set_followup(self, hours: int) -> str:
        """
        Schedule the next customer or internal follow-up.

        Args:
            hours: Number of hours until follow-up.

        Returns:
            Follow-up feedback.
        """

        return self._step(SupportopsXAction(action_type="set_followup", hours=hours))

    def send_customer_reply(self, message: str) -> str:
        """
        Draft a customer-facing reply.

        Args:
            message: Safe, policy-compliant customer update.

        Returns:
            Reply quality and safety feedback.
        """

        return self._step(
            SupportopsXAction(action_type="send_customer_reply", message=message)
        )

    def finalize_case(self, status: str) -> str:
        """
        Finish the case with a final status.

        Args:
            status: One of solved, pending, or escalated.

        Returns:
            Final score feedback.
        """

        return self._step(SupportopsXAction(action_type="finalize", target=status))

    def _step(self, action: SupportopsXAction) -> str:
        if self.done:
            raise ValueError("Episode already complete. Call reset for a new case.")
        obs = self.env.step(action)
        self.done = obs.done
        self.reward += float(obs.reward or 0.0)
        if obs.done:
            self.reward = 10.0 * obs.score - obs.scorecard.get("policy_violations", 0)
        return self._format_observation(obs)

    @staticmethod
    def _format_observation(obs) -> str:
        return (
            f"Task: {obs.task_id}\n"
            f"Ticket: {obs.ticket_id}\n"
            f"Step: {obs.step}\n"
            f"Score: {obs.score:.2f}\n"
            f"Events: {'; '.join(obs.events)}\n"
            f"Visible state:\n{obs.visible_state}\n"
            f"Scorecard: {obs.scorecard}"
        )


def reward_func(environments, **kwargs) -> list[float]:
    """Return the accumulated SupportOps-X reward for TRL."""

    return [float(env.reward) for env in environments]

