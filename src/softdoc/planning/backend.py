"""Injectable model boundary for Planner v0."""

from __future__ import annotations

from typing import Protocol

from softdoc.planning.models import PlannerBackendResponse


class PlannerBackend(Protocol):
    """Interface implemented later by local or hosted language models."""

    @property
    def backend_name(self) -> str: ...

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> PlannerBackendResponse: ...
