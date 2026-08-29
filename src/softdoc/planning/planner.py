"""Initial Planner orchestration without a concrete model dependency."""

from __future__ import annotations

from pydantic import ValidationError

from softdoc.planning.backend import PlannerBackend
from softdoc.planning.models import (
    InitialPlan,
    PlannerConfig,
    PlannerDraft,
    PlannerTrace,
    PlannerWarning,
)
from softdoc.planning.prompt import (
    INITIAL_PLANNER_PROMPT_VERSION,
    build_initial_planner_system_prompt,
    build_initial_planner_user_prompt,
)


class PlannerOutputError(ValueError):
    """Raised when a backend response is not a valid initial plan."""


class InitialPlanner:
    """Question-only Planner v0; it does not retrieve, read, or choose actions."""

    def __init__(
        self,
        backend: PlannerBackend,
        config: PlannerConfig | None = None,
    ) -> None:
        self._backend = backend
        self._config = config or PlannerConfig()

    def create_plan(self, question: str) -> InitialPlan:
        stripped_question = question.strip()
        if not stripped_question:
            raise ValueError("The Planner question must not be blank")

        system_prompt = build_initial_planner_system_prompt(
            max_subquestions=self._config.max_subquestions,
            max_depth=self._config.max_depth,
        )
        initial_user_prompt = build_initial_planner_user_prompt(stripped_question)
        user_prompt = initial_user_prompt
        warnings: list[PlannerWarning] = []
        draft: PlannerDraft | None = None
        response = None
        last_error: PlannerOutputError | None = None
        for attempt in range(1, self._config.max_validation_attempts + 1):
            response = self._backend.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )
            try:
                draft = self._validate_response(response.content, stripped_question)
                break
            except PlannerOutputError as exc:
                last_error = exc
                if attempt == self._config.max_validation_attempts:
                    raise
                warnings.append(
                    PlannerWarning(
                        code="planner_validation_retry",
                        description=f"Attempt {attempt} was rejected: {exc}",
                    )
                )
                user_prompt = self._build_correction_prompt(
                    initial_user_prompt,
                    response.content,
                    str(exc),
                )

        if draft is None or response is None:
            raise last_error or PlannerOutputError("Planner produced no valid plan")

        metadata = dict(response.metadata)
        metadata["validation_attempts"] = len(warnings) + 1
        return InitialPlan(
            original_question=draft.original_question,
            subquestions=draft.subquestions,
            planner_trace=PlannerTrace(
                backend_name=self._backend.backend_name,
                model=response.model,
                prompt_version=INITIAL_PLANNER_PROMPT_VERSION,
                warnings=warnings,
                metadata=metadata,
            ),
        )

    def _validate_response(
        self,
        content: str,
        stripped_question: str,
    ) -> PlannerDraft:
        try:
            draft = PlannerDraft.model_validate_json(content)
        except ValidationError as exc:
            raise PlannerOutputError(
                "Planner backend returned invalid strict JSON or an invalid plan: "
                + str(exc.errors(include_url=False)[0].get("msg", "validation failed"))
            ) from exc

        if draft.original_question != stripped_question:
            raise PlannerOutputError(
                "Planner output must preserve the original question verbatim"
            )

        self._validate_plan_limits(draft)
        return draft

    @staticmethod
    def _build_correction_prompt(
        initial_user_prompt: str,
        rejected_content: str,
        error_message: str,
    ) -> str:
        return (
            initial_user_prompt
            + "\n\nYour previous response was rejected by the deterministic validator.\n"
            + f"Validation error: {error_message}\n"
            + "Correct the error and return the complete strict JSON object again.\n"
            + "Previous response:\n"
            + rejected_content
        )

    def _validate_plan_limits(self, draft: PlannerDraft) -> None:
        if len(draft.subquestions) > self._config.max_subquestions:
            raise PlannerOutputError(
                "Planner output exceeds the configured SubQuestion limit: "
                f"{len(draft.subquestions)} > {self._config.max_subquestions}"
            )

        by_id = {item.subquestion_id: item for item in draft.subquestions}
        cached_depth: dict[str, int] = {}

        def depth(subquestion_id: str) -> int:
            if subquestion_id in cached_depth:
                return cached_depth[subquestion_id]
            dependencies = by_id[subquestion_id].depends_on
            value = 2 if not dependencies else 1 + max(depth(item) for item in dependencies)
            cached_depth[subquestion_id] = value
            return value

        # The Root is the only node in a valid empty plan.
        actual_depth = max((depth(item) for item in by_id), default=1)
        if actual_depth > self._config.max_depth:
            raise PlannerOutputError(
                "Planner output exceeds the configured DAG depth: "
                f"{actual_depth} > {self._config.max_depth}"
            )
