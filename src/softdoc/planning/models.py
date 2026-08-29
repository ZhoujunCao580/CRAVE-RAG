"""Pydantic state models for the initial question planner."""

from __future__ import annotations

from typing import Any, Self

from pydantic import Field, field_validator, model_validator

from softdoc.models import SoftDocModel
def _clean_unique_strings(values: list[str], *, field_name: str) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for value in values:
        stripped = value.strip()
        if not stripped:
            raise ValueError(f"{field_name} entries must not be blank")
        if stripped in seen:
            raise ValueError(f"{field_name} entries must be unique")
        seen.add(stripped)
        cleaned.append(stripped)
    return cleaned


class PlannedSubQuestion(SoftDocModel):
    """One independently checkable information need in the initial plan."""

    subquestion_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    depends_on: list[str] = Field(default_factory=list)

    @field_validator("subquestion_id", "text")
    @classmethod
    def strip_required_text(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("SubQuestion fields must not be blank")
        return stripped

    @field_validator("depends_on")
    @classmethod
    def clean_string_lists(cls, value: list[str], info: Any) -> list[str]:
        return _clean_unique_strings(value, field_name=info.field_name)


class PlannerDraft(SoftDocModel):
    """Strict model-facing output before runtime trace is attached.

    An empty ``subquestions`` list is a valid plan without decomposition: the
    original question itself becomes the first reading target.  The field remains
    required so a missing plan is not confused with an intentional empty plan.
    """

    original_question: str = Field(min_length=1)
    subquestions: list[PlannedSubQuestion]

    @field_validator("original_question")
    @classmethod
    def strip_question(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("The original question must not be blank")
        return stripped

    @model_validator(mode="after")
    def validate_subquestion_graph(self) -> Self:
        by_id: dict[str, PlannedSubQuestion] = {}
        normalized_texts: set[str] = set()
        for subquestion in self.subquestions:
            if subquestion.subquestion_id in by_id:
                raise ValueError("SubQuestion IDs must be unique")
            normalized_text = " ".join(subquestion.text.casefold().split())
            if normalized_text in normalized_texts:
                raise ValueError("SubQuestion texts must not be duplicated")
            by_id[subquestion.subquestion_id] = subquestion
            normalized_texts.add(normalized_text)

        for subquestion in self.subquestions:
            if subquestion.subquestion_id in subquestion.depends_on:
                raise ValueError("A SubQuestion cannot depend on itself")
            unknown = [
                dependency
                for dependency in subquestion.depends_on
                if dependency not in by_id
            ]
            if unknown:
                raise ValueError(
                    "SubQuestion dependencies must reference existing IDs: "
                    + ", ".join(unknown)
                )

        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(subquestion_id: str) -> None:
            if subquestion_id in visiting:
                raise ValueError("SubQuestion dependencies must form a DAG")
            if subquestion_id in visited:
                return
            visiting.add(subquestion_id)
            for dependency in by_id[subquestion_id].depends_on:
                visit(dependency)
            visiting.remove(subquestion_id)
            visited.add(subquestion_id)

        for subquestion_id in by_id:
            visit(subquestion_id)

        return self


class PlannerWarning(SoftDocModel):
    code: str = Field(min_length=1)
    description: str = Field(min_length=1)


class PlannerTrace(SoftDocModel):
    backend_name: str = Field(min_length=1)
    model: str = Field(min_length=1)
    prompt_version: str = Field(min_length=1)
    warnings: list[PlannerWarning] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class InitialPlan(PlannerDraft):
    """Validated initial plan with system-owned provenance."""

    planner_trace: PlannerTrace


class PlannerBackendResponse(SoftDocModel):
    """Raw response returned by an injectable planner backend."""

    content: str = Field(min_length=1)
    model: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("content", "model")
    @classmethod
    def strip_backend_text(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("Planner backend response fields must not be blank")
        return stripped


class PlannerConfig(SoftDocModel):
    """Runtime limits for the initial SubQuestion DAG.

    ``max_depth`` counts the original question as the implicit root at depth 1.
    An empty plan therefore has depth 1, while an independent SubQuestion
    is at depth 2.  Setting ``max_subquestions=0`` or ``max_depth=1`` forces a
    empty plan and is useful for controlled ablations.
    """

    max_subquestions: int = Field(default=6, ge=0)
    max_depth: int = Field(default=4, ge=1)
    max_validation_attempts: int = Field(default=2, ge=1, le=3)
