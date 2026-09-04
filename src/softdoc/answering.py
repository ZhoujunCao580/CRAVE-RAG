"""Frozen Answerer v0 contracts and prompts.

The Answerer receives only the Root Question, a compact question DAG, and
Checker-accepted Evidence.  Source locations stay outside the model context;
the program can expand returned Evidence IDs into citations afterwards.
"""

from __future__ import annotations

import json
from typing import Self

from pydantic import Field, field_validator, model_validator

from softdoc.models import SoftDocModel

from softdoc.prompts import load_prompt_text
from softdoc.reading_state import EvidenceMemory, EvidenceStatus, RootQuestion


ANSWERER_PROMPT_VERSION = "answerer-v0.8"


def _unique_nonblank(values: list[str], *, label: str) -> list[str]:
    if any(not value.strip() for value in values):
        raise ValueError(f"{label} must not contain blank IDs")
    if len(values) != len(set(values)):
        raise ValueError(f"{label} must be unique")
    return values


class AnswerQuestionNode(SoftDocModel):
    """One Planner node exposed as an organizational hint to the Answerer."""

    question_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    depends_on: list[str] = Field(default_factory=list)

    @field_validator("question_id", "text")
    @classmethod
    def strip_required_text(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("Answer question fields must not be blank")
        return stripped

    @field_validator("depends_on")
    @classmethod
    def validate_dependencies(cls, value: list[str]) -> list[str]:
        return _unique_nonblank(value, label="Answer question dependency IDs")


class AnswerEvidence(SoftDocModel):
    """One accepted Evidence statement available for final synthesis."""

    evidence_id: str = Field(min_length=1)
    statement: str = Field(min_length=1)
    supports_question_ids: list[str] = Field(min_length=1)

    @field_validator("evidence_id", "statement")
    @classmethod
    def strip_required_text(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("Answer Evidence fields must not be blank")
        return stripped

    @field_validator("supports_question_ids")
    @classmethod
    def validate_supports(cls, value: list[str]) -> list[str]:
        return _unique_nonblank(value, label="Answer Evidence question IDs")


class AnswerInput(SoftDocModel):
    """Ready-only, transient materialized input for one Answerer call."""

    reading_session_id: str = Field(min_length=1)
    root_question: RootQuestion
    question_graph: list[AnswerQuestionNode] = Field(default_factory=list)
    evidence: list[AnswerEvidence] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_graph_and_evidence(self) -> Self:
        question_ids = [item.question_id for item in self.question_graph]
        _unique_nonblank(question_ids, label="Answer question IDs")
        if self.root_question.question_id in question_ids:
            raise ValueError("Root Question must not be duplicated in question_graph")

        by_id = {item.question_id: item for item in self.question_graph}
        for item in self.question_graph:
            if item.question_id in item.depends_on:
                raise ValueError("An Answer question cannot depend on itself")
            unknown = set(item.depends_on).difference(by_id)
            if unknown:
                raise ValueError(
                    f"Question {item.question_id} has unknown dependencies: "
                    + ", ".join(sorted(unknown))
                )

        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(question_id: str) -> None:
            if question_id in visited:
                return
            if question_id in visiting:
                raise ValueError("Answer question dependencies must form a DAG")
            visiting.add(question_id)
            for dependency_id in by_id[question_id].depends_on:
                visit(dependency_id)
            visiting.remove(question_id)
            visited.add(question_id)

        for question_id in question_ids:
            visit(question_id)

        evidence_ids = [item.evidence_id for item in self.evidence]
        _unique_nonblank(evidence_ids, label="Answer Evidence IDs")
        known_question_ids = {self.root_question.question_id, *question_ids}
        for item in self.evidence:
            unknown = set(item.supports_question_ids).difference(known_question_ids)
            if unknown:
                raise ValueError(
                    f"Evidence {item.evidence_id} supports unknown questions: "
                    + ", ".join(sorted(unknown))
                )
        return self


class AnswerResult(SoftDocModel):
    """Minimal model-owned final answer plus the Evidence it actually used."""

    answer: str = Field(min_length=1)
    used_evidence_ids: list[str] = Field(min_length=1)

    @field_validator("answer")
    @classmethod
    def strip_answer(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("Answer must not be blank")
        return stripped

    @field_validator("used_evidence_ids")
    @classmethod
    def validate_used_evidence(cls, value: list[str]) -> list[str]:
        return _unique_nonblank(value, label="Used Evidence IDs")


class AnswerInputBuilder:
    """Materialize the small Answerer view from canonical EvidenceMemory."""

    def build(
        self,
        *,
        root_question: RootQuestion,
        evidence_memory: EvidenceMemory,
    ) -> AnswerInput:
        if evidence_memory.root_question_id != root_question.question_id:
            raise ValueError("Answer Root Question must match EvidenceMemory")
        if evidence_memory.root_status != EvidenceStatus.READY:
            raise ValueError("Answerer may run only when Root Evidence is ready")
        if evidence_memory.current_target is not None:
            raise ValueError("Ready Answerer input cannot have a current target")
        return AnswerInput(
            reading_session_id=evidence_memory.reading_session_id,
            root_question=root_question,
            question_graph=[
                AnswerQuestionNode(
                    question_id=item.question_id,
                    text=item.text,
                    depends_on=item.depends_on,
                )
                for item in evidence_memory.questions
            ],
            evidence=[
                AnswerEvidence(
                    evidence_id=item.evidence_id,
                    statement=item.statement,
                    supports_question_ids=item.supports_question_ids,
                )
                for item in evidence_memory.evidence
            ],
        )


def validate_answer_result(
    answer_input: AnswerInput,
    result: AnswerResult,
) -> AnswerResult:
    """Reject fabricated Evidence IDs before citation expansion."""

    available_ids = {item.evidence_id for item in answer_input.evidence}
    unknown = set(result.used_evidence_ids).difference(available_ids)
    if unknown:
        raise ValueError(
            "AnswerResult references unavailable Evidence: "
            + ", ".join(sorted(unknown))
        )
    return result


ANSWERER_SYSTEM_PROMPT = load_prompt_text("answerer_v0_8.txt")


def answerer_user_prompt(answer_input: AnswerInput) -> str:
    """Render the deterministic user message for one Answerer invocation."""

    input_json = answer_input.model_dump_json(indent=2)
    output_shape = json.dumps(
        {
            "answer": "<direct answer to the Root Question>",
            "used_evidence_ids": ["<one or more supplied evidence_id values>"],
        },
        indent=2,
    )
    return f"""Answer input:

{input_json}

Return exactly this JSON shape:

{output_shape}
"""
