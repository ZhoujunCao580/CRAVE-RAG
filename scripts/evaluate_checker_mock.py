"""Run a model-only Evidence Checker evaluation against synthetic cases.

This probe deliberately does not call a real Reader.  It constructs validated
EvidenceCheckInput objects, asks a local Ollama model for EvidenceCheckDecision
JSON, materializes the runtime-derived provenance fields, validates and applies
the delta with the production state transition, and writes an auditable report.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib import error, request

from pydantic import ValidationError

from softdoc.checking_prompt import (
    CHECKER_PROMPT_VERSION,
    CHECKER_SYSTEM_PROMPT,
)
from softdoc.reading_state import (
    CurrentTarget,
    EvidenceCheckDecision,
    EvidenceCheckInput,
    EvidenceCheckResult,
    EvidenceItem,
    EvidenceMemory,
    EvidenceStatus,
    ObservationLimitation,
    ObservationSourceRef,
    QuestionState,
    QuestionStatus,
    RootQuestion,
    StoredObservation,
    apply_evidence_check_result,
    materialize_evidence_check_decision,
)


@dataclass(frozen=True)
class Expected:
    used: dict[str, bool]
    add: int
    replace: int = 0
    remove: int = 0
    target_status: str = "incomplete"
    root_status: str = "incomplete"
    gap_required: bool = True
    gap_terms: tuple[str, ...] = ()
    gap_forbidden_terms: tuple[str, ...] = ()
    evidence_required_terms: tuple[str, ...] = ()
    evidence_forbidden_terms: tuple[str, ...] = ()
    next_target: str | None = None


@dataclass
class Case:
    case_id: str
    category: str
    description: str
    checker_input: EvidenceCheckInput
    expected: Expected
    chained_from: str | None = None


def observation(case_id: str, index: int, text: str, input_id: str = "I1") -> StoredObservation:
    action_id = f"action:{case_id}"
    return StoredObservation(
        observation_id=f"obs:{case_id}:{index:02d}",
        action_id=action_id,
        text=text,
        sources=[ObservationSourceRef(input_id=input_id)],
    )


def evidence(case_id: str, index: int, statement: str, supports: str) -> EvidenceItem:
    return EvidenceItem(
        evidence_id=f"evidence:{case_id}:{index:02d}",
        statement=statement,
        observation_ids=[f"obs:prior:{case_id}:{index:02d}"],
        supports_question_ids=[supports],
    )


def make_input(
    case_id: str,
    root_text: str,
    *,
    target_id: str,
    gap: str,
    questions: list[QuestionState] | None = None,
    existing: list[EvidenceItem] | None = None,
    observations: list[StoredObservation],
    limitations: list[ObservationLimitation] | None = None,
) -> EvidenceCheckInput:
    root_id = f"root:{case_id}"
    return EvidenceCheckInput(
        action_id=f"action:{case_id}",
        root_question=RootQuestion(question_id=root_id, text=root_text),
        evidence_memory=EvidenceMemory(
            reading_session_id=f"reading:{case_id}",
            root_question_id=root_id,
            root_status=EvidenceStatus.INCOMPLETE,
            questions=questions or [],
            evidence=existing or [],
            current_target=CurrentTarget(question_id=target_id, gap_description=gap),
        ),
        observations=observations,
        limitations=limitations or [],
    )


def build_cases() -> list[Case]:
    cases: list[Case] = []

    def add(case: Case) -> None:
        cases.append(case)

    cid = "C01"
    root = f"root:{cid}"
    obs = observation(cid, 0, "The audited table reports Acme revenue of USD 12 million in 2023.")
    add(Case(cid, "useful", "One directly useful Observation completes the Root.",
        make_input(cid, "What was Acme's revenue in 2023?", target_id=root,
                   gap="Acme's 2023 revenue is unknown.", observations=[obs]),
        Expected({obs.observation_id: True}, 1, target_status="satisfied",
                 root_status="ready", gap_required=False, next_target=None)))

    cid = "C02"
    root = f"root:{cid}"
    q1 = QuestionState(question_id="Q1", text="What was 2022 revenue?")
    q2 = QuestionState(question_id="Q2", text="What was 2023 revenue?")
    obs = observation(cid, 0, "The 2022 revenue was USD 10 million.")
    add(Case(cid, "subquestion", "Useful Observation completes Q1 but not the Root.",
        make_input(cid, "How much did revenue grow from 2022 to 2023?", target_id="Q1",
                   gap="The 2022 revenue is unknown.", questions=[q1, q2], observations=[obs]),
        Expected({obs.observation_id: True}, 1, target_status="satisfied",
                 root_status="incomplete", gap_required=False, next_target="Q2")))

    cid = "C03"
    root = f"root:{cid}"
    obs = observation(cid, 0, "The company employed 500 people in 2023.")
    add(Case(cid, "irrelevant", "Correct but irrelevant Observation is rejected.",
        make_input(cid, "What was the company's revenue in 2023?", target_id=root,
                   gap="The 2023 revenue is unknown.", observations=[obs]),
        Expected({obs.observation_id: False}, 0, gap_terms=("revenue", "2023"), next_target=root)))

    cid = "C04"
    root = f"root:{cid}"
    old = evidence(cid, 0, "The company's 2023 revenue was USD 12 million.", root)
    obs = observation(cid, 0, "The company's 2023 revenue was USD 12 million.")
    add(Case(cid, "duplicate", "Duplicate Observation is not re-added; existing Evidence is sufficient.",
        make_input(cid, "What was the company's revenue in 2023?", target_id=root,
                   gap="Confirm the 2023 revenue.", existing=[old], observations=[obs]),
        Expected({obs.observation_id: False}, 0, target_status="satisfied",
                 root_status="ready", gap_required=False, next_target=None)))

    cid = "C05"
    root = f"root:{cid}"
    obs = observation(cid, 0, "The 2023 revenue was USD 12 million.")
    add(Case(cid, "partial", "A useful fact is retained while another requested fact remains missing.",
        make_input(cid, "What were revenue and operating income in 2023?", target_id=root,
                   gap="Revenue and operating income for 2023 are unknown.", observations=[obs]),
        Expected({obs.observation_id: True}, 1, gap_terms=("operating income",), next_target=root)))

    cid = "C06"
    root = f"root:{cid}"
    obs = observation(cid, 0, "Method B appears to have an accuracy of 79%.")
    limitation = ObservationLimitation(
        description="The legend labels and bar values are too small to read reliably.",
        input_ids=["I1"],
    )
    add(Case(cid, "limitation", "A directly relevant but unreliable visual reading is rejected.",
        make_input(cid, "What accuracy did Method B achieve?", target_id=root,
                   gap="A reliable Method B accuracy is missing.", observations=[obs],
                   limitations=[limitation]),
        Expected({obs.observation_id: False}, 0, gap_terms=("Method B", "accuracy"), next_target=root)))

    cid = "C07"
    root = f"root:{cid}"
    obs1 = observation(cid, 0, "The audited table clearly reports 2023 revenue of USD 12 million.", "I1")
    obs2 = observation(cid, 1, "A footnote may qualify the reported value.", "I2")
    limitation = ObservationLimitation(
        description="The footnote text in I2 is unreadable.", input_ids=["I2"]
    )
    add(Case(cid, "limitation", "A limitation affects one input but not a separate reliable Observation.",
        make_input(cid, "What was 2023 revenue?", target_id=root,
                   gap="The 2023 revenue is unknown.", observations=[obs1, obs2],
                   limitations=[limitation]),
        Expected({obs1.observation_id: True, obs2.observation_id: False}, 1,
                 target_status="satisfied", root_status="ready", gap_required=False,
                 next_target=None)))

    cid = "C08"
    root = f"root:{cid}"
    old = evidence(cid, 0, "The 2022 revenue was USD 10 million.", root)
    obs = observation(cid, 0, "The audited table shows that USD 10 million is the 2021 value; the 2022 revenue is USD 12 million.")
    add(Case(cid, "conflict", "A clearly diagnosed column error replaces existing Evidence.",
        make_input(cid, "What was revenue in 2022?", target_id=root,
                   gap="Resolve the conflicting 2022 revenue value.", existing=[old], observations=[obs]),
        Expected({obs.observation_id: True}, 0, replace=1, target_status="satisfied",
                 root_status="ready", gap_required=False, next_target=None)))

    cid = "C09"
    root = f"root:{cid}"
    old = evidence(cid, 0, "Audited Table A reports 2023 revenue of USD 10 million.", root)
    obs = observation(cid, 0, "Audited Table B reports 2023 revenue of USD 12 million.")
    add(Case(cid, "conflict", "An unresolved same-scope conflict is retained without overwrite.",
        make_input(cid, "What was revenue in 2023?", target_id=root,
                   gap="Resolve the conflicting 2023 revenue values.", existing=[old], observations=[obs]),
        Expected({obs.observation_id: True}, 1, gap_terms=("conflict",), next_target=root)))

    cid = "C10"
    root = f"root:{cid}"
    old = evidence(cid, 0, "Revenue in 2022 was USD 10 million.", root)
    obs = observation(cid, 0, "Revenue in 2023 was USD 12 million.")
    add(Case(cid, "joint_evidence", "Existing and new Evidence jointly resolve a comparison.",
        make_input(cid, "Did revenue increase from 2022 to 2023?", target_id=root,
                   gap="The 2023 revenue is missing.", existing=[old], observations=[obs]),
        Expected({obs.observation_id: True}, 1, target_status="satisfied",
                 root_status="ready", gap_required=False, next_target=None)))

    cid = "C11"
    root = f"root:{cid}"
    q1 = QuestionState(question_id="Q1", text="What was 2022 revenue?")
    q2 = QuestionState(question_id="Q2", text="What was 2023 revenue?")
    obs = observation(cid, 0, "Revenue in 2022 was USD 100 million.")
    add(Case(cid, "loop_turn_1", "First turn of a real two-turn EvidenceMemory loop.",
        make_input(cid, "What was the percentage revenue growth from 2022 to 2023?",
                   target_id="Q1", gap="The 2022 revenue is missing.",
                   questions=[q1, q2], observations=[obs]),
        Expected({obs.observation_id: True}, 1, target_status="satisfied",
                 root_status="incomplete", gap_required=False, next_target="Q2")))

    # C12 is rebuilt from C11's actual updated memory during each evaluation pass.
    cid = "C12"
    root = f"root:{cid}"
    q1 = QuestionState(question_id="Q1", text="What was 2022 revenue?", status="satisfied")
    q2 = QuestionState(question_id="Q2", text="What was 2023 revenue?")
    prior = evidence(cid, 0, "Revenue in 2022 was USD 100 million.", "Q1")
    obs = observation(cid, 0, "Revenue in 2023 was USD 110 million.")
    add(Case(cid, "loop_turn_2", "Second turn consumes the first turn's updated EvidenceMemory.",
        make_input(cid, "What was the percentage revenue growth from 2022 to 2023?",
                   target_id="Q2", gap="The 2023 revenue is missing.",
                   questions=[q1, q2], existing=[prior], observations=[obs]),
        Expected({obs.observation_id: True}, 1, target_status="satisfied",
                 root_status="ready", gap_required=False, next_target=None), chained_from="C11"))

    cid = "C13"
    root = f"root:{cid}"
    obs = observation(cid, 0, "Revenue in 2022 was USD 9 million.")
    add(Case(cid, "wrong_observation", "A wrong-year Observation is scope-mismatched.",
        make_input(cid, "What was revenue in 2023?", target_id=root,
                   gap="The 2023 revenue is unknown.", observations=[obs]),
        Expected({obs.observation_id: False}, 0, gap_terms=("2023", "revenue"), next_target=root)))

    cid = "C14"
    root = f"root:{cid}"
    obs = observation(cid, 0, "Net income was 14.")
    add(Case(cid, "wrong_observation", "An underspecified value lacks requested company, year, and unit.",
        make_input(cid, "What was Company A's 2023 net income in USD millions?", target_id=root,
                   gap="Company A's 2023 net income in USD millions is unknown.", observations=[obs]),
        Expected({obs.observation_id: False}, 0, gap_terms=("2023", "net income"), next_target=root)))

    cid = "C15"
    root = f"root:{cid}"
    obs1 = observation(cid, 0, "Method A achieved 72% accuracy.", "I1")
    obs2 = observation(cid, 1, "Method B achieved 79% accuracy.", "I2")
    obs3 = observation(cid, 2, "The evaluation dataset contains 1,000 samples.", "I3")
    add(Case(cid, "mixed", "Two useful facts and one irrelevant fact support a comparison.",
        make_input(cid, "Between Method A and Method B, which achieved higher accuracy?",
                   target_id=root, gap="The two methods' accuracies are unknown.",
                   observations=[obs1, obs2, obs3]),
        Expected({obs1.observation_id: True, obs2.observation_id: True,
                  obs3.observation_id: False}, 2, target_status="satisfied",
                 root_status="ready", gap_required=False, next_target=None)))

    cid = "C16"
    root = f"root:{cid}"
    obs1 = observation(cid, 0, "Model X precision is 0.84.", "I1")
    obs2 = observation(cid, 1, "The experiment used three random seeds.", "I2")
    add(Case(cid, "partial", "One requested metric is found while another remains missing.",
        make_input(cid, "What are Model X's precision and recall?", target_id=root,
                   gap="Precision and recall are unknown.", observations=[obs1, obs2]),
        Expected({obs1.observation_id: True, obs2.observation_id: False}, 1,
                 gap_terms=("recall",), next_target=root)))

    cid = "C17"
    root = f"root:{cid}"
    old = evidence(cid, 0, "Method B achieved 81% accuracy.", root)
    obs = observation(cid, 0, "The table header shows that 81% belongs to Method C; Method B achieved 76% accuracy.")
    add(Case(cid, "conflict", "A detected label alignment error replaces a bad Evidence item.",
        make_input(cid, "What accuracy did Method B achieve?", target_id=root,
                   gap="Resolve Method B's accuracy.", existing=[old], observations=[obs]),
        Expected({obs.observation_id: True}, 0, replace=1, target_status="satisfied",
                 root_status="ready", gap_required=False, next_target=None)))

    cid = "C18"
    root = f"root:{cid}"
    old = evidence(cid, 0, "The conference was held in Paris.", root)
    obs = observation(cid, 0, "The conference had 2,000 attendees.")
    add(Case(cid, "joint_evidence", "Existing Evidence is already sufficient despite irrelevant new input.",
        make_input(cid, "In which city was the conference held?", target_id=root,
                   gap="Confirm the conference city.", existing=[old], observations=[obs]),
        Expected({obs.observation_id: False}, 0, target_status="satisfied",
                 root_status="ready", gap_required=False, next_target=None)))

    cid = "C19"
    root = f"root:{cid}"
    q1 = QuestionState(question_id="Q1", text="What changed in revenue?")
    obs = observation(cid, 0, "Revenue increased from USD 10 million to USD 12 million.")
    add(Case(cid, "subquestion", "Plan is exhausted after Q1, but a Root-level explanation is missing.",
        make_input(cid, "How did revenue change, and what caused the change?", target_id="Q1",
                   gap="The direction and amount of the revenue change are unknown.",
                   questions=[q1], observations=[obs]),
        Expected({obs.observation_id: True}, 1, target_status="satisfied",
                 root_status="incomplete", gap_required=False,
                 next_target=root)))

    cid = "C20"
    root = f"root:{cid}"
    obs = observation(cid, 0, "The table in I1 clearly reports 2023 revenue of USD 12 million.", "I1")
    limitation = ObservationLimitation(
        description="The unrelated chart in I2 has unreadable axis labels.", input_ids=["I2"]
    )
    add(Case(cid, "limitation", "An unrelated limitation must not invalidate a clear Observation.",
        make_input(cid, "What was revenue in 2023?", target_id=root,
                   gap="The 2023 revenue is unknown.", observations=[obs],
                   limitations=[limitation]),
        Expected({obs.observation_id: True}, 1, target_status="satisfied",
                 root_status="ready", gap_required=False, next_target=None)))

    cid = "C21"
    root = f"root:{cid}"
    q1 = QuestionState(question_id="Q1", text="What accuracy did Method A achieve?")
    q2 = QuestionState(question_id="Q2", text="What accuracy did Method B achieve?")
    obs = observation(
        cid,
        0,
        "The same table row reports Method A accuracy of 72% and Method B accuracy of 79%.",
    )
    add(Case(
        cid,
        "cross_target_observation",
        "One Observation contains facts for Q1 and Q2, but this turn may update only Q1.",
        make_input(
            cid,
            "Which of Method A and Method B achieved higher accuracy?",
            target_id="Q1",
            gap="Method A's accuracy is unknown.",
            questions=[q1, q2],
            observations=[obs],
        ),
        Expected(
            {obs.observation_id: True},
            1,
            target_status="satisfied",
            root_status="incomplete",
            gap_required=False,
            evidence_required_terms=("Method A", "72"),
            evidence_forbidden_terms=("Method B", "79"),
            next_target="Q2",
        ),
    ))

    cid = "C22"
    root = f"root:{cid}"
    q1 = QuestionState(
        question_id="Q1",
        text="What accuracy did Method A achieve?",
        status="satisfied",
    )
    q2 = QuestionState(
        question_id="Q2",
        text="What accuracy did Method B achieve?",
        status="satisfied",
    )
    existing = [
        evidence(cid, 0, "Method A achieved 72% accuracy.", "Q1"),
        evidence(cid, 1, "Method B achieved 79% accuracy.", "Q2"),
    ]
    obs = observation(
        cid,
        0,
        "The evaluation section states that both accuracy values are measured on the held-out test set.",
    )
    add(Case(
        cid,
        "root_after_plan_ready",
        "After every SubQuestion is satisfied, a Root-level Observation completes the Root.",
        make_input(
            cid,
            "Which method achieved higher accuracy on the held-out test set?",
            target_id=root,
            gap="Confirm that the two reported accuracy values use the held-out test set.",
            questions=[q1, q2],
            existing=existing,
            observations=[obs],
        ),
        Expected(
            {obs.observation_id: True},
            1,
            target_status="satisfied",
            root_status="ready",
            gap_required=False,
            next_target=None,
        ),
    ))

    cid = "C23"
    root = f"root:{cid}"
    q1 = QuestionState(
        question_id="Q1",
        text="What accuracy did Method A achieve?",
        status="satisfied",
    )
    q2 = QuestionState(
        question_id="Q2",
        text="What accuracy did Method B achieve?",
        status="satisfied",
    )
    existing = [
        evidence(cid, 0, "Method A achieved 72% accuracy.", "Q1"),
        evidence(cid, 1, "Method B achieved 79% accuracy.", "Q2"),
    ]
    obs = observation(
        cid,
        0,
        "Method B uses an additional supervised training stage, but the document does not state that this caused the accuracy difference.",
    )
    add(Case(
        cid,
        "root_after_plan_partial",
        "A Root-level fact is useful but does not establish the document's reason for the difference.",
        make_input(
            cid,
            "Which method achieved higher accuracy, and what reason does the document give for the difference?",
            target_id=root,
            gap="The document's stated reason for the accuracy difference is unknown.",
            questions=[q1, q2],
            existing=existing,
            observations=[obs],
        ),
        Expected(
            {obs.observation_id: True},
            1,
            gap_terms=("difference",),
            next_target=root,
        ),
    ))

    cid = "C24"
    root = f"root:{cid}"
    old = evidence(cid, 0, "The company's 2023 revenue was USD 12 million.", root)
    obs = observation(
        cid,
        0,
        "The footnote shows that USD 12 million is the 2022 value, while the 2023 row is not visible in the supplied crop.",
    )
    add(Case(
        cid,
        "remove_invalidated_evidence",
        "A new Observation invalidates existing Evidence without supplying a replacement value.",
        make_input(
            cid,
            "What was the company's revenue in 2023?",
            target_id=root,
            gap="Verify the company's 2023 revenue.",
            existing=[old],
            observations=[obs],
        ),
        Expected(
            {obs.observation_id: False},
            0,
            remove=1,
            gap_terms=("2023", "revenue"),
            next_target=root,
        ),
    ))

    cid = "C25"
    root = f"root:{cid}"
    obs = observation(cid, 0, "The company's 2023 revenue was USD 12 million.")
    add(Case(
        cid,
        "gap_quality",
        "The remaining gap names only the missing operating income, not the established revenue.",
        make_input(
            cid,
            "What were the company's revenue and operating income in 2023?",
            target_id=root,
            gap="The company's 2023 revenue and operating income are unknown.",
            observations=[obs],
        ),
        Expected(
            {obs.observation_id: True},
            1,
            gap_terms=("operating income",),
            gap_forbidden_terms=("revenue",),
            next_target=root,
        ),
    ))

    cid = "C26"
    root = f"root:{cid}"
    limitation = ObservationLimitation(
        description="The requested chart labels are too small to read reliably.",
        input_ids=["I1"],
    )
    add(Case(
        cid,
        "limitation_only",
        "A read may fail with a relevant limitation and no Observation.",
        make_input(
            cid,
            "What accuracy did Method B achieve?",
            target_id=root,
            gap="A reliable Method B accuracy is missing.",
            observations=[],
            limitations=[limitation],
        ),
        Expected(
            {},
            0,
            gap_terms=("Method B", "accuracy"),
            next_target=root,
        ),
    ))

    cid = "C27"
    root = f"root:{cid}"
    q1 = QuestionState(
        question_id="Q1",
        text="What accuracy did Method A achieve?",
        status="satisfied",
    )
    q2 = QuestionState(
        question_id="Q2",
        text="What accuracy did Method B achieve?",
        status="satisfied",
    )
    existing = [
        evidence(cid, 0, "Method A achieved 72% accuracy.", "Q1"),
        evidence(cid, 1, "Method B achieved 79% accuracy.", "Q2"),
    ]
    add(Case(
        cid,
        "state_only_root_finalization",
        "After the plan completes, existing Evidence alone resolves the Root.",
        make_input(
            cid,
            "Which method achieved higher accuracy?",
            target_id=root,
            gap="Which method achieved higher accuracy?",
            questions=[q1, q2],
            existing=existing,
            observations=[],
        ),
        Expected(
            {},
            0,
            target_status="satisfied",
            root_status="ready",
            gap_required=False,
            next_target=None,
        ),
    ))

    return cases


def call_ollama(
    *, base_url: str, model: str, checker_input: EvidenceCheckInput,
    timeout: float, seed: int,
) -> tuple[str, dict[str, Any], float]:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": CHECKER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": checker_input.model_dump_json(indent=2),
            },
        ],
        "stream": False,
        "think": False,
        "format": EvidenceCheckDecision.model_json_schema(),
        "keep_alive": "30m",
        "options": {"temperature": 0.0, "seed": seed, "num_ctx": 8192},
    }
    req = request.Request(
        f"{base_url.rstrip('/')}/api/chat",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    try:
        with request.urlopen(req, timeout=timeout) as response:
            response_payload = json.loads(response.read().decode("utf-8"))
    except (error.URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(f"Ollama request failed: {exc}") from exc
    elapsed = time.perf_counter() - started
    message = response_payload.get("message", {})
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError(f"Ollama returned no content: {response_payload.get('error')}")
    return content, response_payload, elapsed


def normalized_words(value: str | None) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", (value or "").lower()))


def evaluate_result(
    case: Case, result: EvidenceCheckResult,
) -> tuple[dict[str, bool], EvidenceMemory | None, str | None]:
    expected = case.expected
    actual_used = {
        item.observation_id: item.used_for_evidence
        for item in result.observation_assessments
    }
    checks = {
        "used_flags": actual_used == expected.used,
        "add_count": len(result.evidence_updates.add) == expected.add,
        "replace_count": len(result.evidence_updates.replace) == expected.replace,
        "remove_count": len(result.evidence_updates.remove) == expected.remove,
        "current_target_status": result.current_target_status.value == expected.target_status,
        "root_status": result.root_status.value == expected.root_status,
        "gap_presence": (result.remaining_gap_description is not None) == expected.gap_required,
    }
    if expected.gap_terms:
        gap_words = normalized_words(result.remaining_gap_description)
        checks["gap_content"] = all(
            set(term.lower().split()).issubset(gap_words) for term in expected.gap_terms
        )
    else:
        checks["gap_content"] = True
    if expected.gap_forbidden_terms:
        gap_words = normalized_words(result.remaining_gap_description)
        checks["gap_excludes_established_facts"] = not any(
            set(term.lower().split()).issubset(gap_words)
            for term in expected.gap_forbidden_terms
        )
    else:
        checks["gap_excludes_established_facts"] = True
    update_statements = " ".join(
        item.statement
        for item in [
            *result.evidence_updates.add,
            *result.evidence_updates.replace,
        ]
    )
    update_words = normalized_words(update_statements)
    checks["evidence_content_required"] = all(
        set(term.lower().split()).issubset(update_words)
        for term in expected.evidence_required_terms
    )
    checks["evidence_content_excludes_cross_target_facts"] = not any(
        set(term.lower().split()).issubset(update_words)
        for term in expected.evidence_forbidden_terms
    )

    updated: EvidenceMemory | None = None
    apply_error: str | None = None
    try:
        updated = apply_evidence_check_result(case.checker_input, result)
    except (ValueError, ValidationError) as exc:
        apply_error = str(exc)
    checks["delta_applies"] = updated is not None
    checks["next_target"] = (
        updated is not None
        and (updated.current_target.question_id if updated.current_target else None)
        == expected.next_target
    )
    return checks, updated, apply_error


def semantic_signature(result: EvidenceCheckResult) -> dict[str, Any]:
    return {
        "used": {
            item.observation_id: item.used_for_evidence
            for item in result.observation_assessments
        },
        "add": len(result.evidence_updates.add),
        "replace": len(result.evidence_updates.replace),
        "remove": len(result.evidence_updates.remove),
        "current_target_status": result.current_target_status.value,
        "root_status": result.root_status.value,
        "has_gap": result.remaining_gap_description is not None,
    }


def chain_second_turn(previous: EvidenceMemory, template: Case) -> Case:
    old_root_id = previous.root_question_id
    new_root_id = template.checker_input.root_question.question_id
    remapped_questions = [
        item.model_copy() for item in previous.questions
    ]
    remapped_evidence = [
        item.model_copy(
            update={
                "evidence_id": item.evidence_id.replace("C11", "C12"),
                "observation_ids": [oid.replace("C11", "C12-prior") for oid in item.observation_ids],
            }
        )
        for item in previous.evidence
    ]
    target = previous.current_target
    if target is None:
        raise ValueError("C11 did not produce the Q2 target required by C12")
    memory = EvidenceMemory(
        reading_session_id="reading:C12",
        root_question_id=new_root_id,
        root_status=previous.root_status,
        questions=remapped_questions,
        evidence=remapped_evidence,
        current_target=target,
    )
    return Case(
        case_id=template.case_id,
        category=template.category,
        description=template.description,
        checker_input=template.checker_input.model_copy(update={"evidence_memory": memory}),
        expected=template.expected,
        chained_from=template.chained_from,
    )


def run_pass(
    cases: list[Case], *, pass_name: str, base_url: str, model: str,
    timeout: float, seed: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    updated_by_case: dict[str, EvidenceMemory] = {}
    for index, original_case in enumerate(cases, start=1):
        case = original_case
        if case.chained_from:
            previous = updated_by_case.get(case.chained_from)
            if previous is None:
                records.append({
                    "case_id": case.case_id,
                    "pass": pass_name,
                    "category": case.category,
                    "description": case.description,
                    "valid_output": False,
                    "error": f"Required chain source {case.chained_from} was unavailable",
                })
                continue
            case = chain_second_turn(previous, case)
        print(f"[{pass_name}] {index:02d}/{len(cases)} {case.case_id} {case.category}", flush=True)
        raw_output = ""
        try:
            raw_output, response_payload, elapsed = call_ollama(
                base_url=base_url, model=model, checker_input=case.checker_input,
                timeout=timeout, seed=seed,
            )
            decision = EvidenceCheckDecision.model_validate_json(raw_output)
            result = materialize_evidence_check_decision(
                case.checker_input,
                decision,
            )
            checks, updated, apply_error = evaluate_result(case, result)
            if updated is not None:
                updated_by_case[case.case_id] = updated
            records.append({
                "case_id": case.case_id,
                "pass": pass_name,
                "category": case.category,
                "description": case.description,
                "input": case.checker_input.model_dump(mode="json"),
                "expected": {
                    "used": case.expected.used,
                    "add": case.expected.add,
                    "replace": case.expected.replace,
                    "remove": case.expected.remove,
                    "target_status": case.expected.target_status,
                    "root_status": case.expected.root_status,
                    "gap_required": case.expected.gap_required,
                    "gap_terms": list(case.expected.gap_terms),
                    "gap_forbidden_terms": list(case.expected.gap_forbidden_terms),
                    "evidence_required_terms": list(case.expected.evidence_required_terms),
                    "evidence_forbidden_terms": list(case.expected.evidence_forbidden_terms),
                    "next_target": case.expected.next_target,
                },
                "valid_output": True,
                "model_output": decision.model_dump(mode="json"),
                "output": result.model_dump(mode="json"),
                "checks": checks,
                "case_passed": all(checks.values()),
                "apply_error": apply_error,
                "updated_memory": updated.model_dump(mode="json") if updated else None,
                "semantic_signature": semantic_signature(result),
                "elapsed_seconds": round(elapsed, 3),
                "ollama_metrics": {
                    key: response_payload.get(key)
                    for key in ("done_reason", "prompt_eval_count", "eval_count", "total_duration")
                    if key in response_payload
                },
            })
        except (RuntimeError, ValidationError, ValueError) as exc:
            records.append({
                "case_id": case.case_id,
                "pass": pass_name,
                "category": case.category,
                "description": case.description,
                "valid_output": False,
                "raw_output": raw_output,
                "error": str(exc),
                "case_passed": False,
            })
    return records


def write_reports(
    output_dir: Path, *, model: str, cases: list[Case], records: list[dict[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "checker_system_prompt.txt").write_text(
        CHECKER_SYSTEM_PROMPT, encoding="utf-8"
    )
    with (output_dir / "results.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    by_key = {(r["pass"], r["case_id"]): r for r in records}
    stability: list[dict[str, Any]] = []
    for case in cases:
        a = by_key.get(("A", case.case_id), {})
        b = by_key.get(("B", case.case_id), {})
        exact = a.get("output") == b.get("output") and bool(a.get("output"))
        semantic = (
            a.get("semantic_signature") == b.get("semantic_signature")
            and a.get("semantic_signature") is not None
        )
        stability.append({"case_id": case.case_id, "exact": exact, "semantic": semantic})

    called_records = [
        record
        for record in records
        if "output" in record or "raw_output" in record
    ]
    blocked_records = [record for record in records if record not in called_records]
    valid = sum(bool(r.get("valid_output")) for r in called_records)
    passed = sum(bool(r.get("case_passed")) for r in records)
    comparable_stability = [
        item
        for item in stability
        if by_key.get(("A", item["case_id"]), {}).get("semantic_signature") is not None
        and by_key.get(("B", item["case_id"]), {}).get("semantic_signature") is not None
    ]
    semantic_stable = sum(item["semantic"] for item in comparable_stability)
    exact_stable = sum(item["exact"] for item in comparable_stability)
    total_runs = len(records)
    summary = {
        "model": model,
        "prompt_version": CHECKER_PROMPT_VERSION,
        "case_count": len(cases),
        "independent_passes": 2,
        "run_count": total_runs,
        "model_call_count": len(called_records),
        "blocked_before_model_call": len(blocked_records),
        "valid_output_runs": valid,
        "passed_runs": passed,
        "semantic_stability_cases": semantic_stable,
        "exact_stability_cases": exact_stable,
        "stability": stability,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        "# Evidence Checker mock evaluation",
        "",
        f"- Model: `{model}`",
        f"- Prompt: `{CHECKER_PROMPT_VERSION}`",
        f"- Synthetic cases: {len(cases)}",
        f"- Independent runs: {total_runs}",
        f"- Model calls: {len(called_records)} (blocked before call: {len(blocked_records)})",
        f"- Schema-valid outputs: {valid}/{len(called_records)} model calls",
        f"- Fully passing runs: {passed}/{total_runs}",
        f"- Semantically stable comparable cases: {semantic_stable}/{len(comparable_stability)}",
        f"- Byte-for-byte stable comparable cases: {exact_stable}/{len(comparable_stability)}",
        "",
        "## Per-case results",
        "",
        "| Case | Category | A | B | Stable | Description |",
        "|---|---|---:|---:|---:|---|",
    ]
    for case in cases:
        a = by_key.get(("A", case.case_id), {})
        b = by_key.get(("B", case.case_id), {})
        stable = next(item["semantic"] for item in stability if item["case_id"] == case.case_id)
        lines.append(
            f"| {case.case_id} | {case.category} | "
            f"{'PASS' if a.get('case_passed') else 'FAIL'} | "
            f"{'PASS' if b.get('case_passed') else 'FAIL'} | "
            f"{'yes' if stable else 'no'} | {case.description} |"
        )

    lines.extend(["", "## Failures", ""])
    failures = [record for record in records if not record.get("case_passed")]
    if not failures:
        lines.append("No rubric failures.")
    for record in failures:
        lines.extend([
            f"### {record['case_id']} / pass {record['pass']}",
            "",
            f"- Category: `{record['category']}`",
            f"- Error: {record.get('error') or record.get('apply_error') or 'semantic rubric mismatch'}",
        ])
        checks = record.get("checks")
        if checks:
            failed_checks = [name for name, ok in checks.items() if not ok]
            lines.append(f"- Failed checks: {', '.join(failed_checks)}")
        output = record.get("output") or record.get("raw_output")
        lines.extend(["- Model output:", "", "```json", json.dumps(output, ensure_ascii=False, indent=2) if not isinstance(output, str) else output, "```", ""])

    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen3:8b")
    parser.add_argument("--base-url", default="http://localhost:11434")
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/checker_mock_v0"),
    )
    args = parser.parse_args()
    cases = build_cases()
    records = [
        *run_pass(cases, pass_name="A", base_url=args.base_url, model=args.model,
                  timeout=args.timeout, seed=args.seed),
        *run_pass(cases, pass_name="B", base_url=args.base_url, model=args.model,
                  timeout=args.timeout, seed=args.seed),
    ]
    write_reports(args.output, model=args.model, cases=cases, records=records)
    print(f"Wrote reports to {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
