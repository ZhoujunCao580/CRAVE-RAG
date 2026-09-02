from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from PIL import Image

from softdoc.answering import AnswerInput, AnswerResult
from softdoc.controller import ControllerAction, ControllerInput
from softdoc.models import (
    Document,
    Element,
    ElementType,
    Page,
    Provenance,
    Relation,
    RelationSource,
    RelationStatus,
    RelationType,
)
from softdoc.planning.models import InitialPlan, PlannedSubQuestion, PlannerTrace
from softdoc.reading_environment import (
    DeterministicContentReader,
    ReaderContext,
    ReaderObservationDraft,
    ReaderOutput,
    ReadingEnvironment,
    ReadingEnvironmentConfig,
    ReadingRunStatus,
)
from softdoc.reading_state import (
    EvidenceAddition,
    EvidenceCheckInput,
    EvidenceCheckResult,
    EvidenceStatus,
    EvidenceUpdates,
    ObservationAssessment,
    ObservationSourceRef,
    QuestionState,
    QuestionStatus,
    ReaderKind,
    ReadRepresentation,
    RootQuestion,
)
from softdoc.retrieval import SearchSessionConfig


class QueueController:
    def __init__(self, actions: list[dict[str, Any] | Callable[[ControllerInput], dict[str, Any]]]):
        self.actions = list(actions)
        self.inputs: list[ControllerInput] = []

    def decide(self, controller_input: ControllerInput) -> ControllerAction | dict[str, Any]:
        self.inputs.append(controller_input)
        if not self.actions:
            raise AssertionError("The Environment requested an unexpected Controller action")
        item = self.actions.pop(0)
        return item(controller_input) if callable(item) else item


class RejectingController:
    def decide(self, controller_input: ControllerInput) -> ControllerAction | dict[str, Any]:
        raise AssertionError("Exact routing should finish without invoking Controller")


class PredicateChecker:
    """Small teacher stub: policy quality is not part of Environment tests."""

    def __init__(self, useful: Callable[[str], bool]) -> None:
        self.useful = useful
        self.inputs: list[EvidenceCheckInput] = []

    def check(self, checker_input: EvidenceCheckInput) -> EvidenceCheckResult:
        self.inputs.append(checker_input)
        target = checker_input.evidence_memory.current_target
        assert target is not None
        useful_observations = [
            item for item in checker_input.observations if self.useful(item.text)
        ]
        used_ids = {item.observation_id for item in useful_observations}
        additions = [
            EvidenceAddition(
                statement=item.text,
                observation_ids=[item.observation_id],
                supports_question_ids=[target.question_id],
            )
            for item in useful_observations
        ]
        satisfied = bool(additions)
        if not satisfied:
            root_status = EvidenceStatus.INCOMPLETE
        elif target.question_id == checker_input.root_question.question_id:
            root_status = EvidenceStatus.READY
        else:
            remaining = [
                item
                for item in checker_input.evidence_memory.questions
                if item.question_id != target.question_id
                and item.status != QuestionStatus.SATISFIED
            ]
            root_status = (
                EvidenceStatus.READY if not remaining else EvidenceStatus.INCOMPLETE
            )
        return EvidenceCheckResult(
            action_id=checker_input.action_id,
            observation_assessments=[
                ObservationAssessment(
                    observation_id=item.observation_id,
                    used_for_evidence=item.observation_id in used_ids,
                    assessment=(
                        "Useful for the current evidence gap."
                        if item.observation_id in used_ids
                        else "Readable, but it does not resolve the current gap."
                    ),
                )
                for item in checker_input.observations
            ],
            evidence_updates=EvidenceUpdates(add=additions),
            current_target_status=(
                QuestionStatus.SATISFIED
                if satisfied
                else QuestionStatus.INCOMPLETE
            ),
            root_status=root_status,
            remaining_gap_description=(
                None
                if satisfied
                else checker_input.evidence_memory.current_target.gap_description
            ),
        )


class EvidenceAnswerer:
    def answer(self, answer_input: AnswerInput) -> AnswerResult:
        return AnswerResult(
            answer=" | ".join(item.statement for item in answer_input.evidence),
            used_evidence_ids=[item.evidence_id for item in answer_input.evidence],
        )


class PageTeacherReader:
    """Teacher replacement used only where the deterministic Reader cannot see pixels."""

    def __init__(self, page_facts: dict[str, str]) -> None:
        self.page_facts = page_facts
        self.fallback = DeterministicContentReader()

    def read(self, context: ReaderContext) -> ReaderOutput:
        if len(context.inputs) == 1:
            item = context.inputs[0]
            if (
                item.representation == ReadRepresentation.PAGE_VISUAL
                and item.page_id in self.page_facts
            ):
                return ReaderOutput(
                    reader_kind=ReaderKind.PAGE,
                    observations=[
                        ReaderObservationDraft(
                            text=self.page_facts[item.page_id],
                            sources=[ObservationSourceRef(input_id=item.input_id)],
                        )
                    ],
                )
        return self.fallback.read(context)


def _provenance(owner: str) -> Provenance:
    return Provenance(
        provenance_id=f"prov:{owner}",
        adapter="test",
        source_path=Path("fixture.json"),
        source_locator=owner,
    )


def _document(
    tmp_path: Path,
    *,
    page_element_specs: list[list[dict[str, Any]]],
    relations: list[Relation] | None = None,
) -> Document:
    doc_id = "doc:environment-test"
    pages: list[Page] = []
    elements: list[Element] = []
    for page_index, specs in enumerate(page_element_specs):
        page_id = f"page:{page_index + 1}"
        image_path = tmp_path / f"page_{page_index + 1}.png"
        Image.new("RGB", (200, 300), "white").save(image_path)
        element_ids: list[str] = []
        for reading_order, spec in enumerate(specs):
            element_id = str(spec["element_id"])
            element_ids.append(element_id)
            visual = bool(spec.pop("visual", False))
            visual_path = None
            if visual:
                visual_path = tmp_path / f"{element_id.replace(':', '_')}.png"
                Image.new("RGB", (120, 80), "white").save(visual_path)
            elements.append(
                Element(
                    element_id=element_id,
                    document_id=doc_id,
                    page_id=page_id,
                    page_number=page_index + 1,
                    reading_order=reading_order,
                    provenance=_provenance(element_id),
                    image_path=visual_path,
                    **{key: value for key, value in spec.items() if key != "element_id"},
                )
            )
        pages.append(
            Page(
                page_id=page_id,
                document_id=doc_id,
                page_index=page_index,
                page_number=page_index + 1,
                width=200,
                height=300,
                element_ids=element_ids,
                reading_order=element_ids,
                image_path=image_path,
                provenance=_provenance(page_id),
            )
        )
    return Document(
        document_id=doc_id,
        source_path=Path("fixture.pdf"),
        pages=pages,
        elements=elements,
        relations=relations or [],
        provenance=_provenance(doc_id),
    )


def _relation(
    relation_id: str,
    source_id: str,
    target_id: str,
    relation_type: RelationType,
    status: RelationStatus,
) -> Relation:
    return Relation(
        relation_id=relation_id,
        source_id=source_id,
        target_id=target_id,
        relation_type=relation_type,
        confidence=0.9,
        status=status,
        created_by=RelationSource.DETERMINISTIC_RULE,
    )


def test_exact_visual_limitation_then_confirmed_relation_recovers(tmp_path: Path) -> None:
    relation = _relation(
        "rel:caption",
        "caption:1",
        "figure:1",
        RelationType.CAPTION_OF,
        RelationStatus.CONFIRMED,
    )
    document = _document(
        tmp_path,
        page_element_specs=[
            [
                {
                    "element_id": "figure:1",
                    "element_type": ElementType.FIGURE,
                    "reference_label": "Figure 1",
                    "visual": True,
                },
                {
                    "element_id": "caption:1",
                    "element_type": ElementType.CAPTION,
                    "text": "ANSWER: Figure 1 shows the low-light accuracy comparison.",
                },
            ]
        ],
        relations=[relation],
    )
    controller = QueueController(
        [
            {
                "action": "FOLLOW_RELATION",
                "relation_id": relation.relation_id,
                "local_problem": "Identify what Figure 1 shows.",
            }
        ]
    )
    result = ReadingEnvironment(
        document,
        asset_root=tmp_path,
        controller=controller,
        reader=DeterministicContentReader(),
        checker=PredicateChecker(lambda text: "ANSWER:" in text),
        answerer=EvidenceAnswerer(),
    ).run(
        root_question=RootQuestion(
            question_id="root:figure",
            text="What does Figure 1 show?",
        )
    )

    assert result.status == ReadingRunStatus.READY
    assert [item.action_name for item in result.action_trace.entries] == [
        "READ_SOURCE",
        "FOLLOW_RELATION",
    ]
    first_feedback = result.action_trace.entries[0].metadata["controller_feedback"]
    assert "visual Reader is required" in first_feedback["limitations"][0]["description"]
    relation_view = controller.inputs[0].confirmed_relations[0]
    assert relation_view.relation_id == "rel:caption"
    assert relation_view.current_endpoint_id == "figure:1"
    assert relation_view.related_source_preview.source_id == "caption:1"
    assert relation_view.related_source_preview.element_type == ElementType.CAPTION
    assert "low-light accuracy comparison" in (
        relation_view.related_source_preview.label_or_snippet
    )
    assert result.evidence_memory.evidence[0].observation_ids == [
        result.observation_store.observations[0].observation_id
    ]


def test_ordinal_page_anchor_auto_reads_physical_page(tmp_path: Path) -> None:
    document = _document(tmp_path, page_element_specs=[[], []])
    # A conflicting printed label must not change the meaning of "first page".
    document.pages[1].page_label_aliases = ["1"]
    document.pages[1].display_page_label = "1"
    document.pages[1].display_page_label_confidence = 0.95

    result = ReadingEnvironment(
        document,
        asset_root=tmp_path,
        controller=RejectingController(),
        reader=PageTeacherReader({"page:1": "ANSWER: the first page was read."}),
        checker=PredicateChecker(lambda text: "ANSWER:" in text),
        answerer=EvidenceAnswerer(),
    ).run(
        root_question=RootQuestion(
            question_id="root:ordinal-page",
            text="What appears on the first page?",
        )
    )

    assert result.status == ReadingRunStatus.READY
    assert [item.action_name for item in result.action_trace.entries] == [
        "READ_SOURCE"
    ]
    assert result.action_trace.entries[0].target_ids == ["page:1"]
    assert result.answer is not None
    assert "first page" in result.answer.answer


def test_ordinal_exact_read_can_recover_on_adjacent_page(tmp_path: Path) -> None:
    document = _document(tmp_path, page_element_specs=[[], [], []])
    controller = QueueController(
        [
            {
                "action": "READ_ADJACENT_PAGE",
                "from_page_id": "page:2",
                "direction": "next",
                "local_problem": "Check the next physical page for the requested date.",
            }
        ]
    )

    result = ReadingEnvironment(
        document,
        asset_root=tmp_path,
        controller=controller,
        reader=PageTeacherReader(
            {
                "page:2": "The requested date is not visible on this page.",
                "page:3": "ANSWER: the date is 2023-07.",
            }
        ),
        checker=PredicateChecker(lambda text: "ANSWER:" in text),
        answerer=EvidenceAnswerer(),
    ).run(
        root_question=RootQuestion(
            question_id="root:ordinal-recovery",
            text="What date is mentioned on the second page?",
        )
    )

    assert result.status == ReadingRunStatus.READY
    assert [item.action_name for item in result.action_trace.entries] == [
        "READ_SOURCE",
        "READ_ADJACENT_PAGE",
    ]
    assert result.action_trace.entries[0].target_ids == ["page:2"]
    assert result.action_trace.entries[1].target_ids == ["page:3"]
    assert controller.inputs[0].reading_locations[0].page_id == "page:2"
    assert result.answer is not None
    assert "2023-07" in result.answer.answer


def test_irrelevant_table_observation_then_candidate_relation_is_explored(
    tmp_path: Path,
) -> None:
    relation = _relation(
        "rel:continued",
        "table:1",
        "table:2",
        RelationType.CONTINUED_ON,
        RelationStatus.CANDIDATE,
    )
    document = _document(
        tmp_path,
        page_element_specs=[
            [
                {
                    "element_id": "table:1",
                    "element_type": ElementType.TABLE,
                    "html": "<table><tr><td>opening clue</td><td>not enough</td></tr></table>",
                }
            ],
            [
                {
                    "element_id": "table:2",
                    "element_type": ElementType.TABLE,
                    "html": "<table><tr><td>ANSWER:</td><td>42</td></tr></table>",
                }
            ],
        ],
        relations=[relation],
    )
    controller = QueueController(
        [
            {"action": "SEARCH", "operation": "new", "query": "opening clue"},
            {
                "action": "READ_SOURCE",
                "source_ids": ["table:1"],
                "local_problem": "Find the requested value.",
            },
            {
                "action": "EXPLORE_CANDIDATE_RELATION",
                "relation_id": relation.relation_id,
                "local_problem": "Check whether the candidate continuation contains the value.",
            },
        ]
    )
    result = ReadingEnvironment(
        document,
        asset_root=tmp_path,
        controller=controller,
        reader=DeterministicContentReader(),
        checker=PredicateChecker(lambda text: "ANSWER:" in text),
        answerer=EvidenceAnswerer(),
    ).run(
        root_question=RootQuestion(
            question_id="root:table",
            text="What is the value in the continued table?",
        )
    )

    assert result.status == ReadingRunStatus.READY
    assert result.search_sessions[0].opened_candidate_ids == ["table:1"]
    assert controller.inputs[2].candidate_relations[0].relation_id == "rel:continued"
    rejected = result.action_trace.entries[1].metadata["controller_feedback"]
    assert rejected["observation_assessments"][0]["used_for_evidence"] is False
    assert len(result.evidence_memory.evidence) == 1
    assert "42" in result.evidence_memory.evidence[0].statement
    assert document.relations[0].status == RelationStatus.CANDIDATE


def test_search_read_then_adjacent_page_teacher_resolves_gap(tmp_path: Path) -> None:
    document = _document(
        tmp_path,
        page_element_specs=[
            [
                {
                    "element_id": "paragraph:lead",
                    "element_type": ElementType.PARAGRAPH,
                    "text": "entry clue, but no requested amount",
                }
            ],
            [
                {
                    "element_id": "paragraph:next",
                    "element_type": ElementType.PARAGRAPH,
                    "text": "The visual continuation is on this page.",
                }
            ],
        ],
    )
    controller = QueueController(
        [
            {"action": "SEARCH", "operation": "new", "query": "entry clue"},
            {
                "action": "READ_SOURCE",
                "source_ids": ["paragraph:lead"],
                "local_problem": "Find the requested amount.",
            },
            {
                "action": "READ_ADJACENT_PAGE",
                "from_page_id": "page:1",
                "direction": "next",
                "local_problem": "Check the following page for the amount.",
            },
        ]
    )
    result = ReadingEnvironment(
        document,
        asset_root=tmp_path,
        controller=controller,
        reader=PageTeacherReader({"page:2": "ANSWER: the amount is 99."}),
        checker=PredicateChecker(lambda text: "ANSWER:" in text),
        answerer=EvidenceAnswerer(),
    ).run(
        root_question=RootQuestion(
            question_id="root:adjacent",
            text="Figure 99 is unavailable; what amount appears on the following page?",
        )
    )

    assert result.status == ReadingRunStatus.READY
    assert any(
        item.code == "exact_anchor_unresolved" for item in result.diagnostics
    )
    assert result.action_trace.entries[-1].action_name == "READ_ADJACENT_PAGE"
    assert result.action_trace.entries[-1].target_ids == ["page:2"]
    assert "99" in result.answer.answer


def test_program_advances_two_exact_subquestions_and_answerer_combines(
    tmp_path: Path,
) -> None:
    document = _document(tmp_path, page_element_specs=[[], []])
    checker = PredicateChecker(lambda text: "ANSWER:" in text)
    result = ReadingEnvironment(
        document,
        asset_root=tmp_path,
        controller=RejectingController(),
        reader=PageTeacherReader(
            {
                "page:1": "ANSWER: 2022 revenue was 10 million.",
                "page:2": "ANSWER: 2023 revenue was 12 million.",
            }
        ),
        checker=checker,
        answerer=EvidenceAnswerer(),
    ).run(
        root_question=RootQuestion(
            question_id="root:revenue",
            text="How did revenue change?",
        ),
        questions=[
            QuestionState(question_id="Q1", text="What was revenue on Page 1?"),
            QuestionState(question_id="Q2", text="What was revenue on Page 2?"),
        ],
    )

    assert result.status == ReadingRunStatus.READY
    assert [item.question_id for item in result.evidence_memory.questions] == ["Q1", "Q2"]
    assert all(
        item.status == QuestionStatus.SATISFIED
        for item in result.evidence_memory.questions
    )
    assert [
        item.evidence_memory.current_target.question_id for item in checker.inputs
    ] == ["Q1", "Q2"]
    assert len(result.answer.used_evidence_ids) == 2
    assert "10 million" in result.answer.answer
    assert "12 million" in result.answer.answer


def test_next_candidate_batch_remains_in_one_search_session(tmp_path: Path) -> None:
    specs = [
        {
            "element_id": f"paragraph:{index}",
            "element_type": ElementType.PARAGRAPH,
            "text": f"needle candidate {index}",
        }
        for index in range(4)
    ]
    document = _document(tmp_path, page_element_specs=[specs])

    def read_visible(controller_input: ControllerInput) -> dict[str, Any]:
        assert controller_input.visible_search_view is not None
        return {
            "action": "READ_SOURCE",
            "source_ids": [
                controller_input.visible_search_view.candidate_previews[0].element_id
            ],
            "local_problem": "Read one candidate from the second batch.",
        }

    controller = QueueController(
        [
            {"action": "SEARCH", "operation": "new", "query": "needle"},
            lambda value: {
                "action": "SEARCH",
                "operation": "next",
                "search_session_id": value.visible_search_view.search_session_id,
            },
            read_visible,
        ]
    )
    result = ReadingEnvironment(
        document,
        asset_root=tmp_path,
        controller=controller,
        reader=DeterministicContentReader(),
        checker=PredicateChecker(lambda text: "needle" in text),
        answerer=EvidenceAnswerer(),
        config=ReadingEnvironmentConfig(
            action_budget=4,
            search=SearchSessionConfig(batch_size=2),
        ),
    ).run(
        root_question=RootQuestion(
            question_id="root:batches",
            text="Which candidate contains needle?",
        )
    )

    assert result.status == ReadingRunStatus.READY
    assert len(result.search_sessions) == 1
    session = result.search_sessions[0]
    assert session.cursor == 4
    assert len(session.shown_candidate_ids) == 4
    assert len(session.opened_candidate_ids) == 1
    assert controller.inputs[2].visible_search_view.search_session_id == session.search_session_id


def test_validated_initial_plan_maps_into_runtime_question_order(tmp_path: Path) -> None:
    document = _document(
        tmp_path,
        page_element_specs=[
            [
                {
                    "element_id": "paragraph:2022",
                    "element_type": ElementType.PARAGRAPH,
                    "text": "Revenue in 2022 was 10 million.",
                }
            ],
            [
                {
                    "element_id": "paragraph:2023",
                    "element_type": ElementType.PARAGRAPH,
                    "text": "Revenue in 2023 was 12 million.",
                }
            ],
        ],
    )
    plan = InitialPlan(
        original_question="How did revenue change from Page 1 to Page 2?",
        subquestions=[
            PlannedSubQuestion(
                subquestion_id="Q1",
                text="What was revenue on Page 1?",
            ),
            PlannedSubQuestion(
                subquestion_id="Q2",
                text="What was revenue on Page 2?",
            ),
        ],
        planner_trace=PlannerTrace(
            backend_name="teacher",
            model="scripted",
            prompt_version="planner-v0",
        ),
    )
    result = ReadingEnvironment(
        document,
        asset_root=tmp_path,
        controller=RejectingController(),
        reader=PageTeacherReader(
            {
                "page:1": "ANSWER: 2022 revenue was 10 million.",
                "page:2": "ANSWER: 2023 revenue was 12 million.",
            }
        ),
        checker=PredicateChecker(lambda text: "million" in text),
        answerer=EvidenceAnswerer(),
    ).run_with_plan(
        root_question_id="root:planned",
        plan=plan,
        run_key="planned",
    )

    assert result.status == ReadingRunStatus.READY
    assert [item.question_id for item in result.evidence_memory.questions] == [
        "Q1",
        "Q2",
    ]
    assert all(
        item.status == QuestionStatus.SATISFIED
        for item in result.evidence_memory.questions
    )
    assert result.answer is not None


def test_root_direct_initial_plan_runs_without_a_synthetic_subquestion(
    tmp_path: Path,
) -> None:
    document = _document(tmp_path, page_element_specs=[[]])
    plan = InitialPlan(
        original_question="What revenue is reported on Page 1?",
        subquestions=[],
        planner_trace=PlannerTrace(
            backend_name="teacher",
            model="scripted",
            prompt_version="planner-v0.20",
        ),
    )
    checker = PredicateChecker(lambda text: "12 million" in text)

    result = ReadingEnvironment(
        document,
        asset_root=tmp_path,
        controller=RejectingController(),
        reader=PageTeacherReader({"page:1": "Revenue was 12 million."}),
        checker=checker,
        answerer=EvidenceAnswerer(),
    ).run_with_plan(
        root_question_id="root:direct-plan",
        plan=plan,
        run_key="empty-plan",
    )

    assert result.status == ReadingRunStatus.READY
    assert result.evidence_memory.questions == []
    assert checker.inputs[0].evidence_memory.current_target.question_id == (
        "root:direct-plan"
    )
    assert result.evidence_memory.evidence[0].supports_question_ids == [
        "root:direct-plan"
    ]
    assert result.answer is not None


def test_controller_can_stop_cleanly_with_incomplete_evidence(tmp_path: Path) -> None:
    document = _document(tmp_path, page_element_specs=[[]])
    controller = QueueController(
        [
            {
                "action": "STOP",
                "reason": "No visible source is likely to resolve the current gap.",
            }
        ]
    )
    result = ReadingEnvironment(
        document,
        asset_root=tmp_path,
        controller=controller,
        reader=DeterministicContentReader(),
        checker=PredicateChecker(lambda _text: False),
        answerer=EvidenceAnswerer(),
    ).run(
        root_question=RootQuestion(
            question_id="root:stop",
            text="What unavailable fact is reported?",
        )
    )

    assert result.status == ReadingRunStatus.STOPPED_INCOMPLETE
    assert result.evidence_memory.root_status == EvidenceStatus.INCOMPLETE
    assert result.answer is None
    assert result.action_trace.entries[-1].action_name == "STOP"
    assert result.diagnostics[-1].code == "controller_stopped_incomplete"


def test_controller_hides_relations_with_unreadable_related_source_preview(
    tmp_path: Path,
) -> None:
    relation = _relation(
        "rel:document-endpoint",
        "figure:1",
        "doc:environment-test",
        RelationType.BELONGS_TO_SECTION,
        RelationStatus.CONFIRMED,
    )
    document = _document(
        tmp_path,
        page_element_specs=[
            [
                {
                    "element_id": "figure:1",
                    "element_type": ElementType.FIGURE,
                    "reference_label": "Figure 1",
                    "visual": True,
                }
            ]
        ],
        relations=[relation],
    )
    controller = QueueController(
        [
            {
                "action": "STOP",
                "reason": "The remaining relation endpoint cannot be read.",
            }
        ]
    )
    result = ReadingEnvironment(
        document,
        asset_root=tmp_path,
        controller=controller,
        reader=DeterministicContentReader(),
        checker=PredicateChecker(lambda _text: False),
        answerer=EvidenceAnswerer(),
    ).run(
        root_question=RootQuestion(
            question_id="root:relation-filter",
            text="What does Figure 1 show?",
        )
    )

    assert result.status == ReadingRunStatus.STOPPED_INCOMPLETE
    assert controller.inputs[0].confirmed_relations == []


def test_visual_asset_ids_do_not_depend_on_absolute_asset_root(
    tmp_path: Path,
) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    first_document = _document(
        first_root,
        page_element_specs=[
            [
                {
                    "element_id": "figure:1",
                    "element_type": ElementType.FIGURE,
                    "visual": True,
                }
            ]
        ],
    )
    second_document = _document(
        second_root,
        page_element_specs=[
            [
                {
                    "element_id": "figure:1",
                    "element_type": ElementType.FIGURE,
                    "visual": True,
                }
            ]
        ],
    )

    first_environment = ReadingEnvironment(
        first_document,
        asset_root=first_root,
        controller=RejectingController(),
        reader=DeterministicContentReader(),
        checker=PredicateChecker(lambda _text: False),
        answerer=EvidenceAnswerer(),
    )
    second_environment = ReadingEnvironment(
        second_document,
        asset_root=second_root,
        controller=RejectingController(),
        reader=DeterministicContentReader(),
        checker=PredicateChecker(lambda _text: False),
        answerer=EvidenceAnswerer(),
    )

    assert first_environment._read_input("page:1", 0).visual_asset_id == (
        second_environment._read_input("page:1", 0).visual_asset_id
    )
    assert first_environment._read_input("figure:1", 0).visual_asset_id == (
        second_environment._read_input("figure:1", 0).visual_asset_id
    )


def test_linux_read_input_resolves_windows_authored_relative_asset_path(
    tmp_path: Path,
) -> None:
    document = _document(
        tmp_path,
        page_element_specs=[
            [
                {
                    "element_id": "figure:windows-path",
                    "element_type": ElementType.FIGURE,
                    "visual": True,
                }
            ]
        ],
    )
    expected = tmp_path / "assets" / "elements" / "figure.png"
    expected.parent.mkdir(parents=True)
    Image.new("RGB", (120, 80), "white").save(expected)
    element = document.elements[0].model_copy(
        update={"image_path": Path(r"assets\elements\figure.png")}
    )
    document = document.model_copy(update={"elements": [element]})
    environment = ReadingEnvironment(
        document,
        asset_root=tmp_path,
        controller=RejectingController(),
        reader=DeterministicContentReader(),
        checker=PredicateChecker(lambda _text: False),
        answerer=EvidenceAnswerer(),
    )

    read_input = environment._read_input("figure:windows-path", 0)

    assert read_input.representation == ReadRepresentation.ELEMENT_VISUAL
    assert read_input.visual_asset_path == expected
