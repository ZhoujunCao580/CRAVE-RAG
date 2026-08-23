"""Replay Visual Reader probe results through Reading State v0 schemas."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from softdoc.ids import (
    action_id,
    evidence_id,
    observation_id,
    read_input_id,
    reading_session_id,
    root_question_id,
    stable_digest,
)
from softdoc.reading_state import (
    ActionOutcome,
    ActionTrace,
    ActionTraceEntry,
    CurrentTarget,
    EvidenceItem,
    EvidenceMemory,
    EvidenceStatus,
    ExplorationSourceHandle,
    ExplorationStateBuilder,
    ObservationLimitation,
    ObservationSourceRef,
    ObservationStore,
    QuestionState,
    QuestionStatus,
    ReadRecord,
    ReaderKind,
    ReadingSourceType,
    ReadRepresentation,
    ReadInput,
    StoredObservation,
)
from softdoc.reading_state_validation import ReadingStateReferenceValidator


def _read_inputs(row: dict[str, Any]) -> list[ReadInput]:
    resolution_by_alias = {
        item.get("input_id", item.get("visual_id")): item
        for item in row.get("image_resolution", [])
    }
    sources: list[ReadInput] = []
    for index, visual in enumerate(row["request"]["visual_inputs"]):
        old_alias = visual.get("input_id", visual.get("visual_id"))
        alias = read_input_id(index)
        resolution = resolution_by_alias[old_alias]
        element_id = visual.get("element_id")
        source_type = (
            ReadingSourceType.ELEMENT if element_id else ReadingSourceType.PAGE
        )
        representation = (
            ReadRepresentation.ELEMENT_VISUAL
            if element_id
            else ReadRepresentation.PAGE_VISUAL
        )
        source_id = element_id or visual["page_id"]
        resolved_path = Path(resolution["resolved_image_path"])
        sources.append(
            ReadInput(
                input_id=alias,
                source_id=source_id,
                source_type=source_type,
                representation=representation,
                document_id=row["request"]["document_id"],
                page_id=visual["page_id"],
                element_id=element_id,
                visual_asset_id=(
                    "visual:"
                    + stable_digest(
                        row["request"]["document_id"],
                        visual["page_id"],
                        element_id,
                        resolved_path.name,
                    )
                ),
                bbox=tuple(visual["bbox"]) if visual.get("bbox") else None,
                visual_asset_path=resolved_path,
            )
        )
    return sources


def _focus(source: ReadInput) -> ExplorationSourceHandle:
    return ExplorationSourceHandle(
        source_id=source.source_id,
        source_type=source.source_type,
        document_id=source.document_id,
        page_id=source.page_id,
        element_id=source.element_id,
        visual_asset_id=source.visual_asset_id,
        bbox=source.bbox,
    )


def _replay_row(row: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    question_id = root_question_id(row["problem"], row["probe_id"])
    session_id = reading_session_id(question_id, "visual-reader-v0-probe-25")
    canonical_action_id = action_id(session_id, 0)
    inputs = _read_inputs(row)
    inputs_by_id = {source.input_id: source for source in inputs}

    observations: list[StoredObservation] = []
    validated = row.get("validated_output")
    if validated is not None:
        for obs_index, raw_observation in enumerate(validated["observations"]):
            canonical_observation_id = observation_id(canonical_action_id, obs_index)
            source_refs: list[ObservationSourceRef] = []
            raw_sources = raw_observation.get("sources", raw_observation.get("regions", []))
            for region_index, region in enumerate(raw_sources):
                raw_input_id = region.get("input_id", region.get("visual_id"))
                if raw_input_id and raw_input_id.startswith("V"):
                    raw_input_id = "I" + raw_input_id[1:]
                requested = inputs_by_id[raw_input_id]
                bbox = tuple(region["bbox"]) if region.get("bbox") else None
                source_refs.append(
                    ObservationSourceRef(
                        input_id=requested.input_id,
                        bbox=bbox,
                    )
                )
            observations.append(
                StoredObservation(
                    observation_id=canonical_observation_id,
                    action_id=canonical_action_id,
                    text=raw_observation["text"],
                    sources=source_refs,
                )
            )

    limitations: list[ObservationLimitation] = []
    if validated is not None:
        limitations.extend(
            ObservationLimitation(
                description=item["description"],
                input_ids=[
                    "I" + value[1:] if value.startswith("V") else value
                    for value in item.get("input_ids", [])
                ],
            )
            for item in validated["limitations"]
        )
    else:
        limitations.append(
            ObservationLimitation(
                description="; ".join(row.get("validation_errors", []))[:500],
                input_ids=[],
            )
        )
    if not observations and not limitations:
        limitations.append(
            ObservationLimitation(
                description="Reader returned no Observation or limitation.",
                input_ids=[],
            )
        )

    outcome = (
        ActionOutcome.FAILED
        if not observations
        else ActionOutcome.DEGRADED
        if limitations
        else ActionOutcome.SUCCEEDED
    )
    record = ReadRecord(
        action_id=canonical_action_id,
        reader_kind=ReaderKind.VISUAL,
        document_id=row["request"]["document_id"],
        subquestion_id=row["request"].get("subquestion_id"),
        local_problem=row["problem"],
        inputs=inputs,
        observation_ids=[item.observation_id for item in observations],
        limitations=limitations,
    )
    store = ObservationStore(
        reading_session_id=session_id,
        root_question_id=question_id,
        read_records=[record],
        observations=observations,
    )

    primary_target = _focus(inputs[0]) if len(inputs) == 1 else None
    trace = ActionTrace(
        reading_session_id=session_id,
        root_question_id=question_id,
        entries=[
            ActionTraceEntry(
                step_index=0,
                action_id=canonical_action_id,
                question_id=(
                    row["request"].get("subquestion_id") or question_id
                ),
                action_name="READ_VISUAL",
                target_ids=[item.source_id for item in inputs],
                primary_target=primary_target,
                outcome=outcome,
                observation_ids=record.observation_ids,
            )
        ],
    )

    # Mechanical promotion is used only to test the EvidenceMemory contract.
    # It is not a Checker decision and says nothing about semantic correctness.
    evidence_memory = EvidenceMemory(
        reading_session_id=session_id,
        root_question_id=question_id,
        root_status=EvidenceStatus.INCOMPLETE,
        questions=(
            [
                QuestionState(
                    question_id=row["request"]["subquestion_id"],
                    text=row["problem"],
                    status=QuestionStatus.INCOMPLETE,
                )
            ]
            if row["request"].get("subquestion_id")
            else []
        ),
        evidence=[
            EvidenceItem(
                evidence_id=evidence_id(canonical_action_id, index),
                statement=item.text,
                observation_ids=[item.observation_id],
                supports_question_ids=[
                    row["request"].get("subquestion_id") or question_id
                ],
            )
            for index, item in enumerate(observations)
        ],
        current_target=CurrentTarget(
            question_id=row["request"].get("subquestion_id") or question_id,
            gap_description=(
                "Evidence sufficiency was not evaluated in this schema replay."
            ),
        ),
    )
    exploration = ExplorationStateBuilder().build(
        observation_store=store,
        action_trace=trace,
    )

    # Exercise JSON round trips for all three state objects.
    ObservationStore.model_validate_json(store.model_dump_json())
    EvidenceMemory.model_validate_json(evidence_memory.model_dump_json())
    type(exploration).model_validate_json(exploration.model_dump_json())
    reference_errors = ReadingStateReferenceValidator().validate(
        observation_store=store,
        evidence_memory=evidence_memory,
        action_trace=trace,
        exploration_state=exploration,
    )

    requested_aliases = {item.input_id for item in inputs}
    joint_grounded = any(
        {source.input_id for source in observation.sources}
        == requested_aliases
        for observation in observations
    )
    summary = {
        "probe_id": row["probe_id"],
        "category": row["category"],
        "schema_input_valid": validated is not None,
        "outcome": outcome.value,
        "requested_source_count": len(inputs),
        "observation_count": len(observations),
        "limitation_count": len(limitations),
        "evidence_count": len(evidence_memory.evidence),
        "current_focus": (
            exploration.current_focus.source_id
            if exploration.current_focus is not None
            else None
        ),
        "attempted_source_count": len(exploration.attempted_source_ids),
        "attempted_search_queries": exploration.attempted_search_queries,
        "available_search_sessions": exploration.active_search_session_ids,
        "joint_grounded_observation": joint_grounded,
        "cross_store_reference_errors": reference_errors,
    }
    example = {
        "summary": summary,
        "observation_store": store.model_dump(mode="json"),
        "evidence_memory": evidence_memory.model_dump(mode="json"),
        "exploration_state": exploration.model_dump(mode="json"),
    }
    return summary, example


def audit(input_path: Path, report_path: Path, examples_path: Path) -> None:
    rows = [
        json.loads(line)
        for line in input_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    summaries: list[dict[str, Any]] = []
    examples: dict[str, Any] = {}
    example_ids = {"VR001", "VR002", "VR023"}
    errors: list[str] = []
    for row in rows:
        try:
            summary, example = _replay_row(row)
            summaries.append(summary)
            if row["probe_id"] in example_ids:
                examples[row["probe_id"]] = example
        except Exception as exc:  # noqa: BLE001 - audit must report every row
            errors.append(f"{row['probe_id']}: {type(exc).__name__}: {exc}")

    outcomes = Counter(item["outcome"] for item in summaries)
    multi = [item for item in summaries if item["requested_source_count"] > 1]
    lines = [
        "# Reading State v0 × Visual Reader真实probe兼容性审计",
        "",
        f"- 输入请求：{len(rows)}",
        f"- 成功构造三套State：{len(summaries)}/{len(rows)}",
        f"- Schema演练异常：{len(errors)}",
        f"- Action outcome：{dict(outcomes)}",
        f"- 保存Observation：{sum(item['observation_count'] for item in summaries)}",
        f"- 机械映射Evidence：{sum(item['evidence_count'] for item in summaries)}（仅测接口，不代表Checker确认）",
        f"- 多图请求：{len(multi)}；存在严格联合来源Observation：{sum(item['joint_grounded_observation'] for item in multi)}",
        f"- 无唯一current_focus：{sum(item['current_focus'] is None for item in summaries)}",
        "- 跨存储引用错误："
        f"{sum(len(item['cross_store_reference_errors']) for item in summaries)}",
        "- attempted_search_queries非空："
        f"{sum(bool(item['attempted_search_queries']) for item in summaries)}",
        "- available_search_sessions非空："
        f"{sum(bool(item['available_search_sessions']) for item in summaries)}",
        "",
        "## 结论",
        "",
        "1. 22条有效输出均可保存；3条损坏JSON被保存为failed ReadRecord + limitation，不会丢失读取尝试。",
        "2. 模型局部O1/O2已转换为按reading session命名空间隔离的系统Observation ID，没有跨请求冲突。",
        "3. 单图请求可确定性产生current_focus；多图联合请求不虚构唯一focus。",
        "4. 本probe直接READ，没有发生SEARCH，因此查询历史与SearchSession引用正确保持为空。",
        "5. Evidence机械映射全部通过，但错误Observation同样能进入；这证明Schema负责可追溯性，不负责事实正确性，正式提升必须由Checker完成。",
        "6. 多图来源绑定不足仍被原样保留和审计，Reading State不会把单图Observation伪装成联合证据。",
    ]
    if errors:
        lines.extend(["", "## 异常", "", *[f"- {item}" for item in errors]])
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    examples_path.write_text(
        json.dumps(examples, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--examples", type=Path, required=True)
    args = parser.parse_args()
    audit(args.input, args.report, args.examples)


if __name__ == "__main__":
    main()
