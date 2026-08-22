"""Cross-store referential validation for one document-reading run.

Individual Pydantic models validate their own shape and local references.  This
module validates IDs that cross the canonical store boundaries without copying
the referenced objects into another state model.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

from softdoc.models import Relation, RelationStatus
from softdoc.reading_state import (
    ActionOutcome,
    ActionTrace,
    EvidenceMemory,
    ExplorationState,
    ObservationStore,
)
from softdoc.retrieval.models import SearchSession


class ReadingStateReferenceError(ValueError):
    """Raised when references across reading-state stores are inconsistent."""


class ReadingStateReferenceValidator:
    """Validate references across canonical logs, memory, and derived state.

    ``search_sessions`` and ``relations`` are optional registries.  References
    to those objects are checked only when the corresponding registry is
    supplied.  This lets callers validate the three core canonical stores
    before retrieval or relation state has been loaded.
    """

    def validate(
        self,
        *,
        observation_store: ObservationStore,
        evidence_memory: EvidenceMemory,
        action_trace: ActionTrace,
        exploration_state: ExplorationState | None = None,
        search_sessions: Iterable[SearchSession] | None = None,
        relations: Iterable[Relation] | None = None,
        raise_on_error: bool = False,
    ) -> list[str]:
        errors: list[str] = []

        self._validate_run_identity(
            observation_store=observation_store,
            evidence_memory=evidence_memory,
            action_trace=action_trace,
            exploration_state=exploration_state,
            errors=errors,
        )
        self._validate_actions_and_observations(
            observation_store=observation_store,
            action_trace=action_trace,
            errors=errors,
        )
        self._validate_action_questions(
            evidence_memory=evidence_memory,
            action_trace=action_trace,
            errors=errors,
        )
        self._validate_evidence(
            observation_store=observation_store,
            evidence_memory=evidence_memory,
            errors=errors,
        )
        if exploration_state is not None:
            self._validate_exploration_state(
                observation_store=observation_store,
                action_trace=action_trace,
                exploration_state=exploration_state,
                search_sessions=search_sessions,
                relations=relations,
                errors=errors,
            )

        if errors and raise_on_error:
            raise ReadingStateReferenceError("\n".join(errors))
        return errors

    @staticmethod
    def _validate_run_identity(
        *,
        observation_store: ObservationStore,
        evidence_memory: EvidenceMemory,
        action_trace: ActionTrace,
        exploration_state: ExplorationState | None,
        errors: list[str],
    ) -> None:
        expected_session = observation_store.reading_session_id
        expected_question = observation_store.root_question_id
        stores: list[tuple[str, str, str]] = [
            (
                "EvidenceMemory",
                evidence_memory.reading_session_id,
                evidence_memory.root_question_id,
            ),
            (
                "ActionTrace",
                action_trace.reading_session_id,
                action_trace.root_question_id,
            ),
        ]
        if exploration_state is not None:
            stores.append(
                (
                    "ExplorationState",
                    exploration_state.reading_session_id,
                    exploration_state.root_question_id,
                )
            )
        for store_name, session_id, root_question_id in stores:
            if session_id != expected_session:
                errors.append(
                    f"{store_name} reading_session_id {session_id} does not match "
                    f"ObservationStore {expected_session}"
                )
            if root_question_id != expected_question:
                errors.append(
                    f"{store_name} root_question_id {root_question_id} does not match "
                    f"ObservationStore {expected_question}"
                )

    @staticmethod
    def _validate_actions_and_observations(
        *,
        observation_store: ObservationStore,
        action_trace: ActionTrace,
        errors: list[str],
    ) -> None:
        observations_by_id = {
            observation.observation_id: observation
            for observation in observation_store.observations
        }
        records_by_action_id = {
            record.action_id: record for record in observation_store.read_records
        }
        actions_by_id = {entry.action_id: entry for entry in action_trace.entries}
        record_observations_by_action: dict[str, set[str]] = defaultdict(set)

        for record in observation_store.read_records:
            action = actions_by_id.get(record.action_id)
            if action is None:
                errors.append(
                    f"ReadRecord for {record.action_id} references missing Action "
                    f"{record.action_id}"
                )
            else:
                expected_question_id = (
                    record.subquestion_id or observation_store.root_question_id
                )
                if action.question_id != expected_question_id:
                    errors.append(
                        f"Action {record.action_id} question_id "
                        f"{action.question_id} does not match ReadRecord "
                        f"question {expected_question_id}"
                    )
            record_observations_by_action[record.action_id].update(
                record.observation_ids
            )

        for entry in action_trace.entries:
            for observation_id in entry.observation_ids:
                observation = observations_by_id.get(observation_id)
                if observation is None:
                    errors.append(
                        f"Action {entry.action_id} references missing Observation "
                        f"{observation_id}"
                    )
                    continue
                record = records_by_action_id.get(observation.action_id)
                if record is None:
                    # ObservationStore's own validator normally makes this
                    # impossible, but retaining the guard keeps this boundary
                    # safe if models are loaded from a non-Pydantic source.
                    errors.append(
                        f"Observation {observation_id} references missing ReadRecord "
                        f"for Action {observation.action_id}"
                    )
                elif record.action_id != entry.action_id:
                    errors.append(
                        f"Action {entry.action_id} claims Observation {observation_id}, "
                        f"but its ReadRecord belongs to Action {record.action_id}"
                    )

        for action_id, observation_ids in record_observations_by_action.items():
            entry = actions_by_id.get(action_id)
            if entry is None:
                continue
            trace_ids = set(entry.observation_ids)
            missing_from_trace = sorted(observation_ids - trace_ids)
            for observation_id in missing_from_trace:
                errors.append(
                    f"ReadRecord assigns Observation {observation_id} to Action "
                    f"{action_id}, but ActionTrace does not reference it"
                )
            unowned_by_record = sorted(trace_ids - observation_ids)
            for observation_id in unowned_by_record:
                if observation_id in observations_by_id:
                    errors.append(
                        f"Action {action_id} references Observation {observation_id}, "
                        "but no ReadRecord for that Action owns it"
                    )

    @staticmethod
    def _validate_action_questions(
        *,
        evidence_memory: EvidenceMemory,
        action_trace: ActionTrace,
        errors: list[str],
    ) -> None:
        known_question_ids = {
            evidence_memory.root_question_id,
            *(question.question_id for question in evidence_memory.questions),
        }
        for action in action_trace.entries:
            if action.question_id not in known_question_ids:
                errors.append(
                    f"Action {action.action_id} references missing Question "
                    f"{action.question_id}"
                )

    @staticmethod
    def _validate_evidence(
        *,
        observation_store: ObservationStore,
        evidence_memory: EvidenceMemory,
        errors: list[str],
    ) -> None:
        known_question_ids = {
            evidence_memory.root_question_id,
            *(question.question_id for question in evidence_memory.questions),
        }
        observation_ids = {
            observation.observation_id
            for observation in observation_store.observations
        }
        for evidence in evidence_memory.evidence:
            for question_id in evidence.supports_question_ids:
                if question_id not in known_question_ids:
                    errors.append(
                        f"Evidence {evidence.evidence_id} supports missing Question "
                        f"{question_id}"
                    )
            for observation_id in evidence.observation_ids:
                if observation_id not in observation_ids:
                    errors.append(
                        f"Evidence {evidence.evidence_id} references missing "
                        f"Observation {observation_id}"
                    )

    @staticmethod
    def _validate_exploration_state(
        *,
        observation_store: ObservationStore,
        action_trace: ActionTrace,
        exploration_state: ExplorationState,
        search_sessions: Iterable[SearchSession] | None,
        relations: Iterable[Relation] | None,
        errors: list[str],
    ) -> None:
        requested_source_ids = {
            source.source_id
            for record in observation_store.read_records
            for source in record.inputs
        }
        for source_id in exploration_state.attempted_source_ids:
            if source_id not in requested_source_ids:
                errors.append(
                    f"ExplorationState references unknown attempted source {source_id}"
                )

        actions_by_id = {entry.action_id: entry for entry in action_trace.entries}
        for action in exploration_state.recent_actions:
            if action.action_id not in actions_by_id:
                errors.append(
                    f"ExplorationState references missing recent Action "
                    f"{action.action_id}"
                )

        if exploration_state.current_focus is not None:
            focus = exploration_state.current_focus
            successful_focuses = {
                entry.primary_target.source_id
                for entry in action_trace.entries
                if entry.primary_target is not None
                and entry.outcome in {
                    ActionOutcome.SUCCEEDED,
                    ActionOutcome.DEGRADED,
                }
            }
            if focus.source_id not in successful_focuses:
                errors.append(
                    f"ExplorationState current_focus {focus.source_id} is not the "
                    "primary target of a successful or degraded Action"
                )

        if search_sessions is not None:
            session_list = list(search_sessions)
            sessions_by_id = {
                session.search_session_id: session for session in session_list
            }
            if len(sessions_by_id) != len(session_list):
                errors.append("SearchSession registry contains duplicate IDs")
            for session_id in exploration_state.active_search_session_ids:
                if session_id not in sessions_by_id:
                    errors.append(
                        f"ExplorationState references missing SearchSession {session_id}"
                    )

        if relations is not None:
            relation_list = list(relations)
            relations_by_id = {
                relation.relation_id: relation for relation in relation_list
            }
            if len(relations_by_id) != len(relation_list):
                errors.append("Relation registry contains duplicate IDs")
            handles_with_status = [
                *(
                    (handle, RelationStatus.CONFIRMED)
                    for handle in exploration_state.confirmed_relation_handles
                ),
                *(
                    (handle, RelationStatus.CANDIDATE)
                    for handle in exploration_state.candidate_navigation_hints
                ),
            ]
            for handle, expected_status in handles_with_status:
                relation = relations_by_id.get(handle.relation_id)
                if relation is None:
                    errors.append(
                        f"ExplorationState references missing Relation "
                        f"{handle.relation_id}"
                    )
                    continue
                if (
                    handle.source_id != relation.source_id
                    or handle.target_id != relation.target_id
                    or handle.relation_type != relation.relation_type
                ):
                    errors.append(
                        f"ExplorationState Relation handle {handle.relation_id} does "
                        "not match the canonical Relation"
                    )
                if relation.status != expected_status:
                    errors.append(
                        f"ExplorationState Relation handle {handle.relation_id} has "
                        f"status {expected_status.value}, but canonical Relation is "
                        f"{relation.status.value}"
                    )
