# CRAVE-RAG Architecture

**CRAVE-RAG** stands for **Controller-guided Reading and Action Via Evidence Gaps**. SoftDoc remains the parser-neutral document representation, and `softdoc` remains the stable Python package and CLI name.

![CRAVE-RAG overview](assets/crave-rag-overview.svg)

## Core Loop

1. A question and the persistent candidate pool provide possible reading entry points.
2. The Controller chooses the next reading move using the current Evidence state and available document structure.
3. The selected Reader produces source-linked Observations.
4. The Evidence Checker evaluates progress, revises Evidence Memory, and maintains the active gap.
5. If Evidence is incomplete, control returns to the Controller. If it is sufficient, the Answerer responds using accepted Evidence.

## Safety Boundary

- Search results and candidate previews are reading leads, not Evidence.
- Confirmed Relations are navigation handles.
- Candidate Relations are investigation hints, not established facts.
- Readers produce Observations; only the Checker may admit them into Evidence Memory.
- The Answerer receives accepted Evidence rather than raw retrieval or navigation state.
- Accepted Evidence remains traceable through its Observation and read record to the underlying document source.

## Implementation Boundary

The repository implements and tests the SoftDoc foundation, MinerU adapter and deterministic passes, document relations, retrieval stack, resumable search sessions, candidate previews, and the contracts for planning, reading, evidence checking, and answering.

The production Controller policy, model-quality evaluation, deferred planning, Observation Recall, post-training, and full-dataset end-to-end answer evaluation remain research-stage work.
