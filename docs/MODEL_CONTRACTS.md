# Model Contracts

This document is the canonical, human-readable view of CRAVE-RAG's six
model-facing boundaries. The Pydantic models in `src/softdoc` remain the
executable source of truth. Editable frozen prompt text lives in
`src/softdoc/prompts/` and is exposed through `softdoc prompts show
<component>`.

The examples use one coherent question so the data flow is visible. IDs are
opaque handles assigned or validated by the program; models must copy existing
IDs and must not invent document handles.

## 1. Planner

The orchestrator supplies one Root Question. Stable Planner rules are sent in
the system message. The dynamic user message contains the question exactly
once:

```json
{
  "original_question": "How did revenue change from 2022 to 2023?"
}
```

The model returns a conservative question DAG. `subquestions` is required but
may be empty. With an empty plan, the Root Question itself becomes the current
target and is not duplicated as a fake Q1.

```json
{
  "original_question": "What is the title of Figure 3?",
  "subquestions": []
}
```

When the Root explicitly names multiple independent evidence needs, the
Planner may decompose them. Independent needs remain parallel:

```json
{
  "original_question": "How did revenue change from 2022 to 2023?",
  "subquestions": [
    {
      "subquestion_id": "Q1",
      "text": "What was the revenue in 2022?",
      "depends_on": []
    },
    {
      "subquestion_id": "Q2",
      "text": "What was the revenue in 2023?",
      "depends_on": []
    }
  ]
}
```

After validation, the program adds `planner_trace` and stores an `InitialPlan`:

```json
{
  "original_question": "How did revenue change from 2022 to 2023?",
  "subquestions": [
    {
      "subquestion_id": "Q1",
      "text": "What was the revenue in 2022?",
      "depends_on": []
    },
    {
      "subquestion_id": "Q2",
      "text": "What was the revenue in 2023?",
      "depends_on": []
    }
  ],
  "planner_trace": {
    "backend_name": "ollama",
    "model": "qwen3:8b",
    "prompt_version": "planner-v0.21",
    "warnings": [],
    "metadata": {}
  }
}
```

## 2. Visual retrieval indexing

Before question answering, a VLM may create a search-only identity for a real
Figure, Chart, or Table asset. Confirmed captions and the section path may be
supplied as context. Candidate Relations are excluded. The model returns:

```json
{
  "search_summary": "A line chart compares AP and AP50 with and without NMS across decoder layers.",
  "keywords": ["AP", "AP50", "NMS", "decoder layers"]
}
```

The program binds this output to the requested Element, records the visual
asset hash, generator model, Prompt version, and purpose `search_only`, then
adds the summary and keywords to the Element's SearchUnit. BM25 and Dense
retrieval may rank this text, but it is never an Observation or Evidence. A
Reader must still inspect the source image before the Checker can admit a fact
into Evidence Memory.

The frozen Prompt is `visual-retrieval-v0.1`. It deliberately avoids chart
values, trends, rankings, conclusions, inferred identities, and decorative
diagram objects.

## 3. Reader

The example below is the Visual Reader contract. Text, table, page, and visual
Readers are normalized afterward into the same `ReadRecord` and
`StoredObservation` representation.

Input:

```json
{
  "action_id": "action:read:1",
  "subquestion_id": "Q1",
  "document_id": "doc:annual-report",
  "source_name": "Annual Report 2023",
  "problem": "Read the 2022 revenue value shown in this chart.",
  "visual_inputs": [
    {
      "input_id": "I1",
      "visual_asset_id": "visual:chart:revenue",
      "page_id": "page:financial-results",
      "page_number": 18,
      "display_page_label": "16",
      "page_image_path": "assets/elements/revenue_chart.png",
      "element_id": "element:chart:revenue",
      "element_type": "chart",
      "bbox": [0.12, 0.24, 0.88, 0.72]
    }
  ]
}
```

`page_image_path` is a frozen legacy field name. Its value is the visual asset
actually sent to the model, which may be a full-page image or an element crop.
The backend copies the same asset path into the stored `visual_asset_path`.

Output:

```json
{
  "observations": [
    {
      "text": "The chart reports 2022 revenue of 10 million dollars.",
      "sources": [
        {
          "input_id": "I1",
          "bbox": [0.48, 0.31, 0.62, 0.44]
        }
      ]
    }
  ],
  "limitations": []
}
```

The program assigns stable Observation IDs and persists the read, including
failed reads. The canonical stored form for this example is:

```json
{
  "reading_session_id": "reading:1",
  "root_question_id": "root:1",
  "read_records": [
    {
      "action_id": "action:read:1",
      "reader_kind": "visual",
      "document_id": "doc:annual-report",
      "subquestion_id": "Q1",
      "local_problem": "Read the 2022 revenue value shown in this chart.",
      "inputs": [
        {
          "input_id": "I1",
          "source_id": "element:chart:revenue",
          "source_type": "element",
          "representation": "element_visual",
          "document_id": "doc:annual-report",
          "page_id": "page:financial-results",
          "element_id": "element:chart:revenue",
          "table_view_id": null,
          "cell_id": null,
          "visual_asset_id": "visual:chart:revenue",
          "bbox": [0.12, 0.24, 0.88, 0.72],
          "visual_asset_path": "assets/elements/revenue_chart.png"
        }
      ],
      "observation_ids": ["observation:1"],
      "limitations": []
    }
  ],
  "observations": [
    {
      "observation_id": "observation:1",
      "action_id": "action:read:1",
      "text": "The chart reports 2022 revenue of 10 million dollars.",
      "sources": [
        {
          "input_id": "I1",
          "cell_id": null,
          "bbox": [0.48, 0.31, 0.62, 0.44]
        }
      ]
    }
  ]
}
```

## 4. Evidence Checker

The Checker normally receives the complete current Evidence Memory plus only
the new Observations and limitations produced by one read action. After the
last planned SubQuestion is satisfied, it may instead receive a state-only
Root-finalization input with empty Observations and limitations. It returns a
delta; the program applies that delta atomically.

Input:

```json
{
  "action_id": "action:read:1",
  "root_question": {
    "question_id": "root:1",
    "text": "How did revenue change from 2022 to 2023?"
  },
  "evidence_memory": {
    "reading_session_id": "reading:1",
    "root_question_id": "root:1",
    "root_status": "incomplete",
    "questions": [
      {
        "question_id": "Q1",
        "text": "What was the revenue in 2022?",
        "depends_on": [],
        "status": "incomplete"
      },
      {
        "question_id": "Q2",
        "text": "What was the revenue in 2023?",
        "depends_on": [],
        "status": "incomplete"
      }
    ],
    "evidence": [],
    "current_target": {
      "question_id": "Q1",
      "gap_description": "The 2022 revenue is unknown."
    }
  },
  "observations": [
    {
      "observation_id": "observation:1",
      "action_id": "action:read:1",
      "text": "The chart reports 2022 revenue of 10 million dollars.",
      "sources": [
        {
          "input_id": "I1",
          "cell_id": null,
          "bbox": [0.48, 0.31, 0.62, 0.44]
        }
      ]
    }
  ],
  "limitations": []
}
```

Model output (`EvidenceCheckDecision`):

```json
{
  "action_id": "action:read:1",
  "observation_assessments": [
    {
      "observation_id": "observation:1",
      "assessment": "The observation directly and reliably supplies the missing 2022 value."
    }
  ],
  "evidence_updates": {
    "add": [
      {
        "statement": "Revenue in 2022 was 10 million dollars.",
        "observation_ids": ["observation:1"],
        "supports_question_ids": ["Q1"]
      }
    ],
    "replace": [],
    "remove": []
  },
  "current_target_status": "satisfied",
  "root_status": "incomplete",
  "remaining_gap_description": null
}
```

The model does not output `used_for_evidence`. The program first applies the
Evidence delta atomically, then derives that audit flag from the
`observation_ids` referenced by the resulting Evidence set and materializes an
`EvidenceCheckResult` for persisted feedback. This prevents a redundant model
field from disagreeing with the actual Evidence delta.

The program, not the model, assigns the Evidence ID and activates the next
runnable question. Evidence may support either a planned SubQuestion or the
Root Question. With an empty plan, `questions` is empty,
`current_target.question_id` equals `root_question_id`, and accepted Evidence
uses `supports_question_ids: ["root:1"]`.

Completing all planned SubQuestions does not by itself guarantee that the Root
is answerable. The program makes the Root the next current target, copies the
Root Question text as its initial gap, and immediately asks the Checker to
evaluate the complete accepted Evidence without inventing another read. If the
Root is still incomplete, the Checker's specific remaining gap becomes the
Controller's next working target.

## 5. Controller

The Controller receives a derived working view, not the full ObservationStore
or the full document graph. Candidate previews and Relations are navigation
opportunities, not Evidence. When the plan is empty, `subquestions` is empty
and `current_gap.question_id` is the Root ID; no separate Controller schema is
needed.

Input:

```json
{
  "reading_session_id": "reading:1",
  "root_question": {
    "question_id": "root:1",
    "text": "How did revenue change from 2022 to 2023?"
  },
  "root_status": "incomplete",
  "subquestions": [
    {
      "question_id": "Q1",
      "text": "What was the revenue in 2022?",
      "depends_on": [],
      "status": "satisfied"
    },
    {
      "question_id": "Q2",
      "text": "What was the revenue in 2023?",
      "depends_on": [],
      "status": "incomplete"
    }
  ],
  "evidence": [
    {
      "evidence_id": "evidence:1",
      "statement": "Revenue in 2022 was 10 million dollars.",
      "supports_question_ids": ["Q1"]
    }
  ],
  "current_gap": {
    "question_id": "Q2",
    "description": "The 2023 revenue is unknown."
  },
  "reading_locations": [
    {
      "source_id": "element:chart:revenue",
      "source_type": "element",
      "page_id": "page:financial-results"
    }
  ],
  "recent_actions": [
    {
      "action_id": "action:read:1",
      "question_id": "Q1",
      "action_name": "READ_SOURCE",
      "target_ids": ["element:chart:revenue"],
      "execution_status": "succeeded",
      "observation_ids": ["observation:1"],
      "feedback": {
        "limitations": [],
        "observation_assessments": [
          {
            "observation_id": "observation:1",
            "used_for_evidence": true,
            "assessment": "The observation supplied the required 2022 value."
          }
        ]
      }
    }
  ],
  "confirmed_relations": [
    {
      "relation_id": "relation:caption-of:1",
      "relation_type": "caption_of",
      "source_id": "element:caption:revenue",
      "target_id": "element:chart:revenue",
      "current_endpoint_id": "element:chart:revenue",
      "related_source_preview": {
        "source_id": "element:caption:revenue",
        "source_type": "element",
        "page_id": "page:financial-results",
        "element_type": "caption",
        "section_path": ["Financial Results"],
        "label_or_snippet": "Figure 3. Revenue for 2022 and 2023.",
        "content_availability": "text_only"
      }
    }
  ],
  "candidate_relations": [
    {
      "relation_id": "relation:refers-to:1",
      "relation_type": "refers_to",
      "source_id": "element:chart:revenue",
      "target_id": "element:paragraph:revenue-note",
      "current_endpoint_id": "element:chart:revenue",
      "related_source_preview": {
        "source_id": "element:paragraph:revenue-note",
        "source_type": "element",
        "page_id": "page:financial-results",
        "element_type": "paragraph",
        "section_path": ["Financial Results"],
        "label_or_snippet": "Revenue increased because subscription sales grew.",
        "content_availability": "text_only"
      },
      "confidence": 0.76
    }
  ],
  "search_tabs": [
    {
      "search_session_id": "search:2023-revenue",
      "query": "2023 revenue",
      "has_more": true
    }
  ],
  "visible_search_view": {
    "search_session_id": "search:2023-revenue",
    "exact_anchor_matches": [],
    "candidate_previews": [
      {
        "element_id": "element:table:2023-results",
        "element_type": "table",
        "page_id": "page:2023-results",
        "section_path": ["Financial Results"],
        "matched_snippet": "Revenue ... 2023 ... 12 million",
        "content_availability": "structured"
      }
    ]
  },
  "remaining_action_budget": 4
}
```

`related_source_preview` is produced without a model. For an Element, the
builder combines its explicit `reference_label`, own `text`, and (when useful)
HTML converted to plain text, removes repeated whitespace, and truncates the
result to 240 characters. It also exposes the Element type, section path, page
handle, and content availability. For a Page endpoint, it exposes only the
opaque page handle and does not reveal or infer a physical or printed page
number. The preview is not persisted into
SoftDoc, does not copy related-neighbor content into retrieval, and cannot be
promoted to Evidence without a real read.

The builder exposes Relations only when one endpoint matches the most recent
successful or degraded read focus. In this example the focus is
`element:chart:revenue`, so the candidate Relation must touch that chart.

Output (one action only):

```json
{
  "action": "READ_SOURCE",
  "source_ids": ["element:table:2023-results"],
  "local_problem": "Read the reported 2023 revenue value."
}
```

The complete action union also supports `SEARCH` (`new`, `next`, or `switch`),
`FOLLOW_RELATION`, `EXPLORE_CANDIDATE_RELATION`, `READ_ADJACENT_PAGE`, and
`STOP`. Exact Anchor matches are resolved and read by the Environment before an
ordinary Controller search decision when the match is unique and readable.

## 6. Answerer

The Answerer is callable only after the Root is `ready`. It sees accepted
Evidence and the question DAG, but not raw retrieval results, Relations, or
Observations. `question_graph` may be empty when no decomposition was needed.
In that case the accepted Evidence supports the Root ID directly.

Input:

```json
{
  "reading_session_id": "reading:1",
  "root_question": {
    "question_id": "root:1",
    "text": "How did revenue change from 2022 to 2023?"
  },
  "question_graph": [
    {
      "question_id": "Q1",
      "text": "What was the revenue in 2022?",
      "depends_on": []
    },
    {
      "question_id": "Q2",
      "text": "What was the revenue in 2023?",
      "depends_on": []
    }
  ],
  "evidence": [
    {
      "evidence_id": "evidence:1",
      "statement": "Revenue in 2022 was 10 million dollars.",
      "supports_question_ids": ["Q1"]
    },
    {
      "evidence_id": "evidence:2",
      "statement": "Revenue in 2023 was 12 million dollars.",
      "supports_question_ids": ["Q2"]
    }
  ]
}
```

Output:

```json
{
  "answer": "Revenue increased by 2 million dollars from 2022 to 2023, a 20% increase.",
  "used_evidence_ids": ["evidence:1", "evidence:2"]
}
```

The program expands `used_evidence_ids` through Evidence -> Observation ->
ReadRecord -> SoftDoc source when final citations are materialized.

## Runtime order

The headings above follow the requested component presentation order. The
actual stateful loop is:

```text
Planner
  -> initialize either the Root target or the first runnable SubQuestion
  -> initial Exact/Search entry
  -> Controller action
  -> Reader
  -> Evidence Checker
  -> state-only Root finalization when the last SubQuestion completes
  -> Controller action (repeat while incomplete)
  -> Answerer (only when ready)
```

Search results, CandidatePreviews, Relations, and Observations never bypass the
Checker to become answer Evidence.

## Teacher reviews and component-specific SFT export

A model run is already the complete trajectory. Teacher supervision is stored
as a separate, deliberately small review rather than copying the run:

```json
{
  "schema_version": "teacher-review-v0",
  "reading_session_id": "reading:1",
  "episode_status": "accepted",
  "controller_steps": [
    {
      "controller_call_index": 0,
      "action_id": "action:1",
      "training_label_status": "accepted",
      "review_note": "The selected source directly addresses the current gap."
    }
  ],
  "first_corrupted_action_id": null
}
```

`teacher_review.json` is not training data. The exporter joins it with the
corresponding `ModelPipelineRun`, validates exact call/action coverage and the
recorded Controller action, and produces:

```text
controller_sft.jsonl   accepted ControllerInput -> ControllerAction examples
controller_sft_messages.jsonl
                       the same examples as system/user/assistant messages
dataset_info.json      LLaMA-Factory OpenAI-messages dataset registration
dataset_manifest.json Prompt hash/version and source-run lineage
```

The review stores neither Gold-access declarations nor duplicated Evidence.
Gold isolation is enforced by the generation/evaluation protocol, while the
canonical run remains the source of truth for what the model saw and what the
environment executed.

Checker supervision uses the same separation. `checker_review.json` labels
each recorded Checker call and may attach a corrected `replacement_output`
without mutating the raw run. Before export, the corrected or original
canonical `EvidenceCheckResult` is applied to its recorded `EvidenceCheckInput`
using the normal atomic state-transition validator. The exporter then removes
the program-derived audit boolean and emits the model-facing
`EvidenceCheckDecision`. The separate export contains:

```text
checker_sft.jsonl      accepted EvidenceCheckInput -> EvidenceCheckDecision examples
checker_sft_messages.jsonl
                       the same examples as system/user/assistant messages
dataset_info.json      LLaMA-Factory dataset registration
dataset_manifest.json Checker prompt hash/version and source-run lineage
```

Controller and Checker examples are never mixed into one dataset or one
adapter merely because they came from the same full episode.
