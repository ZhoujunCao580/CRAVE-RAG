# Model Contracts

This document is the canonical, human-readable view of CRAVE-RAG's five
model-facing boundaries. The Pydantic models in `src/softdoc` remain the
executable source of truth, and the frozen prompt text is exposed through
`softdoc prompts show <component>`.

The examples use one coherent question so the data flow is visible. IDs are
opaque handles assigned or validated by the program; models must copy existing
IDs and must not invent document handles.

## 1. Planner

The orchestrator supplies one Root Question. The current Planner API accepts
the question text directly; the equivalent JSON envelope is:

```json
{
  "root_question": {
    "question_id": "root:1",
    "text": "How did revenue change from 2022 to 2023?"
  }
}
```

The model returns a conservative question DAG. The Root Question is not
duplicated as a node. Independent evidence needs remain parallel.

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
    "prompt_version": "planner-v0.14",
    "warnings": [],
    "metadata": {}
  }
}
```

## 2. Reader

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
      "page_image_path": "assets/pages/page_0018.png",
      "element_id": "element:chart:revenue",
      "element_type": "chart",
      "bbox": [0.12, 0.24, 0.88, 0.72]
    }
  ]
}
```

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

## 3. Evidence Checker

The Checker receives the complete current Evidence Memory plus only the new
Observations and limitations produced by one read action. It returns a delta;
the program applies that delta atomically.

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

Output:

```json
{
  "action_id": "action:read:1",
  "observation_assessments": [
    {
      "observation_id": "observation:1",
      "used_for_evidence": true,
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

The program, not the model, assigns the Evidence ID and activates the next
runnable question.

## 4. Controller

The Controller receives a derived working view, not the full ObservationStore
or the full document graph. Candidate previews and Relations are navigation
opportunities, not Evidence.

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
      "target_id": "element:chart:revenue"
    }
  ],
  "candidate_relations": [
    {
      "relation_id": "relation:continued-on:1",
      "relation_type": "continued_on",
      "source_id": "element:table:2022",
      "target_id": "element:table:2023",
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

## 5. Answerer

The Answerer is callable only after the Root is `ready`. It sees accepted
Evidence and the question DAG, but not raw retrieval results, Relations, or
Observations.

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
  -> initial Exact/Search entry
  -> Controller action
  -> Reader
  -> Evidence Checker
  -> Controller action (repeat while incomplete)
  -> Answerer (only when ready)
```

Search results, CandidatePreviews, Relations, and Observations never bypass the
Checker to become answer Evidence.
