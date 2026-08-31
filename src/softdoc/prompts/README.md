# CRAVE-RAG Prompts

This directory is the editable source of truth for all model-facing prompt
text. Each filename includes the prompt version so experiments and training
data can bind to an immutable revision.

| Component | Version | File |
| --- | --- | --- |
| Planner | `planner-v0.20` | `planner_v0_20.txt` |
| Visual Reader | `visual-reader-v0.4` | `visual_reader_v0_4.txt` |
| Evidence Checker | `checker-v1.9` | `checker_v1_9.txt` |
| Reading Controller | `controller-policy-v0.7` | `controller_policy_v0_7.txt` |
| Answerer | `answerer-v0.7` | `answerer_v0_7.txt` |

The Python modules that define prompt versions and render dynamic user input
remain stable compatibility APIs. `softdoc.prompt_registry` is the unified
runtime discovery entrypoint.

Current prompts use Markdown headings to separate stable instruction sections.
At runtime, stable instructions are sent as the system message and validated,
and dynamic component input is sent as a separate user message. This directory
keeps only the active `.txt` version of each component. Superseded versions are
recoverable from Git history and are not duplicated in the working tree.

When changing prompt semantics:

1. create the new versioned file in this directory;
2. delete the superseded working-tree file after updating the runtime loader;
3. update the corresponding version constant and loader reference;
4. update focused prompt tests and evaluation cases;
5. regenerate the prompt manifest and record evaluation results;
6. never patch a frozen prompt for only one dataset example.

Prompt files contain instructions only. Pydantic input/output schemas remain
in their owning modules and must not be expanded as an incidental prompt edit.
