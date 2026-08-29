# Archived Prompts

This directory contains superseded CRAVE-RAG prompt versions. They are retained
for experiment reproduction and audit only; the runtime must not load them as
canonical prompts.

| Component | Archived version | File |
| --- | --- | --- |
| Planner | `planner-v0.19` | `planner_v0_19.txt` |
| Planner | `planner-v0.18` | `planner_v0_18.txt` |
| Planner | `planner-v0.17` | `planner_v0_17.txt` |
| Planner | `planner-v0.16` | `planner_v0_16.txt` |
| Planner | `planner-v0.15` | `planner_v0_15.txt` |
| Planner | `planner-v0.14` | `planner_v0_14.txt` |
| Visual Reader | `visual-reader-v0.3` | `visual_reader_v0_3.txt` |
| Visual Reader | `visual-reader-v0.2` | `visual_reader_v0_2.txt` |
| Visual Reader | `visual-reader-v0.1` | `visual_reader_v0_1.txt` |
| Evidence Checker | `checker-v1.8` | `checker_v1_8.txt` |
| Evidence Checker | `checker-v1.7` | `checker_v1_7.txt` |
| Evidence Checker | `checker-v1.6` | `checker_v1_6.txt` |
| Evidence Checker | `checker-v1.5` | `checker_v1_5.txt` |
| Evidence Checker | `checker-v1.4` | `checker_v1_4.txt` |
| Evidence Checker | `checker-v1.3` | `checker_v1_3.txt` |
| Evidence Checker | `checker-v1.2` | `checker_v1_2.txt` |
| Evidence Checker | `checker-v1.1` | `checker_v1_1.txt` |
| Evidence Checker | `checker-v1.0` | `checker_v1_0.txt` |
| Reading Controller | `controller-policy-v0.6` | `controller_policy_v0_6.txt` |
| Reading Controller | `controller-policy-v0.5` | `controller_policy_v0_5.txt` |
| Reading Controller | `controller-policy-v0.4` | `controller_policy_v0_4.txt` |
| Reading Controller | `controller-policy-v0.3` | `controller_policy_v0_3.txt` |
| Answerer | `answerer-v0.6` | `answerer_v0_6.txt` |
| Answerer | `answerer-v0.5` | `answerer_v0_5.txt` |
| Answerer | `answerer-v0.4` | `answerer_v0_4.txt` |
| Answerer | `answerer-v0.3` | `answerer_v0_3.txt` |

When a new prompt version becomes current, move the superseded file into this
directory in the same change that updates the runtime loader and registry. Keep
only the current `.txt` version of each component in the parent `prompts/`
directory.
