# TREVA Research Positioning and Novelty Boundary

Updated: 2026-08-23

This note records what the project is trying to test and what it does **not** claim. It is a public research-positioning snapshot, not a finished novelty or performance claim.

## The proposed research question

Can TREVA's reading Controller use parser-neutral multimodal document structure and uncertain, typed document-functional relations to fill explicit evidence gaps more efficiently than flat search, automatic graph expansion, or a single self-checking perception loop—without allowing parser/navigation errors to contaminate answer Evidence?

The proposed loop is:

```text
find an entry
  -> read the source
  -> produce grounded Observations
  -> independently update EvidenceMemory
  -> choose the next question-conditioned action
  -> stop only when the accepted Evidence set is sufficient
```

The important boundary is not merely “an Agent uses tools.” It is:

```text
navigation lead != Observation != Evidence != Answer
```

## What Q-Guide actually does

[Q-Guide](https://arxiv.org/abs/2608.19739), posted on 2026-08-20, is not only a manga system. It evaluates on the 80-question DocVQA2026 validation split across business reports, comics, engineering drawings, infographics, maps, scientific papers, scientific posters, and slides, plus a Manga109 character-naming benchmark.

Its central contribution is question-conditioned perception recovery over unstructured page images. A single MLLM repeatedly decides which of five recovery tool families to call—text, visual inspection, targeted query, structure, and spatial recovery—accumulates tool observations, self-assesses sufficiency, and submits an answer or Unknown. It intentionally avoids a separate planner/router/critic, and its experiments argue that a compact 2–3-round perception loop can outperform heavier orchestration.

The overlap with this project is real:

- both ask what evidence is still missing;
- both select question-conditioned reading/perception actions;
- both accumulate observations and stop when information appears sufficient.

The boundary is also real:

- Q-Guide starts primarily from unstructured multimodal pages; this project starts from a persistent SoftDoc with typed Elements, Sections, provenance, and Relations;
- Q-Guide uses the same MLLM for tool choice, sufficiency judgment, and final answering; this project separates Reader, Checker, Controller, and Answerer responsibilities;
- Q-Guide's accumulated observations are its evidence state; this project keeps append-only Observations separate from a revisable, Checker-maintained EvidenceMemory;
- Q-Guide does not study confirmed/candidate/rejected document-functional Relations or safe navigation over parser uncertainty.

Therefore “question-guided evidence acquisition” or “human-like active reading” cannot be claimed as unique here. The more specific hypothesis is whether the stronger state and uncertainty boundaries add measurable value in multimodal long documents.

## Comparison with nearby systems

| System | Main representation | Online behavior | Evidence boundary | Main difference from this project |
|---|---|---|---|---|
| [Q-Guide](https://arxiv.org/abs/2608.19739) | Unstructured page images plus recovered content | One MLLM selects perception tools and self-checks | Accumulated tool observations drive submit/continue | No persistent typed SoftDoc or uncertain functional-relation navigation; no independent canonical EvidenceMemory |
| [DocNavRAG](https://arxiv.org/abs/2608.01565) | Document hierarchy plus lateral citation/entity/semantic/community graph | Locate, navigate, expand, fetch with evolving evidence state | Navigation artifacts are separated from fetched content | Primarily paragraph/chunk document navigation; not the same multimodal Element taxonomy or confirmed/candidate functional-edge semantics |
| [MAGE-RAG](https://arxiv.org/abs/2606.15906) | Page and Element evidence graph | Activate, open, search, prune under budget | Builds a query-time evidence subgraph for the reader | Very close on multigranular multimodal graph control; this project instead tests explicit Observation/Evidence admission and uncertain typed relations rather than automatic evidence-subgraph construction |
| [G2-Reader](https://arxiv.org/abs/2601.22055) | Evolving multimodal content graph plus planning DAG | Iterative plan/evidence evolution | Evidence is assembled through dual-graph reasoning | Close on planning and multimodal structure; this project keeps planning optional/conservative and emphasizes environment actions plus a separate Evidence Checker |
| [GraphReader](https://arxiv.org/abs/2406.14550) | Graph of text chunks/atomic facts | Plan, read nodes, inspect neighbors, record insights | Agent gathers graph-node information | Establishes that neighbor-following can work, but does not target parser-produced multimodal document-functional Relations with status uncertainty |

## Defensible project contributions to test

The repository should not claim that any single item below is globally unprecedented. The proposed contribution is the combined system and its ablation:

1. **Parser-neutral SoftDoc IR.** Physical layout, semantic hierarchy, typed multimodal Elements, provenance, assets, and Relations are retained without requiring a graph database.
2. **Uncertainty-aware functional navigation.** `caption_of`, `footnote_of`, `refers_to`, `continued_on`, section, reading-order, and page relations are represented with confirmed/candidate/rejected status. Candidate edges remain hypotheses.
3. **Evidence firewall.** Search results and Relations cannot directly support an answer. A source must be read into a grounded Observation, and the Checker must explicitly admit it into revisable EvidenceMemory.
4. **Question-conditioned reading policy.** Search, candidate paging, source reading, relation following, page navigation, and region inspection are selected around the active evidence target rather than automatically expanded.
5. **Independent cooperative control.** The Controller optimizes exploration; the Checker maintains epistemic state. The design tests whether this separation reduces false-ready decisions and evidence pollution compared with single-agent self-assessment.
6. **Action-to-evidence auditability.** Every accepted fact is traceable through stable IDs to the action, Reader input, Observation, Evidence delta, and document source. This enables action-utility and relation-utility experiments rather than relying only on answer accuracy.

## Handling parser and relation noise

RL alone is not the safety mechanism. The proposed layered treatment is:

1. deterministic validation catches broken IDs, missing assets, invalid bbox values, and inconsistent references;
2. parser recovery preserves degraded content and provenance rather than silently deleting it;
3. Relation status controls how a link is exposed: confirmed handle, candidate navigation hint, or hidden rejected edge;
4. a Controller may inspect a noisy hint, but the hint itself is never Evidence;
5. Reader limitations remain attached to the read attempt;
6. Checker admission, contradiction handling, and selective re-reading prevent known uncertainty from silently entering final Evidence;
7. future policy training optimizes when an uncertain lead is worth investigating under a budget.

The key experiments should report both benefit and harm: useful-evidence rate after relation navigation, false navigation rate, extra actions/tokens, Evidence pollution, false-ready rate, and final answer quality. Relation types must be ablated separately; it is acceptable if some relations prove unhelpful and are removed from the learned action space.

## Public repository timeline

The current Git history verifies development from 2026-07-21 onward. The initial SoftDoc and typed-relation work predates DocNavRAG's 2026-08-03 posting; retrieval sessions and the reading foundation were frozen before Q-Guide's 2026-08-20 posting; the current full evidence-state contract was frozen on 2026-08-22. The repository history is useful evidence of independent development, but it does not establish that an idea was globally first. Any earlier June activity should only be stated publicly after it is supported by a verifiable commit, release, archive, or other timestamp.

## Claims deliberately avoided

- “the first system to imitate human reading”;
- “the first Agent to identify missing evidence”;
- “the first relation-following document Agent”;
- “all SoftDoc Relations improve QA”;
- “separate Controller and Checker is always better”;
- “the current local prototype already outperforms Q-Guide, DocNavRAG, MAGE-RAG, or G2-Reader.”

Those statements require experiments or are contradicted by nearby work. The current repository provides the representation, retrieval, and state contracts needed to test a narrower and more defensible hypothesis.
