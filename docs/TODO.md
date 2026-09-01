# TODO

This file contains only unresolved work that requires evidence from real
experiments. Implemented contracts and workflows belong in
[`MODEL_CONTRACTS.md`](MODEL_CONTRACTS.md) and
[`ARCHITECTURE.md`](ARCHITECTURE.md), not here.

## 1. PDF and SoftDoc integrity

- [ ] On the complete datasets, distinguish MinerU omissions, wrong element
  types, and wrong crops from information lost by the SoftDoc adapter. Fix
  general failures instead of adding rules for individual documents.
- [ ] Review the 22 upstream Table-type error candidates and measure their
  impact on retrieval, reading, and answers before adding general type
  validation or another parser backend.
- [ ] Audit composite Tables as three distinct cases: valid `<img src>`, only
  an `<image N>` placeholder, and no hint even though the visual region contains
  an image. Do not add a VLM call for every Table in v0.
- [ ] Compare the current pipeline, MinerU Hybrid, and visual recovery on a
  larger sample. Remove recovery/normalization rules with no net benefit.
- [ ] Treat splitting MinerU-aggregated cross-page Tables as a development
  strategy validated only on representative-28. On the full datasets, compare
  retaining MinerU aggregation with splitting by physical page and producing a
  confirmed `continued_on` relation. Do not split when row ownership, repeated
  headers, or cross-page `rowspan` cannot be validated uniquely.

## 2. Planner policy

- [ ] Compare No Planner, Initial Planner, and Deferred Planner under identical
  full-loop conditions.
- [ ] Report answer quality, Evidence completeness, action/model-call count,
  latency, and cost. Implement dynamic plan updates only if Deferred Planning
  provides a stable net gain.

## 3. Retrieval, candidate batches, and budget

- [ ] Compare weighted RRF, fixed-quota mixing such as `3 BM25 + 2 Dense`, and
  BM25-first with Dense used only when needed.
- [ ] On evidence-page retrieval, compare text-only BM25/Dense, retrieval-only
  VLM descriptions, native visual page retrieval, and their combinations.
  Generated descriptions may create CandidatePreviews but must never become
  Observations or Evidence without reading the original visual source.
- [ ] Tune CandidatePreview batch sizes such as `3/5/10`, BM25/Dense quotas,
  RRF parameters (including any visual channel), and the policy for
  next/switch/new search.
- [ ] Report more than Top-k recall: include candidate batches, full reads, VLM
  calls, latency, and cost required to obtain sufficient Evidence.
- [ ] Replace the current action-count placeholder budget with configurable
  resource cost. Paging search results, reading text, reading a whole page, and
  inspecting a visual region must not be treated as equal-cost actions.

## 4. Reader and visual actions

- [ ] On real QA, compare direct Controller use of structured content with a
  dedicated Reader producing Observations. Keep the dedicated Reader only if it
  improves quality or cost.
- [ ] Visual Reader v0 handles Page, Figure, and Chart together. Decide from
  experiments whether Page/Element or Figure/Chart Readers should be separated.
- [ ] Default to single-image reads. Decompose multi-image numeric or factual
  comparisons into separate reads followed by Answerer aggregation. Use joint
  multi-image reading only when the visual relationship itself is inseparable.
- [ ] Add `INSPECT_REGION`/zoom, a structured Table Reader, or visual fallback
  only after a real failure taxonomy justifies them; do not expand the action
  schema for hypothetical cases.
- [ ] When `continued_on` is absent, a candidate Relation is uncertain, or a
  structured read fails, let the Controller choose an adjacent page, candidate
  relation, or visual read. The Reader must not navigate automatically.

## 5. Relation and rule audits

- [ ] Ablate `caption_of`, `footnote_of`, `refers_to`, Section, page/reading
  adjacency, and `continued_on` on real trajectories. Measure Evidence gain,
  searches avoided, and incorrect-navigation cost.
- [ ] Audit the remaining aggressive rules, starting with
  `parser_declared_function_target`, `bounded_nearest_compatible_element`, and
  `profile_forced_sibling_level`. Retain, downgrade to candidate, or remove each
  rule based on final QA utility.
- [ ] Keep confirmed and candidate Relations separate in evaluation. A
  candidate is an investigation opportunity and must not become a fact merely
  because the Controller explored it.

## 6. Observation, Evidence Checker, and Recall

- [ ] Decide how to persist mixed Observations (regression case C23). When one
  Observation contains both a reliable local fact and a claim that the current
  source cannot establish causality, test whether to promote the reliable
  sub-fact while keeping the causal insufficiency in the limitation/current
  gap. Observation Recall is absent in v0, so useful facts can otherwise be
  lost silently.
- [ ] Evaluate Observation Recall when later targets could reuse a previously
  rejected Observation. Compare rereading with recalling from ObservationStore
  and resubmitting to the Checker; implement Recall only if it reduces cost
  without increasing errors.
- [ ] Evaluate duplicate reads caused by one Observation potentially supporting
  multiple questions. v0 still evaluates only the current target per Checker
  invocation.

## 7. Controller policy and training

- [ ] With stronger server models, test whether the Controller rejects previews
  that are topically relevant but cannot fill the current field, and whether it
  reliably distinguishes confirmed from candidate Relations.
- [ ] Build Teacher trajectories that record acceptable actions, net Evidence
  gain, cost, and failure cause per step. Start with prompted Teacher/SFT before
  deciding whether preference training or RL is necessary.
- [ ] Prevent meaningless loops: repeat an action only after Evidence, gap,
  readability, or candidate state changes. `STOP` must never turn incomplete
  into ready.
- [ ] Compare giving the Controller only the current gap with giving it concise
  full Evidence and recent feedback. Freeze training inputs only after testing
  which state fields actually improve action selection.
- [ ] Single-step Controller SFT learns only `ControllerInput -> Action` and may
  miss long-horizon credit assignment, route planning, temporarily low immediate
  information gain, and trajectory-level cost control. Preserve the full
  `ModelPipelineRun`, then compare multi-turn trajectory supervision,
  preference learning, and RL before expanding the training target.

## 8. Answerer and final citations

- [ ] With stronger models, test abstention when Evidence proves only that a
  change occurred but the Root asks why it occurred. The primary safeguard is
  still preventing the Checker from declaring `ready` too early.
- [ ] Implement a deterministic Citation Materializer that expands
  `used_evidence_ids -> observation_ids -> ReadRecord.inputs -> SoftDoc source`
  into Document/Page/Element/Region citations. The Answerer must not invent
  source locations.
- [ ] On complete datasets, evaluate end-to-end answer quality, Evidence
  sufficiency, citation correctness, reading efficiency, and cost.
