# TODO

Only unresolved work belongs here. Historical experiments and completed fixes
are recoverable from Git history and are not kept in the working tree.

## 1. Preserve the completed SFT assets

- [ ] Copy the 500+ reviewed Controller SFT dataset, its manifest, exact
  training config, and final adapter from external storage into a durable
  versioned archive. These large artifacts are not committed to Git.
- [ ] Record exact base-model revision, tokenizer revision, adapter SHA-256,
  random seed, package lock, and the scorer summary that produced **56.45%
  Accuracy / 55.32% generalized F1**.
- [ ] Keep Test sealed from further label or checkpoint decisions. Use
  document-isolated Dev or a new Reserve split for selection.

## 2. Expand Controller supervision carefully

- [ ] Add real Teacher corrections at the first wrong Controller decision,
  continue through the actual Reading Environment, and retain a trajectory only
  when the corrected route succeeds.
- [ ] Balance Text/Table/Figure, exact/search/relation/page-context actions,
  single-hop/multi-hop tasks, failure recovery, and efficient successful
  trajectories. Normalize per question so long runs do not dominate.
- [ ] Recheck the two observed SFT regressions on Dev-like states and add
  counterexamples only when the error is attributable to the Controller.
- [ ] Compare checkpoints using JSON validity, action validity, visible-ID
  validity, Teacher route agreement, repeat rate, answer Accuracy/F1, Evidence
  sufficiency, latency, and cost.

## 3. Decide whether Checker SFT is justified

- [ ] Build a new failure funnel after the trained Controller runs. Train the
  Checker only on cases with a correct source and Reader Observation but an
  incorrect Evidence decision.
- [ ] Keep Checker examples and adapters separate from Controller data. Preserve
  atomic Evidence deltas, stable source/page/element identity, target-switch
  rechecks, and Observation Recall behavior.

## 4. Add preference training only after SFT stabilizes

- [ ] Define a versioned Controller preference schema containing one exact
  ControllerInput, one `chosen` legal action, one `rejected` action, prompt
  lineage, source-run identity, and non-model-visible outcome/cost metadata.
- [ ] Add a model-free preference validator: identical state for both actions,
  valid visible IDs, no Gold leakage, no duplicate pair, and no upstream
  retrieval/Reader/Checker corruption.
- [ ] Implement a DPO/ORPO-compatible exporter and trainer configuration.
- [ ] Mine rejected actions from real Student failures and same-state Teacher
  corrections; do not manufacture easy invalid-ID negatives as the main data.
- [ ] Evaluate whether preference training reduces repeated exploration and
  premature STOP without regressing answer quality.

## 5. Remaining system evaluation

- [ ] Evaluate Multimodal Table Reader alignment, cross-page header inheritance,
  unit preservation, and HTML/image disagreement on held-out documents.
- [ ] Measure BM25, text Dense, and visual Dense recall/nDCG separately and as
  the frozen mixed batch; also report reads, VLM calls, latency, and cost.
- [ ] Evaluate question-directed Visual Scan on whole-document, explicit-page,
  and section scopes without restoring the retired MinerU-inventory Coverage
  path.
- [ ] Audit bounded Observation Recall and target-switch Evidence recheck for
  false reuse before expanding to semantic or source recall.
- [ ] Implement deterministic final citation materialization from accepted
  Evidence through Observation and ReadRecord provenance.
- [ ] Run a final untouched Reserve evaluation or the complete official
  MMLongBench-Doc protocol before making public leaderboard claims.
