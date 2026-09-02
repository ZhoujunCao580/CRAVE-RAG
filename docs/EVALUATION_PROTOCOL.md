# Evaluation protocol

The canonical current protocol is
`configs/evaluation/mmlongbench_doc_reference_v0_1.json`.

MMLongBench-Doc is used as one complete development/reference benchmark. The
project does not create a development/test/holdout split from it, because its
documents, questions, and Gold evidence have already influenced system design.
Every report must preserve the disclaimer stored in the protocol.

The protocol freezes four boundaries:

1. the exact annotation artifact and source-corpus revision;
2. the pinned official v1 answer evaluator used for comparison with published
   v1 results;
3. Evidence, retrieval, reading, and efficiency metric definitions;
4. default seed, temperature, context, action budget, candidate batch size, and
   timeouts.

The protocol intentionally does not contain experimental results. Before a
run, create an immutable experiment snapshot that binds the protocol to the
exact code commit, Prompt manifest, model revisions, runtime, SoftDoc pipeline,
retrieval configuration, action contract, corpus audit, hardware, software
environment, and price table.

Validate the protocol:

```bash
python scripts/freeze_experiment.py \
  --protocol configs/evaluation/mmlongbench_doc_reference_v0_1.json \
  --validate-only
```

Create a snapshot:

```bash
python scripts/freeze_experiment.py \
  --protocol configs/evaluation/mmlongbench_doc_reference_v0_1.json \
  --bindings /path/to/experiment_bindings.json \
  --output-root /path/to/runs
```

The configuration fingerprint is deterministic for the protocol and bindings.
The experiment ID adds a UTC creation timestamp. Snapshot creation fails when
the target experiment directory already exists, so a previous run cannot be
silently overwritten.

The official v1 evaluator uses its pinned extraction prompt and scorer. Its
normalization handles answer formats differently (integer, float, string, and
list) and must not be replaced by an undocumented project-local cleanup step.
Any corrected-annotation or semantic-judge score must use another protocol ID
and must not be compared numerically with published v1 scores.
