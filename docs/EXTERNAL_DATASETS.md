# External datasets

CRAVE-RAG separates dataset ingestion from model execution:

```text
native benchmark files
        |
        v
dataset-specific adapter
        |
        v
ExternalDatasetManifest --audit--> verified sources + verified SoftDocs
        |
        v
Gold-free batch JSONL --> scripts/run_model_batch.py
```

`run-model` executes one question against one serialized SoftDoc. It does not
discover which benchmark question belongs to which PDF, find the corresponding
SoftDoc, or verify external assets. The manifest layer performs that mapping
once and makes it auditable.

## MMLongBench-Doc

Build a manifest from the native question file, source PDFs, and generated
SoftDocs:

```bash
softdoc datasets build-mmlongbench \
  --questions /workspace/data/mmlongbench/questions.json \
  --documents /workspace/data/mmlongbench/documents \
  --softdocs /workspace/data/mmlongbench/softdocs \
  --path-root /workspace \
  --output /workspace/data/mmlongbench/dataset_manifest.json
```

For a small pilot, repeat `--question-index` with the zero-based indices to
include. `--hash-sources` adds PDF SHA-256 values when source drift detection is
worth the additional one-time I/O.

Audit before any paid model call:

```bash
softdoc datasets audit \
  /workspace/data/mmlongbench/dataset_manifest.json \
  --output /workspace/data/mmlongbench/audit_report.json
```

The command exits nonzero for a missing source, missing or invalid SoftDoc,
missing visual asset, source/SoftDoc page-count mismatch, source identity
mismatch, unknown question-document mapping, or out-of-range evidence page.
It never treats an absent external corpus as a successful audit.

After the audit passes, export model-visible cases:

```bash
softdoc datasets export-batch \
  /workspace/data/mmlongbench/dataset_manifest.json \
  --output /workspace/data/mmlongbench/batch_cases.jsonl
```

The exported JSONL deliberately contains only question IDs, question text, and
SoftDoc locations. Gold answers, evidence pages, and benchmark metadata remain
outside the model input.

## DocVQA2026 and M109NC

The auditor is dataset-independent. A future adapter only has to normalize the
native release into the same `ExternalDatasetManifest`:

- DocVQA2026 documents are multi-page image lists with nested question IDs and
  question text. Its adapter should materialize each document's images, point
  to the matching SoftDoc, and emit one manifest question per nested question.
- M109NC depends on the licensed Manga109-v2026 books plus the Q-Guide task
  annotations. Its adapter should map each book/page collection and question to
  the matching SoftDoc without copying licensed images into Git.

Do not guess an unpublished or unavailable annotation schema. Once the actual
files are installed, add a thin adapter and adapter-specific fixture test; the
manifest audit and batch exporter do not change.

## Source-versus-Gold disagreements

File integrity is necessary but does not prove that a benchmark answer is
correct. Record source-versus-Gold disagreements during Teacher review and keep
document-grounded Evidence out of a mislabeled SFT target. Gold answer auditing
is intentionally separate from the model-visible batch export.
