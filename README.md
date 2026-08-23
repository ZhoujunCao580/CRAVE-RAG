# Soft-Structured Document Reading Agent

A research prototype for **evidence-driven, multimodal long-document reading**.

The project treats retrieval as the place where reading starts, not the final context-selection step. A future Controller will search for an entry point, read the underlying document source, navigate pages or typed document relations when useful, and continue until an independent Evidence Checker decides that the accumulated evidence is sufficient.

> 当前目标不是“把Top-k一次性交给模型”，而是让系统围绕问题逐步阅读：找到入口、实际读取、发现证据缺口、选择下一动作，并且只让经过检查的Observation进入Evidence。

## Research thesis

```text
Question
  -> optional conservative plan
  -> find a reading entry
  -> read an Element / Page / Region
  -> produce grounded Observations
  -> check and revise EvidenceMemory
  -> navigate, inspect, search again, or stop
  -> answer only from accepted Evidence
```

The current research hypotheses are:

1. **Soft document structure can support active reading without requiring a graph database.** PDFs are represented as typed Elements, Sections, layout metadata, and explicit Relations while retaining parser provenance.
2. **Navigation signals and answer evidence should be different objects.** CandidatePreview, page position, and Relation handles only suggest where to read; they are not facts that may directly support an answer.
3. **Parser uncertainty should remain visible.** Confirmed Relations are normal navigation handles, candidate Relations are investigation hints, and rejected Relations are hidden from the normal Controller view.
4. **Reading should be question-conditioned and stateful.** The next action depends on the current evidence target, what has already been attempted, and the locally available document structure.
5. **Evidence admission should be independent of navigation.** Readers emit grounded Observations; a Checker promotes, replaces, or removes Evidence through validated deltas; an Answerer only receives accepted Evidence.
6. **Every evidence change should be auditable.** `action_id -> ReadRecord -> Observation -> Checker delta -> EvidenceItem` preserves the path from an environment action to the final answer support.

These are research claims to be tested, not claims that every component is novel or already superior. Human-like document navigation, graph traversal, iterative evidence gathering, and sufficiency checks all have close prior work. The intended contribution is the combination of **parser-neutral multimodal soft structure, typed functional relations with explicit uncertainty, and an evidence firewall around an active reading policy**.

See [Research positioning](docs/RESEARCH_POSITIONING.md) for a careful comparison with Q-Guide, DocNavRAG, MAGE-RAG, G2-Reader, and GraphReader.

## Architecture and current status

```text
PDF
  -> MinerUAdapter
  -> SoftDocPipeline
       Document / Page / Section / Element / Relation
  -> Exact Lookup + BM25 + multilingual-E5 + weighted RRF
  -> SearchSession + CandidatePreview
  -> Reading-state contracts
       ObservationStore / ActionTrace / ExplorationState
       EvidenceCheckResult -> EvidenceMemory
  -> Answerer contract
```

Implemented and tested:

- parser-neutral Pydantic v2 SoftDoc models;
- MinerU adapter, deterministic post-processing passes, provenance, assets, overlays, and validation;
- typed relations including `caption_of`, `footnote_of`, `refers_to`, `continued_on`, section membership, and reading/page order;
- Exact anchor lookup, SearchUnit construction, BM25, multilingual E5 dense retrieval, weighted RRF, resumable SearchSession, and deterministic CandidatePreview;
- conservative Planner v0 contracts;
- Visual Reader input/output contracts;
- ObservationStore, ActionTrace, derived ExplorationState, EvidenceMemory, Checker delta validation/application, cross-store reference validation, and Answerer contracts.

Not yet claimed as complete:

- a trained or production Controller policy;
- production Reader and Evidence Checker model quality;
- deferred planning and Observation Recall;
- SFT/RL training;
- full-dataset end-to-end answer evaluation.

The authoritative design is [PROJECT_GUIDE.md](docs/PROJECT_GUIDE.md). Open risks and planned comparisons are in [TODO.md](docs/TODO.md); major decisions are recorded in [HISTORY.md](docs/HISTORY.md).

## Relation safety boundary

```text
confirmed Relation
  -> Controller may FOLLOW_RELATION

candidate Relation
  -> Controller may inspect its target as a navigation hypothesis
  -> it does not establish that the relation is true

Reader output
  -> Observation only

Checker-accepted Observation
  -> Evidence
```

RL is not expected to repair parser truth. Future policy training will learn whether investigating a noisy hint is worth its cost; deterministic validation and the Observation-to-Evidence boundary remain in force.

## Reproducible public timeline

The current Git history records these development checkpoints:

- 2026-07-21: initial Soft Document Structure implementation;
- 2026-07-30: unified pipeline and Milestone 1 freeze;
- 2026-08-11: hybrid retrieval, SearchSession, and CandidatePreview freezes;
- 2026-08-16: reading foundation freeze;
- 2026-08-22: visual reading and evidence-state contracts.

This timeline is evidence of the repository's development history. It is not, by itself, proof of scientific priority or novelty.

## Environment

Python 3.11 is the primary development version. The code uses `pathlib.Path`, requires no GPU for unit tests, and keeps model clients behind injectable interfaces.

```powershell
Set-Location "D:\claude_code_project\multimodal_pdf_rag"
& "D:\Anaconda\shell\condabin\conda-hook.ps1"
conda activate multimodal_pdf_rag
python -m pip install -e .
python -m pytest -q
```

To recreate the environment:

```powershell
conda env create -f environment.yml
conda activate multimodal_pdf_rag
python -m pip install -e .
```

CUDA-enabled PyTorch should be installed for the target machine separately. Unit tests do not download models or call external APIs.

## Parse a PDF

```powershell
$env:HF_HOME = Join-Path $PWD "data\cache\huggingface"
$env:MODELSCOPE_CACHE = Join-Path $PWD "data\cache\modelscope"
$env:MINERU_TOOLS_CONFIG_JSON = Join-Path $PWD "data\cache\mineru.json"
$env:MINERU_MODEL_SOURCE = "local"

mineru -p INPUT.pdf -o MINERU_OUTPUT -b pipeline -m auto
softdoc parse-mineru MINERU_OUTPUT --output SOFTDOC_OUTPUT
softdoc validate SOFTDOC_OUTPUT
```

`MinerUAdapter` only converts parser output into a raw Document. Deterministic recovery, hierarchy, relation, and validation passes are orchestrated by `SoftDocPipeline`.

## Development data

The local development corpus is intentionally not part of the source release:

```text
data/processed/representative_28/
  pdfs/       28 source PDFs
  softdoc/    current SoftDoc outputs
  retrieval/  retrieval results
  reports/    compact experiment reports
```

Current source tag before the Controller contract update: `softdoc-v0.6-reading-state`.
