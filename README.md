# CRAVE-RAG: Controller-guided Reading and Action Via Evidence Gaps

> Start from a clue, follow useful structure, and stop only when the evidence is sufficient.

CRAVE-RAG is a research prototype for question answering over long, visually rich PDFs. Its core is an **agentic reading Controller** that keeps working from the current evidence gap instead of treating retrieved items as a fixed final context. Retrieval only provides a place to begin reading.

The Controller maintains a current question and evidence gap, then chooses how to continue reading. Each reading move produces source-linked Observations. An independent Evidence Checker decides which observations become Evidence, whether the current gap has been closed, and what remains unresolved. The updated gap is returned to the Controller, which chooses another move. The loop ends only when the Evidence is sufficient.

![CRAVE-RAG overview](docs/assets/crave-rag-overview.svg)

## Core Design

This loop gives the system four important properties:

- **Active reading:** it can keep reading from a useful clue instead of repeatedly rebuilding a fixed final context.
- **Structured navigation:** page structure and document relations become explicit reading opportunities rather than automatically expanded answer context.
- **Independent evidence control:** the Controller explores, while the Checker separately decides whether an Observation deserves to enter Evidence.
- **Traceable decisions:** every reading action, Observation, accepted Evidence item, and final citation remains connected to its source.

## Soft Document Structure

PDFs are represented through a parser-neutral intermediate structure:

```text
Document
|-- Pages
|   `-- Heading / Paragraph / Table / Figure / Chart / Caption / ...
|-- Sections
`-- Relations
```

The structure preserves page layout, element bounding boxes, reading order, visual assets, semantic hierarchy, provenance, and confirmed or candidate document relations. Relations guide navigation but never become Evidence by themselves.

## Current Status

The repository currently provides the SoftDoc representation, MinerU adaptation pipeline, deterministic document relations, spatial navigation, exact/sparse/dense retrieval, candidate previews, search sessions, and frozen contracts for planning, reading, evidence checking, and answering.

The next research stage is the Controller and end-to-end evaluation of whether evidence-gap-driven reading improves answer quality and reading efficiency.

## Quick Start

```bash
conda env create -f environment.yml
conda activate multimodal_pdf_rag
python -m pip install -e .
python -m pytest -q
```

```bash
softdoc parse-mineru <MINERU_OUTPUT_DIR> --output <SOFTDOC_OUTPUT_DIR>
softdoc validate <SOFTDOC_OUTPUT_DIR>
```

See [Project Guide](docs/PROJECT_GUIDE.md), [Architecture](docs/ARCHITECTURE.md), and [TODO](docs/TODO.md) for the current implementation boundary and open research questions.
