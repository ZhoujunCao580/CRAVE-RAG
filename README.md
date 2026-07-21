# Soft-Structured Document Reading Agent

Milestone 1 implements a parser-neutral Soft Document Structure for multimodal
PDFs. It contains Pydantic models, a MinerU adapter, deterministic relation
builders, spatial navigation, validation, serialization, debug overlays, and a
small CLI. It intentionally contains no retrieval, embeddings, model calls, or
agent loop.

```powershell
softdoc parse-mineru INPUT_DIR --output OUTPUT_DIR
softdoc validate OUTPUT_DIR
```

The package targets Python 3.11+ and uses only portable `pathlib.Path` paths.

