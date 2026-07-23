# Soft-Structured Document Reading Agent

Milestone 1 implements a parser-neutral Soft Document Structure for multimodal
PDFs. It contains Pydantic models, a MinerU adapter, deterministic relation
builders, spatial navigation, validation, serialization, debug overlays, and a
small CLI. It intentionally contains no retrieval, embeddings, model calls, or
agent loop.

## Windows local environment

This repository uses one Conda environment for both MinerU parsing and the
Soft Document Structure code:

```powershell
Set-Location "D:\claude_code_project\multimodal_pdf_rag"
conda activate multimodal_pdf_rag
python --version
mineru --version
softdoc --help
```

If `conda` is not available in a new PowerShell window, initialize the current
window first:

```powershell
& "D:\Anaconda\shell\condabin\conda-hook.ps1"
conda activate multimodal_pdf_rag
```

To recreate the environment later:

```powershell
conda env create -f environment.yml
conda activate multimodal_pdf_rag
python -m pip install -e .
```

The checked-in dependency configuration installs MinerU's `pipeline` extra. On
this Windows RTX 4060 machine, the CUDA build of PyTorch was installed
separately from the official PyTorch CUDA 13.0 index:

```powershell
python -m pip install --no-cache-dir `
  --index-url https://download.pytorch.org/whl/cu130 `
  "torch==2.13.0+cu130" "torchvision==0.28.0+cu130"
```

Do not copy that CUDA command blindly to a future Linux server. Select the
PyTorch build matching that server's driver.

MinerU 3.4.4's pipeline OCR code imports `six` without declaring it in the
published package metadata. The project dependency files explicitly include
`six>=1.16,<2` so a recreated environment does not fail during OCR startup.

## First MinerU run

Keep model caches and MinerU's configuration on the D drive:

```powershell
Set-Location "D:\claude_code_project\multimodal_pdf_rag"
conda activate multimodal_pdf_rag

$env:HF_HOME = Join-Path $PWD "data\cache\huggingface"
$env:MODELSCOPE_CACHE = Join-Path $PWD "data\cache\modelscope"
$env:MINERU_TOOLS_CONFIG_JSON = Join-Path $PWD "data\cache\mineru.json"
New-Item -ItemType Directory -Force "data\cache" | Out-Null

mineru-models-download -s modelscope -m pipeline
```

The model download is needed once and can take several GB. In later PowerShell
sessions, set the same three environment variables again, but do not repeat the
download.

Parse the downloaded MMLongBench-Doc sample:

```powershell
$env:MINERU_MODEL_SOURCE = "local"
mineru `
  -p "data\raw\mmlongbench_doc\documents\2023.acl-long.386.pdf" `
  -o "data\processed\mineru" `
  -b pipeline `
  -m auto
```

Before parsing, close GPU-heavy programs if possible. To force a reliable CPU
run instead, hide the GPU for that command:

```powershell
$env:CUDA_VISIBLE_DEVICES = "-1"
mineru `
  -p "data\raw\mmlongbench_doc\documents\2023.acl-long.386.pdf" `
  -o "data\processed\mineru_cpu" `
  -b pipeline `
  -m auto
Remove-Item Env:CUDA_VISIBLE_DEVICES
```

Inspect MinerU's generated directory before conversion:

```powershell
Get-ChildItem "data\processed\mineru" -Recurse
```

Then pass the directory containing MinerU's JSON output to the project adapter:

```powershell
softdoc parse-mineru MINERU_OUTPUT_DIR --output "data\processed\softdoc"
softdoc validate "data\processed\softdoc"
```

The adapter supports both the original fixture format
(`layout.json` + `*_content_list_v2.json`) and MinerU 3.4.x output
(`*_middle.json` + `*_content_list_v2.json`). For MinerU 3.4.x it uses the
middle JSON for physical page geometry and the v2 content list for semantic
content. If `*_origin.pdf` is present, it also renders page images for the
SoftDoc debug overlays.

## Heading hierarchy

`HeadingHierarchyBuilder` is parser-neutral. It treats parser heading levels as
hints, then applies deterministic, inspectable rules:

- a first-page top title can become `Document.title` without becoming a
  `Section`;
- numeric headings such as `3` and `3.1` become H1 and H2;
- appendix headings such as `A` and `A.1` become H1 and H2;
- common structural headings such as `Abstract` and `References` are H1;
- unnumbered headings below an explicit anchor become child headings.

The milestone deliberately does not define `SemanticRole`, `SectionRole`, or
other role taxonomies. It preserves the source text and records every heading
decision with its original level, normalized level, confidence, and rule.

After `softdoc parse-mineru`, inspect:

```text
debug/document_outline.md
debug/document_outline.json
debug/heading_decisions.json
debug/page_overlays/
```

Page overlays label headings as `TITLE`, `H1`, `H2`, and so on.

## MinerU types and cross-page code

MinerU 3.4 content-list v2 can emit types including `title`, `paragraph`,
`image`, `table`, `chart`, `code`, `algorithm`, `list`,
`equation_interline`, and auxiliary page content. The adapter maps those
parser-specific names to parser-neutral `ElementType` values and preserves the
original name in `Element.metadata["mineru_type"]`.

`code` and `algorithm` remain separate element types. In the installed MinerU
pipeline they share the same code-body layout path, while a code block
containing inline formulas can be switched to the `algorithm` subtype. Because
that distinction can change at a page boundary, continuation rules treat both
as members of one `code_like` compatibility family.

Code/algorithm `continued_on` candidates are limited to adjacent pages and
require page-boundary and bbox-alignment evidence. A caption attached to the
source closes the listing unless it explicitly says `continued` or `续`; a
caption attached to the target is evidence that a multi-page listing ends
there. These relations remain `candidate` and are listed in
`debug/cross_page_relations.json`.

## Checks

```powershell
python -m pip check
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -m pytest -q
```

The package targets Python 3.11+ and the core code uses portable
`pathlib.Path` paths.
