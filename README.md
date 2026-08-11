# Soft-Structured Document Reading Agent

面向多模态长PDF问答的研究原型。当前已经完成SoftDoc和检索入口层：

```text
PDF -> MinerU -> SoftDoc -> Exact/BM25/Dense -> weighted RRF
    -> SearchSession -> CandidatePreview
```

Reader、Evidence Checker和Agent循环尚未实现。当前详细架构、数据结构、评测结果和
`INSPECT_REGION`说明统一放在：

- [项目指南](docs/PROJECT_GUIDE.md)
- [精简历史与冻结点](docs/HISTORY.md)

## 环境

```powershell
Set-Location "D:\claude_code_project\multimodal_pdf_rag"
& "D:\Anaconda\shell\condabin\conda-hook.ps1"
conda activate multimodal_pdf_rag
python --version
mineru --version
softdoc --help
```

重建环境：

```powershell
conda env create -f environment.yml
conda activate multimodal_pdf_rag
python -m pip install -e .
```

Python目标版本为3.11+。Windows和Linux路径均通过`pathlib.Path`处理。CUDA版PyTorch
应按目标机器的驱动单独安装，不要直接复制另一台机器的CUDA安装命令。

模型缓存保存在：

```text
data/cache/huggingface/
data/cache/modelscope/
```

## 解析一份PDF

```powershell
$env:HF_HOME = Join-Path $PWD "data\cache\huggingface"
$env:MODELSCOPE_CACHE = Join-Path $PWD "data\cache\modelscope"
$env:MINERU_TOOLS_CONFIG_JSON = Join-Path $PWD "data\cache\mineru.json"
$env:MINERU_MODEL_SOURCE = "local"

mineru -p INPUT.pdf -o MINERU_OUTPUT -b pipeline -m auto
softdoc parse-mineru MINERU_OUTPUT --output SOFTDOC_OUTPUT
softdoc validate SOFTDOC_OUTPUT
```

`MinerUAdapter`只转换解析器输出；完整确定性后处理统一由`SoftDocPipeline`编排。

## 当前开发数据

只保留一个代表性开发集：

```text
data/processed/representative_28/
├── pdfs/       28份原始PDF
├── softdoc/    28份当前SoftDoc
├── retrieval/  当前检索结果
└── reports/    少量关键实验结论
```

## 常用命令

```powershell
python -m pytest -q
python -m pip check

python scripts/build_representative_dense_index.py --device cuda
python scripts/evaluate_representative_retrieval.py --device cuda
```

当前本地冻结点：

```text
softdoc-v0.4.1-retrieval-schema
```
