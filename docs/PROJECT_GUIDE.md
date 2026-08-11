# Soft-Structured Document Reading Agent：项目指南

更新日期：2026-08-11

这份文件是项目当前状态的唯一详细说明。README只保留安装和常用命令，历史决策见
[`HISTORY.md`](HISTORY.md)。

## 1. 当前做到哪里

```text
PDF
  -> MinerU Pipeline（默认；Hybrid仅保留为可选实验后端）
  -> MinerUAdapter：只做解析器字段到原始Document的转换
  -> SoftDocPipeline：结构、关系、浮动内容Section修正和验证
  -> Document / Page / Section / Element / Relation
  -> SearchUnitBuilder
  -> Exact Anchor + BM25 + multilingual-e5-small Dense
  -> Element级weighted RRF
  -> SearchSession
  -> 分批CandidatePreview
```

已完成的是“找到值得开始阅读的位置”。尚未实现：

- `READ_ELEMENT`、`READ_PAGE`、`INSPECT_REGION`、`FOLLOW_RELATION`；
- ReadObservation、Evidence State、Evidence Checker；
- 子问题生成、Reading Agent循环和答案生成。

因此当前检索指标不是最终QA正确率。

## 2. SoftDoc

SoftDoc是解析器无关的中间表示：

```text
Document
├── Pages
│   └── Elements: heading / paragraph / table / figure / chart /
│                 caption / footnote / list / equation / code / algorithm
├── Sections
└── Relations
```

Element保存page、原始与归一化bbox、reading order、可选column、section path、原始文本、
HTML、图片引用、parse status、content availability和provenance。

永久Relation包括：`contains`、`next_page`、`next_in_reading_order`、
`belongs_to_section`、`caption_of`、`footnote_of`、`refers_to`和`continued_on`。
位置邻近关系不批量建边，而由`SpatialNavigator`根据bbox实时计算。

`RelationBuilder`只产生确定性或布局启发式关系，不调用LLM/VLM/embedding。
Relation保存status、confidence、created_by和evidence；candidate不等于confirmed。

Pipeline边界：

```text
MinerUAdapter
  -> CoverageRecoveryPass
  -> StructurePass
  -> RelationPass
  -> FloatingSectionPass
  -> ValidationPass
  -> RuleAuditPass
```

Heading归一化、重复页眉页脚识别、Section构建和FloatingContentSectionResolver均在
Adapter之后执行。Hybrid路径不等于另一套SoftDoc模型。

## 3. 检索

### 3.1 SearchUnit

SearchUnit是检索内部片段，不修改Element。短Element通常对应一个Unit，只有超长文本或
表格才切片；所有part仍映射回同一`element_id`。基础索引不复制Caption、Footnote或
Relation邻居，也不生成LLM摘要和关键词。

Section path作为上下文加入`search_text`。当前真实诊断中，BM25前5候选只有25/1375
（1.82%，涉及15题）属于metadata-only，Dense为0/1375，因此暂不增加field weighting。

### 3.2 两条入口

问题包含`Figure 3`、`表2`、`Page 5`、`第4.1节`等Anchor时，Exact返回独立handle；
歧义和未找到会记录，不让搜索失败。Exact不进入普通RRF列表。

没有可用Anchor时：

```text
BM25 Element ranking
Dense Element ranking
  -> RRF = 1.0/(20+bm25_rank) + 1.25/(20+dense_rank)
  -> 完整、去重的Element候选顺序
```

参数可配置。稳定轮转只保留为历史基线，不再是默认策略。

### 3.3 CandidatePreview

Preview是“是否值得打开”的短卡片，不是摘要、ReadObservation或Evidence。它保存：

```json
{
  "element_id": "...",
  "element_type": "figure",
  "display_label": "Figure 5",
  "page_id": "...",
  "page_number": 12,
  "section_path": ["Experiments"],
  "matched_snippet": "...",
  "snippet_source": "search_unit.search_text",
  "snippet_source_id": "search-unit:...",
  "snippet_char_start": 80,
  "snippet_char_end": 240,
  "preview_source": "dense",
  "match_scope": "content",
  "bm25_rank": 17,
  "dense_rank": 2,
  "rrf_score": 0.0712,
  "content_availability": "visual_only"
}
```

字符offset只属于`SearchUnit.search_text`，不能解释为原始Element坐标。BM25只命中
Section/label而Dense覆盖正文时展示Dense片段，其余情况才按RRF贡献选择展示来源。
无label、无文本的纯视觉对象不会伪造空SearchUnit，未来通过READ_PAGE、Relation或视觉
阅读发现。

### 3.4 SearchSession

SearchSession保存完整候选顺序、Exact结果、候选catalog、已经展示/打开的普通候选、
cursor、exhausted和retrieval trace。默认每次展示5条只是上下文预算，不是最终Top-5；
可以继续第6--10条，也可以回看旧候选。BM25、Dense和RRF rank均被保留。

## 4. 当前实测

开发集：28份PDF、1238页、15985个Element、275道问题，其中213道有Gold证据页。
指标表示“前K个Element中是否至少一个位于Gold页”，不是答案准确率。

| 方法 | Hit@1 | Hit@5 | Hit@20 | Hit@50 |
|---|---:|---:|---:|---:|
| BM25 | 38.03% | 58.69% | 77.46% | 84.51% |
| Dense | 37.56% | 59.62% | 82.63% | 92.49% |
| Exact + 旧轮转 | 43.66% | 64.79% | 82.16% | 92.96% |
| Exact + 当前RRF | **48.36%** | **68.54%** | **84.04%** | **92.96%** |

25道问题含支持的Anchor，其中18道有Gold页，Exact命中17/18。当前参数来自开发集，
完整数据集或holdout只需复验一次，不应继续在这28份文档上调参。

Hybrid困难页严格配对只有3道题，Pipeline和Hybrid的RRF Top-1均为3/3。Hybrid约慢
1.73倍、峰值显存约高3.38倍，生成视觉描述还存在错误风险，因此默认继续使用Pipeline。

## 5. 唯一保留的数据目录

```text
data/processed/representative_28/
├── pdfs/       28份原始PDF
├── softdoc/    28份当前SoftDoc
├── retrieval/  当前检索评测
└── reports/    少量不可从代码直接看出的实验结论
```

原始MinerU中间目录、旧SoftDoc版本、旧embedding cache、视觉验收副本和一次性A/B产物已
删除。需要时从`pdfs/`重新运行当前MinerU和Pipeline。模型缓存仍在`data/cache/`。

## 6. Reader v1与INSPECT_REGION

下一阶段先做Reader，不直接做Agent：

- `READ_ELEMENT(element_id)`：读取完整Element，返回文本/HTML/图片引用、bbox和confirmed
  Relation handles；
- `READ_PAGE(page_id)`：读取页面概览、页面图引用和按reading order排列的Element handles；
- `INSPECT_REGION(page_id, bbox)`：查看页面中的一个矩形区域，例如图表局部、表格单元格、
  公式、图例或小字。它返回裁剪图引用、相交Element和已有OCR文本；Reader v1不因此自动
  调VLM，也不把裁剪结果直接当Evidence；
- `FOLLOW_RELATION(relation_id)`：沿confirmed关系打开另一端；candidate默认关闭。

三层必须分开：

```text
CandidatePreview  可能值得读
ReadObservation   实际读到了什么
Evidence          哪段Observation支持哪个子问题事实
```

Exact目标的读取进入统一Action Trace；`opened_candidate_ids`只管理普通搜索候选。

## 7. 常用命令

```powershell
conda activate multimodal_pdf_rag
python -m pytest -q
python -m pip check

softdoc parse-mineru MINERU_OUTPUT_DIR --output OUTPUT_DIR
softdoc validate OUTPUT_DIR

python scripts/build_representative_dense_index.py --device cuda
python scripts/evaluate_representative_retrieval.py --device cuda
```

完整E5缓存被清理后第一次评测会重新编码；之后可复用
`data/processed/representative_28/retrieval/embedding_cache/`。
