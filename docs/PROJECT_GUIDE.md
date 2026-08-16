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
  -> Initial Planner：Question -> SubQuestion DAG（本地Ollama + Qwen3 4B）
```

已完成的是“找到值得开始阅读的位置”。尚未实现：

- `READ_ELEMENT`、`READ_PAGE`、`INSPECT_REGION`、`FOLLOW_RELATION`；
- ReadObservation、Evidence State、Evidence Checker；
- 真实LLM生成SubQuestion、动态REFINE_PLAN、Reading Agent循环和答案生成。

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

## 7. Initial Planner v0

Planner v0只定义“原问题需要回答哪些独立信息需求”：

```text
Question
  -> one or more PlannedSubQuestion
  -> logical depends_on DAG
```

当前已实现严格Pydantic schema、question-only Prompt、可注入`PlannerBackend`、DAG与显式
Anchor校验，以及Mock测试。首个真实backend为本地`OllamaPlannerBackend`，默认模型是
`qwen3:4b-instruct-2507-q4_K_M`。它通过`http://localhost:11434/api/chat`请求JSON Schema
约束输出；模型只能返回`PlannerDraft`，backend名称、模型名称、Prompt版本和重试记录由程序
写入`PlannerTrace`，不接受模型伪造。

边界：

- 简单事实题允许只有一个SubQuestion；
- 原问题是隐式Root；默认最多6个SubQuestion，最大深度4（Root计为第1层）；
- `depends_on`只表示回答当前问题所需的逻辑前置，不表示普通执行顺序；
- 显式Anchor必须从原问题逐字保留，不能预测未出现的Figure/Table/Page；
- 显式Anchor必须是Exact Lookup支持的完整引用（如`Figure 3`），不能拆成`Figure`和`3`；
- Prompt要求数字约束在SubQuestion文本中原样保留；Pydantic不冒充语义裁判去猜哪些数字相关；
- 对Schema、Anchor、ID、依赖或DAG等确定性验证失败，最多自动纠正一次并记录warning；
- 子问题只表达需要从文档取得的新证据；比较、求差和选择由隐式Root基于已取得事实完成；
- 原问题中的实体、指标、时间/条件、输出要求和解释要求必须保持可回答，不能只覆盖部分问法；
- `planner-v0.4`进一步禁止计算/排序/格式化与定位脚手架节点；命名章节、序数位置和页码范围
  保留在问题文本中，但不写入只服务于Exact Lookup的`explicit_anchors`；
- `planner-v0.5`补充count/list、序数、top-N、阈值和视觉条件的忠实性示例，并明确真正的
  未知实体桥接应形成依赖，而同一局部视觉筛选不形成多步流水线；
- `planner-v0.6`补充排除集合、yes/no阈值、最高/最低比较集合、总体样本范围和成对年份
  示例，避免子问题只保留焦点对象却缺少足以完成Root判断的对照证据；
- `planner-v0.7`最终明确yes/no数值阈值必须出现在子问题中、焦点对象与已知比较集合保持
  并行，以及相似性计数只收集参照与候选证据而不新增比较/计数节点；
- `planner-v0.8`和`planner-v0.9`根据275道真实问题审计收紧节点职责边界：兄弟节点不得
  共享或串入彼此的证据需求，比例/差值读取真实操作数，共享年份和范围必须在
  每个相关节点保留，同一局部来源的多值优先合并，计算与补集仍由隐式Root完成；
- `planner-v0.10`删除了重复表达`text`内容、且尚无下游消费者的`answer_requirements`；每个
  节点现在只以`text`表达证据需求，旧字段会被严格Schema拒绝并进入一次可追踪纠错重试；
- `planner-v0.11`删除由开发集错误反向写入的题型案例，只保留隐式Root、最小证据需求、真实
  依赖、语义范围保持等通用原则与两个虚构示例；显式Anchor由独立Exact Lookup负责；
- `planner-v0.12`只增加通用保守边界：无法可靠判断是否应拆分时，将完整原问题保留为一个
  SubQuestion；该单节点计划是合法起点，不代表Planner失败；
- `planner-v0.13`以“是否重复处理已请求的事实”取代差值、比例、总计等题型枚举，并明确同一
  局部证据可直接回答的比较/排序仍可作为一个节点；LLM schema也删除了永远为空的
  `explicit_anchors`；Anchor解析结果由Exact Lookup独立保存，不写回InitialPlan；
- `planner-v0.14`将`depends_on`定义为：前一个答案是否是实例化后一个证据需求所必需；缺失的
  可以是实体、条件、类别、数值、时期或搜索短语，只要后一个问题能从Root独立搜索就保持并行；
- Planner采用Conservative + Deferred边界：初始只拆明显需求，不确定时整体阅读，后续根据
  实际Evidence gap通过`UPDATE_PLAN`细化；
- `planner-v0.14`的Prompt、Schema、默认6节点与4层深度正式冻结；未来价值判断见
  [`TODO.md`](TODO.md)中的No Planner / Initial Planner / Deferred Planner同条件对照；
- Planner不选择SEARCH、READ、FOLLOW_RELATION、INSPECT_REGION等动作；
- Planner不检索、不阅读、不判断Evidence充分性，也不提前回答；
- 当前没有动态`REFINE_PLAN`。

动态重规划将在ReadObservation、Evidence State和Evidence Checker完成后实现；它只能由明确的
missing、ambiguous或conflicting evidence gap触发，不因一次搜索失败而随意改写问题。

本地28份文档对应的275道真实问题曾使用`planner-v0.9`完整重跑，全部通过Planner结构
验证，其中7道需要第二次验证纠错。逐题计划级审计结果为231道正确、39道可接受但有轻微
表达/最小性问题、5道错误。错误集中于含混gap、比较集合缺失、约束丢失、比较对象幻觉和
财务比率漏操作数；这说明Prompt显著改善了整体规划，但4B模型仍不能视为语义验证器。
详细逐题结果保存在`.runlogs/planner_v09_manual_audit.md`和`.jsonl`；它们是旧Schema下的本地
可再生审计产物，不等同于`planner-v0.10`结果、检索命中率或最终回答准确率。

删除重复字段后，`planner-v0.10`只重跑了旧审计中的44道高风险题：44/44结构合法，人工
计划级审计为32道正确、3道可接受但有小问题、9道错误。旧版5道错误中3道完全修复、1道
改善但冗余、1道仍错误；新增错误主要来自4B模型把最终计算结果当作证据需求，以及对病句中
主客体方向的误读。结果见`.runlogs/planner_v10_highrisk_audit.md`；本轮不据此继续追加Prompt
补丁，也不能将这一高风险子集的比例外推为全数据集准确率。

使用通用`planner-v0.11`、相同JSON Schema、温度和`think=false`设置，对同一44题比较
`qwen3:4b-instruct-2507-q4_K_M`与`qwen3:8b`。4B为21正确、15可接受、7语义错误和1个仅由
原问题双空格被正规化造成的结构错误；8B为22正确、12可接受、10语义错误。平均耗时分别为
6.247秒与6.280秒。该结果只说明8B在这个开发高风险回归集上没有稳定优于4B，不是全数据集
泛化准确率。逐题错误与生成计划见`.runlogs/planner_v11_4b_vs_8b_audit.md`。
当前Prompt、实验边界以及“InitialPlan只是可修订假设、Root始终是权威目标”的Controller
契约见[`PLANNER_V0_SUMMARY.md`](PLANNER_V0_SUMMARY.md)。v0.11的4B/8B结果未被后续Prompt覆盖。

## 8. Visual Reader 前的资源审计

在进入视觉阅读实现前，对 `representative_28` 的当前 SoftDoc 做了资源完整性审计：

- 28 份文档、1238 页；
- 1141 个 Figure、160 个 Chart、403 个 Table、39 个 Equation，共 1743 个需要视觉读取的 Element；
- 1743/1743 均有合法 bbox、可读取的视觉资源和可读取的页面图；
- 1741 个使用独立 Element asset，另外 2 个恢复型 Figure 明确使用 `page image + bbox`，读取时必须按 bbox 裁剪；
- 归一化 bbox 中 3 个对象因原始坐标比页面边界多 1 像素而被合法 clamp 到 1.0，不是资源错配；
- 独立 asset 与页面 bbox crop 的像素相关性中位数为 0.9809；唯一较大的低相关样本经人工并排检查，内容与目标区域一致，差异来自缩放/压缩和细线表格。

该审计证明的是“当前 SoftDoc 指向的视觉文件存在，并与所存 bbox 对应”，不能证明 MinerU 没有漏掉任何视觉对象，也不能证明每个语义边界都达到人工标注级精度。Visual Reader 应同时支持两种输入句柄：独立 Element asset，以及 `page image + bbox` 的延迟裁剪。

本轮清理删除了 8 个可由脚本重新生成的 Table review/example 目录，保留原始 PDF、当前 SoftDoc、检索结果、少量结论报告以及 TableView/审计代码。

## 9. 常用命令

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
