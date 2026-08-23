# TREVA：项目指南

更新日期：2026-08-23

TREVA是 **Typed-Relation Evidence-guided Visual Agent** 的缩写；SoftDoc仍是解析器无关的文档中间表示，`softdoc`仍是稳定的Python包名和CLI名称。总流程图见[`ARCHITECTURE.md`](ARCHITECTURE.md)。

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
  -> Visual Reader v0接口与Reading State v0数据模型
```

已完成的是“找到值得开始阅读的位置”。尚未实现：

- `READ_ELEMENT`、`READ_PAGE`、`INSPECT_REGION`、`FOLLOW_RELATION`的正式环境执行闭环；
- Evidence Checker实现与`EvidenceCheckResult`运行时更新；
- 真实LLM生成SubQuestion、动态REFINE_PLAN、Reading Agent循环和答案生成。

`ObservationStore`、`EvidenceMemory`、`ActionTrace`、`ExplorationState`以及Reader/Checker边界已经冻结为本指南第9—10节的v0契约。代码已统一到`action_id + input_id + observation_id + evidence_id`边界；Reader limitation只保留自由文本描述与受影响的动作内输入。

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
- 1139 个 Figure、160 个 Chart、403 个 Table、39 个 Equation，共 1741 个需要视觉读取的 Element；
- 1741/1741 均有合法 bbox、可读取的独立视觉资源和可读取的页面图；
- 曾有 2 个 Figure 由 `ElementNormalizer` 从相邻渐进式幻灯片复制 bbox 合成；消融发现该条件过窄、会级联使用合成结果且 bbox 包含标题，因此已删除。对应视觉内容仍存在于页面图中，但不再伪装成高可靠 Figure Element；
- 归一化 bbox 中 3 个对象因原始坐标比页面边界多 1 像素而被合法 clamp 到 1.0，不是资源错配；
- 独立 asset 与页面 bbox crop 的像素相关性中位数为 0.9809；唯一较大的低相关样本经人工并排检查，内容与目标区域一致，差异来自缩放/压缩和细线表格。

该审计证明的是“当前 SoftDoc 指向的视觉文件存在，并与所存 bbox 对应”，不能证明 MinerU 没有漏掉任何视觉对象，也不能证明每个语义边界都达到人工标注级精度。Visual Reader 默认读取 Element asset；当上游漏检或需要布局上下文时，可显式读取页面图，但 SoftDoc 不再用跨页复制 bbox 的方式合成视觉 Element。

本轮清理删除了 8 个可由脚本重新生成的 Table review/example 目录，保留原始 PDF、当前 SoftDoc、检索结果、少量结论报告以及 TableView/审计代码。

## 9. Visual Reader v0 边界

Visual Reader v0只负责把本次提供的一个或多个视觉输入转换为可追溯的可靠Observation。
`problem`用于限定当前需要观察的内容，不代表Reader需要生成问答式答案。Reader不输出
`answer`、`conclusion`或`supported_by_observation_ids`；Evidence层以后负责判断哪些Observation
可以提升为EvidenceItem，并由EvidenceItem保存其`observation_ids`。

Reader模型只返回局部内容；全局`action_id/observation_id`由Environment在校验后补入。输出Schema冻结为：

```json
{
  "observations": [
    {
      "text": "The tallest visible bar is approximately 79.",
      "sources": [
        {
          "input_id": "I1",
          "bbox": null
        }
      ]
    }
  ],
  "limitations": [
    {
      "description": "The legend in I1 is too small to identify the corresponding method.",
      "input_ids": ["I1"]
    }
  ]
}
```

约束如下：

- `input_id`是一次Action内部的局部别名（`I1/I2/...`），不是全局ID；稳定来源仍由Environment保存的`source_id/visual_asset_id`确定；
- 单图事实Observation通常引用一个`input_id`；真正依赖多图视觉关系的Observation可以引用多个；
- Observation的`sources`只能引用本轮实际提供的输入，`bbox`是相对该输入的可选归一化区域；Table读取还可以引用稳定`cell_id`；
- 不要求每个输入图片都出现在Observation中，未使用且不影响判断的图片不应制造Observation；
- 若只能确定部分内容，保留可靠Observation，并用`limitations`说明没有可靠读出的内容；limitation不评价Observation是否对问题有用；
- limitation v0只保存自然语言`description`及受影响的`input_ids`，不固定错误`code`枚举；
- Reader不判断Evidence是否充分，不生成文档级最终回答，不决定导航、Relation或后续工具动作。

冻结的System Prompt为：

```text
You are a visual reader for long PDF documents.

You receive:
- one local reading problem;
- one or more document images;
- metadata identifying each supplied input as I1, I2, and so on.

The images are provided in the same order as inputs.

Your responsibility is to inspect the supplied images and record the concrete,
reliable visual observations needed for the local reading problem.

The local problem tells you what to inspect. It does not ask you to produce an
answer field or a separate final response. Return observations and limitations
only, even when the observations directly resolve the local problem. Express
relevant visible facts only as Observation.text; do not restate them as an answer.

Rules:

1. Use only information visibly supported by the supplied images.

2. If multiple images are supplied, use them jointly when the local problem
   requires a visual relationship across those images. Do not ignore any supplied
   image that is relevant to the local problem. Do not request additional images
   yourself.

3. Record each concrete fact as an Observation. Keep independently useful facts
   separate. Use the smallest non-duplicative set of Observations needed for the
   local problem, and never repeat an equivalent Observation.

4. Every Observation must list the input_ids that support it through sources.
   Its source regions must refer only to those inputs. If an Observation compares, links, aligns, or
   claims continuity or correspondence between multiple images, express that
   relationship as one Observation whose sources include every input needed
   to establish it, with at least one source entry for each input. Do not
   split one multi-image relationship into separate single-image Observations.
   Do not require an irrelevant supplied image to appear in an Observation.

5. A bbox is optional. If used, it must be an approximate normalized box
   [x1, y1, x2, y2] relative to the corresponding supplied image, with values
   between 0 and 1. Use null when reliable localization is not possible.

6. If information needed by the local problem is too small, blurred, cropped,
   occluded, missing, or otherwise unreadable, do not guess. Preserve any facts
   that can still be observed reliably and describe the remaining uncertainty
   in limitations.

   Every limitation must contain a concise description and the input_ids whose
   unreadable or missing content caused it. A limitation describes what could
   not be read reliably; it does not assess Evidence relevance or sufficiency.

7. Do not invent unreadable text, hidden content, missing values, or details
   from images that were not supplied.

8. Do not output answer, conclusion, supported_by_observation_ids, Evidence
   sufficiency, navigation advice, Relation status, or Controller actions.

9. The top-level JSON object must contain exactly these keys:
   observations and limitations. Do not invent action, observation, Evidence,
   or global source IDs; the program assigns them after validation.

10. Return valid JSON only, without Markdown or additional commentary.

11. Return at most 16 Observations.
```

对应输出模板为：

模板会按本次输入动态列出局部别名：单图请求示例只含`I1`，多图请求示例会在同一个Observation中列出`I1`、`I2`等及各自的source region，避免固定单图示例误导小模型。实际输出仍只引用真正支持该Observation的输入。

```json
{
  "observations": [
    {
      "text": "<one concrete fact visibly supported by the supplied images>",
      "sources": [
        {
          "input_id": "I1",
          "bbox": null
        }
      ]
    }
  ],
  "limitations": []
}
```

如果无法完整判断，不返回答案，而是保留安全的局部观察并报告限制：

```json
{
  "observations": [],
  "limitations": [
    {
      "description": "The legend labels needed by the local problem are too small to read reliably.",
      "input_ids": ["I1"]
    }
  ]
}
```

## 10. Reading State v0

本节是2026-08-22重新冻结的唯一Reading State v0契约。本阶段不实现Controller、Evidence Checker模型或Observation Recall，但已经实现Checker输入/输出数据边界、问题DAG运行状态和delta的安全应用。Reading Session、ObservationStore和EvidenceMemory都以Root Question为范围；Controller与Checker每轮只聚焦`EvidenceMemory.current_target`指向的一个问题。不存在第二个`current_question_id`状态源，也不得为每个子问题建立平行EvidenceMemory。

### 事实源与派生视图

| 对象 | 性质 | 职责 |
|---|---|---|
| `ObservationStore / ReadRecord` | canonical | 永久记录Reader实际收到什么、读出什么及局限 |
| `SearchSession` | canonical | 保存检索候选、cursor、shown/opened状态 |
| `ActionTrace` | canonical | 按顺序记录环境动作及结果 |
| `EvidenceMemory` | canonical | 未来由Checker维护Evidence与当前缺口 |
| `ExplorationState` | derived | 从上述日志压缩出的Controller工作快照，不是第四份历史数据库 |
| `EvidenceCheckResult` | transient result | 一次Checker调用的受验证返回值；不建立独立canonical store |

### ObservationStore

一次Controller动作使用唯一`action_id`贯穿ActionTrace、ReadRecord、本轮Observation和可选Checker更新。v0规定一次Read Action只触发一次Reader调用；视觉fallback、补读另一页或换表示必须由Controller发起新Action，因此不再需要独立`read_request_id`。如果未来确实允许一个Action自动产生多个内部Reader子调用，再在该Action内部增加局部substep，不提前恢复第二套全局ID。

`ReadRecord.inputs`保存结构化的完整Reader输入，因此单图、多图、整页、Region和TableView读取都可复现。`input_id`只是该Action内部的局部别名（如`I1`），`source_id`和`visual_asset_id`才是系统稳定handle。模型不生成全局Observation ID；Environment校验模型输出后确定性分配。`subquestion_id`允许为`null`，所以No-Planner与Deferred Planner同样可以记录读取。

```json
{
  "reading_session_id": "reading:1",
  "root_question_id": "root:1",
  "read_records": [
    {
      "action_id": "action:1",
      "reader_kind": "visual",
      "document_id": "doc:1",
      "subquestion_id": "Q1",
      "local_problem": "Which method has the tallest bar?",
      "inputs": [
        {
          "input_id": "I1",
          "source_id": "element:figure:1",
          "source_type": "element",
          "representation": "element_visual",
          "page_id": "page:1",
          "element_id": "element:figure:1",
          "visual_asset_id": "visual:1",
          "bbox": null
        }
      ],
      "observation_ids": ["obs:abc:00"],
      "limitations": [
        {
          "description": "The legend in I1 is too small to identify the corresponding method.",
          "input_ids": ["I1"]
        }
      ]
    }
  ],
  "observations": [
    {
      "observation_id": "obs:abc:00",
      "action_id": "action:1",
      "text": "The tallest visible bar is approximately 79.",
      "sources": [
        {
          "input_id": "I1",
          "bbox": [0.5, 0.3, 0.7, 0.8]
        }
      ]
    }
  ]
}
```

Table Reader读取整个`TableView`时，Observation source可以在`input_id`之外进一步引用稳定`cell_id`。因此表格事实能够定位到行列单元格，而不是只能引用整个Table Element。limitation只记录Reader没有可靠读出的部分，不判断Observation是否相关、正确或足以成为Evidence；后者属于Checker。

### EvidenceCheckResult

Checker不是Controller可选动作。一次Read Action产生非空Observation后，Environment自动调用Checker；完全没有Observation时通常跳过Checker，把失败或limitation留在ReadRecord和ActionTrace中。Checker读取Root Question，并从完整EvidenceMemory的`current_target.question_id`和`questions`中确定本轮目标；不再另传一份可能冲突的current-question对象。它另外读取本轮Observation及同一Action的limitations，不读取完整探索历史、候选排名或Relation图。

`EvidenceCheckResult`沿用触发它的`action_id`，不新增`evidence_check_id`。它是受验证的临时返回值，不是第四个canonical store。Checker不重写完整EvidenceMemory，只返回本轮delta、`current_target_status`、`root_status`、可选`remaining_gap_description`和简短assessment；程序在当前Memory的副本上应用delta、更新当前问题状态并选择下一目标，验证完整结果后才一次性提交。任一操作或引用无效时，canonical EvidenceMemory完全不变。`remaining_gap_description`只描述**仍未完成的当前target**：`current_target_status=incomplete`时必填，`satisfied`时必须为`null`。Checker不预测下一问题，也不替下一问题写gap。

#### Checker输入contract

delta只减少Checker的**输出**，不会减少它的判断上下文。每轮Checker与Controller都可以读取当前唯一、完整的canonical `EvidenceMemory`；Checker另外只接收本Action的新Observation与limitations：

```json
{
  "action_id": "action:1",
  "root_question": {
    "question_id": "root:1",
    "text": "How did revenue change from 2022 to 2023?"
  },
  "evidence_memory": {
    "reading_session_id": "reading:1",
    "root_question_id": "root:1",
    "root_status": "incomplete",
    "questions": [
      {
        "question_id": "Q1",
        "text": "What was the revenue in 2022?",
        "depends_on": [],
        "status": "satisfied"
      },
      {
        "question_id": "Q2",
        "text": "What was the revenue in 2023?",
        "depends_on": [],
        "status": "incomplete"
      }
    ],
    "evidence": [
      {
        "evidence_id": "evidence:old",
        "statement": "Revenue in 2022 was 10 million.",
        "observation_ids": ["obs:old"],
        "supports_question_ids": ["Q1"]
      }
    ],
    "current_target": {
      "question_id": "Q2",
      "gap_description": "The 2023 revenue is unknown."
    }
  },
  "observations": [
    {
      "observation_id": "obs:abc:00",
      "action_id": "action:1",
      "text": "Revenue in 2023 was 12 million.",
      "sources": [{"input_id": "I1", "bbox": null}]
    }
  ],
  "limitations": []
}
```

Checker不读取完整ObservationStore、ExplorationState、SearchSession、候选排序或Relation图。旧Evidence已经通过其`observation_ids`保持可追溯；只有需要复核原Observation时，才由未来的Observation Recall显式取回，而不是每轮塞入全部历史。

#### Checker输出contract

```json
{
  "action_id": "action:1",
  "observation_assessments": [
    {
      "observation_id": "obs:abc:00",
      "used_for_evidence": true,
      "assessment": "The Observation supplies the missing 2023 value."
    }
  ],
  "evidence_updates": {
    "add": [
      {
        "statement": "Revenue in 2023 was 12 million.",
        "observation_ids": ["obs:abc:00"],
        "supports_question_ids": ["Q2"]
      }
    ],
    "replace": [],
    "remove": []
  },
  "current_target_status": "satisfied",
  "root_status": "ready",
  "remaining_gap_description": null
}
```

`add`不含`evidence_id`，由程序按`action_id + add序号`确定性分配；`replace/remove`必须引用当前Memory中真实存在的`evidence_id`，同一ID不能同时替换和删除。每个新增或替换Evidence必须声明`supports_question_ids`。程序固定执行`remove -> replace -> add -> current target状态更新 -> 下一目标选择`，但只在最终完整EvidenceMemory通过Pydantic状态约束和跨存储引用验证后提交。

`used_for_evidence`表示该Observation是否被最终EvidenceMemory中的任一EvidenceItem引用；程序在delta应用后交叉验证，避免assessment与实际结果自相矛盾。它不限于“首次提升”：Observation也可以补强或修订已有Evidence。每条assessment只说明该Observation的证据作用及原因，不输出隐藏推理，不规定`gap_change/reason_code`枚举，也不命令Controller执行下一工具。

`current_target_status`由未来的Checker模型依据`current_target`、完整EvidenceMemory、本轮Observations和limitations进行语义判断；当前阶段只实现严格数据契约，不假装用确定性程序判断问题是否已经回答。`apply_evidence_check_result()`只把它写回当前目标对应的`QuestionState.status`，不会修改其他问题。若目标就是Root，程序强制该状态与`root_status`一致。

下一轮Controller不是只看delta。它读取提交后的**完整EvidenceMemory**（包括全部已注册问题和唯一`current_target`）和派生的`ExplorationState`；紧接Checker调用的下一步还可以直接读取本轮`EvidenceCheckResult.observation_assessments`，但不把它们复制成长期summary。delta主要用于安全更新、审计“本轮改了什么”和减少Checker输出长度。下一轮Checker同样从提交后的完整EvidenceMemory开始，因此多条Evidence的联合判断不会丢失。

### EvidenceMemory

`EvidenceMemory`是一个Root Question范围内、跨初始和延迟注册子问题共享的工作记忆。它保存全部运行时问题节点（文本、依赖、状态）、已经提升为Evidence的陈述，以及唯一`current_target`。它不是append-only日志：Checker可以通过受验证delta重建、替换或移除Evidence。`root_status=incomplete`必须有`current_target`；`root_status=ready`必须没有`current_target`。

```json
{
  "reading_session_id": "reading:1",
  "root_question_id": "root:1",
  "root_status": "incomplete",
  "questions": [
    {
      "question_id": "Q1",
      "text": "What was the revenue in 2022?",
      "depends_on": [],
      "status": "satisfied"
    },
    {
      "question_id": "Q2",
      "text": "What was the revenue in 2023?",
      "depends_on": [],
      "status": "incomplete"
    }
  ],
  "evidence": [
    {
      "evidence_id": "evidence:abc:0000",
      "statement": "Revenue in 2022 was 10 million.",
      "observation_ids": ["obs:abc:00"],
      "supports_question_ids": ["Q1"]
    }
  ],
  "current_target": {
    "question_id": "Q2",
    "gap_description": "The 2023 revenue is still unknown."
  }
}
```

`root_status=ready`表示当前Evidence集合整体足以回答Root Question，不表示答案已经生成，也不要求某一条Evidence单独包含最终结论；跨Evidence联合推理与答案生成仍由Answerer完成。`questions`是InitialPlan及Deferred Planner节点的运行时物化，不是第二份计划数据库；列表顺序保留Planner稳定顺序。`supports_question_ids`记录Evidence直接支持哪个Root/SubQuestion。v0每轮只评估`current_target`，本轮新增/替换Evidence也只能支持该目标，不会顺便改变其他问题状态；跨问题Observation复用留待TODO中的Recall实验。

依赖与切换由程序执行：目标问题只有在全部`depends_on`已`satisfied`时才可激活；当前目标满足后，程序按Planner原顺序选择第一个依赖就绪且未完成的问题，并直接使用该问题的`text`作为初始`gap_description`。不能由Checker跳到任意问题。现有计划全部完成但Root仍`incomplete`时，目标回到Root，程序生成通用gap：`Determine what evidence is still missing to answer the Root Question: <root question text>`。该通用gap不猜测缺失事实，只表示进入Root级重新审查、直接阅读或Deferred Planning。Deferred Planner若提出新证据需求，必须先由程序分配唯一ID、验证依赖/DAG/数量/深度并注册到`questions`，之后才能成为`current_target`。

### 冻结的普通循环与结果语义

```text
Controller选择Action
  -> Environment/Reader产生ReadRecord与0..N条Observation
  -> 0条Observation：不调用Checker，记录失败/limitation后返回Controller
  -> 非空Observation：Checker只围绕current_target评估并返回delta
  -> 程序验证并原子应用delta
  -> current_target satisfied但Root incomplete：程序按DAG选择下一题；计划耗尽则回到Root等待直接阅读或Deferred Planning
  -> current_target incomplete：Controller依据同一目标的gap、探索状态和本轮规范化结果继续
  -> Root ready：Answerer基于Evidence集合生成最终答案
```

- Observation补齐当前gap时，提升为带`supports_question_ids`的Evidence；Checker返回`current_target_status=satisfied`，但下一问题由程序选择，Checker不指定下一题。
- Observation部分有用时只提升可靠、相关的原子事实；无关、重复、对象/时间/范围不符或已被识别为错误时不提升，`current_target`保持同一问题并更新或保留gap。
- Observation与已有Evidence冲突时不得静默覆盖，当前gap可改写为需要消解的冲突。
- 看似合理但实际错误、且没有冲突或额外信息的Observation不能被架构凭空识别；依靠精确grounding、后续冲突、机会性交叉表示及高风险事实的选择性复核恢复。
- 候选、导航路线或动作预算耗尽时仍保持`incomplete`，由Reading Session记录证据不足而停止；不得伪装成`ready`或无限循环。

### ExplorationState

`ExplorationStateBuilder`从`ReadRecord + SearchSession + ActionTrace + Relation`生成快照。`attempted_source_ids`表示已经发生过读取尝试，不承诺读取成功；实际结果仍查看`ReadRecord.limitations`或`recent_actions.outcome`。`attempted_search_queries`保留实际规范化查询，不调用LLM生成summary。完整候选排序仍由`SearchSession`保存。

`current_focus`由最近一次`outcome=succeeded/degraded`且声明了`primary_target`的Action确定性派生；失败Action不改变focus。多图联合读取没有唯一主要目标时，Action可以不声明`primary_target`：若此前已有focus则保持原focus，否则为`null`，Builder不会擅自从I1/I2中挑选。

```json
{
  "reading_session_id": "reading:1",
  "root_question_id": "root:1",
  "current_focus": {
    "source_id": "element:table:1",
    "source_type": "element",
    "document_id": "doc:1",
    "page_id": "page:1",
    "element_id": "element:table:1",
    "visual_asset_id": null,
    "bbox": null
  },
  "attempted_source_ids": ["element:table:1"],
  "attempted_search_queries": ["2023 revenue table"],
  "active_search_session_ids": ["search_session:1"],
  "confirmed_relation_handles": [
    {
      "relation_id": "relation:1",
      "relation_type": "belongs_to_section",
      "source_id": "element:table:1",
      "target_id": "section:revenue"
    }
  ],
  "candidate_navigation_hints": [
    {
      "relation_id": "relation:2",
      "relation_type": "continued_on",
      "source_id": "element:table:1",
      "target_id": "element:table:2",
      "confidence": 0.82
    }
  ],
  "recent_actions": [
    {
      "action_id": "action:17",
      "question_id": "Q2",
      "action_name": "READ_ELEMENT",
      "target_ids": ["element:table:1"],
      "outcome": "degraded",
      "observation_ids": ["observation:17:00"]
    }
  ]
}
```

`confirmed_relation_handles`表示SoftDoc当前接受、可供正常`FOLLOW_RELATION`使用的关系；`candidate_navigation_hints`只表示target值得调查，不能声称关系成立；`rejected`不暴露。Builder只暴露与`current_focus`直接相连的局部关系，不把整份Document的关系图塞进Controller状态。两类对象都不会自动执行。

`active_search_session_ids`来自当前仍交给Controller使用的canonical `SearchSession`对象，只是引用，不复制ranking、cursor或候选内容。`recent_actions`来自`ActionTrace`最后若干步，只保留动作、目标、`outcome`和产生的`observation_ids`。不再保存自由文本`result_summary`：读取失败细节已经在`ReadRecord.limitations`，当前证据缺口在`EvidenceMemory.current_target`，Checker判断在本轮`EvidenceCheckResult`。Controller处理刚完成的动作时直接读取这些规范化对象；历史快照只保留引用，避免把同一语义复制成另一份自然语言状态。

### Controller v0 设计冻结（尚未实现策略）

Controller的研究职责是：围绕当前`EvidenceMemory.current_target`选择下一次阅读行为，而不是解析PDF、读取像素、判断Evidence充分性或生成最终答案。每一步接收：Root Question、完整且已提交的`EvidenceMemory`、派生`ExplorationState`、当前显示的`CandidatePreview`批次，以及紧接上一动作产生的规范化`ReadRecord`和可选`EvidenceCheckResult`。它不接收重复的Observation历史摘要，也不接收自由文本`result_summary`。

环境层先保留可执行primitive，最终训练时是否合并成更少的抽象动作由消融决定：

- `SEARCH`：没有可用入口、当前gap发生实质变化，或旧候选池与当前目标不匹配时创建新的SearchSession；
- `NEXT_CANDIDATES`：沿同一SearchSession查看下一批Preview，不重新搜索；
- `READ_SOURCE`：读取Element、Page、TableView或已经确定的视觉输入；
- `FOLLOW_RELATION`：只沿confirmed Relation读取另一端；
- `OPEN_NAVIGATION_HINT`：调查candidate Relation的target，但不声称Relation成立；
- `NEXT_PAGE` / `PREV_PAGE`：显式页面导航；
- `INSPECT_REGION`：在已有页面或视觉资产上读取局部区域；
- `STOP`：仅在Root ready或预算耗尽且明确报告证据不足时结束。

`UPDATE_PLAN`在v0关闭；Deferred Planning是否有净收益留待TODO中的对照实验。一次导航动作不会自动伪装成读取：`FOLLOW_RELATION`、翻页或打开候选只改变可访问目标，获得Observation仍需一次明确`READ_SOURCE/INSPECT_REGION`。这一边界使“看到一条边”和“读到可用事实”保持分离。

#### 解析噪声与不确定Relation的安全边界

解析器输出不是Evidence，Relation也不是Evidence。系统分三层处理噪声：

1. 确定性完整性：ID、bbox、资源路径、引用和Schema由程序验证；失败块降级但保留Provenance。
2. 导航不确定性：confirmed Relation可正常导航；candidate只暴露为navigation hint；rejected不暴露。candidate不会自动改变Section、拼接内容或进入Evidence。
3. 证据防火墙：Controller可以花一次动作调查candidate target；Reader产生Observation后，Checker仍须依据当前问题、已有Evidence和limitation决定是否采用。只有被Evidence引用的Observation才能影响最终回答。

未来RL/SFT学习的是“在当前gap、预算和历史下，调查某个不确定线索是否值得”，而不是学习把candidate直接宣布为事实。要验证的指标包括candidate调查率、有效Observation率、Evidence净增益、误导率、额外动作成本及最终答案变化。

### 跨存储引用验证

`ReadingStateReferenceValidator`只验证不同状态边界之间的引用完整性，不判断Observation事实是否正确，也不决定Evidence是否充分。它集中检查：

- `ObservationStore`、`EvidenceMemory`、`ActionTrace`与可选`ExplorationState`是否属于同一reading session和root question；
- 每个`ReadRecord.action_id`是否唯一存在于`ActionTrace`，且每条Observation指回同一Action；
- `ActionTrace.observation_ids`与对应`ReadRecord.observation_ids`是否指向同一批Observation；
- 每条Evidence引用的Observation是否真实存在；
- 可选`ExplorationState`中的source、recent action、SearchSession和Relation引用是否存在。

`ObservationStore`内部的ReadRecord、Observation和input grounding仍由其自身Pydantic验证器负责。跨存储验证器返回全部错误；需要在保存或执行前强制失败时可使用`raise_on_error=True`。未传入SearchSession或Relation registry时，不对相应可选引用作存在性判断。

### ID边界

Document、Page、Section、Element、Relation、SearchSession和视觉asset的现有稳定ID保持不变。ID只作为不透明handle；业务逻辑必须读取显式类型字段，不能从ID字符串反推类型。

Reading v0只新增四类全局运行时ID：`reading_session_id`、`action_id`、`observation_id`和`evidence_id`；`root_question_id`沿用Root Question身份，Action的`question_id`引用Root或`EvidenceMemory.questions`中已注册的问题。它们均由程序按session与序号确定性生成，不接受模型随意生成的`O1/E1`作为全局ID。Deferred Planner只提出内容与依赖，程序负责分配/校验问题ID并注册。`input_id`仅是单个Action内部的`I1/I2/...`局部别名。v0已经删除独立`read_request_id`、`request_source_id`、`request_visual_id`和`evidence_check_id`，不长期兼容两套scheme。

Observation Recall当前只记录在`TODO.md`，以后通过闭环实验决定是否实现。

### Evidence Checker v1 冻结边界

2026-08-22冻结`Evidence Checker v1`：canonical Prompt为
`scripts/evaluate_checker_mock.py`中的`CHECKER_SYSTEM_PROMPT`，输入/输出Schema与delta原子应用
以本章定义和`src/softdoc/reading_state.py`为准。本地`qwen3:8b`的mock evaluation只证明
Prompt、Pydantic验证和状态循环能够执行，不作为Checker质量结论；已发现的漏写gap、协议性delta、
信息补全、false-ready、旧Evidence联合判断和冲突修正问题统一进入`TODO.md`，留待服务器上的正式
模型对照。冻结后不再针对单个合成错例扩充主Prompt；除非contract本身存在可复现错误，否则只允许
修复验证器或运行时实现，不改变Checker语义。

### Answerer v0 冻结边界

`answerer-v0.3`只负责使用Checker已经接受的Evidence回答Root Question。正式Schema与Prompt位于
`src/softdoc/answering.py`。程序只有在`EvidenceMemory.root_status=ready`且
`current_target=null`时才能通过`AnswerInputBuilder`生成输入；Answerer不读取ObservationStore、
文档位置、SearchSession、ExplorationState、ActionTrace、Relation、Reader limitation或Checker
assessment。

Answerer输入只包含：

- Root Question原文；
- 精简`question_graph`，每个节点只有`question_id/text/depends_on`；
- 已接受Evidence的`evidence_id/statement/supports_question_ids`。

`question_graph`只是组织提示，不是事实来源，Root Question始终是权威任务。Answerer可以组合Evidence
并完成Root要求的确定性计算、比较或排序，但不能补充外部知识。模型输出严格收缩为：

```json
{
  "answer": "...",
  "used_evidence_ids": ["evidence:..."]
}
```

模型不生成page、Element、Observation或citation ID。`validate_answer_result`验证所用Evidence真实
存在；随后程序沿`evidence_id -> observation_ids -> ObservationStore sources`展开最终引用。文档位置
因此仍被系统保存，但不占用Answerer上下文。v0不实现真实模型客户端、claim-level citation或答案
质量评测。

## 11. 常用命令

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
