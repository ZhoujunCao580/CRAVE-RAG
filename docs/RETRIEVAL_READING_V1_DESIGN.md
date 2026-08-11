# Retrieval and Reading v1：当前设计与实现状态

> 更新日期：2026-08-11
> 前置版本：Retrieval Entry (`softdoc-v0.2-retrieval-entry`)
> 当前阶段：检索入口和可恢复候选会话；尚未进入 Reader、Evidence Checker 或 Agent 循环

## 1. 当前已经实现

- `ExactAnchorLookup`：Page、Figure、Table、Section，中英文写法；
- `SearchUnitBuilder`：模型无关的离线检索片段，不修改 SoftDoc；
- `BM25Index`：SearchUnit 评分、Element 级最佳 part 合并、完整正分排名；
- `DenseIndex`：`multilingual-e5-small`、归一化 dot product、Element 级合并；
- 可注入 `TextEncoder`：单元测试使用 mock，不要求联网或 GPU；
- E5 超长保护：公共 SearchUnit 不变，Dense 内部无损重叠切片；
- 与文本、模型版本和索引版本绑定的 Embedding cache；
- `CandidatePreview`：由命中 SearchUnit 确定性截取的轻量原文卡片；
- `SearchSession`：保存完整候选顺序、cursor、已展示和已打开状态；
- 加权 RRF：在 Element 级融合 BM25/Dense rank，默认 `k=20`、权重 `1.0/1.25`；
- 默认每批 5 个候选，支持 JSON round-trip 后从下一批继续；
- 28 份真实 SoftDoc 和 275 道 MMLongBench 问题上的离线评测。

当前没有实现：

- `READ_ELEMENT`、`READ_PAGE`、`INSPECT_REGION`、`FOLLOW_RELATION`；
- 子问题自动生成、Evidence State、Evidence Checker；
- LLM/VLM 调用、Agent 循环和答案生成。

## 2. 不变的职责边界

1. `SEARCH` 只寻找阅读入口，不直接生成问答 Evidence。
2. Exact handle、BM25/Dense candidate 都不是 Evidence。
3. Relation 不参与初始检索排名，也不在 SEARCH 中自动展开。
4. 只有完整读取 Element/Page/Region 后，才能形成针对子问题的 Observation。
5. Relation handles 在 `READ_ELEMENT` 后才暴露；默认只显示 confirmed Relation。
6. Retrieval 状态不写入 `Document`，也不是 `SoftDocPipeline` 的 DocumentPass。
7. 不为全部 Element 生成 LLM summary、abstract 或 keywords。
8. 不把固定 Top-k 当成最终上下文；完整排名保留，未来按批次展示。

## 3. 两条在线入口

### 3.1 有明确 Anchor

```text
SubQuestion
  -> ExactAnchorLookup
       -> unique: 直接读取目标
       -> ambiguous: 保留全部轻量句柄，由主 Reading Agent 消歧
       -> unresolved: 启动普通搜索
  -> READ / CHECK_EVIDENCE
  -> 证据仍不足时再启动 BM25 + Dense
```

Exact 命中只是“从这里开始读”。例如问题同时提到 `Figure 3` 和一个跨文档
比较条件时，系统先打开 Figure 3，但仍可能需要普通搜索补足背景或比较证据。

### 3.2 无 Anchor 或 Exact 无法解决缺口

```text
SubQuestion
  -> BM25 完整 Element 排名
  -> Dense 完整 Element 排名
  -> Weighted RRF、Element 去重
  -> SearchSession 每批返回少量 CandidatePreview
  -> 主 Reading Agent 选择 READ_ELEMENT
```

RRF 只使用两个检索器内部的 Element rank，不直接相加 BM25 score 与 cosine score。
稳定轮转仍保留为离线基线，不再是默认在线策略。完整候选顺序保存在
SearchSession 中，默认只显示 5 条，但不存在固定最终 Top-k。

## 4. Exact Anchor Lookup

支持：

- `Page 3`、`page 3`、`第3页`；
- `Figure 3`、`Fig. 3`、`图3`；
- `Table 2`、`表2`；
- `Section 4.1`、`第4.1节`。

Exact 复用 `labels.py` 的类型安全注册表，不建立第二套 Figure/Table 标签。
Page 优先解析可靠的印刷页码别名，无别名时才使用物理 PDF 页码。答案格式中的
`for example` / `e.g.` 假 Anchor 会被忽略。

解析状态是 `unique`、`ambiguous` 或 `unresolved`。歧义不能静默选择；不存在的
Anchor 不导致整次问题失败。

## 5. SearchUnit

SearchUnit 是检索内部视图，不替换或拆分 SoftDoc Element。每个 Unit 必须稳定映射
回唯一 `element_id`。

基础文本：

- Paragraph/List/Caption/Footnote：`section_path + 原文`；
- Heading：标题和祖先 section path，避免重复自身；
- Figure/Chart：自身明确 label、自身可用文本和 section path；
- Table：label、HTML 转纯文本、自身文本和 section path；
- Code/Algorithm/Equation：section path 和自身文本。

基础索引禁止复制 Relation 邻居内容：Caption 不复制到 Figure，Footnote 不复制到
目标图表，`refers_to` 正文也不复制到视觉对象。

公共切片默认使用 256 个模型无关词法 token、32 token overlap。超长 Element 可以
生成多个 SearchUnit；检索结果必须先按最佳 part 合并成一个 Element 候选。

## 6. BM25

BM25 使用确定性英文/数字/Unicode 词法 token，并为连续中文增加 unigram/bigram。
同一 Element 多个 part 命中时使用最高 part 得分。只返回正分候选，不设置固定最终
Top-k；同分时按页、reading order 和稳定 ID 排序。

候选保存命中词、字符 offset 和命中的 SearchUnit，供未来 CandidatePreview 截取
原文窗口。

## 7. Dense Retrieval

模型：`intfloat/multilingual-e5-small`。

- passage 使用 `passage: {search_text}`；
- query 使用 `query: {question}`；
- attention-mask mean pooling；
- 384 维向量归一化后使用 dot product；
- 单文档直接矩阵计算，不使用向量数据库；
- CPU/CUDA 可配置；
- 单元测试使用注入的 mock encoder。

### 7.1 不静默截断

E5 最大输入为 512 model token。公共 SearchUnit 不依赖 E5 tokenizer；Dense 建索引时
先用真实 tokenizer 检查：

```text
<= 512 token -> 一个 DenseSegment
> 512 token  -> 最多 480 token 的内部片段，48 token overlap
```

每个内部片段再次验证完整 `passage:` 输入不超过 512。首片覆盖字符 0，末片覆盖原文
末尾，相邻片段不得留下字符空洞。Reader 最终仍读取完整 Element，而不是
DenseSegment。

### 7.2 Cache 身份

缓存至少绑定：document、SearchUnit、DenseSegment、文本 SHA-256、SearchUnit/index
版本、分片版本、模型和 tokenizer revision、pooling、max length、维度和 dtype。
device 和 batch size 不改变向量语义，因此只记运行配置，不进入 cache identity。

当前 JSON-per-vector 文件缓存适合研究原型且容易审计，但在 Windows 上小文件较多。
扩大到全数据集前应改成批量二进制缓存；这属于工程优化，不改变检索语义。

## 8. 当前加权 RRF 组合策略

28 文档的完整 Element 排名上比较了 BM25-only、Dense-only、两种轮转、多种比例轮转
以及多个 RRF 参数。当前默认值为：

```text
RRF_score(d) = 1.0 / (20 + bm25_rank(d))
             + 1.25 / (20 + dense_rank(d))
```

某一检索器没有找到该 Element 时，该路贡献为零。最终按 RRF score 降序；同分时按
最佳源 rank、页码和稳定 Element ID 排序。它不会直接相加量纲不同的 BM25 score 与
Dense cosine score。

选择目标优先考虑默认第一批 5 个 Preview 是否包含 Gold 页，再比较 MRR、Top-20，
并检查原14份和新增14份文档上的方向是否一致。`k=20`、Dense权重1.25在完整排名上
取得 Top-1 48.36%、Top-5 68.54%、Top-20 84.04%、MRR 0.5671，优于旧 Exact-first
轮转的 43.66%、64.79%、82.16%、0.5403。稳定轮转保留为可复现实验基线。

Exact handles 始终单独保存，不参与 RRF。在线时应先读取 unique Exact 目标；只有证据
不足时才请求普通候选。离线评测把 Exact 放在 RRF 列表之前，只用于模拟最终入口覆盖。
若 Exact Element 同时被 BM25/Dense 命中，它不会在普通候选重复；检索来源、各自 rank
和 RRF score 作为审计元数据合并进 SearchSession 内的 Exact handle。

## 9. CandidatePreview 与 SearchSession

`CandidatePreview` 是“是否值得打开”的轻量卡片，不是 LLM 摘要，也不是 Evidence。
它只包含 Element 类型、可选display label、页码、Section path、命中 SearchUnit 的
有界原文窗口、检索来源、原始 rank 和 content availability。它不包含完整 Element
对象、HTML、高清图片、Relation 目标或任何自动生成结论。

BM25 Preview 围绕 matched offsets 截取；Dense Preview 使用最佳 Dense SearchUnit 的
命中字符范围。offset明确属于`SearchUnit.search_text`，不能解释成原始Element字符坐标。
相同 Element 即使被 BM25 和 Dense 同时发现，也只占一个候选位置，但同时保存两个来源、
各自 rank 与 RRF score。`preview_source`记录实际展示来源，`match_scope`区分content、
metadata、mixed和unknown。BM25只命中Section path/label而Dense覆盖正文时优先展示Dense，
其他情况才按当前RRF贡献选择窗口。Exact Element目标保持单独返回，不在普通序列重复。

有明确label的Figure/Table/Heading通过`display_label`展示。无label且无文本的visual-only
对象不会产生虚假的空SearchUnit，必须由Exact、READ_PAGE、Relation或未来视觉检索发现。

`SearchSession` 是可 JSON 序列化的候选游标，保存：

- 完整 `ranked_candidate_ids`，不存在固定最终 Top-k；
- `shown_candidate_ids`、`opened_candidate_ids`；
- `cursor` 与 `exhausted`；
- Exact matches、unresolved Anchors 和 retrieval trace；
- 生成 Preview 所需的检索元数据，不复制 Relation 邻居或完整 Element。

默认第一次显示 1—5，第二次显示 6—10。Session round-trip 后只要重新加载同一份
SearchUnit index，就能从原 cursor 继续，也可以重新访问之前展示的卡片。创建 Session
不会自动 READ 候选；有 Exact 命中时，调用方可以先直接读取 Exact 目标，只有证据不足
时才请求普通候选批次。

Session trace额外统计两路前5项中metadata-only命中数量，只用于诊断Section path是否
挤占候选，不改变BM25、Dense或RRF排序。未来Exact目标的读取记录进入统一Action Trace；
`opened_candidate_ids`继续只表示已经展示并打开的普通候选。

## 10. 代表性评测快照

数据：28 份 PDF、1238 页、15985 Elements、275 道问题。其中 213 道提供 Gold 物理
证据页；其余 62 道保留结果但不计证据页指标。Gold Element ID 不存在，因此以下是
“候选是否来自 Gold 页”，不是答案准确率。

| K | BM25 | Dense | Exact+旧轮转 | Exact+加权RRF |
|---:|---:|---:|---:|---:|
| 1 | 38.03% | 37.56% | 43.66% | 48.36% |
| 5 | 58.69% | 59.62% | 64.79% | 68.54% |
| 10 | 68.08% | 69.95% | 72.77% | 74.65% |
| 20 | 77.46% | 82.63% | 82.16% | 84.04% |
| 50 | 84.51% | 92.49% | 92.96% | 92.96% |

25 道问题含受支持 Anchor，其中 18 道有 Gold 页；17/18 的 Exact 目标位于 Gold 页。
BM25 和 Dense 在 Top-5 分别有 24 和 26 道独有命中，证明存在互补性；简单轮转在
某些预算下会被较弱通道占位，加权RRF改善了第一批入口质量。
这里的 oracle union@K 表示分别查看 BM25 前 K 和 Dense 前 K 后“任一命中”，最多
检查 2K 个候选，只用于分析互补性，不是可部署的组合排名。

评测产物：

```text
data/processed/representative_28/retrieval_policy_v3/evaluation/
  retrieval_summary.md
  retrieval_summary.json
  retrieval_results.jsonl
```

## 11. 下一步边界

1. 实现 `READ_ELEMENT`、`READ_PAGE`、`INSPECT_REGION` 和 `FOLLOW_RELATION`；
2. 让 READ 后才暴露 confirmed Relation handles；
3. 定义从实际读取结果产生的 Observation / Evidence State；
4. 在完整数据集上重新验证当前 RRF 参数，而不是继续在28份开发文档上调参；
5. 最后接入子问题、Evidence Checker 与主 Agent 循环。

Relation navigation 继续只属于 READ 之后的主动阅读阶段。
