# Retrieval and Reading v1：当前设计与实现状态

> 更新日期：2026-08-04
> 前置版本：SoftDoc Milestone 1 (`softdoc-v0.1-dev`)
> 当前阶段：检索入口原型；尚未进入 Reader、Evidence Checker 或 Agent 循环

## 1. 当前已经实现

- `ExactAnchorLookup`：Page、Figure、Table、Section，中英文写法；
- `SearchUnitBuilder`：模型无关的离线检索片段，不修改 SoftDoc；
- `BM25Index`：SearchUnit 评分、Element 级最佳 part 合并、完整正分排名；
- `DenseIndex`：`multilingual-e5-small`、归一化 dot product、Element 级合并；
- 可注入 `TextEncoder`：单元测试使用 mock，不要求联网或 GPU；
- E5 超长保护：公共 SearchUnit 不变，Dense 内部无损重叠切片；
- 与文本、模型版本和索引版本绑定的 Embedding cache；
- 14 份真实 SoftDoc 和 142 道 MMLongBench 问题上的离线评测。

当前没有实现：

- RRF 或其他分数融合；
- CandidatePreview、SearchSession、cursor 和分批返回接口；
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
  -> 当前：保持两条来源，按批轮转、Element 去重
  -> 未来：是否加入 RRF 由实测决定
  -> CandidatePreview
  -> 主 Reading Agent 选择 READ_ELEMENT
```

当前轮转策略不混合 BM25 与 Dense 分数，也不是 RRF。它只用于测量真实在线
“两条候选通道同时存在”时的入口覆盖；它不是最终冻结的融合算法。

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

## 8. 当前无 RRF 的组合策略

为避免在没有对比数据时提前加入融合，当前评测使用：

```text
Exact handles（如有）
  -> BM25 rank 1
  -> Dense rank 1（若Element未出现）
  -> BM25 rank 2
  -> Dense rank 2（若Element未出现）
  -> ...
```

该顺序只代表未来分批 Preview 的一个简单基线：

- 不比较 BM25 score 与 cosine score；
- 不改变两个检索器内部排名；
- 相同 Element 只出现一次；
- 每个候选保留发现来源和原始 rank；
- Exact 目标仍独立保存。

如果 Exact 命中，真实在线流程会先 READ 并 CHECK_EVIDENCE；普通候选只有在证据不足
时才展示。离线的 `exact_first_dual` 指标表示“最终入口序列”的覆盖，不表示所有通道
会在线同时执行。

## 9. 代表性评测快照

数据：14 份 PDF、505 页、5860 Elements、142 道问题。其中 107 道提供 Gold 物理
证据页；其余 35 道保留结果但不计证据页指标。Gold Element ID 不存在，因此以下是
“候选是否来自 Gold 页”，不是答案准确率。

| K | BM25 | Dense | 双通道轮转 | Exact优先+双通道 |
|---:|---:|---:|---:|---:|
| 1 | 38.32% | 37.38% | 38.32% | 43.93% |
| 5 | 63.55% | 57.94% | 68.22% | 69.16% |
| 10 | 74.77% | 63.55% | 74.77% | 75.70% |
| 20 | 85.05% | 77.57% | 83.18% | 84.11% |
| 50 | 92.52% | 94.39% | 94.39% | 95.33% |

14 道问题含受支持 Anchor，其中 9 道有 Gold 页；9/9 的 Exact 目标位于 Gold 页。
BM25 和 Dense 在 Top-5 分别有 17 和 11 道独有命中，证明存在互补性；但简单轮转在
某些预算下会被较弱通道占位，不能自动获得 oracle union 的全部收益。
这里的 oracle union@K 表示分别查看 BM25 前 K 和 Dense 前 K 后“任一命中”，最多
检查 2K 个候选，只用于分析互补性，不是可部署的组合排名。

评测产物：

```text
data/processed/representative_14/retrieval_dense_e5/evaluation/
  retrieval_summary.md
  retrieval_summary.json
  retrieval_results.jsonl
```

## 10. 下一步边界

当前先不实现 RRF。合理的下一步是：

1. 人工查看 BM25-only、Dense-only 和 neither 的代表性失败；
2. 定义 CandidatePreview 的最小字段与确定性原文窗口；
3. 定义 SearchSession、cursor、批次和来源多样性策略；
4. 重新比较轮转、按页去重和未来 RRF，而不是直接假定某种融合最好；
5. 然后实现 Reader 工具；
6. 最后接入子问题、Evidence State、Checker 与主 Agent 循环。

Relation navigation 继续只属于 READ 之后的主动阅读阶段。
