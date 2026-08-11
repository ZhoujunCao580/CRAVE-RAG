# SoftDoc 到候选预览：端到端审计与当前定案

日期：2026-08-11

## 1. 先说明项目究竟做到哪里

当前已经完成：

```text
PDF
  -> MinerU Pipeline（默认）或 Hybrid（可选）
  -> SoftDoc Document / Page / Section / Element / Relation
  -> SearchUnitBuilder
  -> Exact Anchor Lookup（独立结果）
  -> BM25 + multilingual-e5-small Dense Retrieval
  -> Element 级加权 RRF
  -> SearchSession
  -> 分批 CandidatePreview
```

当前尚未完成：

```text
READ_ELEMENT / READ_PAGE / INSPECT_REGION
FOLLOW_RELATION
Evidence State / Evidence Checker
Reading Agent 循环
答案生成与最终 QA 指标
```

因此，“Agent 已经能看到候选”更准确的说法是：检索层已经提供了未来
Reading Agent 可以消费的候选预览 API；真正的 Agent 还没有实现。

## 2. 各阶段的职责边界

### 2.1 SoftDoc

SoftDoc 保存解析后的事实层：页面、元素、bbox、阅读顺序、Section 和显式关系。
它不保存检索切片，也不因查询而改变。默认解析后端仍是 MinerU Pipeline。

### 2.2 SearchUnitBuilder

SearchUnit 是离线检索内部片段，始终映射回一个原始 `element_id`：

- 短 Element 通常生成一个 SearchUnit；
- 超长段落和表格按边界安全切片；
- Caption、Footnote 不复制到 Figure/Table；
- Relation 邻居文本不注入基础索引；
- 不生成 LLM 摘要和关键词；
- Hybrid 生成但未经核验的视觉描述不进入 SearchUnit。

这样做避免修改 SoftDoc，也避免同一信息因关系复制而在索引中被重复加权。

### 2.3 Exact Anchor Lookup

问题中存在 `Figure 3`、`表2`、`Page 3`、`第4.1节` 等明确指向时，
Exact 返回 Page、Element 或 Section handle。Exact 结果与普通排序分开保存；
不存在或歧义的 Anchor 进入 `unresolved_anchors`，不会让整个搜索失败。
同一个 Exact Element 若也被 BM25/Dense 找到，不会在普通列表重复；对应的来源、
BM25/Dense rank 和 RRF 分数合并记录在 Exact handle 中，便于审计。

### 2.4 BM25 与 Dense

- BM25 擅长编号、年份、术语和原文措辞；
- multilingual-e5-small 擅长中英文表达差异和语义改写；
- 两者都先将 SearchUnit 得分合并为 Element 级排序，同一 Element 不会因多个
  part 占据多个候选位置。

### 2.5 当前候选融合定案

稳定轮转仅保留为历史基线，不再是默认策略。当前默认是 Element 级加权 RRF：

```text
RRF(element)
  = 1.0  / (20 + bm25_rank)
  + 1.25 / (20 + dense_rank)
```

若某元素只被一个通道找到，只计算该通道的贡献。排序完全相同时，依次按最佳来源
rank、页码和稳定 Element ID 打破平局。RRF 使用名次而不是原始分数，因此无需把
BM25 分数和 cosine 分数强行放到同一尺度。

这一参数是在当前 28 份开发文档的完整 Element 排名上比较 5 个 RRF 变体后选择的。
它是当前开发集最优默认值，不代表对完整数据集永久最优；扩大语料后应做一次冻结集
复验，而不是继续在这 28 份文档上反复调参。

## 3. CandidatePreview 到底是什么

CandidatePreview 是让未来 Reading Agent 决定“下一步打开哪个 Element”的小卡片，
不是 LLM 摘要，也不是 Evidence。当前字段为：

```json
{
  "element_id": "...",
  "element_type": "paragraph",
  "page_id": "...",
  "page_number": 12,
  "section_path": ["Experiments", "Robustness"],
  "matched_snippet": "...原始检索文本中的短片段...",
  "snippet_char_start": 80,
  "snippet_char_end": 320,
  "snippet_truncated": true,
  "matched_search_unit_id": "...#part_0001",
  "matched_by": ["bm25", "dense"],
  "bm25_rank": 17,
  "dense_rank": 2,
  "rrf_score": 0.0712,
  "content_availability": "textual"
}
```

Preview 的生成规则：

- BM25 主导时，围绕命中词截取原文窗口；
- Dense 主导时，使用最佳 Dense SearchUnit 的确定性截断文本；
- 两者都命中时，比较二者在当前 RRF 中的贡献，选择贡献更大的来源生成片段，
  避免很弱的 BM25 尾部命中遮住很强的 Dense 命中；
- Caption/Footnote 可直接显示自身短文本；
- `visual_only` Figure 允许空 snippet，只显示类型、label 和位置。

Preview 不包含：完整 Element、大表 HTML、整页图片、高清 crop、Relation 目标、
LLM 摘要或答案。Agent 只有执行未来的 READ/INSPECT 动作后，读到的内容才可能成为
针对子问题的 Evidence。

## 4. SearchSession 到底是什么

SearchSession 是一次子问题搜索的可序列化游标，负责防止“只取一次固定 Top-k 后就丢掉
其余候选”。它保存：

```text
search_session_id
subquestion_id / document_id
config
exact_anchor_matches / unresolved_anchors
ranked_candidate_ids             # 完整、去重后的候选顺序
candidate_catalog                # 生成 Preview 所需的检索元数据
shown_candidate_ids              # 已展示
opened_candidate_ids             # 未来 Reader 已打开
cursor / exhausted
retrieval_trace
```

默认第一次展示第 1--5 个 Preview，下一批展示第 6--10 个。旧候选、完整排序和打开记录
都不会丢失；Agent 可以回看已经展示的候选，也可以在证据不足时请求下一批。
`batch_size=5` 只是界面/上下文预算，不是最终 Top-k，也不会自动 READ 五个元素。

SearchSession JSON round-trip 后保持稳定，但恢复会话时仍需使用相同 document、
`index_version` 和 SearchUnit 索引，才能重新确定性地产生相同 Preview。

## 5. 当前 28 文档检索结果怎样理解

语料共 28 份 PDF、1238 页、15985 个 SoftDoc Element，对应 275 道问题。
其中只有 213 道给出了可用于本评测的 Gold 证据页，其余 62 道不能参与证据页命中率。

下表统计“前 K 个 Element 中，是否至少有一个位于 Gold 证据页”。它不是最终答案准确率，
也不是 Gold Element 精确率，因为数据集没有提供 Gold Element ID。

| 方法 | Hit@1 | Hit@5 | Hit@10 | Hit@20 | Hit@50 |
|---|---:|---:|---:|---:|---:|
| BM25 | 38.03% | 58.69% | 68.08% | 77.46% | 84.51% |
| Dense | 37.56% | 59.62% | 69.95% | 82.63% | 92.49% |
| Exact 优先 + 旧稳定轮转 | 43.66% | 64.79% | 72.77% | 82.16% | 92.96% |
| **Exact 优先 + 当前加权 RRF** | **48.36%** | **68.54%** | **74.65%** | **84.04%** | **92.96%** |

25 道题包含支持的 Anchor，其中 18 道有 Gold 页；Exact 命中 Gold 页 17/18。
BM25 与 Dense 的确互补，例如在 Top-20，BM25-only 命中 16 道、Dense-only 命中
27 道，因此不宜永久删除其中一条通道。

仍未达到 100% 的主要原因包括：SoftDoc 切分/类型错误、视觉内容缺少可检索文本、问题
需要跨多个元素或页面、Gold 证据页并不包含与问题直接相似的措辞，以及当前评测只检索
阅读入口、尚未通过 Relation/Page 阅读继续找证据。

## 6. Hybrid 是否现在默认使用

结论：保留 Hybrid 能力，但当前不设为默认后端。

已完成的困难页面配对实验中，只有 3 道问题满足“全部 Gold 页都落在配对 mini-document”
这一严格条件。Pipeline 与 Hybrid 的 BM25、Dense、RRF 结果完全相同：RRF Top-1 都是
3/3。样本太小，不能证明 Hybrid 能提高全局检索正确率。

此前后端 A/B 还显示：

- Hybrid 的有效解析约慢 1.73 倍；
- 峰值 GPU 显存约为 Pipeline 的 3.38 倍（约 7.2 GB 对 2.1 GB）；
- 对幻灯片/list 和复杂视觉块的切分有时更好；
- 12 个生成式视觉描述中，8 个有依据、3 个部分错误、1 个编造。

未经核验的生成视觉文本现在不会进入检索索引，所以 Hybrid 的潜在语义收益不会直接转化
为 Dense 提升；若直接放开，又会把幻觉带进检索。当前合理策略是 Pipeline 默认，Hybrid
仅作为未来“视觉困难页的按需升级”能力。项目目前还没有可靠的自动困难页路由器，因此也
没有自动混用两个后端。

## 7. 代码审计发现的风险和技术债

1. `relations.py` 与 `normalization.py` 已经很大，分别约 99 KB 和 83 KB；
   `floating_sections.py`、`page_labels.py`、`hierarchy.py` 也包含较多历史规则。这些仍被
   Pipeline 全量文档依赖，不能为了“看起来简洁”直接删除，但不应继续无边界增加补丁。
2. Hybrid 路径会绕过一部分旧 Normalizer，但不是替代整个 SoftDoc Pipeline。
3. 当前 CLI 只有 `parse-mineru` 和 `validate`；检索主要通过 Python API 和评测脚本调用，
   尚无稳定的在线 search CLI/API。
4. 当前 Dense cache 每个向量保存为一个 JSON，小规模开发方便，但全数据集会产生大量
   小文件；后续应改为批量二进制缓存或轻量本地索引。
5. 评测脚本为复现实验会穷举完整排名，时间不能等同于未来单问题在线延迟。
6. Caption/Footnote、continued_on、Heading 等解析仍存在少量误差；检索层不能证明这些
   Relation 已全部正确。
7. SearchSession 为可移植原型同时保存候选 ID 和 catalog 元数据，有少量重复；大规模服务
   化后可拆成持久 SearchStore，但当前没有必要提前增加基础设施。
8. 当前 28 份文档既用于选择融合参数又用于报告结果，因此数字是开发集指标，不是最终
   无偏测试指标。

## 8. 接下来最合理的实施顺序

1. 冻结当前 Exact + BM25 + Dense + 加权 RRF + SearchSession + Preview；
2. 实现最小 Reader 工具：`READ_ELEMENT`、`READ_PAGE`、`INSPECT_REGION`，读取后再暴露
   confirmed Relation handles；
3. 实现 `FOLLOW_RELATION`，candidate Relation 默认关闭；
4. 建立显式 Evidence State 和 `CHECK_EVIDENCE`；
5. 最后再实现同一个 Reading Agent 的选择、回退候选、下一批和回答循环；
6. 在扩大到完整数据集时，只对冻结策略复验一次；如果性能发生实质漂移，再重新选择权重。

当前不建议继续在 28 份开发文档上细调 RRF，也不建议先跑完整 Hybrid 解析。真正缺失的
核心能力已经不是另一个检索器，而是从“候选入口”进入“实际阅读与证据状态”的边界。

## 9. 可复核产物

冻结前验证结果：28/28 SoftDoc 引用验证通过；193/193 pytest 通过；Python
`compileall` 通过；`pip check` 未发现依赖冲突。

- 全量检索：`data/processed/representative_28/retrieval_policy_v3/evaluation/`
- 策略诊断：`data/processed/representative_28/retrieval_policy_v3/policy_analysis/`
- Hybrid 配对探针：`data/processed/mineru_backend_ab_20260806/hybrid_retrieval_probe.md`
- 当前检索设计：`docs/RETRIEVAL_READING_V1_DESIGN.md`
- 项目历史日志：`docs/PROJECT_PROGRESS_LOG.md`
