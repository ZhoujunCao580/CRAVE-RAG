# Representative-28 检索与耗时报告

> 历史快照：本文记录 2026-08-05 的稳定轮转实验。当前在线候选策略已经由
> 2026-08-11 的完整排名复验更新为 Exact 优先 + 加权 RRF。最新结论见
> [`END_TO_END_PIPELINE_AUDIT_20260811.md`](END_TO_END_PIPELINE_AUDIT_20260811.md)，
> 不应再用本文中的轮转数字描述当前默认系统。

日期：2026-08-05

本报告记录当前开发代表集从 14 份扩展到 28 份后的实际运行结果。该集合按
MMLongBench-Doc 的 7 种文档类型均衡选择，每类 4 份文档。它用于开发期暴露问题，
不是未见过的正式测试集，也不应被描述成最终 benchmark 成绩。

## 1. 数据范围

- 文档：28 份
- 页面：1,238 页
- SoftDoc Element：15,985 个
- SearchUnit：14,693 个
- DenseSegment：14,754 个
- 对应问题：275 道
- 有 Gold evidence page 的问题：213 道
- 无 Gold evidence page 的问题：62 道

新增 14 份文档及选择原因见
[`configs/representative_28.json`](../configs/representative_28.json)。新增集包含 733 页，
其中有 170 页设备手册、117 页论文和两份长 10-K。

原计划使用的遗传学幻灯片在 Windows 上使用 MinerU 3.4.4 的 `auto` 和 `txt`
方法均会在初始处理后挂起且不生成文档，因此被保留在 manifest 的
`excluded_after_parser_probe` 中，并用同类型 Tutorial 文档替换。替换文档已成功处理。

## 2. PDF 到可检索状态的冷启动耗时

测量范围：原始 PDF → MinerU → SoftDoc（关闭 debug overlay）→ SearchUnit → BM25
索引 → multilingual-E5-small Dense 索引。下载时间和问答时间不包含在内。

硬件：Windows 笔记本，NVIDIA RTX 4060 Laptop GPU。新增 14 份共 733 页。

| 阶段 | 总耗时 | 平均每 PDF |
|---|---:|---:|
| MinerU 有效产物生成 | 392.000 s | 28.000 s |
| SoftDoc + JSON 序列化 | 86.449 s | 6.175 s |
| E5 模型共享加载 | 6.854 s | 0.490 s（摊销） |
| SearchUnit + BM25 + Dense 索引 | 70.399 s | 5.028 s |
| 合计 | 555.702 s | **39.693 s** |

平均每页耗时为 0.758 秒。当前 SoftDoc 与检索层合计约 11.7 秒/PDF；主要成本仍是
MinerU，而不是 BM25 或 E5 查询。

MinerU 的产物在约 28 秒/PDF 的批量吞吐下完成，但 Windows 临时本地服务没有正常退出。
这属于运行可靠性问题：正式批处理应采用有超时、可重试的文档级作业，并区分“最后产物
时间”和“CLI 退出时间”。不能把无限收尾等待解释为模型计算耗时。

详细机器可读结果：

```text
data/processed/representative_28_extension/pipeline_performance/
  pipeline_performance.json
  pipeline_performance.md
```

## 3. 暖索引在线检索延迟

以下时间只测 Exact + BM25 + Dense 的单题检索，不包含未来的 LLM 子问题生成、Reader、
VLM、Evidence Checker 或答案生成。

| 阶段 | 平均 | P50 | P95 | 最大 |
|---|---:|---:|---:|---:|
| Exact | 1.54 ms | 1.05 ms | 3.92 ms | 8.42 ms |
| BM25 | 26.30 ms | 15.63 ms | 80.22 ms | 166.63 ms |
| Dense | 29.70 ms | 23.04 ms | 63.58 ms | 255.79 ms |
| 三路合计 | **57.54 ms** | **39.93 ms** | **138.39 ms** | **349.75 ms** |

因此，当前在线检索不是 30 秒级瓶颈。未来端到端延迟主要取决于 Agent 的 LLM/VLM
调用次数。首次读取 14,754 个“一向量一 JSON”缓存文件时曾出现明显 Windows I/O/扫描
开销；文件系统缓存变暖后整轮 275 题评测恢复到约 29 秒。扩展到全数据集前应考虑批量
二进制或 SQLite 缓存，但这属于工程优化，不改变检索语义。

## 4. Gold 页严格命中

本地 `questions.json` 的 `evidence_pages` 使用从 1 开始的物理页码，与
`SoftDoc Page.page_number` 比较。为防止把页码约定弄错，我们用 34 道能在 SoftDoc 原文中
精确找到答案字符串的问题进行核验：21 道只符合一基页码，1 道只符合零基索引，9 道
两种口径重叠，3 道受重复答案或解析覆盖影响。因此当前报告采用一基页码。

严格命中表示：候选 Element 所在物理页属于标注的 Gold evidence pages。它不表示候选
Element 本身就是答案证据，也不表示已经回答正确。

| 候选预算 | BM25 | Dense | 双路轮转 | Exact 优先 + 双路轮转 |
|---:|---:|---:|---:|---:|
| 1 | 38.03% | 37.56% | 38.03% | 43.66% |
| 3 | 48.83% | 53.99% | 54.93% | 57.75% |
| 5 | 58.69% | 59.15% | 63.38% | **64.79%** |
| 10 | 68.08% | 69.95% | 71.36% | **72.77%** |
| 20 | 77.93% | 82.63% | 81.22% | **82.16%** |
| 50 | 84.51% | 92.49% | 92.49% | **93.43%** |

Exact 在 18 道既含受支持 Anchor 又有 Gold 页的问题中命中 17 道。唯一异常是
`2311.16502v3.pdf` 的 Fig. 4：Exact 找到 SoftDoc 第 5 页的 Figure 4，而数据集 Gold 标为
第 4 页，需要后续人工确认是 Figure 编号配对还是 Gold 标注问题。

BM25 与 Dense 在 Top-5 分别有 25 和 26 道独占命中，说明两者确实互补。当前双路轮转
只是基线，不是最终融合方案。

## 5. 多页证据与导航上界

“至少命中一个 Gold 页”和“覆盖全部 Gold 页”必须分开：

| 预算 | 至少一页命中 | 全部 Gold 页覆盖 | ±1 页导航上界 |
|---:|---:|---:|---:|
| 5 | 64.79% | 48.83% | 79.81% |
| 10 | 72.77% | 58.69% | 85.45% |
| 20 | 82.16% | 68.54% | 94.84% |
| 50 | 93.43% | 82.16% | 98.59% |

这里的 `±1 页导航上界` 只表示候选页距离任一 Gold 页最多一页，模拟未来
`READ_PAGE(previous/next)` 可能到达的位置。它不代表 Agent 一定知道该向哪边走，也不代表
附近页提供了足够线索，因此不能与严格命中或答案准确率混合。

对当前 Exact 优先 + 双路轮转序列进行 batch size=5 的离线展示模拟：

- 首次 Gold 页前预览数：中位数 2，平均 15.86，P95 为 52；
- 首次 Gold 页前批次数：中位数 1，平均 3.77，P95 为 11；
- 213 道可评题中，212 道最终能在候选序列中找到至少一个 Gold 页；
- 唯一无法通过文本 Element 排名找到 Gold 页的是
  `0e94b...pdf` 第 14 页的视觉图表问题，说明未来仍需要 Page/视觉入口或 Agent 页面导航。

平均数被少量极难视觉问题拉高，因此不能只报告平均，也必须同时看中位数、P95 和动作
预算。batch size=5 只是展示批次，不是最终 Top-k。

## 6. 正确的三层评测口径

1. **严格初始检索**：Gold page Hit@K、首次 Gold rank、完整 Gold 页覆盖率。
2. **可导航性**：从已读候选出发，Agent 在规定动作预算内是否能通过相邻页或 confirmed
   Relation 到达 Gold 页；应报告实际路径长度和动作数，不能用 ±1 上界冒充。
3. **端到端问答**：Agent 实际读取的 Evidence、Evidence Checker 充分性和最终答案准确率。

Gold 页严格命中仍然重要，因为入口越接近证据，Agent 消耗的动作、LLM/VLM 调用和误入
歧路的概率越低。但它不是唯一目标：我们的系统允许继续读页和跟随关系，因此最终应在
“入口召回率”和“导航动作成本”之间共同优化，而不是要求初始 Top-5 覆盖所有证据页。

完整结果：

```text
data/processed/representative_28/retrieval_dense_e5/evaluation/
  retrieval_summary.json
  retrieval_summary.md
  retrieval_results.jsonl
```
