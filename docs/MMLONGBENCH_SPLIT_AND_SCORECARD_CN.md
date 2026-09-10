# CRAVE-RAG：MMLongBench-Doc 拆分与评估记分牌

> 更新日期：2026-09-10  
> 数据集：MMLongBench-Doc v1 当前 135 文档 / 1,091 问题快照  
> 边界：这是内部文档级开发拆分，不是官方公开基准的完整复现，也不能宣称为外部未见测试。

## 1. 冻结拆分

拆分单位是 PDF，同一文档的问题不会跨集合。固定 seed 为 `20260910`。

| 集合 | 文档数 | 题目数 | 原 177 困难题 | 用途 |
| --- | ---: | ---: | ---: | --- |
| Train | 42 | 357 | 140 | Teacher 轨迹、Controller SFT |
| Diagnostic Dev | 12 | 97 | 37 | 已知困难类型诊断，可逐题审计 |
| Clean Dev | 8 | 68 | 0 | checkpoint 与配置选择 |
| Test Core | 10 | 81 | 0 | 代表性密封测试 |
| Test Challenge | 6 | 43 | 0 | 结构困难富集的密封测试 |
| Reserve | 57 | 445 | 0 | 暂不使用，留作扩训或后续验证 |
| **总计** | **135** | **1,091** | **177** |  |

冻结清单：`configs/training/mmlongbench_sft_split_v0_1/split_manifest.json`。

## 2. 当前 CRAVE-RAG 结果

本项目以**答案内容是否正确**作为正式判分口径：直接比较每道轨迹实际保存的最终 `answer`，忽略无害的大小写、解释文字、列表写法和等价缩写。`budget_exhausted` 与 `stopped_incomplete` 轨迹本身已经确定性保存 `answer="Not answerable"`；当 Gold 也为 `Not answerable` 时正常计对，状态名称本身不参与判分。F1 使用 MMLongBench-Doc 官方 generalized F1 的正负类定义，并以内容等价判断作为单题正确性。

此前计算的 `official v1 direct-answer` 48% 一组仅反映未经答案抽取、直接套用严格字符串/List 规则的诊断结果，会把本项目明确认可的格式等价答案重新扣分。它不再作为项目准确率，也不再放入主结果表。

| 集合 | 当前状态 | 正式 Acc | 正式 F1 |
| --- | --- | ---: | ---: |
| Train（357） | 未运行 | — | — |
| Diagnostic Dev（97） | 已完成 | **53/97 = 54.6%** | **56.12%** |
| Clean Dev（68） | 已完成 | **36/68 = 52.9%** | **48.89%** |
| **Dev 合并（165）** | 已完成 | **89/165 = 53.9%** | **53.28%** |
| Test Core（81） | 已完成 | **38/81 = 46.9%** | **47.86%** |
| Test Challenge（43） | 已完成 | **24/43 = 55.8%** | **52.46%** |
| **Test 合并（124）** | 已完成 | **62/124 = 50.0%** | **49.44%** |
| Reserve（445） | 未运行 | — | — |

这里的“正式”表示项目后续选 checkpoint、比较 SFT 和汇报内部结果时统一采用的官方指标口径：Acc 按内容等价判对，F1 使用官方 generalized F1 公式。它不再与 48% 的未抽取严格字符串诊断值并列。若投稿或提交公开榜单，仍需披露本次等价判断含自动/人工校准，并补做官方三阶段复现。

补充历史诊断：原 177 困难题运行的正式 Acc 为 **81/177 = 45.8%**、正式 F1 为 **33.01%**。该批次是困难题诊断，不是独立测试集，也不应与完整基准排名直接比较。

三份现有评分文件中，评分时补写 `Not answerable` 的次数均为 **0**。实际最终输出分别包含：Diagnostic Dev 33 个、Clean Dev 29 个、177 困难题 91 个 `Not answerable`。因此这里没有把空答案事后改成正确答案。

### 格式与语义校准复核

并不是只有一道题存在非精确格式差异。Dev 165 题中共有 **15 道**不能靠简单规范化精确匹配、但内容应判正确：

- **14 道已被自动内容评审正确救回**：其中约 11 道主要是列表引号/顺序、百分号、数字写法、货币符号、冗余解释或标题前缀差异；另外 3 道是语义等价的缩写或改写。
- **1 道 Q377 由人工校准救回**：预测 `['1981-82', '2001-02']`，Gold 为 `['1981', '1982', '2001', '2002']`，是同一组年份的区间写法差异。

14 道自动救回案例为：`Q90, Q119, Q120, Q122, Q338, Q340, Q368, Q771, Q193, Q194, Q223, Q833, Q1087, Q1088`。

对校准后仍错误的 **76 道**再次逐题核查，没有发现第二道“内容正确、仅因格式被判错”的案例。它们由 34 道 Gold 可回答但最终输出 `Not answerable`、11 道 Gold 不可回答但系统给出实质答案，以及 31 道已回答但数值、实体、列表完整性、单位或语义确实错误组成。因此当前内容正确率仍为 **89/165 = 53.9%**，不再追加格式修正。

### Acc 与 generalized F1 的区别

- **Acc**：165 道中答对多少道。当前是 `(61 道可回答题答对 + 28 道不可回答题答对) / 165 = 53.9%`。
- **Precision**：系统给出实质答案的 103 道中，有 61 道内容正确，即 `61/103 = 59.22%`。
- **Recall**：Gold 可回答的 126 道中，有 61 道被正确回答，即 `61/126 = 48.41%`。
- **F1**：Precision 与 Recall 的调和平均，`2PR/(P+R) = 53.28%`。

因此 F1 不是另一种“全题准确率”，也不是普通的答案 token 重合率。它主要衡量系统对**可回答题**是否既敢答、又答对；正确拒答的 28 道不会像在 Acc 中那样直接加入分子。F1 可以高于或低于 Acc：Diagnostic Dev 为 56.12% > 54.6%，Clean Dev 为 48.89% < 52.9%，合并后为 53.28% < 53.9%。

## 3. MAGE-RAG 与论文基线

MAGE-RAG 论文在统一的 Qwen3-VL-8B-Instruct reader 协议下，对完整 MMLongBench-Doc 报告如下主结果：

| 方法 | 类型 | Acc | F1 |
| --- | --- | ---: | ---: |
| Direct MLLM | 直接多模态长上下文 | 43.05% | 43.63% |
| BM25 | Text RAG | 31.16% | 22.03% |
| ColBERTv2 | Text RAG | 30.56% | 20.43% |
| M3DocRAG | Page-level Visual RAG | 38.21% | 37.52% |
| EVisRAG | Page-level Visual RAG | 42.33% | 40.65% |
| G2-Reader | Graph/Agentic RAG | 46.96% | 45.29% |
| MLDocRAG | Graph/Agentic RAG | 47.90% | 未报告 |
| **MAGE-RAG** | Graph/Agentic RAG | **53.26%** | **51.19%** |

MAGE-RAG 的预算扫描中还出现过更高的单个配置 `k=10, T=5, a=3`：Acc **54.10%**、F1 **52.61%**；论文主表采用的是 `k=5, T=10, a=5` 的 **53.26/51.19**。比较时使用主表值，不能挑预算扫描中的最好点替代主结果。

原始 MMLongBench-Doc 页面所说的 GPT-4o **44.9% F1**、GPT-4V **30.5% F1**，是“整份 PDF 页面直接送入 LVLM”的原始论文结果。不同论文版本的 GPT-4V 表格还出现过 31.x 的数值更新，因此引用时应写明采用项目页 headline 的 30.5，而不要混合不同版本表格。

来源：

- MAGE-RAG 论文与 Table III：https://arxiv.org/abs/2606.15906
- MMLongBench-Doc 官方仓库：https://github.com/mayubo2333/MMLongBench-Doc
- MMLongBench-Doc 项目页：https://mayubo2333.github.io/MMLongBench-Doc/

## 4. 我们是否已经超过 MAGE-RAG / GPT-4o

**目前不能这样宣称。**

Dev 合并正式结果 53.9% Acc / 53.28% F1 在数字上略高于 MAGE-RAG 主表的 53.26% / 51.19%，但不能据此宣称超过 MAGE-RAG：我们只评了内部 165 题 Dev，其中 Diagnostic Dev 明确包含已审计困难文档；MAGE-RAG 报告的是完整公开基准，两者问题集合与答案抽取流程不同。

因此当前最稳妥的结论是：

1. CRAVE-RAG 在内部 Dev 上的正式结果为 **53.9% Acc / 53.28% F1**；
2. 该结果已经包含格式等价校准以及实际输出的 `Not answerable`，48% 的严格 direct-answer 诊断值不再代表项目准确率；
3. 这个结果说明系统已经达到可继续做 SFT 的 baseline 水平，但尚无证据证明超过 MAGE-RAG；
4. 内部 Test 124 题的 Prompt-only baseline 为 **50.0% Acc / 49.44% F1**；若要作论文级公开对比，仍需冻结最终系统并运行完整 1,091 题官方协议。

## 5. 评分复现边界

官方 MMLongBench-Doc 是三阶段评估：模型生成长回答、GPT-4o 抽取短答案、规则评分。CRAVE-RAG Answerer 已被约束输出短答案，本项目当前以经过内容等价校准的 Acc/F1 作为正式内部结果，并沿用官方 generalized F1 公式。论文或公开榜单对比时仍需明确披露未调用 GPT-4o extractor，并建议同时提供完整官方复现结果。正式公开报告至少应提供：

- 当前正式 Acc/F1 及内容等价判分规则；
- 完整官方三阶段 Acc/F1（完成复现后）；
- Answerable / Unanswerable、Single-page / Cross-page、Text / Table / Figure 分层结果；
- 检索召回、Evidence sufficiency、动作数、重复动作、延迟与成本。

## 6. 代码一致性快照

2026-09-10 核对结果：

- GitHub `origin/main`：`3582c7d272802d9dcbbf9ea6bbcb5de8e6adde5c`；
- RunPod `/workspace/crave-rag` tracked HEAD：同一提交；
- RunPod tracked code 与 GitHub 一致，无需更新；
- RunPod 有未跟踪的 `configs/training/mmlongbench_sft_split_v0_1/`，它是运行时拆分清单，不代表 tracked code 漂移；
- 本地工作树仍保留尚未提交的 split/scorer 文档与代码，未覆盖、未丢弃。

## 7. Test baseline 完成记录

2026-09-10，Test Core 与 Test Challenge 共 124 题完成内容等价评分：

- 总体：**62/124 = 50.0% Acc，49.44% generalized F1**；
- Gold answerable：99 题，其中 44 题答对；
- Gold `Not answerable`：25 题，其中 18 题正确输出 `Not answerable`；
- 系统最终状态：`ready=79`、`budget_exhausted=31`、`stopped_incomplete=14`；
- `ready` 中可回答题正确率：**44/79 = 55.70%**；可回答题召回：**44/99 = 44.44%**；
- 单轨迹平均耗时 132.0 秒，中位数 120.3 秒，P95 333.3 秒；4 worker 主批次墙钟吞吐约 **35.35 秒/题**。

主批次唯一程序失败 Q657 已在不改变 Prompt、模型、检索策略与预算的前提下恢复完成。修复内容是去重指向同一视觉载荷的 Table/TableView 输入，以及在视觉扫描前去重同一物理页的页码别名。主批次与 Q657 恢复轨迹分别持久化为：

- `.runlogs/test-cap12-w4-trajectories-20260910.tar.gz`；
- `.runlogs/test-q657-recovery-0d7503d-v3.tar.gz`。

本次内容等价评分对 38 个非精确匹配案例做过人工语义复核，因此该集合仍可作为当前 Prompt-only **内部 baseline**，但已不再是对开发者完全密封的盲测集。后续 SFT、Prompt 或架构选择不得参考 Test 逐题错误；checkpoint 只用 Clean Dev 选择。若需要新的最终盲测，应从尚未使用的 Reserve 文档冻结新集合。
