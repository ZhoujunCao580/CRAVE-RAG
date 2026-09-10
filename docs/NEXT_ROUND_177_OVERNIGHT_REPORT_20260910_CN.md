# CRAVE-RAG 177 道困难集过夜实验报告

> 实验日期：2026-09-10  
> 最终代码：`3582c7d`  
> 模型：Qwen3.5-27B，vLLM OpenAI-compatible backend  
> 数据：`next_round_unresolved_177_v0_1`  
> 重要边界：这 177 道来自上一轮尚未解决的问题，是困难诊断集，不是完整 benchmark，也不能把本报告的准确率直接当成全量 Test 准确率。

## 1. 最终结论

177 道全部得到可评分的最终轨迹，最终程序失败为 0。语义评分忽略格式差异，并把 Gold 与预测均为 `Not answerable` 判为正确：

| 指标 | 结果 |
| --- | ---: |
| 总题数 | 177 |
| 语义正确 | 81 |
| 语义准确率 | **45.8%** |
| READY | 86 |
| READY 且正确 | 34 |
| READY precision | **39.5%** |
| `budget_exhausted` | 79 |
| `stopped_incomplete` | 12 |
| 最终程序失败 | **0** |

按 Gold 类型拆分：

| Gold 类型 | 正确 / 总数 | 准确率 |
| --- | ---: | ---: |
| 可回答 | 34 / 120 | **28.3%** |
| Not answerable | 47 / 57 | **82.5%** |

因此，本轮已经把环境、Schema 和恢复链路跑通，但还没有达到“177 道基本做对”的质量目标。当前主要问题已经从程序崩溃转为模型和任务语义：可回答题仍有大量检索/阅读未完成，且部分 READY 是错误地过早收尾。

## 2. 实际运行配置

- 固定每批 `3 text + 2 visual`；视觉简述只给 Controller 看，不参加 BM25、Dense 或 visual-dense 排名。
- 单模型 Qwen3.5-27B；Planner、Controller、Reader、Checker、Answerer 共用同一 vLLM 服务。
- 最大 16 个 Controller actions；Visual Scan、同调用 JSON repair、target-switch recheck 和 Observation Recall 不计入 action budget。
- 正式批次并发 4。
- Answerer thinking 保持开启；Planner 仅在四道持续截断的恢复批次关闭 thinking。
- 每题保存完整 Planner、Controller、Reader/Table Reader、Checker、Answerer、Visual Scan 输入输出及候选、动作、Observation、Evidence、耗时和 token/finish reason。

## 3. 运行分片与恢复

最终 177 条结果按以下优先级合并，任何恢复结果均保留来源，没有覆盖首次轨迹：

1. `next-round-177-a25e9e7-cap16-w2-20260910`：12 题，6 READY、6 budget。
2. `next-round-177-a25e9e7-cap16-w4-remaining-20260910`：165 题，首次完成 158、程序失败 7。
3. `next-round-177-2875270-cap16-recovery2-20260910`：修复 TableView 与不可见 ID 后恢复 Q80、Q341、Q562、Q1006。
4. `next-round-177-3582c7d-cap16-planner-off-recovery3-20260910`：Planner-only thinking off 后恢复 Q169、Q367、Q417、Q702。

7 道首次程序失败全部被恢复为合法轨迹：Q80、Q562、Q169、Q417、Q702 最终 READY；Q341、Q1006、Q367 最终为 `budget_exhausted`。因此最终统计没有把程序错误算作模型错误，也没有静默重跑已成功题目。

## 4. Action budget 的真实边际收益

86 道 READY 首次完成步数如下：

| 首次 READY 区间 | READY 数 | 语义正确 | 语义错误 |
| --- | ---: | ---: | ---: |
| step 1–7 | 71 | 28 | 43 |
| step 8–12 | 13 | 6 | 7 |
| step 13–16 | 2 | 0 | 2 |

结论：把 7 提高到 12 有真实收益，额外救回 13 道，其中 6 道真正答对；继续从 12 提高到 16 只多产生 2 个 READY，而且两题都答错。因此当前数据支持默认 cap=12，而不支持继续无条件增加到 16。预算只能解决“差最后几步”，不能修复错误检索、错误阅读、错误 Coverage 聚合或过早 READY。

## 5. 本轮修复的通用程序问题

### 5.1 Linux/Windows 视觉资产路径

SoftDoc 中部分资产路径为 Windows 风格 `assets\\...`。Linux 会把反斜杠当普通字符，造成“视觉 embedding 可用，但 VLM 找不到原图”的半失效状态。现已统一跨平台路径解析并加入回归测试；完整测试为 **508/508**，后续代码继续增长后本地完整测试为 **570 passed**。

### 5.2 TableView 重读

Q80 暴露 `_read_input` 与 `_primary_handle` 不能正确处理 `table-view:` ID。已让 TableView 保留原表焦点并合法进入 multimodal Table Reader。Q80 不再崩溃，但最终回答仍把一个 benchmark 名称读错，说明程序链路已修复，语义读取尚未完全解决。

### 5.3 Controller 不可见 ID

Q341、Q562 会重复使用历史批次或不可见 source ID。现在同一 Controller 决策最多做三次受控修复；最后一次禁止所有 source-bearing action，只允许合法 SEARCH/STOP。修复不新增 action、不重新搜索、不污染已有输出。Q562 被恢复为 READY，Q341 不再崩溃但仍耗尽预算。

### 5.4 Planner JSON 截断

Q169、Q367、Q417、Q702 在 Planner thinking 开启时，即使 token cap 从 768 提到 4096 仍截断。Planner-only thinking off 后四题均生成完整合法规划；Q169/Q417/Q702 READY，Q367 budget。这个开关没有关闭 Reader/视觉/Controller/Checker/Answerer 的 thinking。

### 5.5 语义评分

新增 `scripts/score_semantic_answers.py`：先做确定性 Not-answerable、精确值和列表集合等价判断，再只对剩余样本调用 Qwen 语义 Judge。177 题 Judge 失败为 0。此评分是诊断口径，不替代论文最终冻结的官方 evaluator。

## 6. Q80、Q642、Q327 的结论

- **Q80**：Table Reader 不再发生原先的整表逐项输出/JSON 截断，TableView 也可正确读取；但语义答案仍把 Gold 的 `POPQA / MS MARCO / SST-2` 回答为 `NQ / PopQA / SST-2`。剩余问题是表格局部对齐或读取选择，不是 JSON 管道崩溃。
- **Q642**：当前两次 Checker 调用分别约 270/436 completion tokens，没有截断。保留逐 Observation assessment，因为它既说明 Observation 为什么被接纳/拒绝，也会作为 feedback 帮助 Controller 避免重复读取。只评估成功 Observation 会丢失最有价值的失败反馈；当前做法是 Evidence 仍只输出本轮 delta，不复述未变化旧 Evidence。
- **Q327**：Answerer thinking off 明显退化：thinking on 得到 `12`，off 得到错误列表，Gold 为 `42`。两边都错，但关闭 thinking 更差，所以正式运行保持 Answerer thinking on，不能把关 thinking 当作通用截断修复。

## 7. Visual Scan / Coverage 的真实表现

Visual Scan 在 47 道题上触发，语义正确 18、错误 29，准确率 **38.3%**。它已经能解决若干纯加法式范围计数：

- Q705：全文 Figure 数为 12；
- Q874：指定页范围内为 5；
- Q197：section 内为 6；
- Q3、Q39、Q113、Q200、Q429、Q430、Q459、Q597、Q614、Q618、Q775、Q816 等也正确。

但当前聚合逻辑把每页 `match_count` 直接相加，只适合“页面内对象可独立相加”的题。它无法可靠处理：

- 去重后的唯一类别或来源；
- 返回页码/标题列表而非数量；
- 同一对象跨页重复出现；
- `argmax/argmin`、条件选择和跨页比较；
- Gold 为 Not answerable，但 VLM 勉强给出 0 或一个猜测；
- section/页码范围本身不确定时的完整性证明。

典型错误包括 Q25 `19` 对 Gold `30`、Q566 `1` 对 `12`、Q1004 `4` 对 `24`，以及 Q169/Q199/Q609/Q662 将本应返回的唯一集合或页码范围聚合成错误答案。

本轮没有为这些题增加单题规则，也没有临时新增第二个 Agent。下一步应在两个通用方向中选择其一后再冻结：

1. 只让 Visual Scan 接管严格可加的 `count/collect-all`，其余回到普通 Controller；或
2. 仍由同一个 VLM 在所有 page findings 完成后做一次 question-shaped final reduction，完成去重、集合/页码输出和歧义保留。

在这个决策完成前，不应把 Visual Scan 的 READY 当作完整性已经证明。

## 8. 剩余失败漏斗

96 道语义错误已全部下载：

| 类别 | 数量 | 含义 |
| --- | ---: | --- |
| READY 但语义错误 | 52 | 已收尾，但证据不完整、Visual Scan 聚合错误、Reader/Checker/Answerer 语义错误或错误地回答 Gold NA |
| 可回答但 `budget_exhausted` | 36 | 16 步内仍未形成足够 Evidence |
| 可回答但 `stopped_incomplete` | 8 | Controller 主动认为当前路线不可继续 |

其中 READY 错误进一步分为：Visual Scan 相关 29、其他 Reader/Checker/Answerer 或问题理解错误 23。非 READY 的 44 道应按候选是否展示、是否 READ、Reader 是否产生关键 Observation、Checker 是否接纳和最后四步是否重复继续审计；不能仅通过加预算处理。

## 9. 本地产物与完整性

本地目录：`.runlogs/next_round_177/20260910/`

- `semantic_scores.json`：177 道最终评分记录；
- `unresolved_96_trajectories.tar.gz`：96 道未解决轨迹归档；
- `unresolved_96/cases/ready_wrong/`：52 道 READY 错答；
- `unresolved_96/cases/answerable_no_answer/`：44 道可答但无正确答案；
- `unresolved_96/unresolved_manifest.json`：每题来源、类别、Gold、预测、评分理由；
- `archive_verification.json`：归档校验。

归档 SHA-256：

`039098959ea0504fe72a5560957c840cce37f1d9ec980c1679b52b76b5129fc7`

本地解压后核对为 96 个题目目录，52 + 44 与 manifest 一致。服务器原始结果、恢复分片、语义评分和归档也均保存在 `/workspace/runs/` 的 Network Volume 上。

## 10. 下一步判断

本轮可以确认“系统能稳定跑完并保留可审计轨迹”，但不能确认“质量已经足够直接生成正式 Teacher 数据”。建议下一步先用这 96 道做失败漏斗：优先审计 52 道 READY 错答，因为它们会产生错误 Teacher 标签；随后解决 Visual Scan 的通用 reduction 语义，再对受影响题做独立 supplement。模型在契约正确、上下文完整时仍做错的案例，再进入 Controller/Checker SFT，而不是继续堆 Prompt 特例。
