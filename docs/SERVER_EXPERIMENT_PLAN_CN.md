# CRAVE-RAG 下一轮服务器验证清单

> 冻结日期：2026-09-10  
> 本文是下一轮唯一有效的服务器实验清单。所有正式输出使用新目录，历史结果不得覆盖。

> **执行状态（2026-09-10）：已完成。** 177 道全部形成最终轨迹，程序失败 0；语义正确 81/177（45.8%）。详细结果、恢复记录、预算曲线与失败桶见 `docs/NEXT_ROUND_177_OVERNIGHT_REPORT_20260910_CN.md`。本文件以下内容保留为实验设计与复现实录，实际偏离项以该报告为准。

## 1. 本轮边界

- 单模型保持 `Qwen3.5-27B`，通过 vLLM OpenAI-compatible 后端运行。
- 正式检索仍是固定配额：每批 3 个文本候选 + 2 个视觉候选；视觉简述只在候选批次冻结后生成并缓存，不参与 BM25/Dense/visual dense 排名。
- 旧 MinerU-inventory Coverage 与 Coverage Checker 保持断路。新计数路径只使用 Planner 的最小 `visual_scan`、Environment 的范围解析、全页 VLM 扫描和现有 Checker/Answerer。
- 允许修复通用程序错误、Schema/ID/去重/截断记录和 Visual Scan 实现；不得为单题写 Gold 特例、改检索大纲、增加新 Agent、修改冻结题集或扩大模型资源。
- 定向重跑写入 supplement 目录，不覆盖首次运行。报告必须区分首次结果与修复后结果。

## 2. 冻结运行配置

- Git commit：运行前记录。
- 输入集：`configs/evaluation/next_round_unresolved_177_v0_1.jsonl`，恰好 177 题。
- 计划 action budget 为 12；本轮为测量边际收益实际运行到 16。`VISUAL_SCAN`、同调用 repair、target-switch recheck 和 Observation Recall 均不消耗 Controller action。实测 step 8–12 新增 13 个 READY（6 个答对），step 13–16 仅新增 2 个 READY且均答错，因此后续默认回到 12。
- 推理并发：先 1 题完成组件 smoke，再用 2 题并发；不得在未测峰值显存前升到 4。
- Reader/Table Reader/视觉简述/Visual Scan 保留 thinking；Q327 的 A/B 显示 Answerer thinking off 退化，正式 177 保持 Answerer thinking on。四道 Planner 持续截断题仅在恢复批次使用 Planner-only thinking off。
- 每题保存 Planner、Controller、Reader/Table Reader、Visual Scan、Checker、Answerer 的原始输入输出，候选批次、动作轨迹、ObservationStore、Evidence delta、最终 EvidenceMemory、耗时、峰值显存、`finish_reason`、token usage 和同调用 repair。

## 3. 阶段 A：环境和接口预检

1. 记录 Pod、GPU、Network Volume、仓库 commit/工作树、模型与缓存路径。
2. 确认 `sentence_transformers` 来自 `/workspace/envs/visual-retrieval-packages`，`ninja` 和 vLLM 来自 `/opt/crave-venv`。
3. 确认 Dense 缓存、ColSmol 全 Table 视觉索引和 visual descriptor cache 可读。
4. 运行完整测试；阶段 0 为 `508/508`，本轮代码完成后的本地完整基线为 `570 passed`。
5. 用一条短 JSON 请求和一张真实图片验证 vLLM 的 JSON Schema 与多模态输入。

## 4. 阶段 B：定向组件与真实模型验证

### B1 — Q658：跨页 Table Reader limitation

- 使用 `multimodal-table-reader-v0.4`。
- 缺可靠表头时，limitation 必须同时包含 `code`、`description`、`input_ids` 和非空 `relevant_visible_content`；其中保留与问题相关的可见行值。
- Table Reader 一次最多 4 个 Observation，通常 1–3 个；相关行、时期或模型变体应合并，不得逐单元格复述整表。
- 通过 validator 后 limitation 必须进入 Controller feedback，不重新 READ、不消耗额外 action。

### B2 — Q366：target-switch Evidence recheck 与 Recall

- 切换 target 后，Checker 先基于完整 EvidenceMemory 做 state-only recheck。
- `satisfied` 必须带真实、可见的 `reused_evidence_ids`；旧 Evidence 不足时返回 `incomplete` 和具体 gap。
- 再以 Q375、Q611、Q838 检查 Observation Recall：只召回少量与新 target 相关的未接纳历史 Observation，不暴露完整 ObservationStore。
- recheck/Recall/同调用 repair 不得新增 Controller action 或重新 READ。

### B3 — Q819：可见 ID 与 Relation 动作

- 保存 Controller 当轮真正可见的 candidate IDs、confirmed/candidate relation 和合法动作。
- 当前批次可见来源可以 `READ_SOURCE`；仅通过 Relation 可达的另一端必须使用对应关系动作。
- 不可见 ID 只在同一次 Controller 调用内受控 repair，不重新搜索、不消耗新 action。

### B4 — JSON 截断与 thinking

- Q80：验证 Table Reader v0.4 的聚合规则是否避免读取整表后逐行输出导致的截断。
- Q642：保留 Checker 对新/Recall Observation 的评估，因为拒绝理由会反馈给 Controller；验证 Reader 聚合后 Checker 输入输出是否已足够短。Checker 只输出本轮 delta，不复述未变化 Evidence。
- Q327：Answerer thinking 开/关做 A/B；同时选至少 3 道已知普通文本/表格/视觉 READY 控制题。只有 Q327 截断改善且控制题语义答案不退化，才在 177 题中关闭 Answerer thinking。
- Q468、Q364 只作旧问题的负向复现控制，不再写成已确认 Table Reader 截断。
- 每次失败必须记录 `finish_reason`、prompt/completion tokens、token cap、raw output 和解析错误；不得猜补 JSON。

### B5 — question-directed Visual Scan

- Planner 只输出目标原问题/子问题和最小范围：`whole_document`、`pages(text)` 或 `section(anchor_text)`；不再输出 operator、item type、predicate 或 grouping rule。
- Environment 先确定页范围，再按每批 5 张完整页面图送给 VLM；问题在所有批次保持不变，输入 ID 在一次 scan 内全局唯一。
- 批结果保留匹配项、局部计数和 limitation。任何页无法判断时不得把它当 0；整个 scan 降级为 incomplete，并把 limitation 交给现有 Checker/Controller。
- `whole_document`：Q705；`pages` 与印刷页映射：Q874；`section` 起点和视觉终点：Q197。
- Q20 作为普通单 Figure 路径控制：不得因为 Visual Scan 接线而误触发全文扫描。
- 缺 Section anchor、页面图、VLM/Schema/Checker 失败时必须回到普通 Controller SEARCH/READ，不得终止整题。
- 不得产生旧 `COUNT_INVENTORY`、`INSPECT_COVERAGE_BATCH` 或 `coverage_checker_calls`。

### B6 — Q1073：gap 更新后的 SearchSession 选择

- READ/Relation/Page Context 后 gap 实质变化时，Controller 重新判断当前批次是否仍覆盖新 gap。
- 当前可见候选覆盖新 gap 时优先 READ；旧 query 不再覆盖新 gap 时使用 `SEARCH new`；只有旧 query 仍服务同一缺口时才用 `SEARCH next`。
- 保留每轮 gap、query、session ID、候选、feedback、SEARCH new/next 和 local_problem。

## 5. 阶段 C：组合 Smoke

运行：`Q80, Q327, Q658, Q366, Q375, Q819, Q1073, Q197, Q705, Q874, Q20`。

通过门槛：

- 无程序崩溃、不可见 ID、重复 Observation source、旧 Coverage 调用；
- 所有模型调用有输入输出或明确失败记录；
- Visual Scan 范围、批次与 Checker 接线可审计；
- Answerer thinking 决策有控制题依据；
- 语义答案人工核对，不只看 READY。

## 6. 阶段 D：冻结 177 题运行

1. 使用阶段 C 冻结的同一 commit、Prompt、Schema、模型、检索配置和 action budget，一次运行全部 177 题。
2. 运行中按增量轨迹审计错误漏斗：Planner、召回/候选、Controller、Reader、Checker、Answerer、Visual Scan、预算终止和程序异常。
3. 只允许通用小修复。修复后仅把受影响题写入独立 supplement；禁止静默覆盖第一次结果。
4. 首次运行与 supplement 合并时保留来源、commit、Prompt 版本和修复原因。

语义评分规则：

- 不因冗长、标点、大小写或列表外壳而判错；只要核心答案与 Gold 语义等价即正确。
- Gold 为 `Not answerable` 且系统也是 `Not answerable`，判正确。
- Gold 可答但系统无答案，或核心事实/数值/单位/集合错误，判错误。
- 同时报告：总语义准确率、answerable/Not-answerable 分层准确率、READY 精确率、有效回答覆盖率、各终止状态和程序失败数。

## 7. 阶段 E：失败下载与收尾

- 将所有语义错误、Gold 可答却无答案、程序异常，以及仍需人工判断的轨迹下载到本地 `.runlogs/next_round_177/`，按失败类型分目录。
- 生成中文报告：配置、每阶段输入/输出、修复记录、首次与 supplement 指标、逐桶原因、可解决项与留给 SFT 的模型能力问题。
- 确认服务器正式结果、日志、缓存和环境快照均在 `/workspace`；确认本地下载可读后再停止 Pod。

## 8. 暂缓问题

- Q7：等价比例证据与硬子问题门槛。
- Q1067：ROA/ROE 等派生指标的按需公式展开。
- 若 Controller/Checker 在契约正确、上下文充分时仍做出稳定语义错误，留给后续 SFT，不为单题堆规则。

## 9. 明确不做

- 不恢复旧 MinerU Coverage Checker。
- 不让视觉简述进入 BM25/Dense 或改变已冻结候选排名。
- 不把 gap 直接覆盖为 local_problem，也不为 Q7/Q1067 编写单题规则。
- 不把定向 smoke 的 READY 比例写成整体准确率。
