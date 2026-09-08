# CRAVE-RAG 阶段 1–7 服务器实验报告

> 日期：2026-09-09  
> 最终代码：`b3dfe59`  
> 范围：视觉/表格 Preview、Reader、Checker/状态机、Planner/Coverage、组合回放及 step-12 预算续跑。  
> 不包含：阶段 8 的 517 题新 development baseline，也不据此宣称新的总体答案准确率。

## 一、先给结论

阶段 1–7 已全部执行到终态，所有输出均已持久化，但系统还没有达到“可直接启动阶段 8 全量 baseline”的门槛。

最重要的结论有四个：

1. **工程稳定性明显改善。**受影响题集 171/171 最终均完整落盘，原主批次的 8 个程序失败及 5 个漏跑案例均由独立补跑补齐；最终没有程序失败和缺失产物。最新完整回归为 **522 passed**。
2. **正式组合流程仍有明显功能缺口。**171 题中仅 49 题到达 `ready`，52 题 `budget_exhausted`，70 题 `stopped_incomplete`。这是故障导向题集的运行终态，不是答案准确率，但足以说明阶段 5 的组合门槛未通过。
3. **视觉简述没有在正式运行时按设计完整接入。**171 题共有 482 个视觉候选展示记录（每题内按 `element_id` 去重），其中 156 个仍是占位 Preview，影响 57 题。85 条 descriptor cache 只能证明简述链路可用，不能代表正式运行已覆盖。
4. **单纯增加动作预算只能救回少数题。**199 道旧 `budget_exhausted` 续跑到 step 12 后，25 题首次 READY，174 题仍未 READY；说明预算不足确实存在，但不是当前主要问题的完整解法。

因此，本轮正确结论是：**阶段 1–7 的实验已经做完，代码和产物可审计；Stage 8 暂缓，下一轮先修正式视觉简述接线和 Coverage 触发/执行，再用小型组合集复测。**

## 二、冻结环境与产物完整性

- GPU：RunPod A100 80GB；常驻 vLLM；模型 `Qwen3.5-27B`；最大并发序列 4。
- Python：`/opt/crave-venv`；持久补充依赖位于 `/workspace/envs/visual-retrieval-packages`。
- SoftDoc：135 份，位于 `/workspace/data/mmlongbench-doc/softdocs`。
- 文本 Dense：`/workspace/models/multilingual-e5-small`；缓存位于 `/workspace/cache/dense-embeddings`。
- 视觉索引：`vidore/colSmol-500M`；132 份可索引文档、7,916 个视觉元素、7,544 张去重图片。
- 视觉简述缓存：85 条，位于 `/workspace/cache/visual-descriptors/qwen35-27b-v0-1/descriptors.jsonl`。
- Stage 0 在跨平台图片路径修复后为 508/508；本轮全部新修复合并后的最终回归为 **522/522**。
- Stage 3–6 的 171 题均有完整要求产物；最终 `cases_with_missing_artifacts=0`。
- 服务器仓库最终为 `main@b3dfe59`，与 GitHub 一致且工作区干净。

## 三、阶段 1：检索与 Candidate Preview

### 1A 视觉简述：部分通过

已确认：

- 修复了 SoftDoc 中 Windows 风格 `assets\...` 路径在 Linux 上无法读取图片的问题；这是之前视觉索引可用、VLM 简述却缺失的直接原因。
- 修复后 descriptor cache 从 6 条增长到 85 条；`Q787` 的视觉摘要能够明确描述地图中的 `piers`，证明图片输入、结构化输出、缓存及 provenance 链路本身可用。
- ColSmol 检索本身可工作；问题不是“没有视觉向量索引”。

未通过点：

- 正式 171 题组合运行共有 482 个视觉候选展示记录（每题内去重），其中 **156/482（32.4%）**仍是 `figure/chart candidate matched from visual content` 一类占位文本，涉及 **57/171（33.3%）**题。
- 阶段 5 的 27 题中也有 21/70 个视觉候选为占位文本，涉及 7 题。
- `Q20` 没有元素级视觉 crop；`Q16/Q604` 连初始 crop 都不存在，无法进入当前页面上下文补救路线。

下一次应验证的正式设计不是全量描述 7,916 个元素，而是：**先冻结当轮 3 文本 + 2 视觉候选，再只对两个真正展示的视觉槽位查缓存/生成简述，按 `element_id` 或内容哈希缓存，然后更新 Preview；生成简述不得反过来改变本轮候选排名。**

### 1B TablePreview：部分通过

- `Q535` 能展示列头、命中行、相邻行和关键错误码。
- 已加入确定性的结构对齐检查；结构不可靠时显示 `Alignment: uncertain` 与原始可见内容，不再伪造精确行列映射。
- `Q658` 仍暴露一个明确缺口：Table Reader 输出 `missing_header_context` 时只生成了 `description`，漏掉契约要求的 `relevant_visible_content`。当前 JSON Schema 无法表达该跨字段条件。
- `Q1086` 继续作为“检索命中不等于 operands 足够”的反例保留。

### 1C Table 双路检索：通过

- 1,688/1,688 张有可解码 crop 的 Table 已进入视觉索引。
- 有文本的 Table 可从 BM25/text-dense 召回；有 crop 的 Table 可从 visual-dense 召回。
- 同一 Table 按 `element_id` 去重，内部保留命中通道，Controller 只看到一个候选。
- 有可靠 TablePreview 时优先展示 TablePreview；没有时才回退到视觉简述。

### 1D Relation Preview：通过，但 Reader limitation 仍需训练/约束

- 扫描 135 份文档；22,745 个无法形成可理解 Preview 的非 Page endpoint 被隐藏，Controller 可见空 endpoint 为 0。
- `contains -> Page` 只保留为内部层级/provenance，不暴露给 Controller。
- `Q444/Q837/Q838`：3/3 选择关系导航，3/3 得到非空另一端 Preview，3/3 完成视觉读取。
- Caption、Footnote 和附近文本作为确定性导航上下文，不因展示 Relation 额外调用 VLM。
- `Q838` 的 Reader 没明确报告目标年份不在图中；这属于 Reader limitation 表达问题，不是 Relation 断链。

### 1E Exact Anchor：组件通过

- 12 个定向案例得到 unique 13、ambiguous 4、unresolved 1。
- `Q906` 正确覆盖 range-too-large/no-anchor；`Q403` 没把答案格式中的 `Chapter 1/3` 当成真实 anchor。
- `Q129` 的 Figure 1–4 被展开为独立 resolution，没有创建过拟合的永久 compound entity。

## 四、阶段 2：Reader：部分通过

- **2A 页面上下文：**19 条记录中 17 completed、2 skipped。`Q16/Q604` 因没有初始 crop，现有“crop 不足 -> limitation -> READ_PAGE_CONTEXT”链路无法起步。
- **2B Multimodal Table Reader：**30 次调用中 22 completed、6 failed、2 skipped。重复 `StoredObservation.sources` 的通用问题已修复；`Q468/Q364` 在后续 P09 组合复测中均能完整运行，但 `Q658` 的 limitation 跨字段约束仍未解决。
- **2C 页面边界：**`Q579` 的末页与非末页两个案例均通过；物理页、总页数和 `is_last_page` 正确进入 Reader 输入。
- **2D 稳定来源：**`Q114/Q378/Q775/Q844` 共 8 次调用全部完成；局部 `I1/I2` 在写入 ObservationStore 前被 Environment 展开为稳定的 source/element/page/physical-page identity。

## 五、阶段 3：Checker、状态机与动作合法性：部分通过

### 已通过或没有再次出现程序失败的部分

- P07：模型不再填写 `used_for_evidence`；程序从最终 Evidence 的 `observation_ids` 反向派生。
- 终止 fallback：`Q877/Q1004` 都以内部 `stopped_incomplete` 保留轨迹，同时输出规范 `Not answerable`。
- P09 定向 6 题全部完成，`Q468` 触发过一次 Checker 同调用 repair；没有留下未处理 JSON EOF。
- 不可见 ID 代表集 6/6 完整运行；本轮没有实际触发 Controller repair，因此只能证明正常路径兼容，不能据此宣称恢复路径效果率。
- P10 的 `Q771/Q165` 均未再次触发原结构错误；修复机制有本地确定性测试，但本轮没有真实 repair 样本可用于衡量模型侧效果。

### 尚未达到语义验收门槛的部分

- Root finalization 4 题中 0 题 READY，且 0 次 state-only Checker；这些从头运行的题没有到达原计划要隔离验证的“最后子问题已满足”状态，因此实验没有真正证明 Root 收尾语义已经闭环。
- Target-switch 3 题中只出现 1 次 state-only Checker，0 条 recalled Observation、0 个 reused Evidence ID；扩大到 Stage 6 的旧 Checker-rejected 14 题后，共出现 14 次 state-only、召回 3 条旧 Observation，但仍无 Evidence reuse。说明机制已接线，但正向复用还未被真实样本证明。
- 全部 171 题一共发生 37 次 Checker repair，说明“同一次模型调用内修复、不新增 Controller action”确实在工作；但 Controller/Answerer repair 本轮均未触发。
- Stage 7 的 `Q338` 首次仍出现 Controller JSON EOF，独立同配置补跑成功。P09 目前是可恢复的随机故障，还不能称根因消失。

## 六、阶段 4：Planner 与 Coverage：未通过完整门槛

### 4A Planner

- 5/5 题最终均完整落盘，没有 Planner 深度异常继续炸批次。
- `Q565` 在两次深度修复仍失败时正确回退为未分解 Root，全局 `max_depth=4` 没被单题放宽。
- 但 5 题中 0 题 READY；`Q7/Q1067/Q1073` 均 budget exhausted，尚不能证明比例口径和 ROA 类指标的语义拆分有效。

### 4B 结构 Coverage

- `Q705` 通过：全文件 scope 确定性解析，12 个 canonical Figure 枚举完整，结构计数 12，1 个动作后 READY。
- `Q906` 通过边界验证：Pages 400–640 在 151 页文档中被确定性判为 unresolved/blocked，没有浪费 Controller 搜索，最终规范 Not answerable。
- `Q874` 未通过：Pages 5–10 被正确解析为印刷页 5–10，但 Planner/requirement 把 `source_type` 生成为 `page`，没有执行目标 Table 的结构计数，最终 incomplete。

### 4C 语义 COUNT

- 13 题终态为 4 ready、1 budget exhausted、8 stopped incomplete；只有 8/13 建立 Coverage plan。
- `Q913` 没进入语义检查：`the figures in Pages 18-19` 被 scope parser 判为 `unsupported_scope_expression`，inventory 直接 blocked。
- 这说明 Coverage Checker 本体存在，但“问题 -> coverage requirement -> scope -> inventory -> 逐项语义检查 -> completion”还没有稳定贯通。

### 4D 非 COUNT Coverage

- 9 题终态为 3 ready、2 budget exhausted、4 stopped incomplete；只有 3/9 建立 Coverage plan。
- collect-all、top-k、argmax、complete-mapping 目前不能被认为已通过。

## 七、阶段 5：27 题组合 Smoke——门槛失败

- 27/27 均程序成功并完整落盘，缺失要求产物为 0。
- 终态：**3 ready、9 budget exhausted、15 stopped incomplete**。
- P50 34.2 秒，P90 164.4 秒，最慢 1,032.2 秒；峰值显存 67,832 MiB。
- 70 个视觉候选中 21 个仍是占位 Preview，影响 7/27 题。
- 仅触发 1 次 state-only Checker、0 次 Recall、0 次 Evidence reuse。

所以阶段 5 只通过“程序能跑完并保存审计产物”，没有通过“新模块在组合流程中可靠生效”的门槛。

## 八、阶段 6：171 道受影响题集回放

主批次为 166 题：158 成功、8 个程序失败。独立补跑包含这 8 题和旧清单漏掉的 `Q88/Q328/Q468/Q716/Q906`，13/13 成功。按 QID 合并后：

- **171/171 程序成功，0 最终失败，0 缺失产物。**
- 终态：49 ready（28.7%）、52 budget exhausted（30.4%）、70 stopped incomplete（40.9%）。
- 平均 75.2 秒，P50 49.4 秒，P90 147.0 秒，最大 1,032.2 秒。
- 共 560 个 Controller action、24 次 state-only Checker、4 条 recalled Observation、0 个 reused Evidence ID、37 次 Checker repair、63 个带 Coverage plan 的案例。
- 482 个视觉候选中 156 个占位，影响 57 题。

这证明批处理、持久化和程序异常恢复已形成闭环，但不证明答案质量；这些是故障导向的重叠题组，不能把 49/171 当作 MMLongBench-Doc 准确率。

## 九、阶段 7：step 7 -> step 12 预算边际收益

199 道旧 `budget_exhausted` 均从保存的 step-7 状态恢复，没有重问前 7 步，也没有覆盖旧输出。4 个主批次程序失败经独立补跑全部补齐。

| 首次 READY step | 新增题数 | 累计救回 | 累计占 199 题 |
| --- | ---: | ---: | ---: |
| 8 | 11 | 11 | 5.5% |
| 9 | 7 | 18 | 9.0% |
| 10 | 1 | 19 | 9.5% |
| 11 | 5 | 24 | 12.1% |
| 12 | 1 | 25 | 12.6% |
| 到 12 仍未 READY | 174 | — | 87.4% |

最终终态为 25 ready、169 budget exhausted、5 stopped incomplete。每题新增运行平均 58.5 秒，P50 47.8 秒，P90 117.9 秒，最大 351.4 秒。

解释：

- 预算确实过短，step 8–9 已救回 18 题；最后一步 SEARCH 后缺 READ 槽位的情况是真问题。
- 但把上限从 7 提到 12 也只多救回 12.6%，无法解决视觉 Preview、Coverage、Reader/Checker 或策略失败。
- 若只看成本，step 9 是明显拐点；若愿意为另外 6 题付费，可考虑 step 11。step 12 只新增 1 题，不值得在本轮直接设为默认值。

## 十、本轮发现并修复的程序问题

1. Linux 无法解析 Windows 风格视觉资源路径：已修复并加入回归测试。
2. 旧 checkpoint 的 SearchUnit ID 随索引表示升级变化：按相同 `element_id` 恢复当前 SearchUnit，保留 cursor/shown/opened 状态。
3. Reader 在同一 Observation 重复引用 `input_id`：写入前稳定去重，不丢 Observation。
4. Coverage Page 字段名不一致：统一为 `physical_page_number`。
5. Planner 连续超深：两次修复失败后安全回退到未分解 Root，不放宽全局深度。
6. Answerer 给出实质答案但漏 Evidence ID：同一次 Answerer 调用最多 repair 一次，不增加 READ/action。
7. Controller repair 审计记录缺失：保存首次 raw output、validator error 和最终 raw output。
8. Stage 7 runner 原先只能一次恢复全部 199 题：新增显式 `--allow-case-subset`，默认仍保持全量安全检查；只有独立非覆盖恢复任务才能选择失败子集。

## 十一、进入阶段 8 前的优先问题

### P0：必须先解决

1. **正式运行时按需视觉简述。**冻结当轮候选后，为真正展示的两个视觉候选查缓存/生成描述，消灭占位 Preview；先复测 Stage 5 的 7 个受影响案例。
2. **Coverage 触发与执行闭环。**优先修 `Q913` 的范围表达解析、`Q874` 的 Table source type，再用 4B/4C/4D 原题确认 requirement、inventory、逐项判断和 completion 全部真实发生。

### P1：紧随其后验证

3. 用可直接恢复到 target-switch/Root-finalization 的保存状态测试 3B/3C，不能再依赖从头运行“碰巧走到”该状态；必须观测到具体 Root gap、正向 Observation recall 或 Evidence reuse。
4. 修 `Q658` 的 discriminated limitation schema，并补无初始 crop 时的页面级入口。
5. 对 JSON EOF 保留 finish reason/token usage；一次同配置重跑成功不等于根因消失。

### P2：模型/训练阶段再处理

6. `Q7/Q1067/Q1073` 的 Planner 语义策略、Reader limitation 质量以及 Controller 候选选择可进入后续 Prompt/SFT 数据设计，但在 P0 修好前不宜用全量 baseline 混合诊断。

## 十二、持久化位置

服务器：

- `/workspace/runs/stage1abc-da4150d-20260908/`
- `/workspace/runs/stage1de-9169d99-20260908/`
- `/workspace/runs/relation-caption-navigation-v0-1/`
- `/workspace/runs/stage2abcd-9169d99-20260908/`
- `/workspace/runs/stage3to6-integrated-958fda3-20260909/`
- `/workspace/runs/stage3to6-supplement-4ad0fbc-20260909/`
- `/workspace/runs/stage7-budget-resume-22a3fba-20260909/`
- `/workspace/runs/stage7-budget-resume-supplement-b3dfe59-20260909/`
- `/workspace/runs/stage1-to7-final-summary-20260909.json`
- `/workspace/runs/stage1-to7-final-validation-b3dfe59-20260909/`

本地机器汇总、manifest 与测试日志：

- `.runlogs/server_environment/stage1_to7_20260909/`

## 十三、最终判定

- 阶段 1：部分通过。
- 阶段 2：部分通过。
- 阶段 3：部分通过，正向状态复用尚未证明。
- 阶段 4：Coverage 完整门槛未通过。
- 阶段 5：组合门槛未通过。
- 阶段 6：工程运行与产物完整性通过，模型行为门槛未通过。
- 阶段 7：实验完成；额外预算救回 25/199，不能单独解决主要失败。
- 阶段 8：本轮未运行，符合用户限定。
