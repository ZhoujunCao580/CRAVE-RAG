# CRAVE-RAG 服务器验证与新 Baseline 计划

> 这是服务器实验的唯一正式清单，纳入 Git。旧 `.runlogs` 清单只保留指向本文的入口。
> 本文区分“本地代码已实现”“服务器组件验证通过”“端到端答案改善”，三者不得混写。

## 一、为什么不选择两个极端

不采用“每改一行就完整跑一道题”，因为很多改动位于检索、Reader、Checker 或状态机边界，
只看最终答案会被下游随机性遮蔽，也会重复支付模型费用。

也不采用“全部改完后直接跑 517 题”，因为一旦结果变化，我们无法判断究竟是视觉简述、
Table Reader、Recall、Coverage，还是某个结构 repair 产生了效果或回归。

固定采用以下三层验证：

1. **组件级因果门槛**：同一输入只检查被修改的那一层；能 A/B 的模块保留旧输出作对照。
2. **组合 Smoke Test**：所有已通过组件同时开启，跑一组跨模块题，检查接口和联动。
3. **受影响题集与新 Baseline**：先回放明确受影响的题集，再续跑预算实验，最后才跑新的
   517 题 development baseline。

组件级测试不要求临时回退整套代码。对于有开关的能力直接 A/B；没有开关的能力，使用旧
`frozen-baseline64-v0` 已保存输入/输出与新组件输出对照。所有新结果写入新目录，不覆盖旧轨迹。

## 二、冻结版本与不可缺失资产

服务器必须检出本计划对应的精确 Git commit，工作树必须干净。正式运行期间不再修改模型、
Prompt、split、检索配额或 action budget。

冻结接口：

- Planner：`planner-v0.22`
- Controller：`controller-policy-v0.12`
- Controller Input / Action：`controller-input-v0.4` / `controller-action-v0.3`
- Visual retrieval descriptor：`visual-retrieval-v0.1`
- Visual Reader：`visual-reader-v0.5`
- Multimodal Table Reader：`multimodal-table-reader-v0.2`
- Checker：`checker-v2.4`
- Coverage Checker：`coverage-checker-v0.1`
- Answerer：`answerer-v0.8`
- Reading Environment：`reading-environment-v0.6`

以下能力必须同时存在，任何一项缺失都不得开始组合 Smoke Test：

- 视觉候选 `search_summary + keywords` 的 Prompt、backend、descriptor provenance、缓存写入、
  SearchUnit metadata 与 CandidatePreview 接线；
- TablePreview、所有可解码 Table crop 的 visual-dense 表示、同 `element_id` 双路去重；
- 类型专用 Relation Preview，且模型不可见 `contains -> Page`；
- `READ_PAGE_CONTEXT(offset=-1/0/+1)` 与页面边界元数据；
- Multimodal Table Reader 及 confirmed cross-page table header inheritance；
- 稳定 Observation source/page identity；
- Checker provenance 派生、Root finalization、target-switch Evidence recheck、bounded Observation
  Recall、Root 状态确定性派生；
- 不可见 source ID 的同调用受控 repair；
- P10 Checker duplicate-replace / Observation-ID 受控 repair；
- Exact Anchor、Coverage scope/inventory、结构 COUNT 与 semantic COUNT；
- `budget_exhausted` / `stopped_incomplete` 的规范 `Not answerable` fallback，以及 step-7 checkpoint
  resume。

## 三、执行顺序

### 阶段 0：服务器与代码完整性预检

只检查，不跑题：

1. 记录 Git commit、`git status --short`、Python 路径、`PYTHONPATH`、模型路径、SoftDoc 路径、
   visual index 路径、descriptor/cache 路径和输出根目录。
2. 运行 `softdoc doctor --profile eval --json`，确认 `torch`、`transformers`、
   `sentence_transformers`、CUDA 和项目包来自预期环境。
3. 导出 Prompt manifest 和 JSON Schemas，核对上面的十个冻结版本；Coverage Checker schema
   必须存在。
4. 运行完整 CPU 测试。当前本地基线为 `507 passed`；服务器不得少测或跳过失败项。
5. 对 3 个真实文档构建检索服务，核对文本、Table、视觉资产数量及 visual index 模型
   `vidore/colSmol-500M`。不得复用与模型/config 不匹配的旧 index state。
6. 随机读取 20 个将实际展示的视觉候选：每个都必须有非空 `search_summary`、合法 provenance
   和可解码原图。若出现空 Preview、`figure candidate matched from visual content` 等占位文本，
   **立即停止**，先补 descriptor/cache 接线；不得继续跑 Agent。
7. 核对所有具有真实可解码 crop 的 Table 都进入 visual index，而不是只收录“单元格内有图”的
   Table；记录总数、成功数、跳过数和原因。

产物：`preflight.json`、Prompt manifest、schema 目录、完整 pytest 日志、视觉 descriptor 覆盖统计。

### 阶段 1：检索与 Preview 的组件级因果测试

这一阶段不调用 Planner/Controller/Checker，只比较候选、排名和 Preview。

#### 1A. 视觉简述与视觉候选 Preview

- 正例：`Q17, Q20, Q412, Q418, Q423, Q431, Q448, Q737, Q787, Q791, Q793, Q818, Q953`
- 完整性反例：`Q426, Q705`
- 比较旧占位 Preview 与新 `search_summary`；保存视觉 rank、最终批次、完整 CandidatePreview。
- 通过：正确视觉资产更可辨识，`Q787` 的摘要能表达 pier 等关键主题；反例不得因发现局部图片
  就宣称全文完整。

#### 1B. TablePreview

- `Q535`：列头、`URLs timedout`、`504` 必须同时可见。
- `Q658`：part/prefix 的行列边界可读，并与另一张 prefix 表区分。
- `Q1086`：年份/财务词重叠不能把缺少 operands 的表伪装成足够证据。
- 通过：新 Preview 提高正确表的可辨识性，且反例不恶化。

#### 1C. Table 双路检索与去重

- `Q24, Q535, Q658, Q716`。
- 所有有文本的 Table 进入 BM25/text-dense；所有有真实 crop 的 Table 进入 visual-dense。
- 相同 `element_id` 合并为一个候选，内部记录 `matched_by`，Controller 不看通道标签；若一路已占位，
  另一路跳过并补下一候选。
- CandidatePreview 优先 TablePreview；没有可用 TablePreview 时才使用 visual `search_summary`。

#### 1D. Relation Preview 与 Page 层级

- 全量扫描可见 Relation：非 Page 的可读另一端不得为空。
- Paragraph/Caption/Footnote 使用 query-centered Text Preview；Table 使用 TablePreview；视觉元素使用
  缓存 `search_summary`。
- `contains -> Page` 只保留为 SoftDoc 内部层级/provenance，不出现在 Controller 可见关系中。
- 用 `Q171, Q396` 核对 Relation 另一端与 Candidate 的 `source_id/type/page_id` 一致。

#### 1E. Exact Anchor

- 定向：`Q129, Q188, Q322, Q389, Q515, Q539, Q614, Q618, Q874, Q889, Q906`。
- 反例：`Q403` 的输出格式 `Chapter 1/3` 不得成为真实 anchor。
- `Q129` 的 `figure 1-4` 必须解析成四个 resolution；多匹配只能返回 anchor set，不创建永久
  compound entity。
- 每个结果必须明确为 unique / ambiguous / unresolved / range-too-large；不得混用物理页与印刷页。

### 阶段 2：Reader 组件级测试

这一阶段给 Reader 固定 source 和 local problem，不让 Controller 选择来源。

#### 2A. 页面上下文与 Visual Reader limitation feedback

- 主正例：`Q171, Q708`。
- 扩展：`Q16, Q201, Q389, Q391, Q474, Q603, Q604, Q790, Q801, Q803, Q856`。
- 边界：`Q539` 只记录错误子图/模型能力，不加单题规则；另取一个 crop 已足够的普通 Figure，
  确认不会无条件读取整页。
- 通过：crop 不足时 Reader limitation 进入 `recent_actions.feedback`；offset 0 同时提供当前整页与
  最近打开 crop；offset ±1 只读相邻物理页。Gold 页存在不等于来源正确。

#### 2B. Multimodal Table Reader 与跨页表

- 单页/Preview：`Q535, Q658, Q1086`。
- confirmed continuation：`Q716`（页 6–10）、`Q468`（160–161）、`Q479`（88–89）。
- 多页聚合：`Q372`（10–11）、`Q320`（30–32）、`Q328`（41–44）。
- confirmed/skipped 混合：`Q88, Q323`。
- 旧表格语义错例：`Q364, Q547, Q548`。
- 单位保真：`Q380` 只要求保存原文 `Rupees in lacs`；其 Gold 与 PDF 单位冲突，不作为最终答案门槛。
- 合成边界：HTML-only、image-only、text-only、双表示、清晰 HTML+模糊图、清晰图+损坏 HTML、
  多级/合并表头、双表示冲突、missing header、相邻无关表不得合并。
- 通过：原 aggregate HTML 留在 Provenance；仅在行归属唯一、顺序完整、无跨页 rowspan 时拆片并
  建 confirmed `continued_on`；后续片段继承 confirmed header。无法确认时不拆、不继承，Reader
  保存问题相关可见行并输出 `missing_header_context`，不得猜行列或单位。

#### 2C. 页面边界

- `Q579`：Reader 输入必须显示 `physical_page_number=20`、`document_page_count=20`、
  `is_last_page=true`。
- 加一个非末页反例，确认印刷页码和缺失 next relation 不会被误当“最后一页”。

#### 2D. 稳定来源与页码

- `Q114, Q775, Q378, Q844`。
- 单次 Reader 内可以用 `I1/I2`，写入 ObservationStore 前必须由 Environment 展开为稳定的
  `source_id/element_id/page_id/physical_page_number/display_page_label`。
- 两次调用都叫 `I1` 的不同图片不得被 Checker 当成同一来源。

### 阶段 3：Checker、状态机与 Controller 合法性

本阶段优先使用保存的 Reader Observation 和 checkpoint；不重新检索或重读。

#### 3A. Checker provenance（P07）

- `Q76, Q79, Q101, Q169, Q378, Q421, Q543, Q620, Q711, Q732, Q747, Q748, Q775, Q844, Q857, Q954`。
- 模型不输出 `used_for_evidence`；程序应用 Evidence delta 后从最终 Evidence 的
  `observation_ids` 反向派生。
- 通过：16 题不再出现 provenance mismatch；未引用 Observation 保持 false，不能偷偷进入 Evidence。

#### 3B. Root finalization 与 Root 状态派生

- 收尾回放：`Q957, Q24, Q10, Q647, Q649, Q76, Q600, Q380, Q34, Q839, Q841, Q451, Q457, Q29, Q119, Q611, Q726`。
- Root 状态矛盾：`Q15, Q840, Q1074`。
- 通过：最后一个子问题 satisfied 后自动切 Root，Root 原题成为初始 gap，同一个 Checker 执行
  state-only recheck；不新增 READ/Controller step。Checker 只输出 current target 状态，Root 状态
  由 Environment 派生。

#### 3C. Target-switch Evidence recheck 与 bounded Observation Recall

- 主正例：`Q611`。
- 真实回归：`Q374, Q375, Q728, Q838, Q951`。
- 合成边界：旧 Evidence 无关、部分支持、非法 reused Evidence ID、state-only 非空 delta、修改非当前 target。
- 通过：正常 READ 只判断当前 target；切 target 后先复核完整 EvidenceMemory，再最多召回 3 条去重的
  未接纳旧 Observation。两次均不伪造 Reader、不新增 action、不扣预算；误满足率必须为 0。

#### 3D. 不可见 source ID 的受控恢复

- 回放旧 55 道程序失败题，重点：`Q581, Q746, Q781, Q786, Q798, Q880`。
- 当前 Candidate/Anchor 才能 `READ_SOURCE`；Page 用 `READ_PAGE_CONTEXT`；Relation endpoint 用合法
  relation action；历史 ID 不是当前句柄。
- validator 拒绝后同一次 Controller 调用最多 repair 一次，不做模糊 ID 匹配，不增加 action/READ。
- 通过：报告首次合法率、一次 repair 后合法率、错误类别和剩余失败数；动作合法不等于答案正确。

#### 3E. P10 Checker 结构 repair

- `Q771`：同一 Evidence duplicate replace。
- `Q165`：Observation ID 多抄一个字符。
- 通过：同调用最多一次完整 JSON repair；合法 ID 必须从 allowlist 精确复制；保存第一次 raw output 与
  validator error；不重新 READ、不新增 action。P09 EOF 不进入该 repair。

#### 3F. P09 JSON 截断只复测、不预改

- Checker：`Q80, Q859`；Answerer：`Q327, Q642`。
- 先用当前 Prompt、token cap 和 vLLM JSON Schema 定向复测，并保存 `finish_reason/token_usage/raw output`。
- 只有仍发生 EOF 才决定紧凑输出、提高对应组件 token cap 或同调用受控重试；不得猜补残缺 JSON。

#### 3G. incomplete 终止输出

- 真实 Gold=Not answerable：`Q877, Q699, Q365, Q176, Q165, Q842, Q124, Q695, Q447, Q952, Q388, Q577, Q612, Q323`。
- 另用一个可 resume 的 budget checkpoint。
- 通过：最终 `budget_exhausted/stopped_incomplete` 产生规范
  `{"answer":"Not answerable","used_evidence_ids":[]}`，但内部失败状态和轨迹保留；resume 后若 ready，
  必须由真实 Answerer 结果替换 fallback。

### 阶段 4：Planner 与 Coverage

#### 4A. Planner semantic closure 与深度

- `Q7`：若分解，子答案必须足以恢复原比例、单位、分母和统计口径；不猜文档表达。
- `Q1067, Q1073`：先按报告指标名直接检索，只有 Reader/Checker 明确产生 operand gap 后才查 operands。
- `Q565`：保持全局 `max_depth=4`，保存 rejected draft、validator error、repair 和最终 DAG；以 `Q564`
  为同构对照，不因单题放宽全局深度。
- 加一个文档直接报告指标值的反例。通过标准是语义完整计划或保留 Root 不拆，而非强行生成 DAG。

#### 4B. Coverage 页码解析与纯结构 COUNT

- `Q705`：全文 Figure，预期 canonical count 12。
- `Q874`：Pages 5–10 Table，预期 5。
- `Q906`：Pages 400–640 超出 151 页，必须 blocked，不能使用部分范围。
- 检查 printed label 与 physical order 两套完整候选；一次范围只能选一个命名空间。人工判断错误时只用
  per-question namespace override 重算，不改 SoftDoc/embedding/旧轨迹。

#### 4C. Semantic COUNT 与完整性证明

- 主例：`Q913`。物理页 18–19 的 5 个 canonical Figure 必须全部进入 Reader/Coverage Checker；
  Environment 只能在 5 项均 resolved 后聚合 Gold 数 9。
- 当前 COUNT 回归集：`Q32, Q89, Q320, Q422, Q452, Q580, Q646, Q721, Q739, Q774, Q802, Q879`。
- Coverage Checker 只能逐 item 输出 matched/not_matched/unresolved 和 grounded Observation IDs；不能输出
  总数、READY 或 complete。一个 unreadable crop 必须使计划 blocked；修复后 resume 只重试 unresolved。
- confirmed 跨页 Table 作为一个 logical inventory item 去重。

#### 4D. 非 COUNT Coverage（COUNT 通过后再做）

- `Q99` collect_all+latest；`Q394` top_k；`Q564` argmax；`Q599` collect_all；`Q644` argmax；
  `Q709` collect_all；`Q710` complete mapping；`Q724` collect_all；`Q804` unique/argmax-like。
- 这 9 题不得使用普通 SEARCH 的局部命中冒充完整性。先冻结 operator 的终止证明，再做组件测试和
  组合回归。

### 阶段 5：组合 Smoke Test

前四阶段全部通过后，固定同一 commit、Prompt、模型、检索配置和 action budget，跑：

`Q7, Q10, Q11, Q24, Q80, Q114, Q165, Q171, Q327, Q372, Q375, Q393, Q476, Q516, Q535, Q565, Q579, Q611, Q642, Q705, Q771, Q786, Q787, Q857, Q859, Q874, Q913`

这些题同时覆盖 Planner、视觉简述、TablePreview、跨页表、页面上下文、稳定来源、Recall、Root 收尾、
ID repair、P09/P10、结构 COUNT 与 semantic COUNT。

每题必须保存 Planner、Controller、Reader/Table Reader、Checker/Coverage Checker、Answerer 的原始输入
输出、候选批次、动作轨迹、Evidence delta、状态变化、耗时、token、峰值显存和所有 repair attempt。
只要 schema、provenance、视觉 Preview 或 checkpoint 写入有一项缺失，就不进入下一阶段。

### 阶段 6：受影响题集回放

组合 Smoke 通过后再批量回放，题目实际执行时去重，但报告可同时归入多个机制：

1. 旧 55 道不可见 ID 失败全集。
2. P07 的 16 道 provenance 全集。
3. 旧 14 道 Checker-rejected：`Q775, Q366, Q375, Q378, Q1059, Q1070, Q1071, Q1074, Q163, Q837, Q844, Q338, Q396, Q859`。
4. 旧 5 道正常 READ 但无状态收益：`Q114, Q476, Q516, Q393, Q11`。
5. 12 道 COUNT 集与 P09/P10 定向题。
6. 所有阶段 1–2 的视觉、Relation、Page-context、Table 与跨页表定向题。

每个机制分别报告：输入是否到位、组件输出是否合法、状态是否推进、最终答案是否变化、额外调用和耗时。
不得只给一个混合准确率。

### 阶段 7：预算边际收益

1. 只读加载 199 道旧 `budget_exhausted` 的 step-7 checkpoint；每题只续跑一次到总 step 12，
   不重问前 7 步、不覆盖旧结果。
2. 记录首次 ready 在 step 8/9/10/11/12/never 的分布、正确数、额外耗时、重复动作和失败漏斗。
3. 重点核验末步才首次暴露未读 Gold 页的 12 题：
   `Q113, Q433, Q597, Q663, Q705, Q737, Q791, Q956, Q1057, Q1072, Q1075, Q1084`。
4. 用边际收益曲线决定新 action budget；在结果出来前不预设 12 为正式默认值。

### 阶段 8：冻结并运行新 development baseline

只有阶段 0–7 的硬门槛通过后才运行：

1. 再次记录 commit、Prompt manifest、模型、索引、descriptor cache、配置哈希与数据清单。
2. 对 517 道 `development_pool_v1` 运行一次全量新 baseline；它不是未见 Test，允许用于本轮工程诊断。
3. 报告 answer accuracy、answer coverage、Evidence success、合法 action/JSON 比例、平均 action/SEARCH/READ、
   Visual/Table/Checker 调用数、延迟、GPU 成本和各终止原因。
4. 与旧 frozen baseline 比较时，同时给出总体结果和按模块受影响题集结果；不得只展示挑选后的提升题。
5. 新 baseline 完成前，不开始 SFT；最终 Test 必须从未参与本轮逐题分析的数据中另行冻结。

## 四、当前暂不扩展的事项

- `Q820, Q821`：Controller 把人数改写成百分比，留作后续 Controller SFT 困难样本。
- `Q77, Q326, Q539, Q634, Q736, Q876`：已逐题归因，不为当前 baseline 堆细碎 Prompt 规则。
- `Q545, Q639, Q1051`：代码表达式、类别层级和财务公式推理，留作 SFT/Answerer 能力回归。
- Source Recall：只有旧 Reader 根本未抽取后续 target 事实时才可能需要；当前只启用 bounded Observation Recall。
- `INSPECT_REGION`/zoom、动态 Planner、RL/DPO 和新的计数动作不进入本轮 baseline。

## 五、完成判定

一个模块只有满足以下条件才标为“服务器验证通过”：

1. 正例得到预期组件输出；
2. 反例没有出现对应误行为；
3. 原始输入输出和验证错误完整持久化；
4. 没有改变实验规则或覆盖旧结果；
5. 组合 Smoke 中接口仍然成立。

最终答案变对是更高一级结果，但不能代替上述组件门槛；反过来，组件通过也不能冒充准确率已经提升。
