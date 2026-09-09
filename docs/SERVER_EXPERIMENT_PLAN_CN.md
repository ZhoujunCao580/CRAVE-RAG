# CRAVE-RAG 下一轮服务器验证清单

> 重写日期：2026-09-09
> 本文是唯一有效的服务器实验文档。旧的阶段报告、未解决问题报告和服务器改动清单已经删除。
> 历史轨迹、原始 JSON、日志、模型缓存、SoftDoc 与环境恢复说明不是实验计划，继续保留作为审计证据。

## 一、这轮实验的边界

本轮只验证已经实现或已经明确需要验证的通用机制，不运行完整 baseline，不修改数据划分，不开始 SFT，
也不为单题增加特例。每个实验使用独立输出目录，不覆盖历史轨迹。

旧 MinerU-inventory Coverage 已从当前运行链路断开，不再继续阶段 4B/4C/4D 或原 T5 实验。
Planner 不再输出 operator、source type、predicate 或 inventory；Environment 也不会调用旧 Coverage
动作或 Coverage Checker。下一版改为 question-directed VLM 范围阅读，须在新契约冻结后另行验证。

开始前只做必要预检：记录 Git commit、工作树状态、Python 环境、Prompt/Schema 版本、模型与缓存路径；
确认 Qwen3.5-27B 的 vLLM 服务能接收结构化 JSON 和图片输入。已有完整测试若代码提交未变化，不重复执行。

所有真实端到端 smoke 与 baseline（即使只有一道题）统一使用
`scripts/run_model_batch.py --runtime-profile crave-baseline-v1 --execution-mode persistent`。
该 Profile 会在首题前强制检查 vLLM、Dense、已完成的视觉索引、视觉简述缓存/按需生成和多模态 Table Reader，
任何模块遗漏均直接失败，不允许静默退化。`softdoc run-model` 只用于低层组件调试，不作为完整架构结果入口。

视觉简述的冻结边界：先由 BM25/Dense 与 visual dense 冻结当前 3 文本 + 2 视觉候选，再只为其中需要视觉
Preview 的候选读取/生成简述。简述仅替换本批次的 Controller `CandidatePreview`，不得写入 SearchUnit，
不得参与 BM25/Dense，也不得让已有缓存改变后续候选排名。

## 二、已确认的概念边界

### `gap` 与 `local_problem`

- `gap` 是 Checker 维护的持久缺口，跨 action 保留，Controller 每轮都能看到。
- `local_problem` 是 Controller 在一次 `READ_SOURCE` 或关系读取动作中，针对已选来源临时写给 Reader 的阅读任务。
- 当前系统本来就要求 Controller 根据最新 `gap` 生成 `local_problem`，不新增“自动复制”机制。
- 不允许用 `local_problem` 覆盖 `gap`。一次局部阅读指令可能过窄或错误，不能反向替代 Checker 对全局证据缺口的判断。
- `SEARCH new/next` 没有 `local_problem`。因此 gap 更新后若 Controller 一直执行旧 SearchSession 的
  `SEARCH next`，新的 local problem 根本不会产生；Q1073 属于这个问题。

## 三、下一次服务器必须执行的验证

### T1：Q658 跨页表缺表头 limitation

状态：代码与 Prompt 已修改，缺真实模型复测。

- 使用 `multimodal-table-reader-v0.4` 和 Q658 保存的真实 Table Reader 输入。
- 一次输出必须同时包含 limitation 的 `code`、`description`、`input_ids` 和非空
  `relevant_visible_content`。
- 输出必须通过 validator，并能进入 Controller feedback；不得静默重读或消耗新的 Controller action。
- 若找到兼容表头，联合重读后必须同时保留原片段的相关可见行与继承表头。

保存：原始输入输出、schema、finish reason、token usage、validator 结果。

### T2：Q366 target-switch Evidence recheck

状态：本地 repair 与状态机测试已通过，缺真实 Checker 验证。

- 切换 target 后，Checker 基于完整 EvidenceMemory 做 state-only recheck。
- 若判断新 target satisfied，必须返回真实且可见的 `reused_evidence_ids`；不得出现
  `satisfied + reused_evidence_ids=[]`。
- 若旧 Evidence 不足，应返回 `incomplete` 和具体 gap，不得由程序猜测复用关系。
- recheck/同调用 repair 不得增加 Controller action 或重新 READ。

扩展反例：Q375、Q611、Q838。无关旧 Evidence 不得被错误复用；Observation Recall 最多返回已配置的
少量相关历史 Observation。

保存：Checker 原始输入输出、repair 输入输出、Recall 选择、Evidence 变化和 action budget。

### T3：Q819 可见来源与 Relation 动作

状态：需要用真实轨迹确认故障归因，暂不预设一定是 Controller 错误。

- 保存 Controller 当轮实际可见的 candidate IDs、confirmed/candidate relations 和允许动作。
- 若目标是当前批次可见的来源，允许 `READ_SOURCE`。
- 若目标只通过 relation 另一端可达，必须使用与 relation 类型匹配的导航动作；不可把不可见端点 ID
  直接交给 `READ_SOURCE`。
- 若输出不可见 ID，只在同一次 Controller 调用中做受控 repair；不得重新搜索或消耗新 action。

通过标准：不再因不可见 ID 崩溃，同时不能把本来合法的直接 READ 强制改成关系导航。

### T4：结构化 JSON 截断诊断

状态：尚未决定修复策略，先复现并保存完整遥测。

- Checker：Q80、Q859。
- Answerer：Q327、Q642。
- Table Reader：Q468、Q364。
- Planner 深度边界：Q565 保存 rejected draft、validator error、repair 与最终 plan。

每次调用必须保存：raw output、finish reason、prompt/completion token usage、token cap、使用的 JSON Schema、
解析错误和同调用 repair 结果。只有证明确为长度截断后，才讨论缩短输出或调整该组件上限；不得猜补 JSON。

### T5：旧 Coverage 断路与 question-directed Visual Scan

状态：**本地接口、Prompt、validator 和 Section anchor 解析已完成；557/557 测试通过；服务器执行链路待接线验证。**

已冻结的本地设计：

- Planner 升级为 `planner-v0.25`。旧兼容字段仍必须输出 `coverage_requirement=null`，不得重新启用
  MinerU inventory、operator、source type、predicate 或 Coverage Checker。
- Root 与每个 SubQuestion 可独立输出最小 `visual_scan`：
  - `{"required":true,"scope":{"kind":"whole_document"}}`；
  - `{"required":true,"scope":{"kind":"pages","text":"Pages 5-10"}}`；
  - `{"required":true,"scope":{"kind":"section","anchor_text":"Academics and Related Resources"}}`。
- 完整 target question 继续表达要找什么；scope 只表达完整性边界，不重复输出 operator、item type、
  predicate、target description 或 grouping rule。
- 普通 Page/Pages/Slide 先使用现有 SoftDoc/PDF 的可见页码映射；能映射则按印刷页，不能映射则按物理页。
  Planner 不猜页码命名空间。
- Section 先只在真实 Heading 中解析起点。找不到时返回 `needs_controller_anchor` 和定位 gap，交给普通
  Controller SEARCH/READ；同名多页时返回 `ambiguous`。不得找不到章节就改扫全文。
- Section 终点不信任 MinerU heading level 直接决定；后续扫描VLM在正常读取页面时确认可见的下一主要章节，
  避免把目标章节内部被误标为同级的子标题当成终点。
- VLM JSON 输入只包含完整问题、scope、batch index/count 和全局唯一 `input_id`。实际页面图片作为多模态
  content 在 JSON 外按 ID 附加；不得伪造 `image`、`physical_page_number` 或 `visible_page_label` 字段。
- 20页分4批时 ID 必须连续保持 `I001-I020`，不能每批重置。结果引用当前批不可见 ID 时由 validator 拒绝。
- 每批输出 `items + partial_count + limitations`。任何页面/相关区域无法可靠判断时，`partial_count` 必须为
  null，不能把“没看清”算作零；已可靠发现的 items 仍须保留。
- 最终聚合输出 `answer_candidate + supporting_input_ids + scope_complete + limitation`，再进入现有 Checker
  与 Answerer，不绕开 EvidenceMemory。

本地真实数据验证：

- Q197 的 MinerU Heading 为粘连且带乱码的
  `Academics and RelatedResources...`，resolver 仍准确定位物理第23页、真实 `page_id/source_id` 和标题 bbox。
- 不存在的标题生成 Controller 定位 gap；不同页面的同名标题被判为 ambiguous。
- 单元/契约测试覆盖：whole document/pages/section schema、全局批次 ID、无伪 image 字段、不可见 ID、
  模糊页非零、partial count 一致性和完整/不完整聚合契约。
- 完整本地测试：`557 passed`。

服务器下一步只做执行接线和真实模型验证，不重新设计 schema：

1. 将 `visual_scan` 从 `InitialPlan` 连接到 ReadingEnvironment、页面批处理、vLLM多图输入、文本汇总、Checker。
2. 先用 Q197 验证 Section anchor、页面中部边界、批次持久化和 Checker 接纳。
3. 用 Q705 验证全文纯视觉计数；用 Q874 验证 Pages 5-10 的页码映射和表格计数。
4. 人为构造一个模糊页和一个截断批次，确认只重试失败页/批次，不覆盖成功结果、不错误输出0。
5. 普通题与上述题都不得生成 `coverage_checker_calls`、`COUNT_INVENTORY` 或
   `INSPECT_COVERAGE_BATCH`；旧 Coverage 继续保持断路。

保存：Planner原始输出、解析后的scope、input ID到page ID的内部映射、多模态请求、每批原始输出、validator、
汇总结果、Checker输入输出、耗时、token usage、重试和峰值显存。该定向测试不作为整体准确率。

### T6：Q1073 新 gap 与旧 SearchSession

状态：通用规则已写入 `controller-policy-v0.13`，缺服务器真实模型验证；本轮不增加 ControllerInput 字段或
SearchSession 状态迁移。

已知失败链路：Reader/Checker 已把 gap 从“找直接报告的 after-tax return on average equity”更新为“寻找
net earnings 与 average equity operands”，但 Controller 随后连续对旧查询执行 `SEARCH next`。

验证目标：

- READ/Relation/Page Context 的反馈使 gap 实质改变后，Controller 必须先按新 gap 重新判断当前仍可见候选。
- 当前批次若有明确覆盖新缺口的候选，优先 READ；Q1073 首批的 Shareholders' Equity 表是核心检查点。
- 当前批次没有合适候选、且旧 query 不覆盖新缺失事实时，Controller 应 `SEARCH new`，而不是仅因旧
  session `has_more=true` 就继续翻页。
- 只有旧 query 仍在追求与 current gap 相同的未解决信息时才允许 `SEARCH next`；`has_more=true` 本身
  不构成继续翻页的理由。
- 新查询命中来源后，Controller 才依据最新 gap 生成新的 `local_problem` 并 READ。

本次只依靠当前已存在的 `current_gap`、SearchSession query、当前 CandidatePreviews 和 recent feedback；
若真实模型仍反复翻页，再评估是否需要显式 `gap_revision/session_stale` 字段，不提前增加状态复杂度。

保存：每轮 gap、search query、session ID、CandidatePreviews、recent feedback、SEARCH new/next 选择和
local problem。

## 四、组合 Smoke Test

T1–T6 各自通过后，只运行一个小型组合集：

`Q658, Q366, Q819, Q565, Q838, Q1073, Q705`

每题保存 Planner、Controller、Reader/Table Reader、Checker/Coverage Checker、Answerer 原始输入输出，
以及候选批次、ActionTrace、ObservationStore、Evidence delta、最终 EvidenceMemory、finish reason、token usage
和逐组件耗时。

组合门槛：无程序崩溃；无不可见 ID；没有被误触发的不可解析 coverage；Q705 的结构计数不回归；
Q1073 在 operand gap 后不再盲翻旧查询。该集合只验证链路，不作为准确率。

## 五、已记录但本轮暂缓设计的问题

### D1：Q7 等价证据与硬子问题门槛

Q7 的“投票民主党人数 / 总人口”拆分在数学上成立，不能强制 Planner 猜文档一定用百分比表达。文档实际给出
“民主党占总人口 31%”和“民主党中投票者占 59%”，当前 Checker 因无法满足绝对人数子问题而拒绝保存，
Root 无法利用 `31% × 59%`。

后续候选设计：允许可靠的部分 Evidence 在 target 仍 incomplete 时保存，并在新 Evidence 可能直接闭合 Root
时做机会性 Root sufficiency recheck。该状态机改动尚未确定，本轮不测试、不改 Prompt。

### D2：Q1067 派生指标按需展开

Q1067 从旧版到当前 Planner 都没有拆分，直接搜索 `ADBE ROA FY2015`；文档没有直接报告 ROA，只给了
net income 和期初/期末 total assets，因此 Controller 一直搜索指标名而没有寻找 operands。

后续候选设计：先按指标原名检索；直接报告路径失败后，才触发一次受控 metric expansion，生成公式与所需
operands，再开始新查询。需要同时考虑 ROA/ROE/margin/growth 等指标及公式口径差异，暂不写硬编码规则。

### D3：语义 coverage 的延迟启用

Q565 说明语义 scope 不能直接交给确定性 Environment，但完全禁止子问题 coverage 也会损失合法能力。后续研究
在 Controller 已定位具体 Table/Section/Page set 后，如何把自然语言目标绑定为可枚举的运行时 scope。

### D4：预算与完整 baseline

当前清单通过前不重跑完整 baseline，也不继续扩大 action cap。覆盖误触发、旧查询翻页和 JSON 稳定性修好后，
再用冻结题集测预算边际收益；最终准确率必须来自独立完整运行，而不是上述定向 smoke。

## 六、明确不做

- 不把 `gap` 直接赋值为 `local_problem`。
- 不把 coverage 限制为只能出现在 Root。
- 不为 Q7 强制生成百分比子问题。
- 不为 Q1067 硬编码单题公式。
- 不覆盖或删除历史原始轨迹、日志、SoftDoc、视觉索引和 descriptor cache。
- 不把定向题集的 READY 比例写成系统准确率。
