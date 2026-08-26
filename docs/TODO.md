# TODO

## Evidence Checker：服务器阶段验证

- [ ] 在服务器上的正式模型重新评估 Checker v1；当前本地 `qwen3:8b` 只证明循环和 Schema 可以运行，不作为最终质量结论。
- [ ] 验证 `incomplete` 时偶尔漏写 `remaining_gap_description`，并采用一次带明确 Pydantic 错误的定向重试，而不是继续加长主 Prompt。
- [ ] 验证 `used_for_evidence=true` 但 `add/replace` 为空的协议错误，以及重复 Observation 应为未采用的边界。
- [ ] 验证模型是否会把问题中的公司、年份、单位或条件擅自补入信息不足的 Observation（如 `Net income was 14`）。
- [ ] 验证旧 Evidence 与新 Observation 的联合充分性、部分问题误判完成、Root 过早 `ready` 和明确冲突的 `replace` 能力。
- [ ] 对比至少一个更强模型；分别报告 Schema 合法率、delta 应用率、false-ready、冲突修正和两轮循环成功率。

只记录尚未完成、需要以后通过实验决定的事项。

## 1. PDF 与 SoftDoc 完整性

- [ ] 在更完整的数据集上检查 MinerU 漏检、错类型、错误裁剪和跨页内容遗漏。
- [ ] 复核当前 22 个上游 Table 类型误判候选；先统计对检索与回答的实际影响，再决定是否增加通用类型校验或更换解析后端。
- [ ] 区分 MinerU 上游问题与 SoftDoc Adapter 丢失；优先修复通用问题，不按单份文档堆规则。
- [ ] 验证当前 Pipeline、MinerU Hybrid 或视觉 fallback 是否值得加入正式流程。
- [ ] 处理复合 Table 的内部图片缺失：`<image N>`不等于有效`img src`，且可能完全没有图片提示；暂不新增检测模块。
- [ ] 跨页聚合表拆分目前只是 `representative-28` 上的暂定开发策略，不得默认推广到后续全部文档。应在完整数据集上比较“保留 MinerU 聚合表”和“按物理页拆分并生成 confirmed `continued_on`”对解析、检索、引用定位和回答的影响，再决定正式默认行为；在此之前，行归属、重复表头或跨页 `rowspan` 无法唯一验证时必须保留 MinerU 原结果，不得强拆。

## 2. Planner 对照

- [ ] 在 Reader、Evidence State 和 Controller 接通后，同条件比较：
  - No Planner；
  - Initial Planner；
  - Deferred Planner（阅读后根据真实 evidence gap 才允许更新计划）。
- [ ] 比较答案质量、证据完整性、动作数、模型调用次数和运行时间。

## 3. 检索策略与批次超参

- [ ] 比较三种候选策略：
  1. weighted RRF；
  2. 固定配额混合，例如每批 `3 BM25 + 2 Dense`；
  3. BM25-first，当前候选不能补齐证据时再启动 Dense。
- [ ] 调整并验证：
  - 每批预览数，如 `3 / 5 / 10`；
  - BM25/Dense 配额，如 `3+2 / 2+3 / 1+4`；
  - RRF 的 `k` 和 BM25/Dense 权重；
  - 何时取下一批、何时切换检索器。
- [ ] 不只看 Top-k recall，还要比较 Agent 找到充分证据所需的批次数、阅读数、延迟和成本。
- [ ] Controller预算不要最终固定为“动作步数”；闭环可运行后，比较token、Reader/VLM调用、图像输入、检索延迟和实际费用，并据此定义可配置的资源成本预算。搜索翻批、文本读取和视觉读取不得默认视为等成本。

## 4. 是否需要专门 Reader

- [ ] 先比较“Controller直接读取Element/TableView”和“专门Reader返回Observation”；只有后者明显改善大表、复杂视觉内容或成本时才实现Reader。
- [ ] Visual Reader v0 暂时统一读取整页和 Figure、Chart 等非 Table 视觉元素；以后再用质量、延迟和成本实验决定是否拆分 Page/Element Reader，以及是否为 Figure/Chart 设置专门Reader。
- [ ] 若实现Reader：Reader只报告本次读到的可靠Observation与limitations，不输出answer/conclusion；`continued_on`缺失、`candidate` Relation和文本/HTML读取失败后的视觉fallback，均交给Controller决策，不自动导航或调用VLM。
- [ ] 用答案质量、证据完整性、动作数、视觉调用和成本决定是否保留Reader。

### Controller 训练备忘（已定设计，不是待办）

- Controller主导选择Visual Reader输入。默认单图读取；能还原为独立事实的比较（如分别读取A、B数值）应分开读取，将事实写入Evidence，最后由Answerer比较。
- 只有无法安全拆成独立事实的视觉比较（如波动、形状、对应关系）以及跨页连续判断，才在一次请求中联合提供多张图，由Reader输出引用多个`input_id`的关系型Observation。
- 训练Controller时应学习“单图读取、补读另一图、联合视觉读取”三种决策；不要让Controller亲自解释图像或生成最终答案。
Visual reading 默认把可独立事实拆开单独读取；事实级比较由 Evidence/Answerer 汇总。只有当前判断本身依赖多个视觉输入间的视觉关系时，Controller 才发起 multi-image joint reading。何时进行单图或多图读取属于未来 Controller policy 的一部分。

## 5. 激进规则审计

- [ ] 在更完整数据集和真实QA轨迹上审计当前激进但仍有通用价值的规则；优先检查 `parser_declared_function_target`、`bounded_nearest_compatible_element` 和 `profile_forced_sibling_level`，再评估其余标题、Section、页码及视觉Caption规则。比较关系准确率、Section变化和最终QA收益后再决定保留、降为candidate或删除。

## 6. Observation Recall

- [ ] Controller闭环可运行后，比较“只使用新Observation”和“`current_target`变化时从未采用的旧Observation中召回少量候选并重新交给Checker”；只有后者改善证据完整性或减少重复阅读时才保留Recall。
- [ ] v0一次只评估当前问题；同一Observation若也能解决后续问题，可能发生重复读取。以后对比“后续问题重新读取”和“从ObservationStore召回并重新交给Checker”，用读取数、成本和错误率决定是否实现跨问题复用。

## 7. Answerer 服务器阶段验证

- [ ] 当Evidence只能证明“数值发生了变化”，但Root Question追问“为什么变化”时，本地`qwen3:8b`可能把变化本身循环表述为原因。首要防线是Checker不得将这类Evidence判为`ready`；Answerer仅保留问题覆盖检查作为兜底。以后用更强模型测量“缺少原因证据”的拒答率，暂不继续为单个错例扩写Prompt。

## 8. Evidence Checker 闭环

- [x] 收缩ID边界：v0以`action_id`串起一次Controller动作、对应`ReadRecord`、其Observation和可选Checker更新；删除独立`read_request_id/request_source_id/request_visual_id`。`EvidenceCheckResult`不新增全局`evidence_check_id`或第四个canonical store。多输入读取仅保留动作内局部`input_id`和稳定`source_id/visual_asset_id`。
- [x] 收缩limitation：它只描述Reader本轮没有可靠读出的内容，不评价Observation是否对问题有用。删除分类`code`，保留简短`description`及受影响的动作内`input_id`；Checker评价仍单独存在。
- [x] 实现最小`EvidenceCheckResult` delta：用`action_id`关联本轮，逐条返回`observation_id + used_for_evidence + assessment`、`add/replace/remove`、`current_target_status`、`root_status`及可选剩余gap。程序在Memory副本上应用、验证完整结果后原子提交；未提及的旧Evidence不会因模型漏抄而消失。
- [x] 固定普通循环：一个Root Question共用唯一ObservationStore与EvidenceMemory；Controller/Checker每轮聚焦一个current question。Evidence用`supports_question_ids`保留目标归属；当前子问题满足但Root仍不充分时切换或细化问题，只有`root_status=ready`才交给Answerer。
- [x] 持久化问题DAG运行状态：`EvidenceMemory.questions`保存全部已注册问题的文本、依赖和状态，`current_target`唯一表示当前问题与gap；Checker只更新当前项，程序按依赖和Planner稳定顺序选择下一项。ActionTrace只引用本步`question_id`，不再维护第二份current-question状态。

### 情况清单与处理

- [ ] **C1：没有Observation，只有读取失败或limitation。** 不调用Checker；在`ReadRecord/ActionTrace`保留结果，Controller看到失败后换候选、表示或读取范围，避免原样重试。
- [ ] **C2：Observation正确且补齐当前gap。** 提升为Evidence并把`current_target_status`设为`satisfied`；程序选择下一个依赖就绪的问题，或在Root充分时设为`ready`。
- [ ] **C3：Observation只有部分有用。** 原子Observation逐条评估，只提升可靠且相关的部分；`current_target`保持同一问题并描述仍缺的事实。
- [ ] **C4：Observation正确但无关、重复、对象/时间/范围不符。** 不提升，当前target与gap保持不变；Controller从本轮`EvidenceCheckResult`读取assessment，自行决定换候选、下一批、重新SEARCH、导航或换表示；不把评价复制到ActionTrace。
- [ ] **C5：Observation明显错误或受limitation影响，Checker能够识别。** 不提升，当前target与gap保持不变；错误读取仍留在append-only ObservationStore中供审计和防重复。
- [ ] **C6：Observation与已有Evidence冲突。** 不静默覆盖；EvidenceMemory保留可追溯来源，并把当前gap改写为需要消解的冲突，Controller再选择独立来源或另一表示核验。
- [ ] **C7：Observation看似合理但实际错误，当前没有冲突或额外信息。** 系统不能凭同一信息必然发现；依靠精确grounding、后续冲突、已有多表示的机会性交叉检查及答案关键事实的选择性复核，不默认全部读两遍。
- [ ] **C8：Checker进行集合级联动。** 每轮读取完整但精简的EvidenceMemory，判断多条Evidence联合后是否充分、重复或冲突；不读取全部历史Observation，`ready`只表示证据集合足以回答，不表示答案已经生成。
- [ ] **C9：候选、导航路线或动作预算耗尽但gap仍未解决。** 不得伪装成`ready`；在Reading Session/Answerer边界记录“证据不足而停止”，避免Controller无限循环。闭环后再决定是否扩展`EvidenceStatus`。
- [ ] 闭环后做错误注入与反馈消融：注入错误数值、错误对象/范围、无关Observation、冲突和不可读结果；比较“只给status/gap”“给简短`EvidenceCheckResult`评价”“给自由文本长reason”在错误恢复率、false-ready率、重复动作、答案质量、动作数和成本上的差异。

### 四项主要风险与预定处理

- [ ] **R1：坏Observation被Checker错误提升为Evidence。** 保留Observation到精确source的grounding和Evidence到Observation的引用；使用原子Observation、同轮limitations、已有Evidence一致性检查和高风险事实的选择性复核；通过错误值、错误对象和错误范围注入测量错误恢复率。没有任何额外信息且错误内容自身完全合理时，不宣称系统能够必然识别。
- [ ] **R2：Checker过早给出`ready`。** `ready`必须由完整Root Question与完整但精简的Evidence集合共同判断，所有答案要求均有来源支持且不存在未解决冲突；Answerer不得把无引用推断补成缺失事实；以false-ready率单独评估，并与最终答案正确率分开报告。
- [ ] **R3：Controller重复无效动作。** 从canonical `ActionTrace`、`ReadRecord`和`SearchSession`派生已尝试source、实际查询、候选cursor、动作outcome及Observation引用；当前轮直接读取规范化`ReadRecord/EvidenceCheckResult`，不生成`result_summary`。对完全相同的动作、目标和查询建立循环保护，除非Evidence/gap或可用输入发生变化。
- [ ] **R4：路线或预算耗尽但Evidence仍不充分。** 保持`EvidenceMemory.root_status=incomplete`，在Reading Session边界记录停止原因并允许Answerer明确拒答或报告证据不足；不得把“无法继续”改写为`ready`。闭环实验后再决定是否需要第三种`EvidenceStatus`，避免现在过早扩展schema。
- [ ] Re-evaluate Controller v0 with the future server model. Measure whether it rejects candidates that match the topic but cannot answer the requested evidence field (for example, a date when the gap asks for a reason), and whether it keeps confirmed Relations separate from candidate Relations. Compare a stronger prompted model first; consider SFT/RL only if these errors persist in real trajectories. Do not add case-specific Prompt rules from synthetic failures.
