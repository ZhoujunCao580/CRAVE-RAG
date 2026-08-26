# TODO

这里只保留尚未完成、必须通过真实实验决定的事项。已实现的契约与流程见
[`MODEL_CONTRACTS.md`](MODEL_CONTRACTS.md) 和
[`ARCHITECTURE.md`](ARCHITECTURE.md)，不在 TODO 重复记录。

## 1. PDF 与 SoftDoc 完整性

- [ ] 在完整数据集上区分 MinerU 上游漏检/错类型/错裁剪与 SoftDoc Adapter
  丢失，优先修复通用问题，不按单份文档堆规则。
- [ ] 复核 22 个上游 Table 类型误判候选，并测量它们对检索、阅读和回答的
  真实影响，再决定是否需要通用类型校验或其他解析后端。
- [ ] 审计复合 Table 内部图片：有效 `<img src>`、仅有 `<image N>` 占位、完全
  无提示但视觉区域含图三类必须分开；暂不增加每表都调用 VLM 的检测模块。
- [ ] 在更大样本上比较当前 Pipeline、MinerU Hybrid 和视觉 recovery；没有净
  收益的 recovery/Normalizer 不保留。
- [ ] 跨页聚合 Table 的拆分只是 representative-28 上的开发策略。完整数据集上
  比较“保留 MinerU 聚合结果”和“按物理页拆分并生成 confirmed
  `continued_on`”。行归属、重复表头或跨页 `rowspan` 无法唯一验证时不得强拆。

## 2. Planner 策略

- [ ] 在完整 Reading Loop 上同条件比较 No Planner、Initial Planner 和 Deferred
  Planner。
- [ ] 同时报告答案质量、Evidence 完整性、动作/模型调用数、延迟和成本；只有
  Deferred Planning 有稳定净收益才实现动态计划更新。

## 3. 检索、候选批次与预算

- [ ] 比较 weighted RRF、固定配额混合（如 `3 BM25 + 2 Dense`）和 BM25-first
  后按需 Dense 三种策略。
- [ ] 调整每批 Preview 数（如 `3/5/10`）、BM25/Dense 配额、RRF 参数，以及何时
  next/switch/new search。
- [ ] 不只报告 Top-k recall；还报告找到充分 Evidence 所需批次、完整读取数、
  VLM 调用、延迟和费用。
- [ ] 将当前 action-count 占位预算替换为可配置资源成本。搜索翻批、文本读取、
  整页视觉和局部视觉不能默认等成本。

## 4. Reader 与视觉动作

- [ ] 用真实 QA 比较 Controller 直接消费结构化内容与专门 Reader 输出
  Observation；只有后者改善质量或成本时才保留专门 Reader。
- [ ] Visual Reader v0 暂时统一 Page/Figure/Chart；以后再以实验决定是否拆分
  Page/Element 或 Figure/Chart Reader。
- [ ] 默认单图读取；可拆成独立数值/事实的多图比较应分别读取并由 Answerer
  汇总。只有视觉关系本身不可拆分时才联合多图读取。
- [ ] `INSPECT_REGION`/zoom、结构化 Table Reader 和视觉 fallback 均等待真实
  failure taxonomy 后再加入动作空间，不为假设场景提前扩 Schema。
- [ ] 当 `continued_on` 未检测到、candidate Relation 可疑或结构化读取失败时，
  由 Controller 决定相邻页、候选关系或视觉读取；Reader 不自动导航。

## 5. Relation 与规则审计

- [ ] 在真实轨迹上逐类消融 `caption_of`、`footnote_of`、`refers_to`、Section、
  page/reading adjacency 和 `continued_on`，测量 Evidence 增益、减少的搜索次数和
  错误导航成本。
- [ ] 审计仍偏激进的规则，优先检查
  `parser_declared_function_target`、`bounded_nearest_compatible_element` 和
  `profile_forced_sibling_level`；依据最终 QA 收益决定保留、降为 candidate 或删除。
- [ ] confirmed Relation 与 candidate Relation 必须持续分开评估；candidate 只能
  作为调查机会，不能因被探索而自动升级为事实。

## 6. Observation、Evidence Checker 与 Recall

- [ ] 在服务器模型上重新评估 Checker。当前本地 `qwen3:8b` 只证明循环与 Schema
  可执行，不是质量结论。
- [ ] 报告 Schema 合法率、delta 应用率、false-ready、冲突处理、两轮循环成功率，
  并至少对比一个更强模型。
- [ ] 覆盖这些 failure cases：无 Observation 只有 limitation；部分有用；无关、
  重复或范围不符；错误 Observation；新旧 Evidence 冲突；多 Evidence 联合才充分；
  路线或预算耗尽但 Root 仍 incomplete。
- [ ] `incomplete` 漏写 `remaining_gap_description` 或违反
  `used_for_evidence`/delta 约束时，最多做一次携带明确 Pydantic 错误的定向重试，
  不继续加长主 Prompt。
- [ ] 评估 Observation Recall：当后续问题可能复用旧但未采纳 Observation 时，
  比较重新读取与从 ObservationStore 召回再交 Checker；只有减少成本且不增加错误
  才实现 Recall。
- [ ] 评估一次 Observation 可支持多个问题造成的重复读取风险；v0 仍一次只评估
  当前 target，不提前扩大 Checker 范围。

## 7. Controller 策略与训练

- [ ] 用更强服务器模型验证 Controller 是否能拒绝“主题相关但不能填补当前字段”的
  Preview，以及是否真正区分 confirmed/candidate Relation。
- [ ] 建立 Teacher 轨迹并记录每一步的可接受动作、Evidence 净增益、成本和失败原因；
  先做 prompted Teacher/SFT，再决定是否需要偏好训练或 RL。
- [ ] 防止无意义循环：重复动作只有在 Evidence、gap、输入可读性或候选状态发生变化
  后才允许；`STOP` 不得把 incomplete 改成 ready。
- [ ] 比较只给当前 gap 与给完整精简 Evidence/近期反馈的策略，确认哪些状态字段确实
  改善动作选择后再冻结训练输入。

## 8. Answerer 与最终引用

- [ ] 在更强模型上验证 Evidence 只证明“发生变化”但 Root 追问“为什么”时的拒答
  行为；首要防线仍是 Checker 不应过早 `ready`。
- [ ] 实现 Citation Materializer：确定性展开
  `used_evidence_ids -> observation_ids -> ReadRecord.inputs -> SoftDoc source`，生成
  Document/Page/Element/Region 引用；Answerer 不得自行编造位置。
- [ ] 在完整数据集上进行端到端答案质量、Evidence 充分性、citation correctness、
  动作效率和成本评估。
