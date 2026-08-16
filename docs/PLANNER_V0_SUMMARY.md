# Planner v0 阶段总结与后续 Controller 边界

这份文档用于向外部讨论者说明当前 Initial Planner 做了什么、没有做什么，以及为什么
Controller 不能盲信一次性的子问题拆分。本文记录的是 `planner-v0.14` 设计；已有 4B/8B
实验使用的是 `planner-v0.11`，两者必须分开解读。

**冻结状态：** `planner-v0.14`的Prompt、模型输出Schema、默认最大节点数6和默认最大深度4
正式冻结。除非建立新版本并重新评测，否则不得因个别数据集错误修改。

## 1. 当前项目走到哪里

项目已经完成两层基础设施：

1. SoftDoc：把 PDF 表示成 Page、Section、Element 和 typed Relation；
2. Retrieval v1：Exact Anchor、BM25、Dense、融合候选、SearchSession 和
   CandidatePreview，负责提供“从哪里开始读”的入口。

当前进入 Planner v0。它只把原问题表达为一个很小的 SubQuestion DAG，不检索文档、
不读取候选、不选择 Agent 动作、不判断证据是否充分，也不生成最终答案。Reader、
Evidence Checker、动态 Controller 仍是后续模块。

## 2. Planner v0 的设计

原问题始终作为隐式 Root，且是最终任务的权威定义。SubQuestion 只是为了寻找证据而提出的
暂定信息需求，不会取代 Root。

当前约束：

- 简单事实问题保留为一个 SubQuestion；
- 只有需要多个可独立检索事实时才拆分；
- 最终计算、比较、排序和格式化由 Root 完成，不单独制造“计算型子问题”；
- 只有后一个问题必须等待前一个问题产生未知实体时，才建立 `depends_on`；
- 最多 6 个 SubQuestion；
- Root 算第 1 层，默认完整 DAG 最大深度为 4；
- 显式 Anchor 由确定性 Exact Lookup 提取，不属于模型面对的 Planner schema；
- 不确定是否需要拆分时，保留完整原问题作为一个 SubQuestion，不猜测隐藏公式或语义角色。

一个 SubQuestion 的最小结构为：

```json
{
  "subquestion_id": "Q1",
  "text": "What was the revenue in 2023?",
  "depends_on": []
}
```

`explicit_anchors` 已从 Planner 的模型输出和运行时 `InitialPlan` 中完全删除。Exact Anchor
Extractor 独立处理原问题或子问题文本，并把匹配保存在检索结果中，而不是写回 Planner。

Pydantic 会确定性检查 JSON 结构、字段类型、空文本、重复 ID、未知依赖、循环依赖、节点数和
深度等问题。它能拦截格式错误，但不能证明子问题在语义上拆得正确。第一次输出结构不合法时，
程序把具体校验错误交给同一模型纠正一次，并在 trace 中记录重试。

## 3. Prompt 演进与实验

早期 Prompt 曾逐渐加入大量由开发题反推的规则和案例。这可以修复个案，但产生了明显的
过拟合风险。随后我们删除了额外的 `answer_requirements` 字段，并在 `planner-v0.11` 中移除
数据集专属案例，只保留通用规则和两个虚构示例。

我们用同一个 `planner-v0.11`、同一个 JSON Schema、相同温度且 `think=false`，在 44 道
历史高风险题上比较本地 Qwen3 4B Instruct-2507 与 Qwen3 8B：

| 模型 | 正确 | 可接受 | 语义错误 | 结构错误 | 平均耗时 |
|---|---:|---:|---:|---:|---:|
| Qwen3 4B | 21 | 15 | 7 | 1 | 约 6.25 秒 |
| Qwen3 8B | 22 | 12 | 10 | 0 | 约 6.28 秒 |

这 44 题是旧实验中挑出的高风险回归集，不代表完整数据集准确率。8B 在这组题上没有显示出
稳定的语义优势，所以当前继续保留 4B 为默认模型，也不再针对这些错误不断增加题型案例。

`planner-v0.12` 增加一条通用保守原则：“不确定时保留原问题”。`planner-v0.13` 进一步把
运算规则抽象为“不要新增只负责处理已请求事实的节点”，澄清局部证据可直接回答的比较仍可
作为一个 SubQuestion，并从 Planner schema 删除 `explicit_anchors`。旧的 v0.11 结果原样保留，
`planner-v0.14` 将依赖的通用定义改成“前一个答案是否是实例化后一个证据需求所必需”；若后者
仅凭原问题即可独立搜索，则保持并行。旧结果不能标成新版本实验；新版本本轮只做代码与结构
测试，不重跑真实模型评测。

## 4. 当前完整英文 Prompt（planner-v0.14）

下面的 `{max_subquestions}`、`{max_depth}` 和 `{serialized_question}` 在运行时由程序填入。

```text
You are the initial planner for a long-document question-answering system.

Decompose the original question into the smallest sufficient set of
SubQuestions for finding evidence in the document.

The original question is an implicit Root.

The Root:
- represents the complete question asked by the user;
- is not a SubQuestion;
- does not retrieve or read document content;
- is not an additional Agent action;
- uses the collected evidence to perform final deterministic operations over
  facts that were already requested separately, and formats the answer.

Every SubQuestion must request new evidence that can be found in the document.

Rules:

1. Keep a simple factual question as one SubQuestion.

2. Split only when answering the original question requires multiple
   independently retrievable facts.

3. Do not create a SubQuestion whose only purpose is to apply a deterministic
   operation to source facts already requested by other SubQuestions.
   The Root performs that final operation.

   If the question asks for a named metric that may be directly reported in the
   document, preserve it as one evidence need. Do not invent a formula or hidden
   operands that are not stated in the question.

4. A comparison, ranking, or selection may remain one SubQuestion when it can be
   answered directly from the same local piece of document evidence. Do not add
   such a node merely to recombine facts already requested separately.

5. Use depends_on only when the answer to an earlier SubQuestion is required to
   instantiate the later evidence need, such as identifying an unknown entity,
   condition, category, value, time period, or search phrase.

   If the later SubQuestion can be searched independently from the original
   question alone, keep it parallel.

6. Preserve the meaning and scope of the original question. Keep all relevant:
   - named entities;
   - dates and numbers;
   - metrics;
   - conditions;
   - exclusions and negations;
   - ranges;
   - comparison subjects;
   - relative locations.

   Do not invent, replace, or silently remove information.

7. When it is unclear whether decomposition is necessary, or when the question
   contains ambiguous wording, an unstated formula, or uncertain semantic roles,
   keep the complete original question as one SubQuestion instead of guessing.
   A one-SubQuestion plan is valid and does not mean planning failed.

8. Do not answer the original question.
   Do not assume facts from the unseen document.
   Do not select retrieval or reading actions.

9. Return no more than {max_subquestions} SubQuestions.

10. The implicit Root is at depth 1. The complete dependency DAG must not exceed
   depth {max_depth}, including the Root.

Example 1 - source facts followed by Root calculation:

Original question:
"How much did Metric A change from Year X to Year Y?"

SubQuestions:
- Q1: What was Metric A in Year X?
- Q2: What was Metric A in Year Y?

Q1 and Q2 are parallel evidence needs.
Do not create Q3 to calculate the change.
The Root calculates the change after Q1 and Q2 have been answered.

Example 2 - true dependency through an unknown entity:

Original question:
"Which system has the highest Metric A, and what method does that system use?"

SubQuestions:
- Q1: Which system has the highest Metric A?
- Q2: What method does the system identified by Q1 use?

Q2 depends on Q1 because the system is unknown until Q1 has been answered.

Return strict JSON only.
Do not return explanations, Markdown, comments, or additional fields.

Return exactly this structure:

{
  "original_question": {serialized_question},
  "subquestions": [
    {
      "subquestion_id": "Q1",
      "text": "...",
      "depends_on": []
    }
  ]
}

Copy the original question exactly into original_question.

Original question:

{serialized_question}
```

## 5. Conservative + Deferred Planning

`Deferred`不是“把Planner推迟到最后”，而是**推迟不确定的拆分决定**：初始Planner只拆明显且
独立的证据需求；其余问题先整体进入搜索和阅读。读到文档后，Evidence Checker才能根据真实
内容指出缺失的是哪个对象、条件、时期或解释，Controller再通过`UPDATE_PLAN`细化计划。

```text
Root Question
  -> optional conservative InitialPlan
  -> SEARCH / READ
  -> observed evidence and concrete gaps
  -> UPDATE_PLAN only when needed
  -> continue reading or answer
```

因此InitialPlan是“当前最小可用路线”，不是一次性必须正确的任务分解。Deferred Planning也不
要求现在新增第二个Planner；未来仍可由同一个Controller调用受约束的计划更新能力。

## 6. Controller 如何知道 Planner 可能是错的

不应该让 Controller 先预测一个不可靠的 `plan_is_correct`，也不应依赖 Planner 自报
confidence。正确边界是让系统从数据结构和停止条件上不允许它盲信计划：

1. **Root 永远保留。** Controller 的最终目标始终是原问题，不是完成所有 SubQuestion。
2. **InitialPlan 明确定义为 provisional。** 它只是第一版搜索假设；一个与 Root 相同的单节点
   计划完全合法。
3. **读证据，而不是机械清单打勾。** Controller 根据当前 Evidence State 决定下一步，不能因
   所有初始节点被标记完成就直接回答。
4. **做两层充分性检查。** Evidence Checker 后续既检查每个 SubQuestion 是否有支持，也独立
   检查已有证据是否覆盖 Root 的实体、条件、时间、比较项和最终回答要求。
5. **发现计划缺陷时使用既有 `UPDATE_PLAN`。** 可以新增、改写、合并、废弃 SubQuestion，
   但每次产生新 revision，保留原因和旧版本，不静默覆盖。
6. **检索失败不等于子问题错误。** 应先区分候选尚未读完、需要 Relation/Page/Region 导航、
   文档确实没有证据，还是问题拆分丢失了条件。

例如原问题是“2022 到 2023 年收入增长了多少？”。初始计划可能是两个年份的收入。如果后来
只找到 2023 年收入，Controller 不会仅因 Q2 已完成就回答；Evidence Checker 会指出 Root 仍缺
2022 年基数。反过来，如果 Planner 因歧义只保留了整个原问题，Controller 仍可先搜索和阅读，
再根据实际发现把它修订成两个更明确的证据需求。

因此“不一定拆分”不是失败模式，而是一种保守起点。动态变化发生在读到文档证据之后，比
Planner 在完全没看文档时强行猜测更可靠。

未来 Controller observation 至少应持续暴露：

```text
root_question
plan_revision
subquestions and their evidence status
current evidence and unresolved root-level gaps
search session / candidates already shown or opened
available reading actions
```

本轮只冻结这项边界，不新增 Controller、Evidence State 或 Checker 代码。

## 7. 建议的后续顺序

1. 冻结 Initial Planner v0，不再按单题扩充 Prompt；
2. 实现 Reader primitive 与统一 ReadObservation；
3. 定义 Evidence State 和 Root-level evidence gap；
4. 实现不调用 LLM 的状态与轨迹基础设施；
5. 接入 Teacher/Controller，允许通过 `UPDATE_PLAN` 产生版本化修订；
6. 对比 no-planner、one-shot planner、dynamic replanning，而不默认拆分必然有效。

正式对照实验已经登记在[`TODO.md`](TODO.md)：No Planner vs Initial Planner vs Deferred
Planner。三者必须共享其余系统组件与预算，避免把检索器或Reader差异误算成Planner收益。

Planner 是否真正提高最终 QA，必须在 Reader、Evidence Checker 和 Controller 接通后用最终
答案质量、证据完整率、阅读动作数、搜索次数与成本共同验证。仅看子问题“像不像人工拆分”
不能证明整个系统有效。
