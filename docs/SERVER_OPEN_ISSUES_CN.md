# CRAVE-RAG 服务器未解决问题报告（2026-09-09）

> 本报告只记录阶段 0、阶段 1A–1E、阶段 2A–2D 及 Relation smoke 中仍未闭环的问题。
> 已通过的测试、已修复的问题和正常结果一律省略；它们以正式实验清单和原始运行产物为准。

## 1. 视觉入口与按批简述

### Q20：Gold 页没有元素级视觉候选

- Gold 物理页为第 17 页，但现有视觉排序中没有可排名的 Gold visual asset。
- 这不是 ColSmol 排名靠后，而是该页没有进入检索的元素级 crop/candidate。
- 后续只审计 Q20 的原 PDF、SoftDoc 元素和资产映射，判断应补页面级视觉入口还是标记为解析数据缺口；
  不重跑全部 135 份 SoftDoc。

### Q16、Q604：页面上下文流程没有初始视觉来源

- 两题在 2A 中均为 `no_visual_crop`。
- `READ_PAGE_CONTEXT` 本身没有失败；问题是没有初始 Figure/Table crop 可供 Reader 先读并产生 limitation，
  因而无法触发“crop 不足后补读整页”的路径。
- 后续需决定是否允许从 Caption/Paragraph/Page handle 建立视觉入口，或将其明确归为数据缺口。

### 后续批次视觉候选的简述完整性仍需验证

- 当前 Stage 1A 产物中仍有 15 个位于第 2/3 批的视觉候选没有 descriptor。
- 正式设计要求：先冻结当前展示批次，再只为该批真正展示的视觉槽位生成并缓存简述；不能先生成简述、
  再因候选重排产生新的未描述候选。
- 下次服务器组合 smoke 需确认每一批在交给 Controller 前均完成该批视觉简述，不做 7,916 张全量生成。

## 2. Multimodal Table Reader

### Q658：missing-header limitation 缺少可见内容

- 一次真实模型输出选择了 `missing_header_context`，但只提供 `description`，漏掉
  `relevant_visible_content`，运行时校验因此拒绝整个结果。
- System/User Prompt 已经写明该字段必须存在；问题在于生成用 JSON Schema 未表达这条跨字段约束。
- 暂缓修复。后续比较 discriminated schema 与同调用一次完整 JSON repair；不得重新读取表格、不得新增
  Controller action，也不得加入 Q658 特例。

### Q468、Q364：Table Reader JSON 在结尾被截断

- 两题出现 EOF/不完整 JSON。
- 归入 P09 定向复测：保存 `finish_reason`、token usage、raw output、token cap 与 schema 使用情况。
- 只有复现后才选择缩短输出、提高 Table Reader 单组件上限或同调用受控重试；不得猜补残缺 JSON。

### Q328、Q88：计划范围内没有找到 Table

- 组件测试返回 `no_table_in_planned_scope`，尚未确认是测试题到 source 的映射错误、SoftDoc 类型缺失，
  还是题目本身不适合作为该组件样例。
- 后续只核对 Gold/source/SoftDoc 映射，不以改变检索规则绕过该问题。

## 3. Reader 的问题边界

### Q838：图中不存在所求另一年份，但 Reader 未显式报告 limitation

- Relation 导航已到正确视觉来源；Reader 只提取了图中可见的 2013 数值。
- 对 local problem 同时要求的 Streaming 年份，Reader 没有明确写出“当前图中未找到”。
- 后续应验证 Reader 在只支持局部答案时能否同时返回已读事实与缺失项 limitation；不得把问题相关
  Observation 缓存成通用视觉 descriptor。

## 4. Exact Anchor 尚未闭环的边界

- `Q129`：`figure 1-4` 中的 `figure 1` 未解析到唯一目标。
- `Q188`：`Figure 3` 匹配多个目标。
- `Q322`：`Table 1-1` 匹配多个目标。
- `Q539`：`Figure 122` 匹配多个目标。
- `Q889`：`Figure 9` 匹配多个目标。
- `Q906`：请求范围过大，未形成可直接读取的唯一 anchor。

这些情况必须继续保留 `ambiguous`、`unresolved` 或 `range-too-large` 语义，并回到普通检索；不得把
多个实体永久合成一个伪元素，也不得混用物理页与印刷页。

## 5. 下一次服务器验证仍需确认的集成边界

- 当前新增的路径规范化、Relation Preview、TablePreview alignment 检查、Table Reader 引用去重等代码
  已有定向测试，但合并后的正式 Controller/Reader smoke 尚未完成。
- 下一次只按正式实验清单逐阶段验证上述未闭环题目；不得因为本报告列出问题而重跑旧 baseline 或
  覆盖已有运行产物。
