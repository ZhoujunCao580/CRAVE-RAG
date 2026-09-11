# CRAVE-RAG SFT 文档级拆分与 Baseline 计划

> 状态：文档级拆分、Dev/Test baseline、Controller SFT 工程 pilot 与 500+ 决策扩展训练均已完成。
> 边界：这是 MMLongBench-Doc 内部开发拆分，不宣称为外部未见 Test。

## 1. 冻结拆分

拆分单位是 PDF，不是问题。同一文档的任何问题不得跨集合。

| 集合 | 文档 | 问题 | 177 困难题 | 用途 |
|---|---:|---:|---:|---|
| Train | 42 | 357 | 140 | Teacher 数据与 SFT |
| Diagnostic Dev | 12 | 97 | 37 | 检查已知困难类型是否改善 |
| Clean Dev | 8 | 68 | 0 | 选择 checkpoint 和配置 |
| Test Core | 10 | 81 | 0 | 代表性内部测试 |
| Test Challenge | 6 | 43 | 0 | 结构困难题富集测试 |
| Reserve | 57 | 445 | 0 | 暂不运行，留作扩训或后续验证 |

总计 135 份文档、1091 道问题。拆分由固定 seed `20260910` 生成。

拆分程序：`scripts/build_sft_document_split.py`  
冻结清单：`configs/training/mmlongbench_sft_split_v0_1/split_manifest.json`

结构困难分层只使用问题文本、证据模态、证据页数量、文档类型和
Not-answerable 标签，不使用当前模型的正确/错误结果。输出给运行器的 JSONL
不包含 Gold answer 或 Gold evidence。

## 2. Baseline 顺序

### Gate 0：服务器 Smoke

从 Diagnostic Dev、Clean Dev 各取 2 道普通题和 2 道困难题，验证：

- 当前 commit、Prompt manifest、JSON Schema 和模型一致；
- Planner、Controller、Reader、Checker、Answerer 原始调用完整保存；
- 检索候选、视觉简述、Table Reader、Relation、Recall 和 Visual Scan 接线存在；
- 无程序失败、JSON 截断或不可见 ID 中断。

### Gate 1：开发集 Baseline

1. 运行 Diagnostic Dev 97 题，可以查看逐题轨迹；
2. 运行 Clean Dev 68 题，可以查看逐题轨迹；
3. 报告语义准确率、READY precision、answerable completion、Evidence 成功、
   action 数、SEARCH/READ 数、重复动作、非法动作、耗时与 token；
4. 只有程序链路稳定且 Clean Dev 不出现系统性退化，才冻结 Prompt-only baseline。

### Gate 2：密封 Test Baseline

在同一冻结 commit/config 上运行 Test Core 81 题和 Test Challenge 43 题。

- 只读取聚合指标，不人工审计 Test 逐题答案或轨迹；
- 保存每题原始产物用于最终配对评分，但训练期间不打开；
- 自动冻结 baseline-correct、baseline-wrong、baseline-no-answer 三个 case ID
  切片，不向 Teacher 暴露 Test 内容。

完成记录（2026-09-10）：Test Core + Test Challenge 共 124 题，内容等价
Acc 为 **62/124 = 50.0%**，generalized F1 为 **49.44%**。为校准格式等价，
本轮已人工复核非精确匹配答案，因此后续不得用 Test 个案制作 Teacher 标签或选择
checkpoint；若需要新的最终盲测，从 Reserve 冻结新文档集合。

### Gate 3：Train Baseline

运行 Train 357 题，完整保存 Student 轨迹。此批次的用途不是提供最终评测分，
而是发现可用于训练的真实状态：

- 成功且高质量的 State -> Action；
- 首个错误决策之前的合法状态；
- 重复 SEARCH/READ、错误 STOP、错选 candidate、Evidence 判断失败等恢复状态。

## 3. Teacher 数据构建

旧失败轨迹不得直接作为 SFT 正样本。Teacher 必须在与 Student 相同的可见状态和
合法动作集合下重新决策，不能看到隐藏 Gold source ID。

第一轮数据配比：

- 45% 普通成功决策；
- 35% 困难题的成功或 Teacher 修正决策；
- 20% 稀有动作与失败恢复。

每题最多采样 4--6 个 Controller 决策，并按问题归一化权重，避免长失败轨迹主导
训练。先用 20--30 题验证训练闭环，再扩到 100--150 题；验证确有收益后再考虑
200--250 题。

## 4. SFT 后评估

1. 先做离线 Controller state 评估：JSON 合法率、Action 合法率、source ID
   合法率、Teacher agreement 和重复动作率；
2. 在 Diagnostic Dev 检查已知失败类型；
3. 在 Clean Dev 选择 checkpoint；
4. 最佳 checkpoint 在 Clean Dev 确认后再运行当前 Test 做配对比较，但不得据其逐题
   结果继续调参；论文级最终盲测应另从 Reserve 冻结；
5. 同时报告 Test 总准确率、Test Challenge 准确率、baseline 错题救回率以及
   baseline 对题退化率。

Controller 是第一阶段唯一训练对象。Checker 继续保留 Prompt-only；只有在
Controller SFT 后的新 Dev 漏斗中确认存在足量“正确来源 + 正确 Observation +
Checker 错判”案例，才单独训练 Checker adapter。

## 5. Pilot 实际结果与下一步门槛

2026-09-11 已完成首个 Controller-only pilot：30 个审阅 episode 中，24 题/78 个决策用于
训练，6 题/14 个决策用于文档隔离的离线评估。60-step QLoRA 训练与部署闭环成功，但离线
route agreement 从 base 的 100% 降至 SFT 的 92.86%；端到端 Test 仍为
**62/124 = 50.0% Acc、49.44% F1**，7 题救回与 7 题退化相互抵消。

因此当前不应直接扩到全量 Teacher 数据，也不进入 DPO。下一轮门槛调整为：

1. 先在 Clean Dev 比较 checkpoint 20、40、60，不默认采用训练 loss 最低的 checkpoint；
2. 增加失败状态上的真实在线 Teacher rollout，而不只是过滤 base 已执行动作；
3. 文档隔离 held-out state 规模扩大后，route agreement 至少不得低于 base；
4. Clean Dev 端到端提升后，才扩大到 100--150 题；
5. Checker 数据继续与 Controller 数据分开构建。

## 6. 500+ 决策扩展 SFT 结果

首轮 78 决策 pilot 只验证了工程闭环，不能代表扩展训练的最终效果。后续使用 **500+ 个经审阅
的 Controller 决策样本**完成扩展 SFT，并在同一内部 Test 124 题上配对比较：

| 指标 | Prompt-only baseline | Controller SFT 500+ | 提升 |
| --- | ---: | ---: | ---: |
| 内容 Acc | 50.00% | **56.45%** | **+6.45 pp** |
| generalized F1 | 49.44% | **约 55.32%** | **约 +5.88 pp** |

题目级变化为：Test Core 救回 6 题；Test Challenge 救回 4 题、退化 2 题；总计净增加 8 道
正确答案。该结果说明扩大且经过审阅的 Controller 监督有效，替代了上一节小样本 pilot 的
“暂不扩训”结论。下一步仍应遵守以下边界：

1. 不用 Test 逐题错误继续选择 checkpoint 或改标签；新一轮选择只看 Clean Dev；
2. 将救回与退化案例按动作类型和文档隔离复核，避免新增样本只覆盖某一困难模式；
3. Checker 训练数据继续单独建集，不能把 Controller、检索或 Reader 的错误混入；
4. 当前 F1 是依据现有汇总推算的近似值，导入完整后训练状态计数后重新精算；
5. 论文级最终结论仍需在 Reserve 冻结新盲测或运行完整官方协议。
