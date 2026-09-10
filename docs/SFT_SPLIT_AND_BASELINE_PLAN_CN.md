# CRAVE-RAG SFT 文档级拆分与 Baseline 计划

> 状态：文档级拆分、Dev baseline 与内部 Test baseline 已完成；下一步进入 Controller Teacher 数据 pilot。  
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
