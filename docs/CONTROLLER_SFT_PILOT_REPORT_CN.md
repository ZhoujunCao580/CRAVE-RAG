# CRAVE-RAG Controller QLoRA/SFT Pilot 报告

> 日期：2026-09-10 至 2026-09-11  
> 状态：训练、离线评估和端到端 Test 均已完成。  
> 边界：只训练 Controller；Planner、Reader、Checker、Answerer 均继续使用原始 Qwen3.5-27B。

## 1. 这次实验回答什么问题

本轮不是追求一次训练就显著提高 QA 分数，而是验证以下闭环：

1. 从真实 Reading Environment 的既有轨迹中审阅 Controller 决策；
2. 严格导出线上同构的 `Controller State -> Action` 监督数据；
3. 用单卡 A100 80GB 对 Qwen3.5-27B 做 Controller-only QLoRA；
4. 用文档隔离的离线状态比较 base 与 SFT Controller；
5. 在冻结的 124 题 Test 上进行一次端到端复测。

本轮不做 DPO，也不把 Checker 错误混入 Controller 标签。

## 2. Teacher rollout 与 distillation

- **Teacher rollout**：Teacher 在真实 Environment 中逐步看到当前 State、选择合法 Action，
  再接收真实检索、Reader、Checker 结果，直到成功或终止。它不是离线写一串看似合理的动作。
- **Teacher distillation**：把 Teacher 路线中的每个决策拆成
  `Controller State -> Teacher Action`，训练 Student Controller 模仿策略。

本 pilot 使用的是**真实执行轨迹上的 Teacher 审阅式 behavior cloning**：Teacher 保留或拒绝已经
执行过的动作，没有伪造未实际执行动作之后的 Observation。真正失败状态上的改道仍需后续在线
Teacher rollout。

答对题可以进入 SFT，但必须是证据正确、路径高质量的成功轨迹；幸运猜对、错误 Evidence 后猜对、
重复搜索或无进展尾部不得作为正样本。

## 3. 数据构建与质量门槛

共审阅 30 道 Diagnostic Dev 题；Clean Dev 和 Test 没有进入数据制作。

| 切分 | 题目 | Controller 决策 | 文档隔离用途 |
|---|---:|---:|---|
| Train | 24 | 78 | QLoRA 训练 |
| Teacher Eval | 6 | 14 | 离线 base/SFT 对比 |
| 合计 | 30 | 92 | 3 份 Eval 文档与 Train 完全隔离 |

30 道中 23 道整条轨迹通过，7 道只保留错误发生前或恢复后的高质量前缀；共拒绝 56 个重复、
无进展、错误选源或 Preview 已明显不匹配的动作。因此这里应准确表述为“30 个审阅 episode、
92 个高质量决策”，而不是声称有 30 条全部成功的在线 Teacher rollout。

训练集覆盖 Text 28、Table 16、Visual 12、Page 1 次 Reader 调用；动作分布如下：

| 动作 | 数量 |
|---|---:|
| `READ_SOURCE` | 35 |
| `SEARCH:new` | 28 |
| `SEARCH:next` | 8 |
| `FOLLOW_RELATION` | 5 |
| `STOP` | 2 |

严格审计结果：JSON、ControllerInput、Action schema、可见 ID 合法率均为 100%；重复 state、
重复 state-action 和连续重复动作均为 0。

## 4. 训练配置与结果

- 基座：Qwen3.5-27B；
- 方法：4-bit QLoRA，BF16，gradient checkpointing；
- LoRA：rank 16，alpha 32，dropout 0.05；
- 最大长度：8192 tokens；
- batch size 1，gradient accumulation 4；
- learning rate `1e-4`，cosine scheduler，60 optimizer steps；
- 优化器：paged AdamW 8-bit；
- 实际训练：60/60 steps，约 59.3 分钟；
- 平均训练 loss：0.05874；
- 采样峰值显存：74,005 MiB；最高温度：73°C；
- 最终 adapter：约 416 MiB。

checkpoint 20 的末步 loss 为 0.01573，checkpoint 40 为 0.0000955，checkpoint 60 为
0.000129。小数据上 loss 很快接近 0，存在明显记忆化风险；最终 60-step adapter 只适合作为
闭环 pilot，不应被当作已选出的正式 checkpoint。

训练期间出现过可恢复的 CUDA workspace/SDPA 分配回退，但没有 `torch.OutOfMemoryError`，
训练正常完成。

需要记录一个非阻塞实现问题：当前 LoRA target 使用通用叶节点名，PEFT 因而也在视觉塔中创建了
同名 adapter 分支；Controller 训练是纯文本 forward，这些视觉分支没有得到有效监督，vLLM
加载时明确将 `visual.blocks.*` 分支忽略，语言层 adapter 仍正常生效。它不使本次结果失效，但
后续正式训练应把 target 精确限制到语言模型层，减少无用权重和加载警告。

## 5. 工程问题与修复

本轮共遇到四类环境/契约问题，均未通过改变实验 Prompt 或检索策略规避：

1. Transformers 5 已移除 `TrainingArguments.warmup_ratio`：改为等价的浮点
   `warmup_steps` 兼容映射；
2. Transformers 5 下 Qwen `apply_chat_template` 返回 `BatchEncoding`：统一提取
   `input_ids`，避免 collator 把字段名字符串当 token；
3. vLLM FlashInfer JIT 找不到 `ninja`：将 `/opt/crave-venv/bin` 加入 `PATH`；
4. Test split 带审计字段 `document_id`，旧 batch loader 将其误判为非法字段：只允许这一已知
   审计字段通过，仍拒绝任意未知字段。

另有一个后续工程债：当前 `/opt/crave-venv` 指向 Network Volume 中包含大量小文件的环境，
首次 import 和进程启动较慢；应在不运行实验时制作压缩归档并恢复到 Container Disk。

## 6. 离线 Controller 对比

离线集合为 6 题、14 个文档隔离 State，只评估 Controller policy，不执行真实检索和 QA。

| 指标 | Base Controller | SFT Controller |
|---|---:|---:|
| 首次 JSON 合法率 | 100% | 100% |
| Action 合法率 | 100% | 100% |
| 可见 ID 合法率 | 100% | 100% |
| repair 后最终合法率 | 100% | 100% |
| Teacher exact agreement | 92.86% | 42.86% |
| Teacher route agreement | 100% | 92.86% |
| 连续结构重复率 | 7.14% | 7.14% |
| 平均延迟 | 3.27 秒 | 3.18 秒 |

两者动作分布完全相同：`FOLLOW_RELATION=1`、`READ_SOURCE=7`、`SEARCH:new=6`。

SFT 的 8 个 exact disagreement 中，7 个只是自由文本 query 或 `local_problem` 的措辞不同；
但 exact 指标按严格字符串仍计错。另 1 个是实际路线错误：Q778 中 Teacher 选择 chart，SFT
选择了附近 paragraph。因此，本轮 SFT 保持了结构化接口合法性，但**没有在 held-out policy
agreement 上证明提升**，route agreement 反而下降 7.14 个百分点。

## 7. 端到端 Test

| 指标 | Prompt-only baseline | SFT Controller | 变化 |
|---|---:|---:|---:|
| 总体 Acc | 62/124 = **50.0%** | 62/124 = **50.0%** | 0 |
| generalized F1 | **49.44%** | **49.44%** | 0 |
| Test Core Acc | 38/81 = 46.91% | 42/81 = 51.85% | +4.94 pp |
| Test Core F1 | 47.86% | 54.24% | +6.38 pp |
| Test Challenge Acc | 24/43 = 55.81% | 20/43 = 46.51% | -9.30 pp |
| Test Challenge F1 | 52.46% | 40.00% | -12.46 pp |

SFT Controller 的 29 道最终预测发生了变化：7 道从错变对，7 道从对变错，其余变化没有
改变正确性；净增益为 0。Core 的收益被 Challenge 的退化完全抵消，说明小样本 adapter
没有形成稳定的跨文档泛化。

状态分布：

| 状态 | Prompt-only baseline | SFT Controller |
|---|---:|---:|
| `ready` | 79 | 79 |
| `budget_exhausted` | 31 | 38 |
| `stopped_incomplete` | 14 | 6 |
| `program_failure` | 0 | 1 |

唯一程序失败 Q1023 是 PlannerDraft JSON 在 768-token 上限处 EOF 截断；相同冻结配置的独立
重试再次发生同一错误。为避免根据 Test 个案临时提高 token 上限或改 Prompt，正式分数将它按
程序失败计错，并保留主批和 recovery 两份原始日志。该错误发生在 Planner，不是 SFT
Controller 的动作输出，但仍属于本次端到端系统结果。

主批 124 题采用 4 workers，墙钟 60 分 49 秒，折合约 29.4 秒/题；123 个有效轨迹的模型
调用时间均值 106.4 秒、中位数 73.1 秒、P95 274.5 秒。该墙钟吞吐较 baseline 记录的
35.35 秒/题快约 16.8%，但受到缓存热度和题目调度影响，只作为工程指标，不作为模型质量结论。

评分继续使用项目冻结口径：忽略无害格式差异；终态 `budget_exhausted` 和
`stopped_incomplete` 的最终答案按 `Not answerable` 处理。Test 曾被人工校准过，因此是内部
配对测试，不再是严格未见盲测；本轮没有根据 Test 个案调参。

## 8. 当前结论

1. 数据导出、审计、QLoRA 训练、vLLM Controller-only adapter 路由和端到端调用闭环已打通；
2. 30 个审阅 episode / 78 个训练决策足够验证工程闭环，不足以训练稳定的新策略；
3. 离线合法率没有退化，但 held-out route agreement 从 100% 降到 92.86%；Test 净增益为 0，
   说明这版 adapter 不能仅凭 train loss 宣称成功；
4. 本轮最有价值的产出是可审计的 distillation pipeline，而不是精度提升；
5. 下一轮应先比较 checkpoint 20/40，并增加真正的在线 Teacher correction 和文档隔离的
   Clean Dev 选择，不能直接扩大相同类型轨迹或开始 DPO；
6. Checker 是否需要 SFT，应等 Controller 训练后的新失败漏斗再决定，避免把上游检索/Reader
   错误或 Controller 错误混入 Checker 数据。

## 9. 产物

服务器 Network Volume 保留原始目录，并额外生成：

- `controller-sft-pilot-model-and-metrics-20260911.tar.gz`；
- `controller-sft-test-trajectories-20260911.tar.gz`；
- `controller-sft-artifacts-20260911.sha256`。

本地已下载至 `.runlogs/controller_sft_pilot_20260911/`。模型/指标归档 SHA256 为
`894110C6643CA902379E9AB5F2517E995828A8D94D2021C033095369F516C87A`；Test 轨迹归档为
`245B8A0A29811850B6830A649CA655694E3313119060527BDE7C928A2A94A906`，均与服务器一致。

## 10. 后续 500+ 决策扩展结果（不属于本 pilot）

本报告第 1--9 节只描述 78 决策的小规模工程 pilot，其“净提升为 0”结论不得外推到后续
扩展训练。2026-09-11，项目使用 **500+ 个经审阅的 Controller SFT 决策样本**完成后续训练，
同一内部 Test 124 题的内容 Acc 从 **50.00% 提升到 56.45%（+6.45 pp）**，generalized
F1 从 **49.44% 提升到约 55.32%（约 +5.88 pp）**。题目级表现为救回 10 题、退化 2 题，
净增加 8 道正确答案。

这证明扩展监督产生了端到端净收益；但约 55.32% 的 F1 仍需在完整后训练状态计数导入后
精算。本节用于连接两个实验阶段，不修改前述 pilot 的原始数据与历史结论。
