# 项目简史与冻结点

这是一份精简历史，不是当前使用说明。当前架构与命令以
[`PROJECT_GUIDE.md`](PROJECT_GUIDE.md)为准。

## Milestone 1：SoftDoc

项目最初目标是把不同PDF解析器输出统一为`Document / Page / Section / Element /
Relation`，保留bbox、reading order、页面图、标题层级和显式关系，同时不依赖图数据库。

真实文档暴露的主要问题及处理：

- MinerU空Heading、空Table、目录型image path和缺内容block会终止转换：Adapter改为
  block级容错，保留warning/provenance，14份代表文档达到14/14转换；
- 标题层级受论文、幻灯片、报告和宣传册版式影响：HeadingHierarchyBuilder从Adapter
  抽离，综合编号、样式、缩进和parser hint，重复页眉页脚不参与Section；
- Caption/Footnote、显式引用和跨页连续关系：RelationBuilder独立生成confirmed或
  candidate关系，复杂空间关系按bbox实时计算；
- 跨页图表属于错误Section：FloatingContentSectionResolver只利用confirmed
  continuation、唯一显式引用和明确caption/footnote关系修正，不因页面邻近迁移；
- 多种启发式逐渐变重：Pipeline被统一为DocumentPass边界，规则增加rule_id和审计，
  Hybrid实验后减少了不必要的后端专用归一化。

冻结点：

```text
softdoc-v0.1-dev              Milestone 1
```

## Retrieval Entry

为了让搜索只负责“从哪里开始读”，实现了：

- Exact Anchor Lookup；
- 不修改SoftDoc的SearchUnitBuilder；
- BM25 Element检索；
- 可注入、可缓存的multilingual-e5-small Dense检索；
- E5 512-token安全分片；
- 不复制Relation邻居、不生成LLM摘要。

冻结点：

```text
softdoc-v0.2-retrieval-entry
```

## SearchSession与候选策略

最初采用BM25优先稳定轮转。扩展到28份文档后，对275道问题重新进行完整Element排名
比较，最终选择Exact独立优先、普通候选使用`k=20`、权重`BM25=1.0/Dense=1.25`的RRF。
SearchSession保存完整排名和游标，CandidatePreview分批展示原始索引片段。

冻结点：

```text
softdoc-v0.3-search-session
softdoc-v0.4-weighted-retrieval
softdoc-v0.4.1-retrieval-schema
```

最后一个schema收尾版本增加display label、明确snippet坐标空间、preview source和
content/metadata match scope。它不改变检索排名，195项测试通过，28文档Hit@K保持不变。

## Initial Planner v0基础设施

检索冻结后增加了question-only Initial Planner边界：严格SubQuestion DAG schema、Prompt、
可注入模型接口、Pydantic/DAG校验和Mock测试。随后接入本地Ollama与
`qwen3:4b-instruct-2507-q4_K_M`，增加最多6节点、Root计入的最大深度4、数字忠实性校验，
以及一次可追踪的验证纠错重试。Planner仍不选择阅读动作，也不实现动态重规划；动态
`REFINE_PLAN`必须等待Evidence State和Evidence Checker提供明确gap。

真实Planner复测后将Prompt升级为`planner-v0.3`：完整Anchor直接复用Exact Lookup语法校验，
拒绝把`Figure 3`拆成`Figure`和`3`；原问题的隐式Root负责比较、求差等无新增证据的合成，
子问题只保留证据需求，并要求覆盖原问题中的全部实体、约束、输出和解释义务。

在28份真实文档对应的275道MMLongBench-Doc问题上完成`planner-v0.3`基线后，将Prompt升级
为`planner-v0.4`：用正反例禁止纯计算、排名、排序、格式化和定位脚手架节点；要求同一局部
证据范围尽量形成一个最小信息需求；明确只有单个编号Exact引用进入`explicit_anchors`，命名
章节、相对位置、序数和页码范围只保留在文本约束中。

对101道高风险真实题复测后，`planner-v0.5`继续收敛少量残余模式：必须逐字复制原问题字段，
不得把count改成list或丢失序数/top-N/阈值；同一视觉区域的多条件筛选保持一个证据需求，
而“先确定未知实体、再到另一处查其属性”保留为真正的依赖节点。

24道残余模式探针显示局部语义仍可能丢失排除范围、yes/no阈值、比较集合或总体样本数，
因此`planner-v0.6`为这些证据充分性边界以及成对年份计算增加通用正例；仍未增加新的
Planner schema或数据集专属规则。

最终残余探针促成`planner-v0.7`：强化yes/no阈值的字面保留、已知比较集合的并行关系，
并将“与参照对象相似的对象数量”限制为参照与候选两类证据需求，比较和计数仍由Root执行。

随后在28份本地文档对应的275道真实问题上完成全量生成和逐题计划级审阅。`planner-v0.8`
修复兄弟节点requirements串线、错误比例操作数、排除条件丢失和相似Figure计算节点；
`planner-v0.9`进一步要求共享年份/范围逐节点保留、同一局部来源的多值合并，并硬性禁止
公式计算节点。最终使用同一`planner-v0.9`对275道题完整重跑，全部结构合法，其中7道经过
一次验证纠错；人工审阅标记231道正确、39道可接受、5道错误。未修改Planner schema、
Pydantic验证、SoftDoc或检索逻辑。

随后将Prompt升级为`planner-v0.10`并收缩Planner schema：删除由开发阶段自行引入、与`text`
重复且没有任何Search、Reader或Evidence消费者的`answer_requirements`。SubQuestion现在只用
`text`表达证据需求；相应的重复字段校验和Prompt要求一并删除，DAG、稳定ID、显式Anchor、
节点数与深度边界保持不变。`planner-v0.9`的275题审计保留为历史结果，不冒充新Schema结果。

随后使用`planner-v0.10`重跑旧审计中的44道非完美高风险题，44道均通过结构验证，其中2道
经过一次自动纠错；人工计划级结果为32道正确、3道可接受、9道错误。旧版5道错误中3道修复、
1道改善为可接受、1道仍错误。残余错误集中于“把最终计算值当成文档证据”以及病句中的语义
角色反转，因此冻结结果并保留审计，不再以个案继续膨胀Prompt。

为避免开发题驱动的Prompt过拟合，`planner-v0.11`移除了BBA、500 MHz、Figure 6、Page 42等
来源于MMLongBench审计题的案例，只保留两个虚构示例和八条Planner通用边界。隐式Root明确
不属于SubQuestion、不执行检索且不增加Agent动作；Exact Anchor继续由确定性组件独立处理。

随后以同一`planner-v0.11`在44道历史高风险题上对比Qwen3 4B Instruct-2507与Qwen3 8B。
两者均显式关闭Thinking以隔离参数规模差异；8B可在RTX 4060 Laptop 8GB上100% GPU运行。
4B得到21正确、15可接受、7语义错误、1结构错误，8B得到22正确、12可接受、10语义错误；
平均耗时均约6.3秒。因此暂不切换默认模型，也不根据这批开发题继续追加Prompt案例。

`planner-v0.12`只增加一条与数据集无关的保守原则：当拆分必要性、隐藏公式或语义角色不明确
时，保留完整原问题作为一个SubQuestion，不强行猜测。InitialPlan同时冻结为provisional
计划：原问题Root始终是最终目标，未来Controller必须依据Root级Evidence gap通过既有
`UPDATE_PLAN`做版本化修订，而不能盲信或机械完成初始节点。v0.11的4B实验结果继续作为原始
基线，不冒充v0.12评测。完整说明见`docs/PLANNER_V0_SUMMARY.md`。

根据外部设计复核，`planner-v0.13`继续做通用化收缩：删除差值、比例、总计、百分比等题型
枚举，改为禁止新增“只对其他节点已请求事实执行确定性运算”的节点；同时明确若比较、排序或
选择可由同一局部证据直接回答，它本身仍可作为一个SubQuestion。Planner的模型输出与运行时
`InitialPlan`均删除`explicit_anchors`，确定性Anchor结果由Exact Lookup独立保存。
这些变化未重跑真实模型，不能借用v0.11的结果。

`planner-v0.14`把依赖关系从“未知实体或短语”推广为“前序答案是否为实例化后续证据需求所
必需”，覆盖条件、类别、数值、时期等未知槽位；只要后续需求能够仅凭Root独立搜索，就保持
并行。同时冻结Conservative + Deferred Planning边界：初始Planner不确定时不猜，真实阅读后
再由Evidence gap触发`UPDATE_PLAN`。此后不再按开发错误题扩充Prompt。

冻结决定：`planner-v0.14`成为Initial Planner v0正式冻结版本，Prompt、LLM输出Schema、默认
6节点与4层深度均由快照测试保护。未来只在Reader和Evidence闭环完成后，以相同下游组件和预算
比较No Planner、Initial Planner和Deferred Planner；该实验已登记在`docs/TODO.md`，执行前不
宣称Planner或动态修订具有净收益。

## 已完成但不再保留的大型产物

- representative-14与extension-14的旧SoftDoc版本；
- 旧Dense/RRF embedding cache；
- MinerU Pipeline/Hybrid困难页A/B的中间PDF、图片和JSON；
- 逐页visual audit副本；
- 多轮review packet和一次性候选策略分析。

关键结论已保存在Git、`PROJECT_GUIDE.md`和当前retrieval summary中。上述产物都可由28份
原始PDF、当前代码和本地模型重新生成。

## Reading foundation 冻结

在进入视觉阅读前，冻结以下基础设施：

- `planner-v0.14` 的 Conservative + Deferred Initial Planner 边界；
- parser-neutral TableView：只序列化真实 HTML 单元格，`rowspan/colspan` 由运行时 occupancy grid 解析；
- MinerU Table HTML 和视觉资源恢复；
- 保守的跨页聚合表重组：只有页归属唯一、顺序完整、无跨边界 `rowspan` 且重组校验通过时，才按物理页拆分并生成 confirmed `continued_on`；否则整个组保持原样；
- 删除 `progressive_slide_caption_target_repaired`：该规则曾从相邻幻灯片复制 bbox 并合成 Figure，28 份文档中只触发 2 次，且出现“合成对象继续充当下一次模板”的级联和标题被框入图片的问题。宁可保留可审计的上游漏检，也不把不可靠区域写成确认的 Figure。
- 删除上述 2 个合成对象后，28 份开发文档仍有 1741/1741 个 Figure/Chart/Table/Equation 具备合法 bbox 和可读取的独立视觉资源；原始页面图片仍完整保留，可供后续页面级视觉 fallback 使用。

冻结标签：

```text
softdoc-v0.5-reading-foundation
```

下一阶段只实现 Visual Reading Environment / Observation 边界，不在本次冻结中实现 Evidence Checker 或 Agent 循环。

Visual Reader v0进一步收缩为纯观察器：输入中的`problem`只限定阅读目标，输出只包含可追溯
`observations`和`limitations`。删除设计中的`answer`、`conclusion`和
`supported_by_observation_ids`；可分解的多图事实由后续Evidence/Answerer汇总，无法分解的视觉
关系则直接表示为引用多个`input_id`的关系型Observation。Evidence充分性、最终回答和下一步
导航仍不属于Reader职责。

Reading State v0随后冻结：`ObservationStore`和`ActionTrace`分别作为读取历史与动作历史的
canonical source of truth，`SearchSession`继续保存候选状态，`ExplorationState`只由这些对象
派生为Controller working view。`ReadRecord`保存结构化`inputs`，使单图、多图、整页、Region及
TableView输入可复现；动作内局部`input_id`与稳定资源`visual_asset_id`明确分开，Table
Observation可进一步引用稳定`cell_id`。

Relation入口同时拆成`confirmed_relation_handles`与`candidate_navigation_hints`：前者表示可正常
导航的已确认关系，后者只表示值得调查的目标，不把candidate升级为事实；rejected不暴露。
运行时只新增Session、Action、Observation和Evidence全局ID；Reader输入只使用动作内局部
`input_id`。核心SoftDoc ID保持不变且一律作为不透明handle使用。

第二次接口审计后，`ExplorationState`把`visited`改成语义更准确的`attempted_source_ids`，只保留
实际`attempted_search_queries`；`current_focus`由最近一次成功或degraded且带`primary_target`的
Action确定性派生。Relation handle/hint只暴露与当前focus直接相连的局部关系，不再dump全图。
`EvidenceMemory`明确为Checker可修订的工作记忆，`ready`只代表Evidence集合充分，不代表答案已生成。
随后将Checker输出从完整Memory重写改为`add/replace/remove` delta：Checker输入仍读取完整当前
EvidenceMemory，程序只在副本上应用delta并验证完整结果，成功后一次性提交，避免模型漏抄旧
Evidence造成静默删除。

Reading State v0最终补齐多子问题边界：一个Root Question只建立一份ObservationStore和
EvidenceMemory，但Controller与Checker每轮只聚焦一个`current_question`。Checker输入用
`QuestionContext`同时保存Root与current question；每条Evidence增加`supports_question_ids`，
使共享Memory仍能区分Q1/Q2证据。`current_question_status`与`root_status`正式分离，结构化
`active_gap`同时保存目标问题ID和缺口描述；当前子问题完成不再被误写成整个Root已ready。

全项目一致性复核发现临时`EvidenceCheckResult.current_question_status`原本未写入任何canonical状态。
因此在不修改Planner DAG的前提下，为Root级`EvidenceMemory`增加精简`question_progress`：每次应用
Checker delta时自动更新当前SubQuestion并保留旧状态；当current question就是Root时，程序强制
其状态与`root_status`一致。`ActionTraceEntry`和派生`RecentActionSummary`同时记录
`current_question_id`，跨存储验证器检查ReadRecord与ActionTrace的问题归属一致。

2026-08-22再次收口后，删除了重复的`QuestionContext/current_question_id`状态源：
`EvidenceMemory.questions`现在物化全部已注册问题的文本、依赖与运行状态，唯一
`current_target`同时给出当前问题ID和证据缺口。Checker只评估当前目标并返回delta；程序负责
依赖校验、原子应用以及按Planner稳定顺序选择下一项。Initial Plan耗尽而Root仍不充分时目标回到
Root；Deferred Planner的新问题必须先由程序分配/验证ID、注册进DAG，之后才能激活。

同日冻结`Evidence Checker v1` Prompt与contract。本地`qwen3:8b`合成评测仅用于验证结构化输出、
Pydantic拦截和delta状态循环可以运行；模型质量缺陷不再通过逐题补Prompt处理，而是登记到TODO，
留待服务器模型对照。清理三轮可重建的Checker临时报告、旧运行日志、pytest缓存和Python字节码；
模型缓存、28份真实文档、SoftDoc与检索产物均保留。

随后冻结`answerer-v0.1`。Answerer输入由ready状态的EvidenceMemory确定性物化，只包含Root
Question、精简问题DAG以及`evidence_id/statement/supports_question_ids`；不把Observation文本、
文档位置、检索、动作或Checker上下文交给模型。输出只保留`answer`和`used_evidence_ids`，引用位置
由程序在模型返回后通过Evidence与ObservationStore展开。该阶段只冻结Schema、Builder、验证器和
Prompt，不接真实Answerer模型。

随后将Answerer Prompt更新为`answerer-v0.2`：增加回答前的Evidence一致性与问题
覆盖检查，但不改变输入输出contract。使用同一组14个合成case和本地`qwen3:8b`
复测仍为12/14；模型仍会在相互冲突的Evidence中擅自选值，并把数值上涨本身
当作上涨原因。这两项保留为模型能力/指令遵循风险，不再为单个错例追加题型
规则。

经责任边界复核，Answerer Prompt更新为`answerer-v0.3`：删除Evidence冲突检测和仲裁
要求，因为相互冲突的Evidence应由Checker拦截，不应进入`root_status=ready`的
Answerer输入。Answerer仅保留“Evidence是否直接覆盖Root Question的每一部分”检查。

2026-08-23冻结Controller v0设计边界。删除`ActionTrace/ExplorationState`中的自由文本
`result_summary`，以`outcome + observation_ids`保留可追溯结果；失败细节、证据缺口和Checker
评价分别只存在于`ReadRecord.limitations`、`EvidenceMemory.current_target`和本轮
`EvidenceCheckResult`。同时明确confirmed Relation、candidate navigation hint和Evidence之间的
安全边界：不确定关系只能触发调查，不能直接成为事实。
## 2026-08-23：项目代号确定为 TREVA

项目公开展示名称由“Soft-Structured Document Reading Agent”收敛为 **TREVA**（**Typed-Relation Evidence-guided Visual Agent**）。SoftDoc继续表示解析器无关的文档中间结构，Python包和CLI继续使用`softdoc`，避免品牌调整破坏稳定ID、导入路径与现有脚本。新增`ARCHITECTURE.md`，明确离线文档处理、在线阅读循环以及“导航线索不等于Observation，Observation不等于Evidence”的安全边界。该命名不改变任何SoftDoc语义规则、检索排序或运行时contract。
