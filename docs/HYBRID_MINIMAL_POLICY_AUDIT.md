# Hybrid 最小规则策略与复验报告

日期：2026-08-06

## 1. 本轮结论

MinerU Hybrid high 不能直接替代当前 Pipeline，但可以作为更可信的版面类型来源。当前实现采用双路径：

- Pipeline 输出继续使用原有兼容性修正规则，以覆盖旧解析器在幻灯片、宣传册、表单和财报上的已知缺陷。
- Hybrid 输出跳过旧的 `ElementNormalizer` 规则库，只保留重复页眉页脚确认、明显错误 Heading 排除、标题层级、Section、Relation 和 coverage recovery 等结构必要步骤。
- Hybrid 生成的图片/图表文字完整保存在 Element 中并记录来源，但标为未验证，不进入 BM25 或 Dense 基础索引。

这不是“Hybrid 输出全部可信”。Hybrid 只被信任用于较好的 block 类型和分组；VLM 生成内容、caption/footnote 配对、Section 与跨页 Relation 仍由 SoftDoc 单独处理或留给后续阅读验证。

## 2. 规则修改

### 2.1 修复页眉页脚的错误定义

旧行为会把 MinerU 单页标成 `page_header` / `page_footer` 的元素直接确认为重复页眉页脚。这会误删首次出现的章节标题、手册 Chapter 标题和正文边缘内容。

新行为要求同时满足：

1. 文本跨至少若干页面重复；
2. 位于相似的页面顶部或底部区域；
3. 出现频率达到阈值；
4. 大型幻灯片页标题不会仅因重复而成为页眉。

MinerU 的原始 header/footer 类型现在只是一个信号。单页出现的边缘文本会保留为 Element，但不会自动进入当前语义 Section。

28 份文档中，重复页眉页脚标记从 1,236 个降到 732 个，排除了 504 个缺少跨页重复证据的标记。732 个 decision 与 732 个 Element metadata 完全一致。

### 2.2 Hybrid 最小策略

Hybrid 文档通过 `Document.metadata.parser_backend == "hybrid"` 进入最小策略：

- 不执行旧 `ElementNormalizer`；
- 不用 caption 临时创建 Section；
- 不用末页 checklist 临时创建 Section；
- Heading 候选只排除已确认的重复页眉页脚、空标题、联系方式/URL、明显完整句子等强负例；
- Parser 原始类型仍不是最终 Heading level，层级继续由 HeadingHierarchyBuilder 归一化；
- 不改变 RelationType，也不改变现有 RelationBuilder 阈值。

策略名记录为 `hybrid_parser_assisted_minimal_v1`。本次 12 页 Hybrid 样本的 normalizer firing 为 0；Pipeline 路径仍保留 145 次兼容性修正。

### 2.3 Hybrid 视觉文字隔离

Hybrid 的 image/chart `body` 可能是 VLM 生成内容。Adapter 现在保存：

- `text_source = vlm_generated_visual_description`
- `retrieval_text_status = unverified`
- `generated_visual_text = true`

SearchUnitBuilder 不索引这类内容，跳过原因为 `unverified_generated_visual_text`。图片本体、bbox、crop、label 和 provenance 都仍保留，供后续 `READ_ELEMENT` / `INSPECT_REGION` 使用。

## 3. Pipeline 与 Hybrid 难页复验

同一组 12 个困难页面（论文、Apple 宣传册、Best Buy 财报、幻灯片、图表页和跨页表格）结果如下：

| 指标 | Pipeline | Hybrid high |
|---|---:|---:|
| 有效解析时间 | 31.31 s | 54.15 s |
| 平均每页 | 2.61 s | 4.51 s |
| 峰值 GPU 显存 | 2,129 MiB | 7,188 MiB |
| image/chart blocks | 27 | 32 |
| 生成视觉文字 blocks | 0 | 12 |

Hybrid 约慢 1.73 倍、峰值显存约为 3.38 倍。它明显改善了稀疏幻灯片、列表分组、单独视觉对象和图表可读性，但对论文首页、密集财务表格和跨页表格边界没有稳定优势。

12 条视觉生成内容人工抽查中，8 条基本可靠、3 条部分错误、1 条明显编造。因此当前隔离策略是必要的。

转换后的两组 SoftDoc 均通过验证：

- Pipeline：12 页，143 Elements，14 Sections，415 Relations；
- Hybrid：12 页，145 Elements，19 Sections，421 Relations；
- Hybrid 12 条生成视觉文字全部未进入 SearchUnit。

可视化抽查仍发现一个未解决问题：幻灯片版权行 `© Disciplined Agile Consortium` 被当作 caption 与照片连接。说明 Hybrid 提升 block 类型并不等于功能关系已经正确，caption/footnote Relation 仍需单独评估。

## 4. 28 文档完整重建

新策略重新构建了 28 份开发文档：

- 28/28 Document validate 成功；
- 1,238 页；
- 15,985 Elements；
- Section 总数由 1,976 变为 1,994；
- 新恢复的 Section 主要是以前被错误视作页眉的真实 Chapter、Appendix 和首次出现标题。

输出：

- `data/processed/representative_14/softdoc_policy_v2`
- `data/processed/representative_28_extension/softdoc_policy_v2`

## 5. 检索复验

范围为 28 份文档、275 道题，其中 213 道有 Gold evidence page。这里测量的是“候选 Element 所在页是否命中 Gold 页”，不是回答准确率。

| K | BM25 旧/新 | Dense 旧/新 | Exact-first dual 旧/新 |
|---:|---:|---:|---:|
| 1 | 38.03 / 38.03 | 37.56 / 37.56 | 43.66 / 43.66 |
| 5 | 58.69 / 58.69 | 59.15 / 59.62 | 64.79 / 64.79 |
| 20 | 77.93 / 77.46 | 82.63 / 82.63 | 82.16 / 82.16 |
| 50 | 84.51 / 84.51 | 92.49 / 92.49 | 93.43 / 92.96 |

结论：结构修正没有使检索显著变好或变坏，变化都在 1 道题（0.47 个百分点）范围内。Dense Top-5 略升，BM25 Top-20 和组合 Top-50 各少 1 道。Exact 对有 Gold 的 Anchor 问题命中 17/18。

新报告：`data/processed/representative_28/retrieval_policy_v2/evaluation/retrieval_summary.md`。

本次完整评测的运行时间不能直接与旧报告解释为在线算法变慢：两次运行的缓存状态、索引重建和系统负载不同。排名质量可直接比较，延迟需要单独的 warm-cache benchmark。

## 6. 工程审计

### 保留的必要内容

- Parser-neutral Pydantic models、DocumentStore、serialization；
- MinerU Adapter 与 coverage recovery；
- Heading hierarchy、Section builder；
- RelationBuilder、SpatialNavigator、FloatingContentSectionResolver；
- Exact、SearchUnit、BM25、Dense 检索及其单元测试；
- Pipeline 兼容规则，因为当前 28 份完整文档仍主要来自 Pipeline。

### 仍然存在的技术债

1. `normalization.py` 和 `relations.py` 仍较大。Hybrid 路径已不执行 legacy normalizer，但不能在未完成完整 Hybrid 迁移前直接删除，否则会降低现有 Pipeline corpus 覆盖率。
2. coverage recovery 对 Pipeline 很重要，但重建时成本较高；后续可按输入文件 hash 缓存，不应删除。
3. 当前 retrieval evaluation 是离线全量分析脚本，不是最终在线 SearchSession 延迟实现。
4. Relation 规则对 Hybrid 与 Pipeline 共用；Hybrid 新类型尚未形成专门的、经过 Gold 验证的关系收益。
5. Hybrid 原始产物 `hybrid_high_clean` 主要用于计时，其中部分目录不是标准可转换的 content-list 布局；正式转换应使用带完整 JSON 的 Hybrid artifact。

### 可清理但本轮未删除的产物

为了保留前后对比，本轮没有破坏性删除。确认采用 v2 后，可在旧版和新版中二选一保留：

- `representative_14` / `representative_28_extension` 下旧 `softdoc_final` 与新 `softdoc_policy_v2`；
- `representative_28` 下旧 `retrieval_dense_e5` 与新 `retrieval_policy_v2`；
- A/B 目录中的中间 probe、重复 overlay 和不完整计时输出。

## 7. 验证

- `python -m compileall -q src tests scripts`：通过；
- `pytest -q`：182 passed；
- 28/28 新 SoftDoc：validate 通过；
- 12/12 Hybrid 难页：转换通过；
- Hybrid 未验证视觉文字进入 SearchUnit：0 条。

## 8. 当前建议

现在不应把所有文档切换到 Hybrid high。更稳妥的运行方式是：

1. Pipeline 作为批量默认解析；
2. 对视觉密集、低 coverage、类型明显异常的页按需升级到 Hybrid；
3. Hybrid 视觉生成内容只作为阅读提示，必须经过实际 region read 后才能成为 Evidence；
4. 下一阶段优先实现 SearchSession / CandidatePreview 与 Reading Agent，而不是继续扩大文档类型补丁库。
