# TODO

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

## 4. 是否需要专门 Reader

- [ ] 先比较“Controller直接读取Element/TableView”和“专门Reader返回Observation”；只有后者明显改善大表、复杂视觉内容或成本时才实现Reader。
- [ ] 若实现Reader：Reader只报告本次读到的内容与限制；`continued_on`缺失、`candidate` Relation和文本/HTML读取失败后的视觉fallback，均交给Controller决策，不自动导航或调用VLM。
- [ ] 用答案质量、证据完整性、动作数、视觉调用和成本决定是否保留Reader。
