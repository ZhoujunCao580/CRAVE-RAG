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

## 已完成但不再保留的大型产物

- representative-14与extension-14的旧SoftDoc版本；
- 旧Dense/RRF embedding cache；
- MinerU Pipeline/Hybrid困难页A/B的中间PDF、图片和JSON；
- 逐页visual audit副本；
- 多轮review packet和一次性候选策略分析。

关键结论已保存在Git、`PROJECT_GUIDE.md`和当前retrieval summary中。上述产物都可由28份
原始PDF、当前代码和本地模型重新生成。

## 下一冻结目标

Reader v1：`READ_ELEMENT / READ_PAGE / INSPECT_REGION / FOLLOW_RELATION`，统一输出
ReadObservation，但暂不实现Evidence Checker或Agent循环。
