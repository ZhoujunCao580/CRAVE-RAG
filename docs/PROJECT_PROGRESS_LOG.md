# Soft-Structured Document Reading Agent：项目阶段日志

> 更新日期：2026-08-11
> 当前阶段：SoftDoc 与 Retrieval Entry 已冻结；正在进入 Reader 工具阶段
> 当前状态：28 份代表性真实 PDF 已完成 SoftDoc；Exact、SearchUnit、BM25、Dense、加权RRF、CandidatePreview 与 SearchSession 已实现并统一评测；尚未实现 Reader、Evidence Checker 和 Agent。
>
> Retrieval and Reading v1 的当前边界与实测结果详见
> [`RETRIEVAL_READING_V1_DESIGN.md`](RETRIEVAL_READING_V1_DESIGN.md)。

## 1. 这份日志的用途

本文记录项目从最初构想到当前版本的设计演进、遇到的问题、解决方案、验证结果和剩余限制。它面向刚接触项目的开发者或研究者，重点回答：

- 这个项目最终想做什么；
- 当前里程碑实际做到了什么；
- 为什么不能直接使用 MinerU 的 JSON；
- SoftDoc 的核心数据结构和关系是怎样设计的；
- 在真实 PDF 上遇到了哪些典型问题；
- 我们增加了哪些模块解决这些问题；
- 哪些能力仍然没有实现，不能误认为已经完成。

这不是逐个 Git commit 的机械列表，而是一份按技术问题组织的研究开发日志。

## 2. 最初目标与当前边界

项目暂定名称为 **Soft-Structured Document Reading Agent**，目标是研究面向多模态长 PDF 的文档阅读和问答系统。

完整系统最终希望具备：

1. 将 PDF 解析为 Page 和 Element；
2. 同时保留页面物理布局与文档语义层级；
3. 保存可解释的、有类型的文档关系；
4. 让 Agent 通过显式工具收集证据；
5. 显式维护证据是否充分；
6. 输出答案、证据引用和行动轨迹。

第一个里程碑——**与具体解析器解耦的 Soft Document Structure 中间表示**——已经冻结。
当前正在其外部实现 Retrieval v1；检索状态不回写 SoftDoc。

当前明确没有实现：

- Reader 工具；
- LLM 或 VLM 调用；
- Agent 循环；
- Evidence Checker；
- 模型训练或强化学习；
- Neo4j 或其他图数据库；
- LangChain / LangGraph；
- 前端。

因此，当前成果是后续阅读 Agent 的文档基础设施，不是一个已经可以端到端回答问题的 RAG 系统。

## 3. 前期调研与关键决策

### 3.1 G2 Reader 与 MAGE-RAG 带来的启发

项目早期参考了 G2 Reader 和 MAGE-RAG。

G2 Reader 的启发主要是：文档不应只是一串文本块，还可以表示为包含页面、元素和关系的结构，Agent 可以沿关系寻找上下文。它更明显地采用图式表达。

MAGE-RAG 的启发主要是：长文档问答需要同时利用版面、页面、文本和视觉内容，不能把 PDF 简单降级成纯文本。

我们的取舍是：

- 接受“文档元素之间存在关系”这一思想；
- 不要求图数据库；
- 不把所有空间邻近都预先建成边；
- Relation 单独保存即可，未来需要时可以映射成图；
- 第一阶段只保存确定性较强、可解释的关系；
- 不生成 `explains`、`supports`、`contradicts`、`semantic_similar` 等需要模型判断的主观关系。

### 3.2 数据集选择

为了尽早发现真实问题，我们没有只依赖人工小样例，而是选择了 MMLongBench-Doc 作为长文档问答数据集候选，并从真实文档开始测试。

后来又建立了 `representative_14`，包含论文、报告、幻灯片、宣传册、表单和手册等多种版式。这样做的目的不是宣称已经完成基准评测，而是用少量但多样的文档暴露解析和结构化问题。

### 3.3 “软结构”而不是“硬知识图谱”

SoftDoc 中的“软”体现在：

- 元素类型和层级被明确保存；
- 关系有 `confidence`、`status`、`created_by` 和 `evidence`；
- 无法完全确定的跨页关系可以保留为 `candidate`；
- 位置关系大多由 bbox 按需计算，而不是永久固化；
- 原始解析器 payload 始终保留，可追溯和重新解释；
- 后续可以加入 DoclingAdapter、模型验证器或 Agent，而无需推翻核心结构。

## 4. 工程环境与基础设施

### 4.1 Python 环境

最初本机已有旧的 `pdf-rag` 环境，但它不是围绕整个项目设计的，也没有完整覆盖 MinerU 与 SoftDoc 的依赖。后来统一建立了项目级 Conda 环境：

```text
multimodal_pdf_rag
```

当前工程要求：

- Python 3.11；
- Pydantic v2；
- pytest；
- `src` 目录结构；
- `pathlib.Path`；
- Windows 和 Linux 均可运行；
- 测试不依赖 GPU；
- 测试不调用外部 API。

MinerU 3.4.4 被放入同一项目环境，因此 MinerU 解析和 SoftDoc 转换可以在一个环境中完成。模型缓存放在 D 盘项目数据目录，避免占用系统盘。未来迁移到 Linux GPU 服务器时，需要根据服务器驱动重新选择 PyTorch/CUDA 版本，不能直接复制本机 CUDA 安装命令。

### 4.2 `pyproject.toml`

`pyproject.toml` 是当前 Python 工程的统一配置文件，负责：

- 包名和版本；
- Python 版本范围；
- 运行依赖；
- MinerU 可选依赖；
- pytest 配置；
- `src` 包发现；
- CLI 入口 `softdoc`。

它让项目可以通过 `python -m pip install -e .` 以可编辑模式安装，也让 Windows 和 Linux 使用相同的包定义。

## 5. 第一版 SoftDoc 核心模型

我们建立了统一的 Pydantic 文档模型：

```text
Document
├── Pages
│   └── Page
│       ├── Heading
│       ├── Paragraph
│       ├── Table
│       ├── Figure
│       ├── Chart
│       ├── Code
│       ├── Algorithm
│       ├── Caption
│       ├── Footnote
│       ├── List
│       └── Equation
├── Sections
└── Relations
```

Pydantic 在这里的作用是：用带类型的 Python 模型描述数据，并在创建、赋值和反序列化时自动验证字段。相比任意嵌套的 `dict`，它可以尽早发现 bbox 非法、关系状态错误、字段缺失等问题。

### 5.1 页面物理结构

每个 Page 保存：

- 页码和页面索引；
- 页面宽高；
- 页面元素 ID；
- reading order；
- 页面图片路径；
- provenance 和 metadata。

每个 Element 保存：

- 所属 `page_id`；
- 原始 bbox；
- 0 到 1 的归一化 bbox；
- reading order；
- 可选列号；
- 原始文本；
- 表格 HTML；
- 图片或裁剪图路径；
- section 归属和 section path；
- 原始解析器来源。

### 5.2 稳定 ID

Document、Page、Element、Section、Relation、BoundingBox 和 Provenance 都使用稳定、确定性的字符串 ID。相同输入和相同来源位置在重复转换时应得到相同 ID，便于：

- JSON round-trip；
- 关系引用；
- 增量调试；
- 将来缓存或索引；
- 对比两次规则修改前后的结果。

### 5.3 BoundingBox

BoundingBox 同时保存原始坐标和归一化坐标，并验证：

- `x1 < x2`；
- `y1 < y2`；
- 坐标必须有限；
- 归一化值必须位于 `[0, 1]`。

原始坐标保证对解析器输出忠实，归一化坐标让不同页面尺寸之间的空间规则可比较。

### 5.4 Provenance

Provenance 保存：

- adapter 名称；
- 来源文件；
- 来源位置；
- parser 版本；
- 原始 payload；
- 附加 metadata。

它解决了一个重要问题：SoftDoc 可以对 MinerU 输出进行归一化，但不能把无法识别或暂时不用的原始信息静默丢弃。

### 5.5 一页 SoftDoc 实际长什么样

概念图中的 `Page -> Element` 是包含关系，但在落盘数据里，Page 不会把所有 Element 再复制一遍。Page 保存本页元素的稳定 ID 和顺序，Element 统一保存在 Document 的 `elements` 数组以及 `elements.jsonl` 中。这样既能从页面进入元素，也避免数据重复。

下面是当前 ACL 真实输出第 2 页的简化示例。为了便于阅读，长 ID、完整 provenance 和部分元素被省略：

```json
{
  "page_id": "doc:2023.acl-long.386:9064dd38:page:0001",
  "document_id": "doc:2023.acl-long.386:9064dd38",
  "page_index": 1,
  "page_number": 2,
  "width": 595.0,
  "height": 841.0,
  "element_ids": [
    "...:page:0001:element:0000:figure:main",
    "...:page:0001:element:0000:caption:figure",
    "...:page:0001:element:0001:paragraph:main",
    "...:page:0001:element:0005:heading:main"
  ],
  "reading_order": [
    "...:page:0001:element:0000:figure:main",
    "...:page:0001:element:0000:caption:figure",
    "...:page:0001:element:0001:paragraph:main",
    "...:page:0001:element:0005:heading:main"
  ],
  "image_path": "assets/pages/9494f8dc2f7f.png"
}
```

其中 Figure 是一个独立 Element：

```json
{
  "element_id": "...:page:0001:element:0000:figure:main",
  "page_id": "...:page:0001",
  "page_number": 2,
  "element_type": "figure",
  "reading_order": 0,
  "bbox": {
    "raw": [70.0, 72.0, 524.0, 266.0],
    "normalized": [0.1176, 0.0856, 0.8807, 0.3163]
  },
  "section_id": "...:section:9a54d1758c47",
  "section_path": ["1 Introduction"],
  "image_path": "assets/elements/cfb240973125.jpg",
  "content_availability": "visual_only",
  "summary": null,
  "keywords": []
}
```

Figure 下方的说明文字不是塞进 Figure 的某个私有字段，而是一个可单独读取、定位和引用的 Caption Element：

```json
{
  "element_id": "...:page:0001:element:0000:caption:figure",
  "page_id": "...:page:0001",
  "element_type": "caption",
  "reading_order": 1,
  "bbox": {
    "raw": [67.0, 277.0, 526.0, 315.0],
    "normalized": [0.1126, 0.3294, 0.8840, 0.3746]
  },
  "section_path": ["1 Introduction"],
  "text": "Figure 1: Overview of our PROGRAMFC model, ...",
  "content_availability": "text_only"
}
```

因此，“一页”可以从三个角度读取：

1. 看 `Page.image_path` 和 bbox，恢复物理版面；
2. 按 `Page.reading_order` 依次取得本页 Element；
3. 沿 Relation 取得 Caption、Footnote、Section、显式引用和跨页延续。

### 5.6 一条 Relation 实际长什么样

同一页 Caption 与 Figure 的配对如下：

```json
{
  "relation_id": "rel:caption_of:45948d02759b",
  "source_id": "...:page:0001:element:0000:caption:figure",
  "target_id": "...:page:0001:element:0000:figure:main",
  "relation_type": "caption_of",
  "confidence": 1.0,
  "status": "confirmed",
  "created_by": "parser",
  "evidence": [
    {
      "rule": "parser_declared_function_target",
      "description": "caption is associated with a compatible document element.",
      "source_ids": ["caption-id", "figure-id"],
      "data": {"cross_page": false}
    }
  ],
  "metadata": {}
}
```

ACL 第 1 页正文写有 “as illustrated in Figure 1”，而 Figure 1 位于第 2 页。SoftDoc 将它表示为跨页显式引用：

```json
{
  "source_id": "...:page:0000:element:0008:paragraph:main",
  "target_id": "...:page:0001:element:0000:figure:main",
  "relation_type": "refers_to",
  "confidence": 1.0,
  "status": "confirmed",
  "created_by": "explicit_reference",
  "evidence": [
    {
      "rule": "explicit_numbered_reference",
      "description": "Text explicitly names Figure 1.",
      "data": {
        "matched_text": "Figure 1",
        "source_page_id": "...:page:0000",
        "target_page_id": "...:page:0001"
      }
    }
  ]
}
```

这里要特别区分：

- `caption_of`：说明文字属于哪个图；
- `refers_to`：正文提到了哪个图；
- `continued_on`：前后两个块是不是同一段、同一表或同一代码块的跨页延续；
- `belongs_to_section`：元素在语义上属于哪个章节。

## 6. Relation 设计的修正

### 6.1 最初问题

早期关系设计容易被理解为主要描述 reading order，不能充分表达文档层级、功能关系和跨页连续性；另一方面，如果给页面内所有元素建立 `near`、`above`、`below`、`same_column` 等边，又会造成关系数量平方增长，而且很多边没有长期价值。

### 6.2 当前 Relation 模型

每条 Relation 包含：

- `relation_id`；
- `source_id`；
- `target_id`；
- `relation_type`；
- `confidence`；
- `status`；
- `created_by`；
- `evidence`；
- `metadata`。

状态包括：

- `confirmed`：规则证据足够强，可以被下游直接使用；
- `candidate`：存在可能性，但当前确定性规则不足以确认；
- `rejected`：保留被否决的判断时使用。

来源包括：

- `parser`；
- `deterministic_rule`；
- `explicit_reference`；
- `layout_heuristic`；
- `llm`；
- `human`。

虽然枚举允许未来使用 `llm`，但当前阶段没有任何关系由 LLM 创建。

### 6.3 永久保存的关系

当前只永久保存：

文档层级与顺序：

- `contains`；
- `next_page`；
- `next_in_reading_order`；
- `belongs_to_section`。

文档功能：

- `caption_of`；
- `footnote_of`；
- `refers_to`。

跨页连续：

- `continued_on`。

明确不保存：

- 全元素两两 `near`；
- `above` / `below`；
- `same_column`；
- `semantic_similar`；
- `explains`；
- `supports`；
- `contradicts`。

## 7. SpatialNavigator：为什么空间关系按需计算

页面中的 bbox 是稳定事实，“A 是否靠近 B”却取决于查询距离和使用场景。因此我们增加 `SpatialNavigator`，实时计算：

- 某元素附近的元素；
- 上方元素；
- 下方元素；
- 同列元素；
- 重叠元素；
- 相邻页面。

例如，查找 Figure 下方的 Caption 时，系统根据两者 bbox、方向和距离计算，而不是提前给页面内所有元素建立 `near` 边。

这样既避免关系爆炸，也保留了未来调整空间阈值的能力。

## 8. MinerU 接入：从“能解析”到“能转换”

### 8.1 MinerU 输出目录与 MinerUAdapter

MinerU 输出目录是 MinerU 对某个 PDF 的解析产物，可能包含：

- `*_middle.json` 或旧版 `layout.json`；
- `*_content_list_v2.json`；
- 原 PDF；
- 页面和元素图片；
- 其他中间文件。

`MinerUAdapter` 是转换层。它读取 MinerU 专属字段，再生成 parser-neutral 的 Document、Page 和 Element。核心模型中不出现 MinerU 专属字段，因此以后可以增加 `DoclingAdapter`，而不修改 SoftDoc 模型。

### 8.2 第一个真实问题：MinerU 成功，Adapter 失败

最初 MinerU 已经能解析测试 PDF，但 Adapter 只适配了人工 fixture 或另一种输出形状，无法转换真实 MinerU 3.4 输出。

解决方式：

- 同时支持旧式 `layout.json` 和 MinerU 3.4 的 `*_middle.json`；
- 使用 middle JSON 获取物理页面几何；
- 使用 content list v2 获取语义元素；
- 对两侧 block 做对齐；
- 渲染原始 PDF 页面作为页面资产；
- 将 MinerU 类型映射为内部 ElementType；
- 在 metadata 中保留 `mineru_type`；
- 在 Provenance 中保留原始 payload。

这一步后才真正完成“MinerU 能识别”到“SoftDoc 能消费”的闭环。

## 9. Adapter 健壮性修复

### 9.1 问题来源

在 14 份代表性文档第一次批量转换时，只有 7 份成功。失败并不是整份文档不可读，而是某个异常 block 触发模型验证错误，导致整份文档终止。

典型异常包括：

- 空 heading；
- 空 table；
- `image_source.path` 指向目录而不是图片文件；
- HTML 和 text 同时为空；
- 有 bbox、页面上也有表格，但没有结构化内容的整页 table；
- 单个 block 字段形态异常。

### 9.2 修复原则

MinerUAdapter 现在遵循“忠实转换、局部降级、整文档继续”的原则：

- 空 Heading 不创建空 Section；
- 无法可靠创建 Heading 时跳过或降级，并保留 warning 和 raw payload；
- 空 Table 仍然创建 Table Element；
- Table 增加 `parse_status` 和 `content_availability`；
- 没有 HTML/text、但有页面图和 bbox 时，生成 fallback crop；
- 此类表格标记为 `visual_only` 或 `degraded`；
- 单 block 转换异常记录为 block-level warning；
- 后续 block 继续转换；
- 只有输入目录或文档级 JSON 完全不可读时才允许整份失败。

### 9.3 结果

为异常场景建立了最小回归 fixture。修复后，`representative_14` 达到 **14/14 转换成功**，并且所有最终 Document 都能通过引用完整性验证。

## 10. Heading Hierarchy 的演进

### 10.1 最初问题

真实 PDF 表明 MinerU 的 `title` 或原始 heading level 不能直接等同于文档标题层级：

- 作者和单位常被识别为 paragraph；
- 粗体文本可能被识别为标题，也可能只是强调；
- Algorithm、Code、Prompt 等内容可能被误当标题；
- References 结构不稳定；
- 幻灯片每页重复的页标题可能被层层嵌套成几十级；
- 页眉页脚可能参与 Section 构建；
- 文档总标题可能被错误当成普通 Section；
- 不同文档类型的视觉样式完全不同。

我们讨论过为每个 Element 增加广泛的 `SemanticRole`，但最终放弃，因为作者、单位、bibliography、claim、evidence 等角色会快速膨胀，而且后续 LLM 仍可能重新解释。当前只解决真正影响结构检索的 Heading Hierarchy。

### 10.2 架构拆分

Section 构建从 MinerUAdapter 内部抽离，当前顺序是：

1. Adapter 忠实生成页面和元素；
2. `SoftDocumentStructureBuilder` 做解析器无关的结构归一化；
3. `HeadingHierarchyBuilder` 决定文档标题和 H1/H2/H3；
4. `SectionBuilder` 根据归一化 heading level 构建 Section。

### 10.3 RepeatedHeaderFooterDetector

为了排除页眉页脚，增加重复区域检测器，综合：

- 跨页重复文本；
- 顶部或底部相似 bbox；
- 出现频率。

被判为 page header/page footer 的元素不参与 Heading 和 Section 构建，但原始元素不会被删除。

### 10.4 标题层级规则

标题归一化的优先信号依次包括：

1. 数字编号，如 `3`、`3.1`、`3.1.1`；
2. 附录编号，如 `A`、`A.1`；
3. `Part`、`Item`、`Chapter` 等明确结构模式；
4. 文档内部一致的字体或 bbox 高度样式；
5. 缩进；
6. Parser 原始 level，且只作为一个信号。

文档总标题进入 `Document.title`，不创建普通 Section。

对于幻灯片，同一视觉样式、跨页反复出现的页标题默认成为同级 Section，不能因为 parser level 波动形成几十层嵌套。

### 10.5 文档类型 Profile

我们没有选择“为每个文件名打补丁”，而是增加 `DocumentProfileDetector`，从布局和内容特征判断：

- academic；
- slides；
- brochure；
- report；
- form；
- manual。

Profile 只调整通用规则的权重和适用条件，不绕过统一模型。它用于处理例如：

- 论文中的编号标题；
- 幻灯片中重复页标题；
- 10-K 中 Item 层级；
- 表单中的题号和表格；
- 手册中的章节与索引。

### 10.6 可检查输出

每份文档输出：

- `debug/document_outline.md`；
- `debug/document_outline.json`；
- `debug/heading_decisions.json`；
- 带 `TITLE`、`H1`、`H2`、`H3` 和原始 MinerU level 的 overlay。

这样标题判断不再只藏在 JSON 中。

## 11. Element 归一化与页面覆盖恢复

### 11.1 Algorithm 与 Code

我们保留 `algorithm` 和 `code` 两种 ElementType：

- Algorithm 表示步骤化算法、伪代码或过程描述；
- Code 表示源代码或更接近可执行语法的代码块。

MinerU 可能在跨页时对同一内容给出不同类型，因此 `continued_on` 判断中将两者视为兼容的 `code_like` 家族，但不会在最终数据中强行合并类型。

Prompt 类内容不再被误当 Heading；在有充分结构信号时可归一化为 Code，并把来源判断写入 metadata，而不是建立无限扩张的 SemanticRole。

### 11.2 ElementNormalizer 与 HeadingEligibilityDetector

后处理增加：

- `ElementNormalizer`：纠正影响阅读方式的类型；
- `HeadingEligibilityDetector`：先判断某个候选是否有资格参与标题层级；
- 再由 HeadingHierarchyBuilder 分级。

这个拆分防止“类型纠正”“能否作为标题”“标题是几级”三个问题混成一组不可解释的规则。

### 11.3 页面覆盖恢复

BigData 第 9 页暴露了另一个问题：页面上有很大的可见文字，但 MinerU 没有生成对应元素。仅仅看 MinerU block 无法恢复它。

因此增加 PDF 原生文本层覆盖恢复：

- 扫描 PDF text layer；
- 判断文本是否已被 MinerU 元素覆盖；
- 对未覆盖且有意义的文本创建可追溯 Element；
- 避免把极端边缘页码、纯装饰字符、视觉容器内部重复文本再次加入；
- 标记来源为 text-layer coverage recovery。

它是 MinerU 的保守补充，不是第二套 OCR，也不调用模型。

## 12. RelationBuilder 的实现与修正

### 12.1 独立 RelationBuilder

关系规则不写死在 Pydantic 模型或 MinerUAdapter 中，而是集中在 `RelationBuilder`，包括：

1. containment；
2. page order；
3. reading order；
4. section membership；
5. caption relations；
6. footnote relations；
7. explicit references；
8. cross-page continuation candidates。

核心模型只定义“关系是什么”，RelationBuilder 决定“怎样生成关系”。

### 12.2 Caption 关系

最初出现真正的 Figure caption 被 MinerU 识别为 paragraph，导致 Figure 没有编号入口，后续 `Figure 1` 引用也无法定位。

解决方式不是修改 MinerU，而是在归一化和标签注册阶段：

- 识别段落开头的 `Figure/Fig./图/Table/表` 编号；
- 判断其是否具备 caption 的位置和版面特征；
- 将编号登记到对应 Figure/Table；
- 创建 `caption_of`；
- 让显式引用可以通过编号找到视觉目标。

Caption 与 Figure/Table 可以位于同页，也允许位于相邻页。

### 12.3 显式 `refers_to`

当前支持：

- `Figure 1a`；
- `Figure 1(a)`；
- `Figure 3b-c`；
- `Fig. 3(b–c)`；
- `图3`；
- `表2`；
- `第4.1节`。

关系允许跨页。没有独立子图节点时，引用回退到主 Figure/Table，同时把子图编号保存在 relation metadata 中。

ACL 第 7 页正文中的 “in Figure 1” 已能生成 confirmed 的跨页 `refers_to`，目标是第 2 页 Figure 1。这种关系不是 `continued_on`：它表示正文引用一个视觉对象，而不是两个元素是同一内容的跨页延续。

### 12.4 FootnoteRelationValidator

最初 parser 给出的 footnote 绑定被直接信任，真实文档中会误连。后来增加 FootnoteRelationValidator：

- parser 绑定只作为一个信号；
- 检查 footnote 文本前缀或标记；
- 检查页面区域；
- 检查 bbox 方向和距离；
- 检查目标元素是否含对应标记；
- 强证据生成 `confirmed`；
- 弱证据只生成 `candidate`；
- 不删除原始 Footnote Element。

后来进一步支持 `footnote_of` 一对多，因为一条共享脚注可能同时解释同页多个图表或表格目标。例如 Independents 报告中的共享注释现在可连接多个目标。

### 12.5 `continued_on`

`continued_on` 表示两个 Element 是同一逻辑内容在后页的延续，例如：

- 跨页段落；
- 续表；
- 跨页代码或算法；
- 跨页列表。

为了避免全篇两两比较，候选范围限制在：

- 当前页与下一页；
- 必要时当前页与后两页；
- 页面尾部元素与后页开头元素；
- 同类型或兼容类型。

段落/列表参考：

- 前页末尾是否缺少句号；
- 后页开头能否形成词法接缝；
- 中间是否出现新标题；
- section、栏位和样式是否一致；
- 是否位于真实页面边界。

表格参考：

- 是否相邻页；
- 表格编号是否一致；
- caption 是否含 `continued` 或“续”；
- 列数与表头是否相似；
- bbox 宽度和列位置是否相似。

Code/Algorithm 参考：

- 是否位于相邻页边界；
- bbox/栏位是否对齐；
- 前块是否缺少块终止；
- 后块是否明显开始新结构；
- caption 是否明确结束或表示 continued。

第一阶段的 `continued_on` 全部保留为 `candidate`，不自动确认。原因是视觉和规则证据可以提供候选，但没有 LLM/VLM 或人工标注时，自动确认的误伤代价较高。Candidate 当前主要用于：

- 调试和人工审计；
- 后续检索时作为可选扩展线索；
- 未来交给模型或人工确认；
- FloatingContentSectionResolver 记录候选 section，但不据此直接迁移。

### 12.6 跨页误报的通用负证据

在 14 份文档中，`continued_on` 一度既漏报又误报。最终没有继续按文件名堆补丁，而是加入通用负证据：

- 不可读 OCR 噪声；
- 独立页码；
- 独立 URL；
- 单独的子图标签行；
- 页边距脚注；
- 过短且语义独立的片段；
- 不处于页边界的大写标题或独立文本；
- 新标题之后的内容。

这些规则修复了 ACL 页脚、epros URL、2210 子图标签、AFE OCR 噪声、DSA 页码/噪声、Independents 页码等误报。

## 13. FloatingContentSectionResolver

### 13.1 为什么需要它

页面顺序中的 Section 归属不一定等于语义归属。典型例子：

```text
第 17 页正文（Section 4）提到 Figure 7
第 18 页出现 Figure 7
但第 18 页顶部已经出现 Section 5 标题
```

如果只按当前位置向上寻找最近标题，Figure 7 会属于 Section 5；语义上它可能属于引用它的 Section 4。这就是“跨页浮动内容的 Section 归属”。

处理对象包括：

- figure；
- chart；
- table；
- code；
- algorithm；
- equation；
- caption；
- footnote。

### 13.2 当前规则

`FloatingContentSectionResolver` 在基础 Section 和关系生成之后运行：

1. confirmed `continued_on`：target 继承 source 的 section；
2. 唯一、明确、距离有限且位于目标之前的 `refers_to`：浮动目标可继承引用正文所属 section；
3. Caption/Footnote：继承明确目标元素的最终 section；
4. candidate `continued_on`：不修改 section，只记录 `candidate_section_id`；
5. 只有空间邻近或页面邻近：不修改；
6. 多个 Section 有同等强度引用：保留原 section，并记录 ambiguous。

每次决定都保存：

- original section；
- resolved section；
- confidence；
- status；
- created_by；
- evidence relation IDs。

完成迁移后重新生成 `belongs_to_section`，保证 Relation 与 `Element.section_id` 一致。

调试输出位于：

```text
debug/section_resolution_decisions.json
```

## 14. 表单、续表与特殊版式问题

哈希命名的表单 PDF 暴露了：

- `title` 和 page header 混淆；
- 大号题号没有被框出；
- 相似内容有时是 list、有时是 paragraph；
- 第 2、3、4 页以及后续多页表格没有连成 continued；
- MinerU 对同一种表单结构跨页给出不同类型。

对应修复包括：

- form 文档 Profile；
- 标题资格与重复页眉分离；
- 对表单题号和列表结构进行归一化；
- 表格身份、列布局、标题和相邻页信号联合判断；
- 允许连续多页形成 Table `continued_on` 链；
- 保留 MinerU 原类型到 metadata，SoftDoc 类型负责统一下游读取。

这类修复被写成“表单类型规则”，而不是针对文件哈希写条件。

## 15. 可视化与人工验收

### 15.1 Overlay 的第一次问题

最初 debug overlay 只有白底和标注框，没有原 PDF 页面。这样无法判断 bbox 和类型是否正确。

后来修复为：

- 从原 PDF 渲染页面图片；
- 在原页面图上叠加 bbox；
- 显示 element type；
- 显示 element ID；
- 显示 reading order；
- 显示 Heading 的归一化等级和原始等级；
- 绘制 `caption_of`、`footnote_of`、`refers_to`；
- 提供跨页关系调试图和 JSON。

### 15.2 当前调试文件

每份文档通常包含：

```text
debug/
├── page_overlays/
├── cross_page_relations.json
├── cross_page_overlays/
├── document_outline.md
├── document_outline.json
├── heading_decisions.json
├── section_resolution_decisions.json
└── adapter_warnings.json
```

其中：

- `page_overlays` 用于检查每页元素；
- `cross_page_relations.json` 用于检查 `refers_to` 和 `continued_on` 等跨页关系；
- `cross_page_overlays` 将跨页两页并排显示并连线；
- outline 和 decisions 用于检查标题与 Section；
- warnings 用于定位降级 block 和未知字段。

### 15.3 验收方式

最终验收不只看 SoftDoc JSON，还配对检查：

1. 原 PDF 页面；
2. MinerU layout 标注；
3. MinerU content list；
4. SoftDoc page overlay；
5. SoftDoc elements/relations；
6. 跨页关系输出；
7. Section outline 与迁移记录。

曾生成过用于人工浏览的临时 contact sheet 和审计中间文件。根据“减少磁盘占用、只保留最终真实文档结果”的要求，这些临时验证产物和旧 `sample_output` 已删除；单元测试及必要的最小 fixture 保留，因为它们是代码回归保护，不是冗余真实数据副本。

## 16. 代表性 14 文档

当前最终输出位于：

```text
data/processed/representative_14/softdoc_final/
```

文档包括：

| 文档 | 类型 | 页数 | 元素数 |
|---|---:|---:|---:|
| 2023.acl-long.386 | academic | 24 | 242 |
| 2024.ug.eprospectus | brochure | 27 | 1102 |
| 2210.02442v1 | academic | 24 | 300 |
| 936c0e… | form | 15 | 127 |
| afe620… | report | 20 | 412 |
| bigdatatrends… | slides | 53 | 277 |
| catvsdog… | slides | 68 | 449 |
| COSTCO_2021_10K | report | 76 | 802 |
| DSA-278777 | report | 21 | 375 |
| e79deb… | report | 17 | 287 |
| earthlinkweb… | slides | 44 | 274 |
| GPL brochure | brochure | 17 | 369 |
| Independents-Report | report | 23 | 224 |
| Macbook_air | manual | 76 | 620 |

合计：

- 14 份 PDF；
- 505 页；
- 5860 个 Element；
- 695 个 Section；
- 17624 条 Relation。

当前关系分布：

| RelationType | 数量 |
|---|---:|
| contains | 6365 |
| next_page | 491 |
| next_in_reading_order | 5355 |
| belongs_to_section | 5010 |
| caption_of | 175 |
| footnote_of | 101 |
| refers_to | 85 |
| continued_on | 42 |

42 条 `continued_on` 当前全部为 candidate。没有生成 `near` 或 `semantic_similar` 永久关系。

## 17. 当前处理流水线

当前从 MinerU 输出到最终 SoftDoc 的执行顺序是：

```text
PDF
  ↓
MinerU
  ↓
MinerUAdapter
  ├─ Page / Element / bbox / reading order
  ├─ 原始 payload 与 warning
  └─ 页面图与元素裁剪图
  ↓
SoftDocPipeline
  ↓
CoverageRecoveryPass
  └─ 原生 PDF 文本层覆盖恢复
  ↓
StructurePass
  ├─ DocumentProfileDetector
  ├─ ElementNormalizer
  ├─ RepeatedHeaderFooterDetector
  ├─ HeadingEligibilityDetector
  ├─ HeadingHierarchyBuilder
  └─ SectionBuilder
  ↓
RelationPass
  ├─ contains / page order / reading order
  ├─ belongs_to_section
  ├─ caption_of / footnote_of
  ├─ explicit refers_to
  └─ continued_on candidates
  ↓
FloatingSectionPass
  ├─ 高置信关系修正浮动内容 Section
  ├─ candidate 只记录建议，不强行迁移
  └─ 重建 belongs_to_section
  ↓
ValidationPass
  └─ DocumentStore.validate_references
  ↓
RuleAuditPass
  └─ 只读收集 rule firing，不修改 Document
  ↓
JSON / JSONL / Assets / Debug Overlays
```

`MinerUAdapter.parse()` 现在只负责 MinerU 到原始 Document 的转换，不再运行 coverage、结构、关系或 Section 修正。`SoftDocPipeline` 是唯一完整编排入口。每个 pass 都实现统一的 `DocumentPass` 接口，声明 `name`、`requires`、`provides` 和 `apply(document, context)`，并返回外置 `PassReport`。

PassReport 和规则审计报告不写回 Document，因此不会改变 SoftDoc JSON 语义。冻结 tag 与新 Pipeline 对 14 份完整 `document.json` 的严格深比较结果为 14/14 相等、0 个字段差异。

## 18. 主要源代码模块

`src/softdoc/` 下的文件分工如下：

| 文件 | 作用 |
|---|---|
| `models.py` | Pydantic 核心模型、枚举和字段验证 |
| `ids.py` | 稳定 ID 生成 |
| `parser.py` | 统一 DocumentParser Protocol |
| `adapters/mineru.py` | 将 MinerU 输出忠实转换为 SoftDoc |
| `pipeline.py` | DocumentPass、PassReport 与唯一 SoftDocPipeline 编排入口 |
| `coverage.py` | 从 PDF 原生文本层保守恢复 MinerU 漏掉的文本 |
| `profiles.py` | 判断 academic/slides/form/report 等文档 Profile |
| `normalization.py` | Element 类型归一化 |
| `repetition.py` | 重复页眉页脚检测 |
| `eligibility.py` | 判断元素能否参与 Heading 构建 |
| `hierarchy.py` | 归一化 Document title 和 Heading level |
| `sections.py` | 根据 Heading 构建 Section tree 和元素初始归属 |
| `outline.py` | 生成可读文档大纲 |
| `labels.py` | Figure/Table/Section 等编号注册与查找 |
| `relations.py` | RelationBuilder、Footnote validator 和跨页规则 |
| `floating_sections.py` | 修正浮动元素的语义 Section 归属 |
| `spatial.py` | 基于 bbox 的实时空间查询 |
| `store.py` | 内存 DocumentStore、关系跟随和引用验证 |
| `serialization.py` | 输出和加载 JSON/JSONL/debug 文件 |
| `visualization.py` | 页面与跨页关系 overlay |
| `structure.py` | 编排解析器无关的结构后处理流水线 |
| `rule_audit.py` | 稳定 rule_id、规则触发统计和覆盖报告 |
| `semantic_diff.py` | 完整 JSON 语义深比较与差异报告 |
| `cli.py` | `softdoc parse-mineru` 和 `softdoc validate` |

`src/soft_structured_document.egg-info/` 如果存在，是 Python 可编辑安装生成的包元数据，不是业务代码。

`tests/` 保存单元测试和最小 MinerU fixture。`.pytest_cache/` 是 pytest 为加速重复测试保存的缓存，可安全删除，删除后下次测试会自动创建。

## 19. 测试与结构验证

当前测试覆盖：

- Pydantic JSON 序列化/反序列化；
- 稳定 ID；
- bbox 合法性；
- relation 端点有效性；
- caption/footnote 配对；
- Section membership；
- 跨页 `refers_to`；
- `continued_on` candidate；
- SpatialNavigator；
- MinerU 异常 block 降级；
- Heading hierarchy；
- form 归一化；
- coverage recovery；
- FloatingContentSectionResolver；
- CLI 和输出文件；
- 所有 DocumentPass 的幂等性；
- 规则审计只读性、稳定 rule_id 和稳定报告；
- 完整 JSON 语义 diff；
- JSON round-trip 后关系不变。

最近一次完整测试结果为：

```text
127 passed
```

14 份最终输出的结构不变量检查结果：

- 14/14 转换成功；
- validation error 为 0；
- 没有重复 ID；
- 没有悬空 relation source/target；
- Page 引用的 Element 均存在；
- Element 引用的 Page 均存在；
- Section 引用有效；
- `belongs_to_section` 与最终 `Element.section_id` 一致；
- 505/505 页面资产和 overlay 存在且尺寸匹配；
- 14 份 layout PDF 与 SoftDoc 页数一致。
- 冻结 tag 与新 Pipeline 的 14 份完整 JSON 语义 diff 为 14/14 相等；
- `rule_coverage_report.json/.md` 记录 rule_id、触发次数、受影响文档/元素/关系、状态和 evidence。

这些结果证明结构完整性较高，但不等于在人工 ground truth 上测得了 100% 语义准确率。

## 20. 已知剩余问题

当前仍有以下已知限制：

1. **ACL References 多栏 bbox 不完整**  
   第 10–13 页 References 的完整语义文本保存在 List 中，但 bbox 主要覆盖第一栏片段，因此 overlay 不能完整反映多栏参考文献的空间范围。

2. **部分源文档 OCR 噪声**  
   AFE 前几页和 DSA 个别页面存在 MinerU/OCR 噪声。SoftDoc 保留页面图、bbox、原始 payload 和降级状态，但没有训练 OCR 模型修复字符。

3. **AFE 重复区域检测仍有边界案例**  
   AFE 第 6 页有少量有效内容可能被标记为 FOOTER。元素没有被删除，但不会进入 Heading/Section 构建。

4. **空白页保留空元素**  
   MacBook 第 4、24、38、74、76 页在原 PDF 中确实近乎空白。MinerU 生成过全页空 paragraph，SoftDoc 将其保留并标记 `unavailable`，没有真实内容丢失。

5. **宣传册和幻灯片元素较碎**  
   密集视觉页面可能被拆成多个较小 Element。页面本身没有丢失，但粒度未必总是最适合未来问答。

6. **`continued_on` 仍是候选，不是人工真值**  
   当前 42 条关系经过通用正负证据过滤，但仍全部是 candidate。未来需要人工标注集、LLM/VLM 验证或下游任务评估决定确认策略。

7. **尚无定量语义准确率**  
   当前“效果较高”来自逐页视觉审计、关系抽查、回归测试和结构不变量，不是基于完整人工标注的 precision/recall。

## 21. 当前 Git 状态

仓库已经有过以下阶段提交：

```text
chore: integrate MinerU project environment
fix: add MinerU pipeline OCR runtime dependency
feat: support MinerU 3.4 soft document conversion
feat: normalize document heading hierarchy
feat: detect cross-page code continuations
feat: freeze soft document milestone 1
```

Milestone 1 冻结提交为 `41419a7`，对应 annotated tag：

```text
softdoc-v0.1-dev
```

Pipeline 边界统一、pass 幂等性、规则审计和语义 diff 作为冻结 tag 之后的独立工程提交保存，没有重写历史。交接时仍应先查看 `git status`，不得使用 `git reset --hard`。

## 22. 当前推荐的暂停点

Milestone 1 已形成一个可交接的研究原型：

- 有解析器无关模型；
- 有真实 MinerU Adapter；
- 有标题和 Section；
- 有高置信功能关系与可审计候选关系；
- 有浮动内容 Section 修正；
- 有空间查询；
- 有完整输出、验证和可视化；
- 有 14 类真实文档结果；
- 有 127 个回归测试。

在进入下一阶段前，最值得做的不是继续添加启发式补丁，而是：

1. 建立一小套人工标注的 Element/Heading/Relation ground truth；
2. 测量各文档类型的 precision/recall；
3. 确定哪些错误真正影响 MMLongBench-Doc 问答；
4. 再决定检索是按 Element、Section、Page 还是多粒度组合；
5. 最后才设计 Agent 的 SEARCH、READ_ELEMENT、FOLLOW_RELATION、INSPECT_REGION、CHECK_EVIDENCE、UPDATE_PLAN、ANSWER。

当前应停止在这里，不继续实现检索、embedding、Agent、LLM/VLM 或模型训练。

> **后续决策说明：** 上述内容记录的是 Milestone 1 当时的暂停点。
> 2026-08-04 已完成 Exact、SearchUnit、BM25 和 Dense，并将 Relation navigation
> 明确放在 READ 之后，而不是 SEARCH 阶段自动扩展。当前边界以
> [`RETRIEVAL_READING_V1_DESIGN.md`](RETRIEVAL_READING_V1_DESIGN.md) 为准。

## 23. 常用入口

激活环境：

```powershell
Set-Location "D:\claude_code_project\multimodal_pdf_rag"
& "D:\Anaconda\shell\condabin\conda-hook.ps1"
conda activate multimodal_pdf_rag
```

运行测试：

```powershell
python -m pytest -q
```

转换一份 MinerU 输出：

```powershell
softdoc parse-mineru MINERU_OUTPUT_DIR --output OUTPUT_DIR
softdoc validate OUTPUT_DIR
```

查看 14 份当前结果：

```text
data/processed/representative_14/softdoc_final/
```

运行统一检索评测：

```powershell
python scripts/evaluate_representative_retrieval.py --device cuda
```

最重要的人工检查入口：

```text
debug/page_overlays/
debug/cross_page_relations.json
debug/cross_page_overlays/
debug/document_outline.md
debug/heading_decisions.json
debug/section_resolution_decisions.json
```

## 24. Retrieval v1 实现与评测（2026-08-04）

Milestone 1 冻结后，检索代码放在独立的 `softdoc.retrieval` 包中，没有修改
SoftDoc 语义规则或 Relation 状态。当前实现：

- Exact Anchor Lookup；
- 模型无关 SearchUnitBuilder；
- BM25 Element 排名；
- 可注入 Encoder 的 multilingual-E5 Dense 排名；
- E5 512-token 无损分片保护；
- 与文本、模型、tokenizer 和索引版本绑定的 Embedding cache。

真实运行覆盖 14 份 PDF、505 页、5860 Elements，生成 5242 SearchUnits 和
5265 DenseSegments。仅 23 个 SearchUnit 需要 E5 内部二次切分，没有静默截断。

在 142 道对应的 MMLongBench 问题中，107 道提供 Gold 物理证据页。统一评测结果：

| K | BM25 | Dense | 双通道轮转 | Exact优先+双通道 |
|---:|---:|---:|---:|---:|
| 1 | 38.32% | 37.38% | 38.32% | 43.93% |
| 5 | 63.55% | 57.94% | 68.22% | 69.16% |
| 10 | 74.77% | 63.55% | 74.77% | 75.70% |
| 20 | 85.05% | 77.57% | 83.18% | 84.11% |
| 50 | 92.52% | 94.39% | 94.39% | 95.33% |

Exact 共识别 14 道 Anchor 问题，其中 9 道有 Gold 页，Gold 页命中为 9/9。
BM25 与 Dense 存在明显互补，但简单轮转不能在所有预算下超过最佳单通道，因此
暂不实现 RRF，先保留两条来源并研究 CandidatePreview 与批次策略。

这些指标只表示候选是否来自 Gold 页。数据集没有 Gold Element ID，当前也没有
Reader 和答案生成，因此不能解释为最终 QA 准确率。

本轮清理后完整测试套件为 180 个测试；没有删除仍覆盖核心模型、Pipeline、关系、
空间查询、Exact、SearchUnit、BM25 或 Dense 行为的源码测试。

## 25. 加权 RRF、SearchSession 与 CandidatePreview（2026-08-11）

在 representative-28 的完整 Element 排名上重新比较稳定轮转和 5 组 RRF 参数后，
默认候选策略由稳定轮转更新为：

```text
Exact handles 单独保留
普通候选 RRF = 1.0 / (20 + bm25_rank) + 1.25 / (20 + dense_rank)
```

旧轮转仍作为可复现基线保留。当前 Exact + RRF 的 Gold 证据页入口命中为：
Hit@1 48.36%、Hit@5 68.54%、Hit@10 74.65%、Hit@20 84.04%、Hit@50 92.96%。
这不是答案准确率，也不是 Gold Element 指标。

同时完成了可序列化 `SearchSession` 和确定性 `CandidatePreview`：候选完整排名不因
默认 5 条展示批次而丢失，游标可继续取下一批，已经展示和打开的候选可追踪。
Preview 只显示原始索引片段与位置，不包含完整 Element、Relation 目标、LLM 摘要或证据。

困难页面的 Pipeline/Hybrid 严格配对检索只有 3 道可核验问题，两者 RRF Top-1 均为
3/3，暂时没有证据证明 Hybrid 能提高检索；结合更高显存、耗时和生成视觉描述幻觉风险，
当前继续以 Pipeline 为默认，Hybrid 仅保留为可选后端。

完整边界、数据模型、风险与下一步见：
[`END_TO_END_PIPELINE_AUDIT_20260811.md`](END_TO_END_PIPELINE_AUDIT_20260811.md)。

冻结前验证：28/28 SoftDoc 引用完整性通过，193/193 pytest 通过，`compileall` 通过，
`pip check` 未发现依赖冲突。

## 26. Retrieval schema收尾（2026-08-11）

在不改变Exact、BM25、Dense和RRF候选顺序的前提下，完成CandidatePreview展示层收尾：

- `display_label`显式保留Figure/Table/Heading等可读标签；
- `snippet_char_start/end`明确属于`SearchUnit.search_text`，不再暗示原始Element坐标；
- `preview_source`记录实际使用BM25或Dense生成片段；
- `match_scope`区分content、metadata、mixed和unknown；
- BM25只命中Section path/label而Dense覆盖正文时，Preview展示Dense正文；
- Session trace统计两路前5项metadata-only数量，但不影响分数和排序；
- 无label且无文本的visual-only对象继续不建立伪造SearchUnit。

28文档、213道可核验问题的完整评测结果与冻结前逐项相同：Exact+RRF的Hit@1/5/20/50
仍为48.36%/68.54%/84.04%/92.96%。BM25前5项的metadata-only命中为25/1375
（1.82%，涉及15题），Dense为0/1375，未发现需要立即调整字段权重的系统性问题。
完整测试为195/195通过。
