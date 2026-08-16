# TableView 设计、实现状态与 Reader 暂缓决策

TableView 是从 SoftDoc Table Element 派生出的确定性结构，不修改原始
SoftDoc，也不实现 Reader。本文同时记录已经冻结的坐标语义、资源边界和
representative-28 的生成结果。

## 1. 当前确认的事实

SoftDoc的Table已经保留：

- 原始HTML；
- 整个Element的bbox和外层图片；
- 页面、Section和Provenance；
- HTML中真实存在的内部`img src`。

必须严格区分：

- `<img src="...">`：存在一个可解析路径；只有路径对应的文件真实存在时，才是**已解析内部图片**。
- `<image 1>`或正文中的`image`、`figure`：只是文本或占位信号，不能证明存在可访问的内部图片资源。
- `Element.image_path`：整个Table/复合Element的外层图片，不等于HTML内部图片。

## 2. 最小TableView草案

TableView是从`Element.html`确定性生成的派生视图，不替换、不修改SoftDoc Element。

```json
{
  "table_view_id": "table-view:<element_id>",
  "element_id": "<stable Element ID>",
  "row_count": 0,
  "column_count": 0,
  "cells": [
    {
      "cell_id": "<element_id>#r0c0",
      "row": 0,
      "column": 0,
      "rowspan": 1,
      "colspan": 1,
      "text": null,
      "visual_asset_ids": []
    }
  ],
  "visual_assets": [
    {
      "visual_asset_id": "<stable visual ID>",
      "path": "assets/elements/example.jpg",
      "source": "html_img_src"
    }
  ],
  "outer_visual_path": "assets/elements/whole_table.jpg"
}
```

约束：

- 空单元格仍存在：`text=null`且`visual_asset_ids=[]`。
- 单元格可以只有文字、只有图片、同时有文字和图片，或者完全为空。
- `rowspan`和`colspan`直接来自HTML，不推断表头。
- `visual_assets`只收录文件真实存在的`img src`。
- `<image 1>`不会伪装成`visual_asset`。
- 当前不增加`header_rows`、`header_columns`、Reader状态或LLM摘要。

### 2.1 合并单元格坐标语义

- `cells`只序列化HTML中真实存在的`td`/`th`，即span的anchor cell。
- `rowspan`/`colspan`覆盖的位置不创建重复的假cell。
- 运行时构造不序列化的occupancy grid；例如`r0c0`的cell具有
  `rowspan=2`时，`get_cell_at(1, 0)`仍返回这个anchor cell。
- Evidence以后引用的是anchor cell ID与坐标，不会同时出现“真实cell”和
  “展开后的假cell”两套身份。

### 2.2 资源路径与Controller句柄

- TableView内部继续保存`Path`，用于本地持久化、验证和跨平台解析。
- `visual_asset_id`是稳定资源身份；未来Environment向Controller暴露
  `visual_asset_id`或handle，而不要求模型理解Windows/Linux磁盘路径。
- 这属于未来Agent API边界，本阶段不实现Controller接口。

### 2.3 转换正确不等于上游识别正确

TableView保证“忠实地把当前MinerU HTML变成可寻址结构”，但不会猜测或修正
MinerU的OCR、行列分割和类型判断。例如：

- 普通财务表可以与crop逐行逐列对应；
- 勾号若已被MinerU识别成`V`，TableView仍保存`V`；
- 视觉上十几条记录的复杂表若被MinerU HTML拆成40行，TableView仍忠实得到
  40行。这属于上游表格恢复质量问题，不是TableView materialization错误。

## 3. 真实普通Table示例

来源：COSTCO 2021 10-K，第5页会员数量表。

```text
Element ID:
doc:costco_2021_10k:2a296fa7:page:0004:element:0006:table:main

HTML grid: 6 rows x 4 columns
Internal img src: 0
Outer Element image: assets/elements/24c39219a831.jpg
```

TableView中的关键内容如下（仅截取部分cells展示结构）：

```json
{
  "table_view_id": "table-view:doc:costco_2021_10k:2a296fa7:page:0004:element:0006:table:main",
  "element_id": "doc:costco_2021_10k:2a296fa7:page:0004:element:0006:table:main",
  "row_count": 6,
  "column_count": 4,
  "cells_excerpt": [
    {
      "cell_id": "...#r0c0",
      "row": 0,
      "column": 0,
      "rowspan": 1,
      "colspan": 1,
      "text": null,
      "visual_asset_ids": []
    },
    {
      "cell_id": "...#r0c1",
      "row": 0,
      "column": 1,
      "rowspan": 1,
      "colspan": 1,
      "text": "2021",
      "visual_asset_ids": []
    },
    {
      "cell_id": "...#r1c0",
      "row": 1,
      "column": 0,
      "rowspan": 1,
      "colspan": 1,
      "text": "Gold Star",
      "visual_asset_ids": []
    },
    {
      "cell_id": "...#r1c1",
      "row": 1,
      "column": 1,
      "rowspan": 1,
      "colspan": 1,
      "text": "50,200",
      "visual_asset_ids": []
    }
  ],
  "visual_assets": [],
  "outer_visual_path": "assets/elements/24c39219a831.jpg"
}
```

## 4. 真实带内部图片Table示例

来源：2021 Apple Catalog，第11页产品对比表。

```text
Element ID:
doc:2021-apple-catalog:a5ef0dd6:page:0010:element:0000:table:main

HTML grid: 12 rows x 4 columns
Valid internal img src: 3
Outer Element image: assets/elements/a3f3a51e7f62.jpg
```

第一行的三个产品图片分别属于三个单元格：

```json
{
  "table_view_id": "table-view:doc:2021-apple-catalog:a5ef0dd6:page:0010:element:0000:table:main",
  "element_id": "doc:2021-apple-catalog:a5ef0dd6:page:0010:element:0000:table:main",
  "row_count": 12,
  "column_count": 4,
  "cells_excerpt": [
    {
      "cell_id": "...#r0c0",
      "row": 0,
      "column": 0,
      "rowspan": 1,
      "colspan": 1,
      "text": null,
      "visual_asset_ids": []
    },
    {
      "cell_id": "...#r0c1",
      "row": 0,
      "column": 1,
      "rowspan": 1,
      "colspan": 1,
      "text": null,
      "visual_asset_ids": ["visual:1e13ad3c4e38"]
    },
    {
      "cell_id": "...#r1c1",
      "row": 1,
      "column": 1,
      "rowspan": 1,
      "colspan": 1,
      "text": "One55",
      "visual_asset_ids": []
    },
    {
      "cell_id": "...#r5c1",
      "row": 5,
      "column": 1,
      "rowspan": 1,
      "colspan": 1,
      "text": null,
      "visual_asset_ids": []
    }
  ],
  "visual_assets": [
    {
      "visual_asset_id": "visual:1e13ad3c4e38",
      "path": "assets/elements/1e13ad3c4e38.jpg",
      "source": "html_img_src"
    },
    {
      "visual_asset_id": "visual:a9e7189deb63",
      "path": "assets/elements/a9e7189deb63.jpg",
      "source": "html_img_src"
    },
    {
      "visual_asset_id": "visual:818b0342b23a",
      "path": "assets/elements/818b0342b23a.jpg",
      "source": "html_img_src"
    }
  ],
  "outer_visual_path": "assets/elements/a3f3a51e7f62.jpg"
}
```

## 5. 跨页聚合表的物理页恢复

MinerU有时把多页表格的完整HTML挂在第一页Table上，同时后续页仍保存各自的
局部Table。当前在Adapter之后运行保守的reconciliation pass：

- 只有各页行能唯一、按顺序、完整重组聚合HTML时才拆分；
- 允许识别可验证的重复表头；
- 跨页`rowspan`、歧义匹配或缺行时不修改任何Element；
- 每页Table只保存本页HTML，原聚合HTML继续保留在Provenance；
- 相邻物理片段生成`confirmed continued_on`；
- 诊断写入`debug/cross_page_table_reconciliation.json`。

因此“不拆并把完整表视为第一页内容”只是校验失败时的安全回退，不是正常输出；
这样既不伪造后续页为空，也不会在不确定时强拆MinerU结果。

## 6. 为什么暂时不实现Reader

最初考虑Reader，是因为我们希望把MinerU HTML转换成可查询的行、列和单元格，并在结构失败时调用视觉fallback。

检查真实数据后，结论发生了收缩：

1. 多数正常表格的HTML已经保存了行列结构；确定性HTML解析足以生成TableView。
2. 带有效`img src`的Table也可以确定性地把图片绑定到具体单元格。
3. 真正困难的是内部图片丢失、类型误判、跨页表格和视觉fallback；这些问题不是增加一组Reader状态字段就能解决。
4. 如果Reader自动决定continued_on、candidate relation或是否调用VLM，它会与Controller职责重叠，并提前引入大量复杂逻辑。

因此当前方案是：

```text
SoftDoc Table Element
        ↓
确定性HTML解析
        ↓
TableView（行、列、单元格、已解析内部图片）
        ↓
未来由Controller直接读取和决策
```

暂时不做：

- 专门Table Reader模型；
- Reader自动调用VLM；
- Reader自动沿continued_on或candidate relation导航；
- 对`<image N>`进行虚假的图片绑定；
- 为静默视觉遗漏增加新检测模块。

以后只有实验表明“Controller直接读取TableView”明显不够时，才实现Reader，并比较答案质量、证据完整性、动作数和视觉调用成本。

## 7. 希望外部审查的问题

1. 这个最小TableView是否足以表示普通表格、合并单元格和带真实内部图片的表格？
2. `cells + visual_assets + outer_visual_path`是否比直接保留HTML更适合Controller和Evidence引用？
3. 是否应当继续保持`<image N>`仅为未解析信号，而不创建假的visual asset？
4. 在不实现专门Reader的情况下，让Controller直接读取TableView是否合理？
5. 对silent visual omission，是否应等真实任务证明影响严重后再设计低成本检测机制？

## 8. representative-28 全量生成结果

当前实现已对28份文档的全部403个Table Element生成TableView：

- 403个TableView；
- 19,086个真实HTML anchor cells；
- 13个Table含可解析内部图片，共52个visual assets；
- 379个结构化纯文本Table；
- 11个Table仍含未解析的`<image N>`占位符；
- 自动结构检查失败数为0。

跨页恢复共确认17组、40个物理Table片段，并生成23条相邻片段间的
`confirmed continued_on`。内部图片从55降到52，是因为3个原本错误泄露到
前一页聚合HTML中的跨页图片现已回到实际物理页片段，而不是资源丢失。

自动检查覆盖：一表一视图、独立HTML解析器坐标一致、资源存在、span
occupancy、幂等构建和JSON round-trip。它们不证明MinerU已经正确识别PDF中
的每一行、每一列和每一个值；后者仍需将outer crop、原始HTML和TableView
三栏进行视觉对照。
