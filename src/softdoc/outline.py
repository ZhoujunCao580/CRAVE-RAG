"""Readable document-outline artifacts derived from normalized Sections."""

from __future__ import annotations

from pydantic import Field

from softdoc.models import Document, SoftDocModel


class OutlineSection(SoftDocModel):
    section_id: str
    title: str
    level: int
    page_number: int
    heading_element_id: str
    children: list["OutlineSection"] = Field(default_factory=list)


class DocumentOutline(SoftDocModel):
    document_id: str
    title: str | None = None
    sections: list[OutlineSection] = Field(default_factory=list)


def build_document_outline(document: Document) -> DocumentOutline:
    elements = {element.element_id: element for element in document.elements}
    nodes = {
        section.section_id: OutlineSection(
            section_id=section.section_id,
            title=section.title,
            level=section.level,
            page_number=elements[section.heading_element_id].page_number,
            heading_element_id=section.heading_element_id,
        )
        for section in document.sections
    }
    roots: list[OutlineSection] = []
    for section in document.sections:
        node = nodes[section.section_id]
        if section.parent_section_id and section.parent_section_id in nodes:
            nodes[section.parent_section_id].children.append(node)
        else:
            roots.append(node)
    return DocumentOutline(
        document_id=document.document_id,
        title=document.title,
        sections=roots,
    )


def outline_markdown(outline: DocumentOutline) -> str:
    lines = [f"# {outline.title or outline.document_id}", ""]

    def append_nodes(nodes: list[OutlineSection], depth: int) -> None:
        for node in nodes:
            indent = "  " * depth
            lines.append(
                f"{indent}- H{node.level} {node.title} "
                f"(page {node.page_number}, `{node.section_id}`)"
            )
            append_nodes(node.children, depth + 1)

    append_nodes(outline.sections, 0)
    return "\n".join(lines).rstrip() + "\n"
