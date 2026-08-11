"""Parser-neutral orchestration for deterministic SoftDoc passes."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, ClassVar, Protocol, Sequence, runtime_checkable

from pydantic import Field

from softdoc.coverage import recover_pdf_text_layer_coverage
from softdoc.floating_sections import FloatingContentSectionResolver
from softdoc.models import Document, SoftDocModel
from softdoc.page_labels import PageLabelResolver
from softdoc.parser import DocumentParser
from softdoc.relations import RelationBuilder
from softdoc.rule_audit import collect_rule_coverage
from softdoc.store import DocumentStore
from softdoc.structure import SoftDocumentStructureBuilder


logger = logging.getLogger(__name__)


class PassContext(SoftDocModel):
    """Runtime-only context shared by pipeline passes.

    The context and reports are deliberately kept outside ``Document`` so a
    pipeline refactor cannot silently change the serialized SoftDoc semantics.
    """

    input_path: Path
    output_dir: Path
    available: set[str] = Field(default_factory=set)
    metadata: dict[str, Any] = Field(default_factory=dict)


class PassReport(SoftDocModel):
    """Deterministic report returned by one pipeline pass."""

    name: str
    requires: list[str] = Field(default_factory=list)
    provides: list[str] = Field(default_factory=list)
    changed: bool
    skipped: bool = False
    before_fingerprint: str
    after_fingerprint: str
    details: dict[str, Any] = Field(default_factory=dict)


@runtime_checkable
class DocumentPass(Protocol):
    """Uniform interface for every document post-processing pass."""

    name: str
    requires: frozenset[str]
    provides: frozenset[str]

    def apply(self, document: Document, context: PassContext) -> PassReport:
        """Apply this pass in place and return an external audit report."""
        ...


class PipelineResult(SoftDocModel):
    """Document plus runtime pass reports."""

    document: Document
    pass_reports: list[PassReport] = Field(default_factory=list)


class _BasePass:
    name: ClassVar[str]
    requires: ClassVar[frozenset[str]] = frozenset()
    provides: ClassVar[frozenset[str]] = frozenset()

    def _report(
        self,
        document: Document,
        *,
        before: str,
        skipped: bool = False,
        details: dict[str, Any] | None = None,
    ) -> PassReport:
        after = document_fingerprint(document)
        return PassReport(
            name=self.name,
            requires=sorted(self.requires),
            provides=sorted(self.provides),
            changed=before != after,
            skipped=skipped,
            before_fingerprint=before,
            after_fingerprint=after,
            details=details or {},
        )


class CoverageRecoveryPass(_BasePass):
    """Recover native PDF text omitted by the parser, when a source PDF exists."""

    name = "coverage_recovery"
    requires = frozenset({"raw_document"})
    provides = frozenset({"coverage_recovered"})

    def apply(self, document: Document, context: PassContext) -> PassReport:
        before = document_fingerprint(document)
        if "coverage_recovery" in document.metadata:
            return self._report(
                document,
                before=before,
                skipped=True,
                details={"reason": "coverage_recovery_already_present"},
            )

        source_pdf = _find_source_pdf(context.input_path)
        if source_pdf is None:
            return self._report(
                document,
                before=before,
                skipped=True,
                details={"reason": "source_pdf_not_found"},
            )

        try:
            result = recover_pdf_text_layer_coverage(
                document,
                source_pdf.resolve(),
            )
            document.metadata["coverage_recovery"] = {
                "source": "native_pdf_text_layer",
                "scanned_line_count": result.scanned_line_count,
                "recovered_count": result.recovered_count,
                "recovered_element_ids": list(result.recovered_element_ids),
            }
            details = {
                "source_pdf": source_pdf.as_posix(),
                "scanned_line_count": result.scanned_line_count,
                "recovered_count": result.recovered_count,
            }
        except Exception as exc:
            warning = {
                "code": "pdf_text_layer_recovery_failed",
                "message": (
                    "Native PDF text-layer recovery failed; continuing with "
                    "MinerU elements."
                ),
                "payload": {
                    "source_pdf": source_pdf.as_posix(),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            }
            warnings = document.metadata.setdefault("adapter_warnings", [])
            if not isinstance(warnings, list):
                raise ValueError(
                    "document.metadata['adapter_warnings'] must be a list"
                )
            warnings.append(warning)
            logger.warning("%s: %s", warning["code"], warning["message"])
            details = {
                "source_pdf": source_pdf.as_posix(),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        return self._report(document, before=before, details=details)


class PageLabelPass(_BasePass):
    """Resolve printed page-label aliases without requiring every page to have one."""

    name = "page_label_resolver"
    requires = frozenset({"coverage_recovered"})
    provides = frozenset({"page_labels_resolved"})

    def __init__(self, resolver: PageLabelResolver | None = None) -> None:
        self.resolver = resolver or PageLabelResolver()

    def apply(self, document: Document, context: PassContext) -> PassReport:
        before = document_fingerprint(document)
        source_pdf = _find_source_pdf(context.input_path)
        try:
            result = self.resolver.resolve(
                document,
                source_pdf=source_pdf.resolve() if source_pdf else None,
            )
            pdf_error: dict[str, str] | None = None
        except Exception as exc:
            # Printed labels are an optional address layer.  A damaged PDF text
            # layer must not prevent the rest of SoftDoc from being built.
            result = self.resolver.resolve(document, source_pdf=None)
            pdf_error = {
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        return self._report(
            document,
            before=before,
            details={
                "source_pdf": source_pdf.as_posix() if source_pdf else None,
                "pages_with_printed_labels": sum(
                    bool(decision.page_label_aliases)
                    for decision in result.decisions
                ),
                "multi_label_pages": sum(
                    len(decision.page_label_aliases) > 1
                    for decision in result.decisions
                ),
                "pdf_error": pdf_error,
            },
        )


class StructurePass(_BasePass):
    """Apply the existing parser-neutral structure builder unchanged."""

    name = "document_structure"
    requires = frozenset({"page_labels_resolved"})
    provides = frozenset({"document_structure"})

    def __init__(
        self,
        builder: SoftDocumentStructureBuilder | None = None,
    ) -> None:
        self.builder = builder or SoftDocumentStructureBuilder()

    def apply(self, document: Document, context: PassContext) -> PassReport:
        before = document_fingerprint(document)
        if (
            "document_profile" in document.metadata
            and "heading_decisions" in document.metadata
            and "heading_eligibility_decisions" in document.metadata
            and "repeated_header_footer_decisions" in document.metadata
        ):
            return self._report(
                document,
                before=before,
                skipped=True,
                details={"reason": "document_structure_already_present"},
            )
        result = self.builder.apply(document)
        return self._report(
            document,
            before=before,
            details={
                "profile": result.profile_decision.profile.value,
                "normalization_decisions": len(
                    result.normalization_decisions
                ),
                "heading_eligibility_decisions": len(
                    result.heading_eligibility_decisions
                ),
                "heading_decisions": len(result.heading_decisions),
                "repeated_region_decisions": len(
                    result.repeated_region_decisions
                ),
            },
        )


class RelationPass(_BasePass):
    """Build the existing deterministic relation set unchanged."""

    name = "relation_builder"
    requires = frozenset({"document_structure"})
    provides = frozenset({"relations_built"})

    def apply(self, document: Document, context: PassContext) -> PassReport:
        before = document_fingerprint(document)
        relations = RelationBuilder(document).build_all()
        return self._report(
            document,
            before=before,
            details={"relation_count": len(relations)},
        )


class FloatingSectionPass(_BasePass):
    """Apply the existing floating-content section resolver unchanged."""

    name = "floating_section_resolver"
    requires = frozenset({"relations_built"})
    provides = frozenset({"floating_sections_resolved"})

    def apply(self, document: Document, context: PassContext) -> PassReport:
        before = document_fingerprint(document)
        decisions = FloatingContentSectionResolver(document).resolve()
        return self._report(
            document,
            before=before,
            details={"decision_count": len(decisions)},
        )


class ValidationPass(_BasePass):
    """Validate all stable IDs and references without changing the document."""

    name = "reference_validation"
    requires = frozenset({"floating_sections_resolved"})
    provides = frozenset({"validated"})

    def apply(self, document: Document, context: PassContext) -> PassReport:
        before = document_fingerprint(document)
        DocumentStore(document).validate_references(raise_on_error=True)
        return self._report(
            document,
            before=before,
            details={"validation_errors": []},
        )


class RuleAuditPass(_BasePass):
    """Collect deterministic rule coverage without mutating the Document."""

    name = "rule_audit"
    requires = frozenset({"validated"})
    provides = frozenset({"rule_audited"})

    def apply(self, document: Document, context: PassContext) -> PassReport:
        before = document_fingerprint(document)
        report = collect_rule_coverage([document])
        context.metadata["rule_coverage_report"] = report.model_dump(
            mode="json"
        )
        return self._report(
            document,
            before=before,
            details={
                "rule_count": report.rule_count,
                "total_firings": report.total_firings,
            },
        )


DEFAULT_PASSES: tuple[DocumentPass, ...] = (
    CoverageRecoveryPass(),
    PageLabelPass(),
    StructurePass(),
    RelationPass(),
    FloatingSectionPass(),
    ValidationPass(),
    RuleAuditPass(),
)


class SoftDocPipeline:
    """The single orchestration entry from parser artifacts to final SoftDoc."""

    def __init__(
        self,
        parser: DocumentParser,
        *,
        passes: Sequence[DocumentPass] | None = None,
    ) -> None:
        self.parser = parser
        self.passes = tuple(passes) if passes is not None else DEFAULT_PASSES
        for document_pass in self.passes:
            if not isinstance(document_pass, DocumentPass):
                raise TypeError(
                    f"Invalid document pass: {type(document_pass).__name__}"
                )

    def run(self, input_path: Path, output_dir: Path) -> PipelineResult:
        input_path = Path(input_path)
        output_dir = Path(output_dir)
        document = self.parser.parse(input_path, output_dir)
        context = PassContext(
            input_path=input_path,
            output_dir=output_dir,
            available={"raw_document"},
        )
        reports: list[PassReport] = []
        for document_pass in self.passes:
            missing = document_pass.requires - context.available
            if missing:
                raise RuntimeError(
                    f"Pass {document_pass.name!r} requires unavailable "
                    f"capabilities: {sorted(missing)}"
                )
            report = document_pass.apply(document, context)
            reports.append(report)
            context.available.update(document_pass.provides)
        return PipelineResult(document=document, pass_reports=reports)

    def parse(self, input_path: Path, output_dir: Path) -> Document:
        """Compatibility-shaped convenience method returning only Document."""

        return self.run(input_path, output_dir).document


def document_fingerprint(document: Document) -> str:
    """Canonical SHA-256 for semantic/idempotency checks."""

    payload = json.dumps(
        document.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _find_source_pdf(input_path: Path) -> Path | None:
    """Use the same deterministic preference order as the frozen adapter."""

    preferred = sorted(input_path.glob("*_origin.pdf"))
    if preferred:
        return preferred[0]
    candidates = [
        path
        for path in sorted(input_path.glob("*.pdf"))
        if not path.stem.endswith(("_layout", "_span"))
    ]
    return candidates[0] if candidates else None
