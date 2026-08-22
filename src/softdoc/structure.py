"""Parser-neutral post-processing for the Soft Document Structure."""

from __future__ import annotations

from collections import Counter
import re

from pydantic import Field

from softdoc.eligibility import (
    HeadingEligibilityDecision,
    HeadingEligibilityDetector,
)
from softdoc.hierarchy import HeadingDecision, HeadingHierarchyBuilder
from softdoc.models import Document, SoftDocModel
from softdoc.normalization import (
    ElementNormalizationDecision,
    ElementNormalizer,
)
from softdoc.profiles import (
    DocumentProfileDecision,
    DocumentProfileDetector,
)
from softdoc.repetition import (
    RepeatedHeaderFooterDetector,
    RepeatedRegionDecision,
)
from softdoc.sections import SectionBuilder


class StructureBuildResult(SoftDocModel):
    profile_decision: DocumentProfileDecision
    normalization_decisions: list[ElementNormalizationDecision] = Field(
        default_factory=list
    )
    heading_eligibility_decisions: list[HeadingEligibilityDecision] = Field(
        default_factory=list
    )
    heading_decisions: list[HeadingDecision] = Field(default_factory=list)
    repeated_region_decisions: list[RepeatedRegionDecision] = Field(
        default_factory=list
    )


class SoftDocumentStructureBuilder:
    """Apply parser-independent structure rules after element conversion."""

    def __init__(
        self,
        *,
        repeated_region_detector: RepeatedHeaderFooterDetector | None = None,
        profile_detector: DocumentProfileDetector | None = None,
        element_normalizer: ElementNormalizer | None = None,
        heading_eligibility_detector: HeadingEligibilityDetector | None = None,
        heading_builder: HeadingHierarchyBuilder | None = None,
        section_builder: SectionBuilder | None = None,
    ) -> None:
        self.repeated_region_detector = (
            repeated_region_detector or RepeatedHeaderFooterDetector()
        )
        self.profile_detector = profile_detector or DocumentProfileDetector()
        self.element_normalizer = element_normalizer or ElementNormalizer()
        self.heading_eligibility_detector = (
            heading_eligibility_detector or HeadingEligibilityDetector()
        )
        self.heading_builder = heading_builder or HeadingHierarchyBuilder()
        self.section_builder = section_builder or SectionBuilder()

    def apply(self, document: Document) -> StructureBuildResult:
        parser_backend = str(
            document.metadata.get("parser_backend") or "pipeline"
        ).casefold()
        trusted_parser_types = parser_backend == "hybrid"
        profile = self.profile_detector.detect(
            document.pages,
            document.elements,
            source_path=document.source_path,
        )
        if trusted_parser_types:
            # Hybrid already supplies stronger title/list/visual typing.  Do
            # not run the legacy document-specific repair catalogue on top of
            # it; preserve parser output and apply only generic structure
            # checks below.
            normalization_decisions = []
            repeated = self.repeated_region_detector.detect(
                document.pages, document.elements
            )
        else:
            normalization_decisions = self.element_normalizer.normalize(
                document,
                profile.profile,
            )
            repeated = self.repeated_region_detector.detect(
                document.pages, document.elements
            )
        eligibility_decisions = self.heading_eligibility_detector.detect(
            document.pages,
            document.elements,
            profile.profile,
            trusted_parser_types=trusted_parser_types,
        )
        hierarchy = self.heading_builder.build(
            document.document_id,
            document.pages,
            document.elements,
            profile=profile.profile,
        )
        if not document.title and hierarchy.document_title:
            document.title = hierarchy.document_title
        if not document.title:
            document.title = (
                self._entity_title_fallback(document)
                or self._source_title_fallback(document)
            )
        section_builder = (
            SectionBuilder(
                allow_caption_anchors=False,
                allow_terminal_checklist=False,
            )
            if trusted_parser_types
            else self.section_builder
        )
        document.sections = section_builder.build(
            document.document_id,
            document.pages,
            document.elements,
        )
        document.metadata["heading_hierarchy"] = {
            "created_by": "deterministic_rule",
            "strategy": (
                "hybrid_parser_assisted_minimal_v1"
                if trusted_parser_types
                else "profiled_document_style_plus_conservative_rules"
            ),
            "document_profile": profile.profile.value,
            "parser_backend": parser_backend,
            "legacy_element_normalizer_applied": not trusted_parser_types,
            "caption_derived_sections_enabled": not trusted_parser_types,
            "terminal_checklist_sections_enabled": not trusted_parser_types,
        }
        document.metadata["document_profile"] = profile.model_dump(mode="json")
        document.metadata["element_normalization_decisions"] = [
            decision.model_dump(mode="json")
            for decision in normalization_decisions
        ]
        document.metadata["heading_eligibility_decisions"] = [
            decision.model_dump(mode="json")
            for decision in eligibility_decisions
        ]
        document.metadata["heading_decisions"] = [
            decision.model_dump(mode="json")
            for decision in hierarchy.decisions
        ]
        document.metadata["repeated_header_footer_decisions"] = [
            decision.model_dump(mode="json")
            for decision in repeated.decisions
        ]
        return StructureBuildResult(
            profile_decision=profile,
            normalization_decisions=normalization_decisions,
            heading_eligibility_decisions=eligibility_decisions,
            heading_decisions=hierarchy.decisions,
            repeated_region_decisions=repeated.decisions,
        )

    @staticmethod
    def _entity_title_fallback(document: Document) -> str | None:
        candidates: Counter[str] = Counter()
        pattern = re.compile(
            r"\b([A-Z][A-Za-z&'-]+(?:\s+[A-Z][A-Za-z&'-]+){1,6}\s+"
            r"(?:Limited|Ltd\.?|Inc\.?|Corporation|Company))\b"
        )
        for element in document.elements:
            for match in pattern.finditer(element.text or ""):
                candidates[" ".join(match.group(1).split())] += 1
        if not candidates:
            return None
        title, count = candidates.most_common(1)[0]
        return title if count >= 2 else None

    @staticmethod
    def _source_title_fallback(document: Document) -> str | None:
        stem = document.source_path.stem
        stem = stem.removesuffix("_origin").removesuffix("-origin")
        if not stem or len(stem) >= 80:
            return None
        # Opaque dataset hashes are not useful titles.
        if all(character in "0123456789abcdefABCDEF" for character in stem):
            return None
        words = stem.replace("_", " ").replace("-", " ").split()
        if not words:
            return None
        return " ".join(
            word if any(character.isupper() for character in word[1:])
            else word.capitalize()
            for word in words
        )
