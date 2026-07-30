"""Soft document intermediate representation."""

from softdoc.floating_sections import (
    FloatingContentSectionResolver,
    SectionResolutionDecision,
    SectionResolutionStatus,
)
from softdoc.hierarchy import (
    HeadingAction,
    HeadingDecision,
    HeadingHierarchyBuilder,
    HeadingHierarchyResult,
)
from softdoc.eligibility import (
    HeadingEligibilityDecision,
    HeadingEligibilityDetector,
)
from softdoc.normalization import (
    ElementNormalizationDecision,
    ElementNormalizer,
)
from softdoc.profiles import (
    DocumentProfile,
    DocumentProfileDecision,
    DocumentProfileDetector,
)
from softdoc.models import (
    BoundingBox,
    Document,
    Element,
    ElementType,
    Page,
    Provenance,
    Relation,
    RelationEvidence,
    RelationSource,
    RelationStatus,
    RelationType,
    Section,
)
from softdoc.repetition import (
    RepeatedHeaderFooterDetector,
    RepeatedHeaderFooterResult,
    RepeatedRegion,
    RepeatedRegionDecision,
)
from softdoc.sections import SectionBuilder
from softdoc.structure import SoftDocumentStructureBuilder, StructureBuildResult

__all__ = [
    "BoundingBox",
    "Document",
    "Element",
    "ElementType",
    "Page",
    "Provenance",
    "Relation",
    "RelationEvidence",
    "RelationSource",
    "RelationStatus",
    "RelationType",
    "Section",
    "HeadingAction",
    "HeadingDecision",
    "HeadingHierarchyBuilder",
    "HeadingHierarchyResult",
    "HeadingEligibilityDecision",
    "HeadingEligibilityDetector",
    "ElementNormalizationDecision",
    "ElementNormalizer",
    "DocumentProfile",
    "DocumentProfileDecision",
    "DocumentProfileDetector",
    "RepeatedHeaderFooterDetector",
    "RepeatedHeaderFooterResult",
    "RepeatedRegion",
    "RepeatedRegionDecision",
    "SectionBuilder",
    "SoftDocumentStructureBuilder",
    "StructureBuildResult",
    "FloatingContentSectionResolver",
    "SectionResolutionDecision",
    "SectionResolutionStatus",
]
