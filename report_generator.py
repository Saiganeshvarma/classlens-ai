"""
report_generator.py — Assembles the final AssessmentReport and generates
downloadable JSON output.

This module is the last step in the pipeline.  It:
  1. Takes all pipeline outputs (metadata, dimensions, strengths, etc.).
  2. Calls scoring.py to calculate the final score and rating.
  3. Constructs and validates a complete AssessmentReport.
  4. Serialises to JSON for download.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import List, Optional

from schemas import (
    AssessmentReport,
    ConfidenceLevel,
    DimensionAssessment,
    EvidencePackage,
    Rating,
    RecordingMetadata,
)
from scoring import finalise_report_scores, validate_score_rating_consistency

logger = logging.getLogger(__name__)


# ── Exceptions ────────────────────────────────────────────────────────────────

class ReportGenerationError(Exception):
    """Raised when the final report cannot be assembled or validated."""


# ── Assembly ──────────────────────────────────────────────────────────────────

def build_report(
    pkg: EvidencePackage,
    dimensions: List[DimensionAssessment],
    strengths: List[str],
    improvements: List[str],
    overall_rationale: str,
    confidence: ConfidenceLevel,
    limitations: List[str],
    scoring_warnings: Optional[List[str]] = None,
) -> AssessmentReport:
    """
    Build and validate the final AssessmentReport.

    The overall_score and rating are always computed here by the Python
    scoring engine — never trusted from the LLM.

    Raises ReportGenerationError if the report fails Pydantic validation.
    """
    # 1. Calculate score + rating using Python (enforces config weights)
    try:
        score, rating, calc_warnings = finalise_report_scores(dimensions)
    except ValueError as exc:
        raise ReportGenerationError(
            f"Score calculation failed: {exc}"
        ) from exc

    all_warnings = (scoring_warnings or []) + calc_warnings
    if all_warnings:
        for w in all_warnings:
            logger.warning("Scoring warning: %s", w)

    # 2. Double-check consistency (belt-and-braces)
    consistency_error = validate_score_rating_consistency(score, rating)
    if consistency_error:
        logger.error("Score/rating inconsistency: %s", consistency_error)
        # Re-derive the rating from the score rather than crashing
        from scoring import score_to_rating
        rating = score_to_rating(score)

    # 3. Merge processing limitations with any scoring warnings
    all_limitations = list(limitations)
    for w in calc_warnings:
        if w not in all_limitations:
            all_limitations.append(w)

    # 4. Enrich metadata with transcription status
    meta = pkg.recording_metadata
    ps = pkg.processing_summary

    # 5. Assemble the report
    try:
        report = AssessmentReport(
            recording_metadata=meta,
            overall_score=score,
            rating=rating,
            dimensions=dimensions,
            strengths=strengths or ["No specific strengths identified from the available evidence."],
            improvements=improvements or ["No specific improvements identified."],
            overall_rationale=overall_rationale,
            confidence=confidence,
            limitations=all_limitations,
            processing_summary=ps,
        )
    except Exception as exc:
        raise ReportGenerationError(
            f"Report failed Pydantic validation: {exc}"
        ) from exc

    logger.info(
        "Report generated: score=%.1f, rating=%s, confidence=%s",
        score, rating.value, confidence.value,
    )
    return report


# ── JSON serialisation ────────────────────────────────────────────────────────

def report_to_json(report: AssessmentReport) -> str:
    """
    Serialise the report to a pretty-printed JSON string suitable for download.

    Adds a top-level metadata envelope with the generation timestamp.
    """
    envelope = {
        "classlens_version": "1.0.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "assessment": report.model_dump(mode="json"),
    }
    # Remove local filesystem paths from video frames before serialising
    assessment = envelope["assessment"]
    for dim in assessment.get("dimensions", []):
        pass  # paths live in EvidencePackage.frames, not in DimensionAssessment
    # EvidencePackage is not part of AssessmentReport, so frame paths never leak

    return json.dumps(envelope, indent=2, ensure_ascii=False)


def report_to_json_bytes(report: AssessmentReport) -> bytes:
    """Return the report JSON as UTF-8 bytes for Streamlit download_button."""
    return report_to_json(report).encode("utf-8")


# ── Human-readable summary ────────────────────────────────────────────────────

def report_to_summary(report: AssessmentReport) -> str:
    """
    Generate a concise plain-text summary of the report for the UI header.
    This is NOT the downloadable JSON — it is a short textual digest.
    """
    lines = [
        f"Overall Rating: {report.rating.value}  ({report.overall_score:.1f}/100)",
        f"Confidence: {report.confidence.value.capitalize()}",
        "",
        "Dimension Scores:",
    ]
    for dim in report.dimensions:
        if dim.score is not None:
            lines.append(f"  • {dim.name}: {dim.score:.0f}/100 (weight {dim.weight:.0f}%)")
        else:
            lines.append(f"  • {dim.name}: not assessed")

    if report.strengths:
        lines.append("\nStrengths:")
        for s in report.strengths[:3]:
            lines.append(f"  + {s}")

    if report.improvements:
        lines.append("\nAreas for Improvement:")
        for imp in report.improvements[:3]:
            lines.append(f"  → {imp}")

    if report.limitations:
        lines.append("\nLimitations:")
        for lim in report.limitations[:2]:
            lines.append(f"  ⚠ {lim}")

    return "\n".join(lines)


# ── Rating colour map (for Streamlit UI) ─────────────────────────────────────

RATING_COLOURS: dict[Rating, str] = {
    Rating.POOR: "#e74c3c",       # red
    Rating.AVERAGE: "#e67e22",    # orange
    Rating.GOOD: "#2ecc71",       # green
    Rating.EXCELLENT: "#1abc9c",  # teal
}

RATING_EMOJI: dict[Rating, str] = {
    Rating.POOR: "🔴",
    Rating.AVERAGE: "🟡",
    Rating.GOOD: "🟢",
    Rating.EXCELLENT: "⭐",
}
