"""
test_report_generator.py — Tests for report assembly, JSON serialisation,
and summary generation.

Tests:
- build_report() assembles a valid AssessmentReport
- Weighted score is calculated by Python, not trusted from LLM
- Unavailable dimensions are handled without crashing
- Invalid dimension sets raise ReportGenerationError
- report_to_json() produces valid JSON with envelope
- report_to_json_bytes() returns bytes
- report_to_summary() includes key fields
- RATING_COLOURS and RATING_EMOJI cover all four ratings
"""

from __future__ import annotations

import json

import pytest

from report_generator import (
    RATING_COLOURS,
    RATING_EMOJI,
    ReportGenerationError,
    build_report,
    report_to_json,
    report_to_json_bytes,
    report_to_summary,
)
from schemas import (
    AssessmentReport,
    ConfidenceLevel,
    DimensionAssessment,
    DimensionStatus,
    EvidenceItem,
    EvidencePackage,
    EvidenceType,
    ProcessingSummary,
    Rating,
    RecordingMetadata,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _evidence() -> EvidenceItem:
    return EvidenceItem(
        observation="Teacher introduced the lesson topic at the start.",
        evidence_type=EvidenceType.TRANSCRIPT,
        impact="Students understand what they will learn in the session.",
    )


def _dim(
    name: str,
    score: float | None,
    weight: float,
    status: DimensionStatus = DimensionStatus.ASSESSED,
) -> DimensionAssessment:
    evidence = [_evidence()] if status != DimensionStatus.UNAVAILABLE else []
    rationale = f"Assessment rationale for dimension {name} based on observed evidence."
    return DimensionAssessment(
        name=name,
        score=score,
        weight=weight,
        status=status,
        evidence=evidence,
        rationale=rationale,
    )


def _five_dims(scores=None, statuses=None) -> list[DimensionAssessment]:
    from config import get_settings
    names_weights = list(get_settings().dimension_weights().items())
    if scores is None:
        scores = [70.0, 65.0, 68.0, 75.0, 60.0]
    if statuses is None:
        statuses = [DimensionStatus.ASSESSED] * 5
    return [
        _dim(name, scores[i], weight, statuses[i])
        for i, (name, weight) in enumerate(names_weights)
    ]


def _pkg() -> EvidencePackage:
    meta = RecordingMetadata(
        filename="test_class.mp4",
        duration_seconds=300.0,
        audio_available=True,
        video_available=True,
        file_size_bytes=15728640,
    )
    return EvidencePackage(
        recording_metadata=meta,
        processing_summary=ProcessingSummary(
            transcription_completed=True,
            frames_extracted=8,
        ),
    )


# ── build_report() ────────────────────────────────────────────────────────────

class TestBuildReport:

    def test_returns_assessment_report(self):
        report = build_report(
            pkg=_pkg(),
            dimensions=_five_dims(),
            strengths=["Clear explanation of key concepts."],
            improvements=["Add more interaction with students."],
            overall_rationale="Good class overall with structured delivery and clear audio.",
            confidence=ConfidenceLevel.MEDIUM,
            limitations=[],
        )
        assert isinstance(report, AssessmentReport)

    def test_score_computed_by_python(self):
        """Even if we tamper with dim weights, Python recalculates correctly."""
        dims = _five_dims(scores=[80.0] * 5)
        # Tamper one weight
        dims[0] = dims[0].model_copy(update={"weight": 99.0})
        report = build_report(
            pkg=_pkg(),
            dimensions=dims,
            strengths=["Excellent teaching."],
            improvements=[],
            overall_rationale="Outstanding class quality demonstrated throughout the session.",
            confidence=ConfidenceLevel.HIGH,
            limitations=[],
        )
        # All scores are 80 → score must be exactly 80 (config weights enforced)
        assert abs(report.overall_score - 80.0) < 0.01
        assert report.rating == Rating.EXCELLENT

    def test_poor_rating_for_low_scores(self):
        report = build_report(
            pkg=_pkg(),
            dimensions=_five_dims(scores=[25.0] * 5),
            strengths=[],
            improvements=["Major improvements needed across all dimensions."],
            overall_rationale="Class quality is poor based on the available evidence.",
            confidence=ConfidenceLevel.LOW,
            limitations=["Limited evidence due to short recording."],
        )
        assert report.rating == Rating.POOR
        assert report.overall_score <= 39

    def test_one_unavailable_dim_handled(self):
        statuses = [DimensionStatus.ASSESSED] * 4 + [DimensionStatus.UNAVAILABLE]
        scores = [70.0, 65.0, 68.0, 75.0, None]
        report = build_report(
            pkg=_pkg(),
            dimensions=_five_dims(scores=scores, statuses=statuses),
            strengths=["Good audio quality throughout."],
            improvements=["Visual aids would strengthen the lesson."],
            overall_rationale="Good class with strong audio but visual support unavailable.",
            confidence=ConfidenceLevel.MEDIUM,
            limitations=["No video stream — visual assessment skipped."],
        )
        assert report.rating in (Rating.GOOD, Rating.AVERAGE)
        # Limitation about visual support should be present
        all_lims = " ".join(report.limitations)
        assert "unavailable" in all_lims.lower() or "visual" in all_lims.lower()

    def test_all_unavailable_raises_report_generation_error(self):
        statuses = [DimensionStatus.UNAVAILABLE] * 5
        scores = [None] * 5
        with pytest.raises(ReportGenerationError, match="Score calculation failed"):
            build_report(
                pkg=_pkg(),
                dimensions=_five_dims(scores=scores, statuses=statuses),
                strengths=[],
                improvements=[],
                overall_rationale="No evidence available for any dimension.",
                confidence=ConfidenceLevel.LOW,
                limitations=["Recording was empty."],
            )

    def test_strengths_default_when_empty(self):
        report = build_report(
            pkg=_pkg(),
            dimensions=_five_dims(),
            strengths=[],
            improvements=["Add more student interaction."],
            overall_rationale="The class showed adequate teaching quality overall.",
            confidence=ConfidenceLevel.MEDIUM,
            limitations=[],
        )
        assert len(report.strengths) >= 1

    def test_rating_colour_and_emoji_present_for_all(self):
        for rating in Rating:
            assert rating in RATING_COLOURS
            assert rating in RATING_EMOJI

    def test_boundary_score_39_is_poor(self):
        dims = _five_dims(scores=[39.0] * 5)
        report = build_report(
            pkg=_pkg(),
            dimensions=dims,
            strengths=[],
            improvements=["Significant improvements needed."],
            overall_rationale="Class quality is very poor based on all available evidence.",
            confidence=ConfidenceLevel.LOW,
            limitations=[],
        )
        assert report.rating == Rating.POOR

    def test_boundary_score_40_is_average(self):
        dims = _five_dims(scores=[40.0] * 5)
        report = build_report(
            pkg=_pkg(),
            dimensions=dims,
            strengths=["Some positive aspects noted."],
            improvements=["Multiple areas need improvement."],
            overall_rationale="Class is average overall with room for improvement.",
            confidence=ConfidenceLevel.MEDIUM,
            limitations=[],
        )
        assert report.rating == Rating.AVERAGE

    def test_boundary_score_80_is_excellent(self):
        dims = _five_dims(scores=[80.0] * 5)
        report = build_report(
            pkg=_pkg(),
            dimensions=dims,
            strengths=["Outstanding delivery across all dimensions."],
            improvements=[],
            overall_rationale="Excellent class quality demonstrated throughout the session.",
            confidence=ConfidenceLevel.HIGH,
            limitations=[],
        )
        assert report.rating == Rating.EXCELLENT


# ── report_to_json() ──────────────────────────────────────────────────────────

class TestReportToJson:

    def _base_report(self) -> AssessmentReport:
        return build_report(
            pkg=_pkg(),
            dimensions=_five_dims(),
            strengths=["Good pacing and clarity."],
            improvements=["Improve student engagement."],
            overall_rationale="Good class with clear explanations and structured delivery.",
            confidence=ConfidenceLevel.MEDIUM,
            limitations=[],
        )

    def test_produces_valid_json(self):
        report = self._base_report()
        raw = report_to_json(report)
        parsed = json.loads(raw)
        assert isinstance(parsed, dict)

    def test_json_contains_envelope_keys(self):
        report = self._base_report()
        parsed = json.loads(report_to_json(report))
        assert "classlens_version" in parsed
        assert "generated_at" in parsed
        assert "assessment" in parsed

    def test_json_assessment_has_rating(self):
        report = self._base_report()
        parsed = json.loads(report_to_json(report))
        assert "rating" in parsed["assessment"]

    def test_json_assessment_has_five_dims(self):
        report = self._base_report()
        parsed = json.loads(report_to_json(report))
        assert len(parsed["assessment"]["dimensions"]) == 5

    def test_json_score_matches_report(self):
        report = self._base_report()
        parsed = json.loads(report_to_json(report))
        assert abs(parsed["assessment"]["overall_score"] - report.overall_score) < 0.01

    def test_json_no_local_paths_leaked(self):
        """Temp filesystem paths should not appear in the serialised JSON."""
        report = self._base_report()
        raw = report_to_json(report)
        # Check for temp dir paths specifically (not the "classlens_version" envelope key)
        assert "/tmp/classlens_" not in raw
        assert "classlens_ai_" not in raw  # old prefix guard

    def test_report_to_json_bytes_returns_bytes(self):
        report = self._base_report()
        b = report_to_json_bytes(report)
        assert isinstance(b, bytes)
        assert len(b) > 0

    def test_report_to_json_bytes_is_valid_utf8(self):
        report = self._base_report()
        b = report_to_json_bytes(report)
        decoded = b.decode("utf-8")
        json.loads(decoded)  # should not raise


# ── report_to_summary() ───────────────────────────────────────────────────────

class TestReportToSummary:

    def _base_report(self) -> AssessmentReport:
        return build_report(
            pkg=_pkg(),
            dimensions=_five_dims(),
            strengths=["Clear delivery."],
            improvements=["More student interaction needed."],
            overall_rationale="Good class with solid teaching foundations.",
            confidence=ConfidenceLevel.MEDIUM,
            limitations=["Transcript truncated after 6000 words."],
        )

    def test_summary_contains_overall_rating(self):
        report = self._base_report()
        summary = report_to_summary(report)
        assert "Overall Rating" in summary

    def test_summary_contains_score(self):
        report = self._base_report()
        summary = report_to_summary(report)
        assert str(int(report.overall_score)) in summary or f"{report.overall_score}" in summary

    def test_summary_contains_all_dimension_names(self):
        report = self._base_report()
        summary = report_to_summary(report)
        for dim in report.dimensions:
            assert dim.name in summary

    def test_summary_contains_strengths(self):
        report = self._base_report()
        summary = report_to_summary(report)
        assert "Strengths" in summary

    def test_summary_contains_improvements(self):
        report = self._base_report()
        summary = report_to_summary(report)
        assert "Improvement" in summary

    def test_summary_contains_limitations(self):
        report = self._base_report()
        summary = report_to_summary(report)
        assert "Limitations" in summary


# ── Rating metadata completeness ──────────────────────────────────────────────

class TestRatingMetadata:

    def test_all_ratings_have_colour(self):
        for r in Rating:
            assert r in RATING_COLOURS, f"Missing colour for {r}"
            assert RATING_COLOURS[r].startswith("#")

    def test_all_ratings_have_emoji(self):
        for r in Rating:
            assert r in RATING_EMOJI, f"Missing emoji for {r}"
            assert len(RATING_EMOJI[r]) > 0
