"""
test_schemas.py — Pydantic schema validation tests.

Tests every important validation rule in schemas.py including:
- Valid and invalid ratings
- Score range enforcement
- Missing required fields
- EvidenceItem placeholder rejection
- DimensionAssessment consistency rules
- AssessmentReport score/rating consistency
- TranscriptSegment end >= start
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from schemas import (
    AssessmentReport,
    ConfidenceLevel,
    DimensionAssessment,
    DimensionStatus,
    EvidenceItem,
    EvidenceType,
    ProcessingSummary,
    Rating,
    RecordingMetadata,
    TranscriptSegment,
    TranscriptResult,
    AudioFeatures,
    VideoFrame,
)


# ── Shared factories ──────────────────────────────────────────────────────────

def _valid_evidence() -> EvidenceItem:
    return EvidenceItem(
        observation="Teacher explained the concept of photosynthesis with a clear diagram.",
        evidence_type=EvidenceType.TRANSCRIPT,
        impact="Students can follow the explanation without confusion.",
    )


def _valid_dim(
    name: str = "Teaching Clarity",
    score: float = 70.0,
    weight: float = 30.0,
    status: DimensionStatus = DimensionStatus.ASSESSED,
) -> DimensionAssessment:
    evidence = []
    if status in (DimensionStatus.ASSESSED, DimensionStatus.PARTIAL):
        evidence = [_valid_evidence()]
    return DimensionAssessment(
        name=name,
        score=score if status != DimensionStatus.UNAVAILABLE else None,
        weight=weight,
        status=status,
        evidence=evidence,
        rationale="A detailed rationale for the score based on observed evidence.",
    )


def _five_valid_dims(overall_score: float = 70.0) -> list[DimensionAssessment]:
    return [
        _valid_dim("Teaching Clarity", overall_score, 30),
        _valid_dim("Engagement and Checks for Understanding", overall_score, 25),
        _valid_dim("Lesson Structure", overall_score, 20),
        _valid_dim("Audio Intelligibility", overall_score, 15),
        _valid_dim("Visual Instructional Support", overall_score, 10),
    ]


def _valid_metadata() -> RecordingMetadata:
    return RecordingMetadata(
        filename="sample_class.mp4",
        duration_seconds=600.0,
        audio_available=True,
        video_available=True,
        file_size_bytes=52428800,
    )


def _valid_report(score: float = 70.0, rating: Rating = Rating.GOOD) -> AssessmentReport:
    return AssessmentReport(
        recording_metadata=_valid_metadata(),
        overall_score=score,
        rating=rating,
        dimensions=_five_valid_dims(score),
        strengths=["Teacher used real-world examples effectively."],
        improvements=["Add more explicit comprehension checks after each topic."],
        overall_rationale="The lesson demonstrated clear teaching with good audio quality "
                          "and logical structure, though engagement could be improved.",
        confidence=ConfidenceLevel.MEDIUM,
        limitations=["Only 12 frames sampled from a 10-minute recording."],
        processing_summary=ProcessingSummary(
            transcription_completed=True,
            frames_extracted=12,
        ),
    )


# ── EvidenceItem ──────────────────────────────────────────────────────────────

class TestEvidenceItem:

    def test_valid_evidence_accepted(self):
        ev = _valid_evidence()
        assert ev.observation is not None

    def test_short_observation_rejected(self):
        with pytest.raises(ValidationError):
            EvidenceItem(
                observation="Too short",
                evidence_type=EvidenceType.TRANSCRIPT,
                impact="Affects quality.",
            )

    def test_short_impact_rejected(self):
        with pytest.raises(ValidationError):
            EvidenceItem(
                observation="Teacher clearly explained photosynthesis with diagrams.",
                evidence_type=EvidenceType.TRANSCRIPT,
                impact="Bad",
            )

    def test_placeholder_observation_rejected(self):
        # "N/A" is rejected by min_length=10 before the custom validator runs
        with pytest.raises(ValidationError):
            EvidenceItem(
                observation="N/A",
                evidence_type=EvidenceType.TRANSCRIPT,
                impact="This matters for teaching quality assessment.",
            )

    def test_placeholder_observation_long_rejected(self):
        # A long placeholder that passes min_length but is in the banned set
        with pytest.raises(ValidationError, match="Placeholder"):
            EvidenceItem(
                observation="placeholder",
                evidence_type=EvidenceType.TRANSCRIPT,
                impact="This matters for teaching quality assessment.",
            )

    def test_placeholder_impact_rejected(self):
        # "todo" is rejected by min_length=10 before the custom validator runs
        with pytest.raises(ValidationError):
            EvidenceItem(
                observation="Teacher used slides throughout the lesson.",
                evidence_type=EvidenceType.VISUAL_FRAME,
                impact="todo",
            )

    def test_placeholder_impact_long_rejected(self):
        # A placeholder that passes min_length but is banned
        with pytest.raises(ValidationError, match="Placeholder"):
            EvidenceItem(
                observation="Teacher used slides throughout the lesson.",
                evidence_type=EvidenceType.VISUAL_FRAME,
                impact="placeholder",
            )

    def test_negative_timestamp_rejected(self):
        with pytest.raises(ValidationError):
            EvidenceItem(
                observation="Teacher explained the topic clearly at this point.",
                evidence_type=EvidenceType.TRANSCRIPT,
                timestamp_seconds=-1.0,
                impact="Timestamp context matters for review.",
            )

    def test_all_evidence_types_accepted(self):
        for et in EvidenceType:
            ev = EvidenceItem(
                observation=f"A specific observation of type {et.value}.",
                evidence_type=et,
                impact="This observation is relevant to teaching quality.",
            )
            assert ev.evidence_type == et

    def test_optional_fields_can_be_none(self):
        ev = EvidenceItem(
            observation="Teacher demonstrated problem-solving step by step.",
            evidence_type=EvidenceType.DERIVED,
            impact="Step-by-step demonstration aids understanding.",
        )
        assert ev.timestamp_seconds is None
        assert ev.source_excerpt is None


# ── DimensionAssessment ───────────────────────────────────────────────────────

class TestDimensionAssessment:

    def test_valid_assessed_dimension(self):
        d = _valid_dim()
        assert d.score == 70.0

    def test_score_above_100_rejected(self):
        with pytest.raises(ValidationError):
            _valid_dim(score=101.0)

    def test_score_below_0_rejected(self):
        with pytest.raises(ValidationError):
            _valid_dim(score=-1.0)

    def test_unavailable_dim_no_score_required(self):
        d = _valid_dim(status=DimensionStatus.UNAVAILABLE)
        assert d.score is None

    def test_assessed_dim_requires_score(self):
        with pytest.raises(ValidationError, match="score is None"):
            DimensionAssessment(
                name="Teaching Clarity",
                score=None,
                weight=30.0,
                status=DimensionStatus.ASSESSED,
                evidence=[_valid_evidence()],
                rationale="A detailed rationale explaining the score based on observations.",
            )

    def test_assessed_dim_requires_evidence(self):
        with pytest.raises(ValidationError, match="no evidence"):
            DimensionAssessment(
                name="Teaching Clarity",
                score=70.0,
                weight=30.0,
                status=DimensionStatus.ASSESSED,
                evidence=[],
                rationale="A detailed rationale explaining the score based on observations.",
            )

    def test_partial_status_accepted_with_score(self):
        d = DimensionAssessment(
            name="Lesson Structure",
            score=55.0,
            weight=20.0,
            status=DimensionStatus.PARTIAL,
            evidence=[_valid_evidence()],
            rationale="Limited evidence available; partial assessment only is possible.",
        )
        assert d.status == DimensionStatus.PARTIAL

    def test_rationale_too_short_rejected(self):
        with pytest.raises(ValidationError):
            DimensionAssessment(
                name="Teaching Clarity",
                score=70.0,
                weight=30.0,
                status=DimensionStatus.ASSESSED,
                evidence=[_valid_evidence()],
                rationale="Too short",
            )

    def test_weight_zero_rejected(self):
        with pytest.raises(ValidationError):
            DimensionAssessment(
                name="Teaching Clarity",
                score=70.0,
                weight=0.0,
                status=DimensionStatus.ASSESSED,
                evidence=[_valid_evidence()],
                rationale="A detailed rationale explaining the score based on observations.",
            )


# ── AssessmentReport ──────────────────────────────────────────────────────────

class TestAssessmentReport:

    def test_valid_good_report(self):
        report = _valid_report(70.0, Rating.GOOD)
        assert report.rating == Rating.GOOD
        assert report.overall_score == 70.0

    def test_valid_poor_report(self):
        report = _valid_report(30.0, Rating.POOR)
        assert report.rating == Rating.POOR

    def test_valid_average_report(self):
        report = _valid_report(50.0, Rating.AVERAGE)
        assert report.rating == Rating.AVERAGE

    def test_valid_excellent_report(self):
        report = _valid_report(85.0, Rating.EXCELLENT)
        assert report.rating == Rating.EXCELLENT

    def test_score_rating_mismatch_raises(self):
        """Score 70 (Good) with rating Poor should fail validation."""
        with pytest.raises(ValidationError, match="implies Good"):
            _valid_report(70.0, Rating.POOR)

    def test_score_above_100_rejected(self):
        with pytest.raises(ValidationError):
            AssessmentReport(
                recording_metadata=_valid_metadata(),
                overall_score=101.0,
                rating=Rating.EXCELLENT,
                dimensions=_five_valid_dims(100.0),
                strengths=["Good class."],
                improvements=["None needed."],
                overall_rationale="Excellent class with strong evidence throughout.",
                confidence=ConfidenceLevel.HIGH,
                limitations=[],
                processing_summary=ProcessingSummary(),
            )

    def test_score_below_0_rejected(self):
        with pytest.raises(ValidationError):
            AssessmentReport(
                recording_metadata=_valid_metadata(),
                overall_score=-1.0,
                rating=Rating.POOR,
                dimensions=_five_valid_dims(0.0),
                strengths=[],
                improvements=["Everything needs work."],
                overall_rationale="Very poor class quality based on available evidence.",
                confidence=ConfidenceLevel.LOW,
                limitations=["Limited evidence."],
                processing_summary=ProcessingSummary(),
            )

    def test_fewer_than_five_dims_rejected(self):
        with pytest.raises(ValidationError):
            AssessmentReport(
                recording_metadata=_valid_metadata(),
                overall_score=70.0,
                rating=Rating.GOOD,
                dimensions=_five_valid_dims()[:3],   # only 3
                strengths=["Good teaching."],
                improvements=["More checks needed."],
                overall_rationale="Good class overall with clear explanations.",
                confidence=ConfidenceLevel.MEDIUM,
                limitations=[],
                processing_summary=ProcessingSummary(),
            )

    def test_duplicate_dimension_names_rejected(self):
        dims = _five_valid_dims()
        dims[1] = dims[0].model_copy()  # duplicate first dim
        with pytest.raises(ValidationError, match="distinct"):
            AssessmentReport(
                recording_metadata=_valid_metadata(),
                overall_score=70.0,
                rating=Rating.GOOD,
                dimensions=dims,
                strengths=["Good."],
                improvements=["Improve engagement."],
                overall_rationale="Good class overall with clear teaching methods.",
                confidence=ConfidenceLevel.MEDIUM,
                limitations=[],
                processing_summary=ProcessingSummary(),
            )

    def test_invalid_rating_string_rejected(self):
        with pytest.raises(ValidationError):
            AssessmentReport(
                recording_metadata=_valid_metadata(),
                overall_score=70.0,
                rating="Outstanding",  # type: ignore[arg-type]
                dimensions=_five_valid_dims(),
                strengths=[],
                improvements=[],
                overall_rationale="Good class overall with clear teaching methods.",
                confidence=ConfidenceLevel.MEDIUM,
                limitations=[],
                processing_summary=ProcessingSummary(),
            )


# ── TranscriptSegment ─────────────────────────────────────────────────────────

class TestTranscriptSegment:

    def test_valid_segment(self):
        seg = TranscriptSegment(start=0.0, end=5.0, text="Hello class.")
        assert seg.start == 0.0

    def test_end_before_start_rejected(self):
        with pytest.raises(ValidationError, match="end .* before start"):
            TranscriptSegment(start=10.0, end=5.0, text="Invalid segment.")

    def test_end_equals_start_accepted(self):
        seg = TranscriptSegment(start=5.0, end=5.0, text="Instant segment.")
        assert seg.end == seg.start

    def test_negative_start_rejected(self):
        with pytest.raises(ValidationError):
            TranscriptSegment(start=-1.0, end=5.0, text="Negative start.")


# ── AudioFeatures ─────────────────────────────────────────────────────────────

class TestAudioFeatures:

    def test_valid_features(self):
        af = AudioFeatures(
            duration_seconds=120.0,
            has_speech=True,
            silence_ratio=0.1,
            estimated_snr_db=25.0,
        )
        assert af.has_speech is True

    def test_silence_ratio_above_1_rejected(self):
        with pytest.raises(ValidationError):
            AudioFeatures(duration_seconds=60.0, has_speech=True, silence_ratio=1.5)

    def test_silence_ratio_negative_rejected(self):
        with pytest.raises(ValidationError):
            AudioFeatures(duration_seconds=60.0, has_speech=True, silence_ratio=-0.1)

    def test_duration_negative_rejected(self):
        with pytest.raises(ValidationError):
            AudioFeatures(duration_seconds=-1.0, has_speech=False)


# ── VideoFrame ────────────────────────────────────────────────────────────────

class TestVideoFrame:

    def test_valid_frame(self):
        f = VideoFrame(timestamp_seconds=30.0, frame_index=1)
        assert f.timestamp_seconds == 30.0

    def test_negative_timestamp_rejected(self):
        with pytest.raises(ValidationError):
            VideoFrame(timestamp_seconds=-5.0, frame_index=0)

    def test_path_is_optional(self):
        f = VideoFrame(timestamp_seconds=0.0, frame_index=0)
        assert f.path is None
