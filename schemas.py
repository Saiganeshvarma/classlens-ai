"""
schemas.py — Pydantic models for ClassLens AI.

All assessment data flows through these models.  Every field has an
explicit type, validator, and docstring so that malformed AI output is
caught before it reaches the UI.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator, model_validator


# ── Enumerations ──────────────────────────────────────────────────────────────

class Rating(str, Enum):
    """The four permitted quality classifications."""
    POOR = "Poor"
    AVERAGE = "Average"
    GOOD = "Good"
    EXCELLENT = "Excellent"


class DimensionStatus(str, Enum):
    """Whether a dimension could be assessed."""
    ASSESSED = "assessed"
    PARTIAL = "partial"       # assessed with limited evidence
    UNAVAILABLE = "unavailable"  # no usable evidence


class EvidenceType(str, Enum):
    """The source of a specific observation."""
    TRANSCRIPT = "transcript"
    AUDIO_FEATURE = "audio_feature"
    VISUAL_FRAME = "visual_frame"
    DERIVED = "derived"       # inference from multiple sources


class ConfidenceLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# ── Sub-models ────────────────────────────────────────────────────────────────

class EvidenceItem(BaseModel):
    """A single concrete observation supporting a dimension score."""
    observation: str = Field(
        ...,
        min_length=10,
        description="What was observed — specific, not generic.",
    )
    evidence_type: EvidenceType
    timestamp_seconds: Optional[float] = Field(
        default=None,
        ge=0,
        description="Timestamp in the recording; None if not reliably available.",
    )
    source_excerpt: Optional[str] = Field(
        default=None,
        max_length=500,
        description="Short verbatim transcript excerpt or frame description.",
    )
    impact: str = Field(
        ...,
        min_length=10,
        description="Why this observation matters for teaching quality.",
    )
    is_direct_observation: bool = Field(
        default=True,
        description="False when the item is an interpretation rather than a fact.",
    )

    @field_validator("observation", "impact")
    @classmethod
    def no_generic_placeholder(cls, v: str) -> str:
        banned = {"n/a", "none", "not available", "placeholder", "tbd", "todo"}
        if v.strip().lower() in banned:
            raise ValueError(f"Placeholder text is not a valid observation: '{v}'")
        return v


class DimensionAssessment(BaseModel):
    """Assessment for one of the five evaluation dimensions."""
    name: str = Field(..., description="Human-readable dimension name.")
    score: Optional[float] = Field(
        default=None,
        ge=0,
        le=100,
        description="Score 0–100, or None when status is 'unavailable'.",
    )
    weight: float = Field(
        ...,
        gt=0,
        le=100,
        description="Rubric weight in percent — set by config, not by the LLM.",
    )
    status: DimensionStatus
    evidence: List[EvidenceItem] = Field(default_factory=list)
    rationale: str = Field(
        ...,
        min_length=20,
        description="Explanation of the score referencing the evidence.",
    )

    @model_validator(mode="after")
    def score_required_when_assessed(self) -> "DimensionAssessment":
        if self.status in (DimensionStatus.ASSESSED, DimensionStatus.PARTIAL):
            if self.score is None:
                raise ValueError(
                    f"Dimension '{self.name}' has status '{self.status}' "
                    "but score is None."
                )
        return self

    @model_validator(mode="after")
    def evidence_required_when_assessed(self) -> "DimensionAssessment":
        if self.status == DimensionStatus.ASSESSED and len(self.evidence) == 0:
            raise ValueError(
                f"Dimension '{self.name}' is marked 'assessed' but has no evidence items."
            )
        return self


class RecordingMetadata(BaseModel):
    """Facts extracted from the recording before analysis."""
    filename: str
    duration_seconds: float = Field(..., ge=0)
    audio_available: bool
    video_available: bool
    has_speech: bool = False
    detected_language: Optional[str] = None
    file_size_bytes: int = Field(..., ge=0)


class ProcessingSummary(BaseModel):
    """Which pipeline stages completed successfully."""
    validation_completed: bool = False
    audio_extraction_completed: bool = False
    transcription_completed: bool = False
    audio_feature_analysis_completed: bool = False
    frame_extraction_completed: bool = False
    visual_analysis_completed: bool = False
    evaluation_completed: bool = False
    frames_extracted: int = 0
    transcript_word_count: int = 0
    transcription_confidence: Optional[float] = Field(default=None, ge=0, le=1)


# ── Main report ───────────────────────────────────────────────────────────────

class AssessmentReport(BaseModel):
    """
    The complete output of one ClassLens AI analysis run.

    The overall_score and rating are always computed by Python (scoring.py),
    not by the LLM.  The LLM supplies dimension scores and evidence only.
    """
    recording_metadata: RecordingMetadata
    overall_score: float = Field(..., ge=0, le=100)
    rating: Rating
    dimensions: List[DimensionAssessment] = Field(
        ...,
        min_length=5,
        max_length=5,
        description="Exactly five dimension assessments, one per rubric dimension.",
    )
    strengths: List[str] = Field(
        default_factory=list,
        description="Concrete positive observations from the recording.",
    )
    improvements: List[str] = Field(
        default_factory=list,
        description="Specific, actionable recommendations.",
    )
    overall_rationale: str = Field(
        ...,
        min_length=30,
        description="Holistic explanation of the final assessment.",
    )
    confidence: ConfidenceLevel
    limitations: List[str] = Field(
        default_factory=list,
        description="Missing evidence or processing gaps that affect reliability.",
    )
    processing_summary: ProcessingSummary

    @model_validator(mode="after")
    def score_matches_rating(self) -> "AssessmentReport":
        """
        Validate that the Python-calculated score is consistent with the rating.
        Thresholds are checked here against the default values; full validation
        lives in scoring.py where the configurable thresholds are applied.
        """
        score = self.overall_score
        rating = self.rating
        # Default thresholds (same as config defaults)
        if score <= 39 and rating != Rating.POOR:
            raise ValueError(
                f"Score {score:.1f} implies Poor but rating is {rating}."
            )
        if 40 <= score <= 59 and rating != Rating.AVERAGE:
            raise ValueError(
                f"Score {score:.1f} implies Average but rating is {rating}."
            )
        if 60 <= score <= 79 and rating != Rating.GOOD:
            raise ValueError(
                f"Score {score:.1f} implies Good but rating is {rating}."
            )
        if score >= 80 and rating != Rating.EXCELLENT:
            raise ValueError(
                f"Score {score:.1f} implies Excellent but rating is {rating}."
            )
        return self

    @model_validator(mode="after")
    def five_distinct_dimensions(self) -> "AssessmentReport":
        names = [d.name for d in self.dimensions]
        if len(set(names)) != 5:
            raise ValueError(f"Expected 5 distinct dimension names; got: {names}")
        return self


# ── Intermediate pipeline objects ─────────────────────────────────────────────

class TranscriptSegment(BaseModel):
    """One timed segment from the transcription provider."""
    start: float = Field(..., ge=0)
    end: float = Field(..., ge=0)
    text: str
    confidence: Optional[float] = Field(default=None, ge=0, le=1)

    @model_validator(mode="after")
    def end_after_start(self) -> "TranscriptSegment":
        if self.end < self.start:
            raise ValueError(
                f"Segment end ({self.end}) is before start ({self.start})."
            )
        return self


class TranscriptResult(BaseModel):
    """Full transcript with segments and metadata."""
    full_text: str
    segments: List[TranscriptSegment] = Field(default_factory=list)
    language: Optional[str] = None
    duration_seconds: Optional[float] = Field(default=None, ge=0)
    word_count: int = Field(default=0, ge=0)


class AudioFeatures(BaseModel):
    """Measurable audio characteristics."""
    duration_seconds: float = Field(..., ge=0)
    has_speech: bool
    silence_ratio: Optional[float] = Field(default=None, ge=0, le=1,
        description="Fraction of recording that is silent (>2 s gaps).")
    long_silence_count: Optional[int] = Field(default=None, ge=0,
        description="Number of silences longer than 5 seconds.")
    estimated_snr_db: Optional[float] = Field(default=None,
        description="Rough signal-to-noise ratio estimate (higher = cleaner).")
    clipping_detected: bool = False
    notes: List[str] = Field(default_factory=list)


class VideoFrame(BaseModel):
    """Metadata for one extracted frame."""
    timestamp_seconds: float = Field(..., ge=0)
    frame_index: int = Field(..., ge=0)
    description: Optional[str] = None   # filled by vision model if available
    path: Optional[str] = None          # local temp path (never serialised to JSON)

    model_config = {"json_schema_extra": {"example": {"timestamp_seconds": 30.0, "frame_index": 1}}}


class EvidencePackage(BaseModel):
    """
    The complete evidence bundle passed to the AI evaluator.
    Contains all available facts; the evaluator must not fabricate anything
    beyond what is present here.
    """
    recording_metadata: RecordingMetadata
    transcript: Optional[TranscriptResult] = None
    audio_features: Optional[AudioFeatures] = None
    frames: List[VideoFrame] = Field(default_factory=list)
    processing_summary: ProcessingSummary
    processing_notes: List[str] = Field(default_factory=list,
        description="Warnings or limitations from the pipeline steps.")
