"""
test_scoring.py — Unit tests for the Python scoring engine.

Tests:
- score_to_rating() at every threshold boundary
- calculate_weighted_score() with all dims, unavailable dims, partial dims
- finalise_report_scores() end-to-end
- enforce_config_weights() overrides LLM-supplied weights
- Edge cases: score=0, score=100, all dims unavailable
"""

from __future__ import annotations

import os
import pytest

# Ensure no .env leaks alter thresholds during tests
os.environ.setdefault("THRESHOLD_POOR_MAX", "39")
os.environ.setdefault("THRESHOLD_AVERAGE_MAX", "59")
os.environ.setdefault("THRESHOLD_GOOD_MAX", "79")

from config import get_settings
from schemas import (
    DimensionAssessment,
    DimensionStatus,
    EvidenceItem,
    EvidenceType,
    Rating,
)
from scoring import (
    calculate_weighted_score,
    enforce_config_weights,
    finalise_report_scores,
    normalised_weights,
    score_breakdown,
    score_to_rating,
    validate_score_rating_consistency,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_dim(
    name: str,
    score: float | None,
    weight: float,
    status: DimensionStatus = DimensionStatus.ASSESSED,
) -> DimensionAssessment:
    """Helper — build a minimal valid DimensionAssessment."""
    evidence = []
    if status in (DimensionStatus.ASSESSED, DimensionStatus.PARTIAL):
        evidence = [
            EvidenceItem(
                observation=f"Specific observation for {name} dimension.",
                evidence_type=EvidenceType.TRANSCRIPT,
                impact="Directly affects the quality of teaching.",
            )
        ]
    rationale = f"Assessment rationale for {name} based on available evidence."
    return DimensionAssessment(
        name=name,
        score=score,
        weight=weight,
        status=status,
        evidence=evidence,
        rationale=rationale,
    )


def _all_five_dims(
    scores: list[float | None] | None = None,
    statuses: list[DimensionStatus] | None = None,
) -> list[DimensionAssessment]:
    """Return all five canonical dimensions with optional overrides."""
    settings = get_settings()
    names_weights = list(settings.dimension_weights().items())
    if scores is None:
        scores = [70.0, 65.0, 68.0, 75.0, 60.0]
    if statuses is None:
        statuses = [DimensionStatus.ASSESSED] * 5
    return [
        _make_dim(name, scores[i], weight, statuses[i])
        for i, (name, weight) in enumerate(names_weights)
    ]


# ── score_to_rating() ─────────────────────────────────────────────────────────

class TestScoreToRating:

    def test_score_0_is_poor(self):
        assert score_to_rating(0) == Rating.POOR

    def test_score_39_is_poor(self):
        assert score_to_rating(39) == Rating.POOR

    def test_score_39_point_9_is_average(self):
        # 39.9 > 39 (integer threshold), so it correctly maps to Average
        assert score_to_rating(39.9) == Rating.AVERAGE

    def test_score_40_is_average(self):
        assert score_to_rating(40) == Rating.AVERAGE

    def test_score_50_is_average(self):
        assert score_to_rating(50) == Rating.AVERAGE

    def test_score_59_is_average(self):
        assert score_to_rating(59) == Rating.AVERAGE

    def test_score_59_point_9_is_good(self):
        # 59.9 > 59 (integer threshold), so it correctly maps to Good
        assert score_to_rating(59.9) == Rating.GOOD

    def test_score_60_is_good(self):
        assert score_to_rating(60) == Rating.GOOD

    def test_score_70_is_good(self):
        assert score_to_rating(70) == Rating.GOOD

    def test_score_79_is_good(self):
        assert score_to_rating(79) == Rating.GOOD

    def test_score_79_point_9_is_excellent(self):
        # 79.9 > 79 (integer threshold), so it correctly maps to Excellent
        assert score_to_rating(79.9) == Rating.EXCELLENT

    def test_score_80_is_excellent(self):
        assert score_to_rating(80) == Rating.EXCELLENT

    def test_score_90_is_excellent(self):
        assert score_to_rating(90) == Rating.EXCELLENT

    def test_score_100_is_excellent(self):
        assert score_to_rating(100) == Rating.EXCELLENT


# ── calculate_weighted_score() ────────────────────────────────────────────────

class TestCalculateWeightedScore:

    def test_all_five_available_equal_scores(self):
        """All dims at 70 → score == 70.0 exactly."""
        dims = _all_five_dims(scores=[70.0] * 5)
        score, warnings = calculate_weighted_score(dims)
        assert abs(score - 70.0) < 0.01
        assert warnings == []

    def test_correct_weighted_formula(self):
        """Manual check: (70*30 + 65*25 + 68*20 + 75*15 + 60*10) / 100."""
        dims = _all_five_dims(scores=[70.0, 65.0, 68.0, 75.0, 60.0])
        expected = (70*30 + 65*25 + 68*20 + 75*15 + 60*10) / 100
        score, _ = calculate_weighted_score(dims)
        assert abs(score - round(expected, 1)) < 0.01

    def test_one_unavailable_dim_excluded(self):
        """Visual (weight 10) unavailable → denominator = 90."""
        statuses = [DimensionStatus.ASSESSED] * 4 + [DimensionStatus.UNAVAILABLE]
        scores = [70.0, 65.0, 68.0, 80.0, None]
        dims = _all_five_dims(scores=scores, statuses=statuses)
        score, warnings = calculate_weighted_score(dims)
        expected = (70*30 + 65*25 + 68*20 + 80*15) / 90
        assert abs(score - round(expected, 1)) < 0.01
        assert len(warnings) == 1
        assert "unavailable" in warnings[0].lower()

    def test_all_dims_unavailable_raises(self):
        """If no dimension is scoreable, ValueError must be raised."""
        statuses = [DimensionStatus.UNAVAILABLE] * 5
        scores = [None] * 5
        dims = _all_five_dims(scores=scores, statuses=statuses)
        with pytest.raises(ValueError, match="No assessable dimensions"):
            calculate_weighted_score(dims)

    def test_score_clamped_at_100(self):
        """Scores of 100 on all dims → 100.0."""
        dims = _all_five_dims(scores=[100.0] * 5)
        score, _ = calculate_weighted_score(dims)
        assert score == 100.0

    def test_score_clamped_at_0(self):
        """Scores of 0 on all dims → 0.0."""
        dims = _all_five_dims(scores=[0.0] * 5)
        score, _ = calculate_weighted_score(dims)
        assert score == 0.0

    def test_partial_status_included(self):
        """PARTIAL dims should still contribute to the score."""
        statuses = [DimensionStatus.PARTIAL] * 5
        scores = [60.0] * 5
        dims = _all_five_dims(scores=scores, statuses=statuses)
        score, warnings = calculate_weighted_score(dims)
        assert abs(score - 60.0) < 0.01
        assert warnings == []

    def test_three_unavailable_renormalises(self):
        """Three unavailable dims → renormalise remaining two."""
        statuses = [
            DimensionStatus.ASSESSED,   # Teaching Clarity w=30
            DimensionStatus.UNAVAILABLE,
            DimensionStatus.UNAVAILABLE,
            DimensionStatus.UNAVAILABLE,
            DimensionStatus.ASSESSED,   # Visual w=10
        ]
        scores = [80.0, None, None, None, 50.0]
        dims = _all_five_dims(scores=scores, statuses=statuses)
        score, warnings = calculate_weighted_score(dims)
        expected = (80 * 30 + 50 * 10) / 40
        assert abs(score - round(expected, 1)) < 0.01
        assert len(warnings) == 3


# ── finalise_report_scores() ─────────────────────────────────────────────────

class TestFinaliseReportScores:

    def test_returns_correct_tuple(self):
        dims = _all_five_dims(scores=[70.0] * 5)
        score, rating, warnings = finalise_report_scores(dims)
        assert isinstance(score, float)
        assert isinstance(rating, Rating)
        assert isinstance(warnings, list)

    def test_rating_matches_score(self):
        dims = _all_five_dims(scores=[75.0] * 5)
        score, rating, _ = finalise_report_scores(dims)
        assert rating == Rating.GOOD

    def test_enforces_config_weights(self):
        """Dims with wrong weights should be corrected before scoring."""
        settings = get_settings()
        names_weights = list(settings.dimension_weights().items())
        # Tamper with the first dimension's weight
        tampered = _all_five_dims(scores=[80.0] * 5)
        tampered[0] = tampered[0].model_copy(update={"weight": 99.0})
        # Should still produce the correct score (config weight restored)
        score, rating, _ = finalise_report_scores(tampered)
        # With all 80s, score must be 80 → Excellent
        assert abs(score - 80.0) < 0.01
        assert rating == Rating.EXCELLENT


# ── enforce_config_weights() ─────────────────────────────────────────────────

class TestEnforceConfigWeights:

    def test_overrides_llm_supplied_weight(self):
        settings = get_settings()
        dims = _all_five_dims(scores=[70.0] * 5)
        # Tamper every weight
        tampered = [d.model_copy(update={"weight": 20.0}) for d in dims]
        corrected = enforce_config_weights(tampered)
        for dim in corrected:
            expected_w = settings.dimension_weights()[dim.name]
            assert abs(dim.weight - expected_w) < 0.001

    def test_weights_sum_to_100_after_enforcement(self):
        dims = _all_five_dims(scores=[70.0] * 5)
        tampered = [d.model_copy(update={"weight": 99.0}) for d in dims]
        corrected = enforce_config_weights(tampered)
        total = sum(d.weight for d in corrected)
        assert abs(total - 100.0) < 0.001


# ── validate_score_rating_consistency() ──────────────────────────────────────

class TestValidateConsistency:

    def test_consistent_returns_none(self):
        assert validate_score_rating_consistency(75.0, Rating.GOOD) is None

    def test_inconsistent_returns_message(self):
        msg = validate_score_rating_consistency(75.0, Rating.POOR)
        assert msg is not None
        assert "75" in msg

    @pytest.mark.parametrize("score,rating", [
        (0,   Rating.POOR),
        (39,  Rating.POOR),
        (40,  Rating.AVERAGE),
        (59,  Rating.AVERAGE),
        (60,  Rating.GOOD),
        (79,  Rating.GOOD),
        (80,  Rating.EXCELLENT),
        (100, Rating.EXCELLENT),
    ])
    def test_boundary_values_consistent(self, score, rating):
        assert validate_score_rating_consistency(score, rating) is None


# ── score_breakdown() and normalised_weights() ────────────────────────────────

class TestBreakdownHelpers:

    def test_score_breakdown_length(self):
        dims = _all_five_dims()
        bd = score_breakdown(dims)
        assert len(bd) == 5

    def test_score_breakdown_contains_weighted_contribution(self):
        dims = _all_five_dims(scores=[80.0] * 5)
        bd = score_breakdown(dims)
        for item in bd:
            assert item["weighted_contribution"] is not None

    def test_unavailable_dim_has_none_score_in_breakdown(self):
        statuses = [DimensionStatus.ASSESSED] * 4 + [DimensionStatus.UNAVAILABLE]
        scores = [70.0, 70.0, 70.0, 70.0, None]
        dims = _all_five_dims(scores=scores, statuses=statuses)
        bd = score_breakdown(dims)
        last = bd[-1]
        assert last["score"] is None
        assert last["weighted_contribution"] is None

    def test_normalised_weights_excludes_unavailable(self):
        statuses = [DimensionStatus.ASSESSED] * 4 + [DimensionStatus.UNAVAILABLE]
        scores = [70.0, 70.0, 70.0, 70.0, None]
        dims = _all_five_dims(scores=scores, statuses=statuses)
        nw = normalised_weights(dims)
        assert len(nw) == 4
        total = sum(nw.values())
        assert abs(total - 100.0) < 0.1
