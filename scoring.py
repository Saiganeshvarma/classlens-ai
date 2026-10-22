"""
scoring.py — Weighted scoring engine for ClassLens AI.

The Python engine (not the LLM) always calculates the final score and
determines the rating category.  This module is intentionally free of
any AI/network calls and is fully unit-testable.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from config import get_settings
from schemas import (
    AssessmentReport,
    DimensionAssessment,
    DimensionStatus,
    Rating,
)


# ── Rating lookup ─────────────────────────────────────────────────────────────

def score_to_rating(score: float) -> Rating:
    """
    Map a 0–100 score to a Rating category using the configured thresholds.

    Thresholds are read from settings so they can be adjusted via .env
    without touching source code.  The mapping is:
        0 – POOR_MAX   → Poor
        POOR_MAX+1 – AVERAGE_MAX  → Average
        AVERAGE_MAX+1 – GOOD_MAX  → Good
        GOOD_MAX+1 – 100          → Excellent
    """
    s = get_settings()
    if score <= s.threshold_poor_max:
        return Rating.POOR
    if score <= s.threshold_average_max:
        return Rating.AVERAGE
    if score <= s.threshold_good_max:
        return Rating.GOOD
    return Rating.EXCELLENT


# ── Core weighted calculation ─────────────────────────────────────────────────

def calculate_weighted_score(dimensions: List[DimensionAssessment]) -> Tuple[float, List[str]]:
    """
    Calculate the weighted overall score from dimension assessments.

    Rules:
    - Dimensions with status UNAVAILABLE are excluded from both the
      numerator and denominator (re-normalised).
    - Dimensions with status PARTIAL contribute their score at full weight
      (the partial evidence is all we have).
    - Returns (rounded_score, list_of_warnings).

    Raises ValueError if no dimension has a usable score.
    """
    warnings: List[str] = []
    numerator = 0.0
    denominator = 0.0

    for dim in dimensions:
        if dim.status == DimensionStatus.UNAVAILABLE:
            warnings.append(
                f"Dimension '{dim.name}' is unavailable and excluded from scoring."
            )
            continue
        if dim.score is None:
            warnings.append(
                f"Dimension '{dim.name}' has status '{dim.status}' but no score; "
                "excluding it from scoring."
            )
            continue
        numerator += dim.score * dim.weight
        denominator += dim.weight

    if denominator == 0:
        raise ValueError(
            "No assessable dimensions found — cannot calculate an overall score. "
            "The recording may lack sufficient usable content."
        )

    raw = numerator / denominator
    # Clamp to [0, 100] to absorb any floating-point edge cases
    clamped = max(0.0, min(100.0, raw))
    rounded = round(clamped, 1)
    return rounded, warnings


def apply_weights_from_config(
    dimensions: List[DimensionAssessment],
) -> List[DimensionAssessment]:
    """
    Overwrite the weight on each dimension with the value from config.

    The LLM is instructed not to change weights, but this function
    enforces it programmatically so a misbehaving model cannot alter scoring.
    """
    settings = get_settings()
    weight_map = settings.dimension_weights()

    for dim in dimensions:
        if dim.name in weight_map:
            # Pydantic v2: use model_copy to get an updated instance
            dim = dim.model_copy(update={"weight": weight_map[dim.name]})
        else:
            # Unknown dimension name — keep the weight as-is and warn
            pass  # caller can detect this by checking returned list

    return dimensions


def enforce_config_weights(dimensions: List[DimensionAssessment]) -> List[DimensionAssessment]:
    """
    Return a new list of dimensions with weights replaced by config values.
    This is the public-facing guard used in the pipeline.
    """
    settings = get_settings()
    weight_map = settings.dimension_weights()
    updated = []
    for dim in dimensions:
        correct_weight = weight_map.get(dim.name, dim.weight)
        updated.append(dim.model_copy(update={"weight": correct_weight}))
    return updated


# ── Score breakdown for UI ────────────────────────────────────────────────────

def score_breakdown(dimensions: List[DimensionAssessment]) -> List[Dict]:
    """
    Return a list of dicts suitable for charting in the Streamlit dashboard.
    Unavailable dimensions are included with score=None for transparency.
    """
    return [
        {
            "dimension": dim.name,
            "score": dim.score,
            "weight": dim.weight,
            "status": dim.status.value,
            "weighted_contribution": (
                round(dim.score * dim.weight / 100, 1)
                if dim.score is not None
                else None
            ),
        }
        for dim in dimensions
    ]


def normalised_weights(dimensions: List[DimensionAssessment]) -> Dict[str, float]:
    """
    Return weights re-normalised to sum to 100 after excluding unavailable dims.
    Useful for displaying "effective weight" in the UI.
    """
    available = [
        d for d in dimensions if d.status != DimensionStatus.UNAVAILABLE
    ]
    total = sum(d.weight for d in available)
    if total == 0:
        return {}
    return {d.name: round(d.weight * 100 / total, 1) for d in available}


# ── Validation helpers ────────────────────────────────────────────────────────

def validate_score_rating_consistency(
    score: float, rating: Rating
) -> Optional[str]:
    """
    Return an error message if score and rating are inconsistent, else None.
    Uses current config thresholds.
    """
    expected_rating = score_to_rating(score)
    if rating != expected_rating:
        return (
            f"Score {score:.1f} should map to {expected_rating.value} "
            f"but the report says {rating.value}."
        )
    return None


# ── Entry point used by evaluator.py ─────────────────────────────────────────

def finalise_report_scores(
    dimensions: List[DimensionAssessment],
) -> Tuple[float, Rating, List[str]]:
    """
    The single function the pipeline calls to get the final score + rating.

    1. Enforces config weights (removes LLM interference).
    2. Calculates the weighted score.
    3. Maps score → Rating.
    4. Returns (score, rating, warnings).
    """
    enforced = enforce_config_weights(dimensions)
    score, warnings = calculate_weighted_score(enforced)
    rating = score_to_rating(score)
    return score, rating, warnings
