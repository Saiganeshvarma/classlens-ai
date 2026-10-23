"""
evaluator.py — AI-powered teaching quality evaluator for ClassLens AI.

Builds an evidence package from the pipeline outputs, sends it to the
configured LLM with a structured prompt, parses the JSON response through
Pydantic, and returns validated dimension assessments.

The LLM never calculates the overall score or determines the final rating —
those are always done by scoring.py.
"""

from __future__ import annotations

import json
import logging
import textwrap
import time
from typing import List, Optional, Tuple

from config import get_settings
from schemas import (
    ConfidenceLevel,
    DimensionAssessment,
    DimensionStatus,
    EvidenceItem,
    EvidencePackage,
    EvidenceType,
)

logger = logging.getLogger(__name__)


# ── Exceptions ────────────────────────────────────────────────────────────────

class EvaluatorError(Exception):
    """Raised when evaluation cannot be completed."""


class MalformedResponseError(EvaluatorError):
    """Raised when the LLM response cannot be parsed or validated."""


# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""
You are an expert educational quality evaluator assessing online classes for children.

Your task is to evaluate the quality of a class recording using the evidence provided
(transcript segments, audio measurements, and video frame descriptions).

EVALUATION DIMENSIONS (with fixed weights you must not alter):
1. Teaching Clarity (weight: 30%) — clarity of explanation, logical progression, examples.
2. Engagement and Checks for Understanding (weight: 25%) — questions, student interaction opportunities, adaptation.
3. Lesson Structure (weight: 20%) — introduction, coherent flow, transitions, recap.
4. Audio Intelligibility (weight: 15%) — speech clarity, background noise, distortion.
5. Visual Instructional Support (weight: 10%) — on-screen content relevance, readability.

STRICT RULES:
1. Base every conclusion only on the evidence supplied in this message.
2. Label observations as "direct_observation": true/false.
3. NEVER invent transcript quotations, timestamps, student responses, or video events.
4. Do NOT judge based on teacher appearance, accent, personality, or demographics.
5. Do NOT assume high energy or constant talking equals excellent teaching.
6. Do NOT assume silence means disengagement — it may be think time.
7. Do NOT infer learning outcomes without supporting evidence.
8. If a dimension cannot be assessed (e.g. no video available, NO TRANSCRIPT, no audio), set its status to "unavailable".
9. If the transcript shows 0 words, mark Teaching Clarity, Engagement, Lesson Structure, and Audio Intelligibility as "unavailable".
10. Cite real timestamps from the evidence where available.
11. Give specific, actionable improvement recommendations.
12. Be honest about uncertainty and gaps.
13. Do NOT calculate an overall score — the scoring engine handles that.
14. Return ONLY valid JSON matching the schema below. No markdown, no prose outside JSON.

REQUIRED JSON SCHEMA:
{
  "dimensions": [
    {
      "name": "<exact dimension name from the list above>",
      "score": <integer 0-100 or null if unavailable>,
      "status": "assessed" | "partial" | "unavailable",
      "evidence": [
        {
          "observation": "<specific observation>",
          "evidence_type": "transcript" | "audio_feature" | "visual_frame" | "derived",
          "timestamp_seconds": <float or null>,
          "source_excerpt": "<short verbatim excerpt or null>",
          "impact": "<why this matters>",
          "is_direct_observation": true | false
        }
      ],
      "rationale": "<explanation of the score citing evidence>"
    }
  ],
  "strengths": ["<specific positive observation>", ...],
  "improvements": ["<specific actionable recommendation>", ...],
  "overall_rationale": "<holistic explanation of the teaching quality>",
  "confidence": "low" | "medium" | "high",
  "limitations": ["<gap or caveat>", ...]
}
""").strip()


# ── Evidence formatter ────────────────────────────────────────────────────────

def _format_evidence_for_prompt(pkg: EvidencePackage) -> str:
    """
    Serialise the EvidencePackage into a readable prompt section.
    Truncates long transcripts to avoid exceeding context limits.
    """
    lines: List[str] = []

    meta = pkg.recording_metadata
    lines.append("=== RECORDING METADATA ===")
    lines.append(f"Filename: {meta.filename}")
    lines.append(f"Duration: {meta.duration_seconds:.1f} seconds")
    lines.append(f"Audio stream available: {meta.audio_available}")
    lines.append(f"Video stream available: {meta.video_available}")
    if meta.detected_language:
        lines.append(f"Detected language: {meta.detected_language}")
    lines.append("")

    # Audio features
    if pkg.audio_features:
        af = pkg.audio_features
        lines.append("=== AUDIO FEATURES ===")
        lines.append(f"Duration analysed: {af.duration_seconds:.1f}s")
        lines.append(f"Speech detected: {af.has_speech}")
        if af.silence_ratio is not None:
            lines.append(f"Silence ratio: {af.silence_ratio:.0%}")
        if af.long_silence_count is not None:
            lines.append(f"Long silences (>5s): {af.long_silence_count}")
        if af.estimated_snr_db is not None:
            lines.append(f"Estimated SNR: {af.estimated_snr_db:.1f} dB")
        lines.append(f"Clipping detected: {af.clipping_detected}")
        for note in af.notes:
            lines.append(f"Note: {note}")
        lines.append("")

    # Transcript
    if pkg.transcript:
        tr = pkg.transcript
        lines.append("=== TRANSCRIPT ===")
        if tr.language:
            lines.append(f"Language: {tr.language}")
        lines.append(f"Total words: {tr.word_count}")
        lines.append("")

        # Include up to 6000 words to stay within context limits
        MAX_WORDS = 6000
        included_words = 0
        truncated = False

        for seg in tr.segments:
            words_in_seg = len(seg.text.split())
            if included_words + words_in_seg > MAX_WORDS:
                truncated = True
                break
            ts = f"[{seg.start:.0f}s–{seg.end:.0f}s]"
            lines.append(f"{ts} {seg.text}")
            included_words += words_in_seg

        if truncated:
            lines.append(
                f"\n[TRANSCRIPT TRUNCATED — showing first {included_words} of "
                f"{tr.word_count} words due to context limits. "
                "Later portions were not included in this evaluation.]"
            )
        lines.append("")
    else:
        lines.append("=== TRANSCRIPT ===")
        lines.append("No transcript available.")
        lines.append("")

    # Video frames
    if pkg.frames:
        lines.append("=== VIDEO FRAMES ===")
        lines.append(
            f"{len(pkg.frames)} frames extracted at "
            f"~{get_settings().frame_interval_seconds}s intervals."
        )
        for frame in pkg.frames:
            ts_str = f"{frame.timestamp_seconds:.0f}s"
            desc = frame.description or "(no visual description available)"
            lines.append(f"  Frame @ {ts_str}: {desc}")
        lines.append(
            "Note: Only sampled frames were inspected — full visual content was not analysed."
        )
        lines.append("")
    else:
        lines.append("=== VIDEO FRAMES ===")
        lines.append("No frames available (audio-only recording or extraction failed).")
        lines.append("")

    # Processing notes
    if pkg.processing_notes:
        lines.append("=== PROCESSING NOTES ===")
        for note in pkg.processing_notes:
            lines.append(f"• {note}")
        lines.append("")

    return "\n".join(lines)


# ── OpenAI evaluation ─────────────────────────────────────────────────────────

def _call_openai_with_retry(client, model: str, messages: list, max_wait: int = 60) -> str:
    """
    Call the OpenAI chat completions API with one automatic retry on rate limit.
    Returns the raw response text.
    """
    import openai

    def _do_call():
        return client.chat.completions.create(
            model=model,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.2,
            max_tokens=4096,
        )

    try:
        response = _do_call()
        return response.choices[0].message.content or ""
    except openai.RateLimitError:
        logger.warning("Rate limit hit — waiting %ds before retry...", max_wait)
        time.sleep(max_wait)
        try:
            response = _do_call()
            return response.choices[0].message.content or ""
        except openai.RateLimitError as exc2:
            raise EvaluatorError(
                "OpenAI rate limit reached even after waiting. "
                "Try switching EVALUATION_MODEL=gpt-4o-mini in your .env for higher limits, "
                "or wait a few minutes and retry."
            ) from exc2


def _evaluate_openai(evidence_text: str) -> dict:
    """Call GPT-4o (or configured model) with the evidence and return parsed JSON dict."""
    import openai  # local import

    settings = get_settings()
    client = openai.OpenAI(
        api_key=settings.openai_api_key,
        timeout=settings.llm_timeout_seconds,
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Please evaluate the following class recording evidence and return "
                "your assessment as JSON matching the schema in the system prompt.\n\n"
                + evidence_text
            ),
        },
    ]

    try:
        raw = _call_openai_with_retry(client, settings.evaluation_model, messages)
    except EvaluatorError:
        raise
    except openai.AuthenticationError as exc:
        raise EvaluatorError(
            "OpenAI authentication failed. Check your OPENAI_API_KEY."
        ) from exc
    except openai.APITimeoutError as exc:
        raise EvaluatorError(
            f"Evaluation timed out after {settings.llm_timeout_seconds}s."
        ) from exc
    except openai.APIError as exc:
        raise EvaluatorError(f"OpenAI API error: {exc}") from exc

    if not raw:
        raise MalformedResponseError("LLM returned an empty response.")

    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise MalformedResponseError(
            f"LLM response is not valid JSON: {exc}\n"
            f"Raw response (first 500 chars): {raw[:500]}"
        ) from exc


# ── Ollama evaluation ─────────────────────────────────────────────────────────

def _evaluate_ollama(evidence_text: str) -> dict:
    """Call Ollama with the evidence and return parsed JSON dict."""
    import requests

    settings = get_settings()
    
    user_message = (
        "Please evaluate the following class recording evidence and return "
        "your assessment as JSON matching the schema in the system prompt.\n\n"
        + evidence_text
        + "\n\nReturn ONLY the JSON object, no other text."
    )

    try:
        response = requests.post(
            "http://localhost:11434/api/chat",
            json={
                "model": settings.evaluation_model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message}
                ],
                "stream": False,
                "format": "json",
                "options": {
                    "temperature": 0.2,
                    "num_predict": 4096,
                }
            },
            timeout=settings.llm_timeout_seconds,
        )
        response.raise_for_status()
        result = response.json()
        raw = result.get("message", {}).get("content", "")
    except requests.exceptions.ConnectionError as exc:
        raise EvaluatorError(
            "Cannot connect to Ollama. Make sure Ollama is running: ollama serve"
        ) from exc
    except requests.exceptions.Timeout as exc:
        raise EvaluatorError(
            f"Ollama request timed out after {settings.llm_timeout_seconds}s."
        ) from exc
    except requests.exceptions.RequestException as exc:
        raise EvaluatorError(f"Ollama API error: {exc}") from exc

    if not raw:
        raise MalformedResponseError("Ollama returned an empty response.")

    # Strip markdown fences if present
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise MalformedResponseError(
            f"Ollama response is not valid JSON: {exc}\n"
            f"Raw (first 500): {raw[:500]}"
        ) from exc


# ── Gemini evaluation ─────────────────────────────────────────────────────────

def _evaluate_gemini(evidence_text: str) -> dict:
    """Call Google Gemini with the evidence and return parsed JSON dict."""
    import google.generativeai as genai
    import time

    settings = get_settings()
    genai.configure(api_key=settings.gemini_api_key)

    model = genai.GenerativeModel(
        model_name=settings.evaluation_model,
        system_instruction=SYSTEM_PROMPT,
        generation_config=genai.types.GenerationConfig(
            temperature=0.2,
            max_output_tokens=4096,
            response_mime_type="application/json",
        ),
    )

    user_message = (
        "Please evaluate the following class recording evidence and return "
        "your assessment as JSON matching the schema in the system prompt.\n\n"
        + evidence_text
    )

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = model.generate_content(user_message)
            raw = response.text
            break
        except Exception as exc:
            err = str(exc).lower()
            if "api_key" in err or "invalid" in err or "403" in err or "api key" in err or "authentication" in err:
                raise EvaluatorError(
                    "Gemini authentication failed. Check your GEMINI_API_KEY in .env."
                ) from exc
            if "quota" in err or "rate" in err or "429" in err or "resource_exhausted" in err:
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 20  # 20s, 40s, 60s
                    logger.warning(f"Gemini rate limit hit. Waiting {wait_time}s before retry {attempt + 2}/{max_retries}...")
                    time.sleep(wait_time)
                    continue
                else:
                    raise EvaluatorError(
                        "Gemini rate limit reached. Wait a moment and retry, "
                        "or check your quota at https://aistudio.google.com."
                    ) from exc
            raise EvaluatorError(f"Gemini API error: {exc}") from exc
    else:
        raise EvaluatorError("Gemini evaluation failed after all retries")

    if not raw:
        raise MalformedResponseError("Gemini returned an empty response.")

    # Strip markdown fences if present
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise MalformedResponseError(
            f"Gemini response is not valid JSON: {exc}\n"
            f"Raw (first 500): {raw[:500]}"
        ) from exc


# ── Anthropic evaluation ──────────────────────────────────────────────────────

def _evaluate_anthropic(evidence_text: str) -> dict:
    """Call Claude with the evidence and return parsed JSON dict."""
    import anthropic  # local import

    settings = get_settings()
    client = anthropic.Anthropic(
        api_key=settings.anthropic_api_key,
        timeout=settings.llm_timeout_seconds,
    )

    user_message = (
        "Please evaluate the following class recording evidence and return "
        "your assessment as JSON matching the schema in the system prompt.\n\n"
        + evidence_text
        + "\n\nReturn ONLY the JSON object, no other text."
    )

    try:
        response = client.messages.create(
            model=settings.evaluation_model,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )
    except anthropic.AuthenticationError as exc:
        raise EvaluatorError(
            "Anthropic authentication failed. Check your ANTHROPIC_API_KEY."
        ) from exc
    except anthropic.RateLimitError as exc:
        raise EvaluatorError("Anthropic rate limit reached.") from exc
    except anthropic.APIError as exc:
        raise EvaluatorError(f"Anthropic API error: {exc}") from exc

    raw = response.content[0].text if response.content else ""
    if not raw:
        raise MalformedResponseError("Anthropic returned an empty response.")

    # Strip markdown code fences if present
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])

    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise MalformedResponseError(
            f"Anthropic response is not valid JSON: {exc}\n"
            f"Raw (first 500): {raw[:500]}"
        ) from exc


# ── Response parser / validator ───────────────────────────────────────────────

_CANONICAL_DIMENSION_NAMES = [
    "Teaching Clarity",
    "Engagement and Checks for Understanding",
    "Lesson Structure",
    "Audio Intelligibility",
    "Visual Instructional Support",
]


def _parse_dimension(raw: dict) -> DimensionAssessment:
    """
    Parse one dimension dict from the LLM response into a DimensionAssessment.
    Raises MalformedResponseError with a descriptive message on failure.
    """
    settings = get_settings()
    weight_map = settings.dimension_weights()

    name = str(raw.get("name", "")).strip()
    if name not in weight_map:
        raise MalformedResponseError(
            f"Unknown dimension name '{name}'. "
            f"Expected one of: {list(weight_map.keys())}"
        )

    # Always use config weight — the LLM cannot change this
    weight = weight_map[name]

    raw_status = str(raw.get("status", "assessed")).lower()
    try:
        status = DimensionStatus(raw_status)
    except ValueError:
        status = DimensionStatus.ASSESSED  # default gracefully

    raw_score = raw.get("score")
    score: Optional[float] = None
    if raw_score is not None and status != DimensionStatus.UNAVAILABLE:
        try:
            score = max(0.0, min(100.0, float(raw_score)))
        except (ValueError, TypeError):
            score = None

    # Parse evidence items
    evidence: List[EvidenceItem] = []
    for ev in raw.get("evidence", []):
        try:
            ts = ev.get("timestamp_seconds")
            ts_float: Optional[float] = float(ts) if ts is not None else None
            et_raw = str(ev.get("evidence_type", "derived")).lower()
            try:
                et = EvidenceType(et_raw)
            except ValueError:
                et = EvidenceType.DERIVED
            item = EvidenceItem(
                observation=str(ev.get("observation", ""))[:500],
                evidence_type=et,
                timestamp_seconds=ts_float,
                source_excerpt=str(ev.get("source_excerpt", ""))[:500] or None,
                impact=str(ev.get("impact", "Supports quality assessment."))[:300],
                is_direct_observation=bool(ev.get("is_direct_observation", True)),
            )
            evidence.append(item)
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("Skipping malformed evidence item: %s", exc)

    rationale = str(raw.get("rationale", "No rationale provided.")).strip()
    if len(rationale) < 10:
        rationale = f"Score {score} assigned for {name}. No detailed rationale provided."

    try:
        return DimensionAssessment(
            name=name,
            score=score,
            weight=weight,
            status=status,
            evidence=evidence,
            rationale=rationale,
        )
    except Exception as exc:
        raise MalformedResponseError(
            f"Dimension '{name}' failed Pydantic validation: {exc}"
        ) from exc


def _parse_llm_response(
    raw_json: dict,
) -> Tuple[List[DimensionAssessment], List[str], List[str], str, ConfidenceLevel, List[str]]:
    """
    Parse and validate the full LLM JSON response.

    Returns:
        (dimensions, strengths, improvements, overall_rationale, confidence, limitations)
    """
    raw_dims = raw_json.get("dimensions", [])
    if not isinstance(raw_dims, list) or len(raw_dims) == 0:
        raise MalformedResponseError("LLM response has no 'dimensions' list.")

    # Build a map by name so we can handle out-of-order responses
    by_name: dict[str, DimensionAssessment] = {}
    for raw_dim in raw_dims:
        try:
            dim = _parse_dimension(raw_dim)
            by_name[dim.name] = dim
        except MalformedResponseError as exc:
            logger.warning("Dimension parse failed: %s", exc)

    # Ensure all five canonical dimensions are present
    settings = get_settings()
    weight_map = settings.dimension_weights()
    dimensions: List[DimensionAssessment] = []

    for canonical_name in _CANONICAL_DIMENSION_NAMES:
        if canonical_name in by_name:
            dimensions.append(by_name[canonical_name])
        else:
            logger.warning(
                "Dimension '%s' missing from LLM response — marking unavailable.",
                canonical_name,
            )
            dimensions.append(
                DimensionAssessment(
                    name=canonical_name,
                    score=None,
                    weight=weight_map[canonical_name],
                    status=DimensionStatus.UNAVAILABLE,
                    evidence=[],
                    rationale=(
                        f"This dimension was not included in the AI evaluation response. "
                        "It cannot be scored."
                    ),
                )
            )

    strengths = [str(s) for s in raw_json.get("strengths", []) if str(s).strip()]
    improvements = [str(i) for i in raw_json.get("improvements", []) if str(i).strip()]
    overall_rationale = str(raw_json.get("overall_rationale", "")).strip()
    if not overall_rationale or overall_rationale.lower() in ["none", "n/a", "null"]:
        overall_rationale = "No assessable dimensions were available for this recording. The content lacks sufficient speech or visual elements to evaluate teaching quality."

    raw_confidence = str(raw_json.get("confidence", "medium")).lower()
    try:
        confidence = ConfidenceLevel(raw_confidence)
    except ValueError:
        confidence = ConfidenceLevel.MEDIUM

    limitations = [str(l) for l in raw_json.get("limitations", []) if str(l).strip()]

    return dimensions, strengths, improvements, overall_rationale, confidence, limitations


# ── Vision analysis helpers ───────────────────────────────────────────────────

def _analyse_frames_with_vision(
    frames_with_paths: list,
    transcript_snippet: str,
) -> List[str]:
    """
    Use GPT-4o vision to describe video frames.

    Returns a list of description strings (one per frame that was sent).
    Falls back gracefully if vision is not available or API fails.
    """
    import base64

    settings = get_settings()
    if settings.ai_provider != "openai" or not settings.provider_ready():
        return []

    try:
        import openai
    except ImportError:
        return []

    client = openai.OpenAI(
        api_key=settings.openai_api_key,
        timeout=60,
    )

    # Build content list
    content = [
        {
            "type": "text",
            "text": (
                "These are frames sampled from an online class recording. "
                "For each frame, briefly describe: what is shown on screen "
                "(slides, whiteboard, teacher, etc.), whether it supports "
                "the lesson topic, and whether any text is readable. "
                "Be specific and factual. Do not infer the teacher's quality "
                "from appearance.\n\n"
                f"Transcript context (first 500 chars): {transcript_snippet[:500]}"
            ),
        }
    ]

    for frame in frames_with_paths:
        if not frame.path or not __import__("os").path.exists(frame.path):
            continue
        try:
            with open(frame.path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{b64}",
                    "detail": "low",   # cheaper; adequate for frame content detection
                },
            })
            content.append({
                "type": "text",
                "text": f"Frame at {frame.timestamp_seconds:.0f}s:",
            })
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("Could not encode frame at %.0fs: %s", frame.timestamp_seconds, exc)

    if len(content) == 1:
        return []  # no frames encoded

    try:
        response = client.chat.completions.create(
            model=settings.evaluation_model,
            messages=[{"role": "user", "content": content}],
            max_tokens=1024,
        )
        raw = response.choices[0].message.content or ""
        # Split into per-frame descriptions (best effort)
        descriptions = [line.strip() for line in raw.split("\n") if line.strip()]
        return descriptions
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("Vision analysis failed: %s", exc)
        return []


# ── Public entry point ────────────────────────────────────────────────────────

def evaluate(
    pkg: EvidencePackage,
) -> Tuple[List[DimensionAssessment], List[str], List[str], str, ConfidenceLevel, List[str]]:
    """
    Run the AI evaluation and return validated dimension assessments.

    Steps:
    1. Optionally enrich frames with vision descriptions.
    2. Format evidence into a structured prompt.
    3. Call the configured LLM (with retry on malformed response).
    4. Parse and validate the response through Pydantic.

    Returns:
        (dimensions, strengths, improvements, overall_rationale, confidence, limitations)

    Raises EvaluatorError if the provider is not configured or all retries fail.
    """
    settings = get_settings()

    if not settings.provider_ready():
        raise EvaluatorError(
            f"AI provider '{settings.ai_provider}' is not configured. "
            + "\n".join(settings.setup_errors())
        )

    # Vision enrichment (best-effort — failure doesn't block evaluation)
    if pkg.frames and pkg.recording_metadata.video_available:
        transcript_snippet = (
            pkg.transcript.full_text[:500]
            if pkg.transcript and pkg.transcript.full_text
            else ""
        )
        descriptions = _analyse_frames_with_vision(pkg.frames, transcript_snippet)
        # Attach descriptions back to frames (mutate in place for simplicity)
        for i, desc in enumerate(descriptions):
            if i < len(pkg.frames):
                pkg.frames[i] = pkg.frames[i].model_copy(update={"description": desc})

    evidence_text = _format_evidence_for_prompt(pkg)
    logger.debug("Evidence prompt length: %d chars", len(evidence_text))

    last_error: Optional[Exception] = None
    for attempt in range(1, settings.llm_max_retries + 1):
        logger.info("Evaluation attempt %d/%d.", attempt, settings.llm_max_retries)
        try:
            if settings.ai_provider == "openai":
                raw_json = _evaluate_openai(evidence_text)
            elif settings.ai_provider == "gemini":
                raw_json = _evaluate_gemini(evidence_text)
            elif settings.ai_provider == "ollama":
                raw_json = _evaluate_ollama(evidence_text)
            else:
                raw_json = _evaluate_anthropic(evidence_text)

            result = _parse_llm_response(raw_json)
            logger.info("Evaluation succeeded on attempt %d.", attempt)
            return result

        except MalformedResponseError as exc:
            logger.warning("Attempt %d malformed response: %s", attempt, exc)
            last_error = exc
            # Add a note to the evidence so the next attempt knows to be more careful
            evidence_text += (
                f"\n\n[PREVIOUS ATTEMPT {attempt} FAILED VALIDATION: {exc}. "
                "Please ensure your response is valid JSON exactly matching the schema.]"
            )
        except EvaluatorError:
            raise  # provider errors are not retryable

    raise EvaluatorError(
        f"Evaluation failed after {settings.llm_max_retries} attempts. "
        f"Last error: {last_error}"
    )
