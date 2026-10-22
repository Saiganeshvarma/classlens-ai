"""
audio_analyzer.py — Transcription and audio feature analysis for ClassLens AI.

Two responsibilities:
  1. Transcribe speech using OpenAI Whisper (returns timestamped segments).
  2. Analyse measurable audio characteristics from the WAV file without
     calling any external API (silence, clipping, SNR estimate).

Provider-specific code is isolated to _transcribe_openai().  Adding a
second provider means implementing a matching _transcribe_<name>() and
wiring it in transcribe_audio().
"""

from __future__ import annotations

import logging
import math
import os
import wave
from typing import List, Optional, Tuple

import numpy as np

from config import get_settings
from schemas import AudioFeatures, TranscriptResult, TranscriptSegment

logger = logging.getLogger(__name__)


# ── Exceptions ────────────────────────────────────────────────────────────────

class TranscriptionError(Exception):
    """Raised when transcription cannot be completed."""


class AudioAnalysisError(Exception):
    """Raised when audio feature extraction fails."""


# ── OpenAI Whisper transcription ──────────────────────────────────────────────

def _transcribe_openai(audio_path: str) -> TranscriptResult:
    """
    Transcribe audio using OpenAI Whisper hosted API.

    Requires OPENAI_API_KEY to be set.
    Returns a TranscriptResult with timestamped segments.
    Raises TranscriptionError on any failure.
    """
    try:
        import openai  # local import so missing key doesn't crash at module load
    except ImportError as exc:
        raise TranscriptionError(
            "openai package is not installed. Run: pip install openai"
        ) from exc

    settings = get_settings()
    if not settings.transcription_ready():
        raise TranscriptionError(
            "OPENAI_API_KEY is not configured. "
            "Set it in your .env file to enable transcription."
        )

    api_key = settings.openai_api_key
    client = openai.OpenAI(api_key=api_key, timeout=settings.llm_timeout_seconds)

    file_size = os.path.getsize(audio_path)
    logger.info(
        "Sending audio to Whisper (%s, %.1f MB).",
        settings.transcription_model, file_size / (1024 * 1024),
    )

    try:
        with open(audio_path, "rb") as audio_file:
            response = client.audio.transcriptions.create(
                model=settings.transcription_model,
                file=audio_file,
                response_format="verbose_json",
                timestamp_granularities=["segment"],
            )
    except openai.AuthenticationError as exc:
        raise TranscriptionError(
            "OpenAI authentication failed. Check your OPENAI_API_KEY."
        ) from exc
    except openai.RateLimitError as exc:
        raise TranscriptionError(
            "OpenAI rate limit reached. Wait a moment and try again."
        ) from exc
    except openai.APITimeoutError as exc:
        raise TranscriptionError(
            f"OpenAI Whisper timed out after {settings.llm_timeout_seconds}s."
        ) from exc
    except openai.APIError as exc:
        raise TranscriptionError(f"OpenAI API error during transcription: {exc}") from exc

    # Parse response
    full_text: str = getattr(response, "text", "") or ""
    raw_segments = getattr(response, "segments", []) or []
    language: Optional[str] = getattr(response, "language", None)
    duration: Optional[float] = getattr(response, "duration", None)

    segments: List[TranscriptSegment] = []
    for seg in raw_segments:
        try:
            seg_dict = seg if isinstance(seg, dict) else seg.__dict__
            start = float(seg_dict.get("start", 0))
            end = float(seg_dict.get("end", start))
            text = str(seg_dict.get("text", "")).strip()
            if not text:
                continue
            segments.append(
                TranscriptSegment(
                    start=start,
                    end=max(end, start),
                    text=text,
                )
            )
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("Skipping malformed segment: %s", exc)

    word_count = len(full_text.split()) if full_text else 0
    logger.info(
        "Transcription complete: %d words, %d segments, language='%s'.",
        word_count, len(segments), language,
    )

    return TranscriptResult(
        full_text=full_text,
        segments=segments,
        language=language,
        duration_seconds=duration,
        word_count=word_count,
    )


def _transcribe_gemini(audio_path: str) -> TranscriptResult:
    """
    Transcribe audio using Google Gemini's multimodal API.
    Gemini 1.5 Flash/Pro can process audio files directly.
    Free tier: 15 RPM, 1500 requests/day — no billing required.
    """
    try:
        import google.generativeai as genai
    except ImportError as exc:
        raise TranscriptionError(
            "google-generativeai package not installed. Run: pip install google-generativeai"
        ) from exc

    settings = get_settings()
    if not settings.gemini_api_key or settings.gemini_api_key == "your-gemini-api-key-here":
        raise TranscriptionError(
            "GEMINI_API_KEY is not configured. Set it in your .env file."
        )

    genai.configure(api_key=settings.gemini_api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")

    file_size = os.path.getsize(audio_path)
    logger.info("Sending audio to Gemini for transcription (%.1f MB).", file_size / (1024*1024))

    try:
        audio_file = genai.upload_file(audio_path, mime_type="audio/wav")
        response = model.generate_content([
            audio_file,
            (
                "Please transcribe this audio recording completely and accurately. "
                "Format your response as a plain transcript only — no timestamps, "
                "no speaker labels, no commentary. Just the spoken words."
            ),
        ])
        full_text = response.text.strip() if response.text else ""
    except Exception as exc:
        err = str(exc).lower()
        if "quota" in err or "rate" in err or "429" in err:
            raise TranscriptionError(
                "Gemini rate limit reached during transcription. Wait a moment and retry."
            ) from exc
        if "api_key" in err or "invalid" in err or "403" in err:
            raise TranscriptionError(
                "Gemini authentication failed. Check your GEMINI_API_KEY."
            ) from exc
        raise TranscriptionError(f"Gemini transcription error: {exc}") from exc

    if not full_text:
        raise TranscriptionError("Gemini returned an empty transcript.")

    word_count = len(full_text.split())
    # Gemini doesn't return timestamps — create one segment for the full text
    segments = [TranscriptSegment(start=0.0, end=0.0, text=full_text)]

    logger.info("Gemini transcription complete: %d words.", word_count)
    return TranscriptResult(
        full_text=full_text,
        segments=segments,
        language=None,
        word_count=word_count,
    )


# ── Public transcription entry point ─────────────────────────────────────────

def transcribe_audio(audio_path: str) -> TranscriptResult:
    """
    Transcribe audio using the configured provider.

    - OpenAI Whisper: requires OPENAI_API_KEY, best timestamp accuracy
    - Gemini: requires GEMINI_API_KEY, free tier available, no timestamps
    """
    if not os.path.exists(audio_path):
        raise TranscriptionError(f"Audio file not found: '{audio_path}'")

    file_size = os.path.getsize(audio_path)
    if file_size == 0:
        raise TranscriptionError("Audio file is empty.")

    settings = get_settings()

    # Route to Gemini if that's the configured provider
    if settings.ai_provider == "gemini":
        return _transcribe_gemini(audio_path)

    # OpenAI Whisper — 25 MB upload limit
    max_bytes = 25 * 1024 * 1024
    if file_size > max_bytes:
        raise TranscriptionError(
            f"Audio file is {file_size / (1024*1024):.1f} MB, which exceeds "
            "Whisper's 25 MB limit."
        )

    if not settings.transcription_ready():
        raise TranscriptionError(
            "OPENAI_API_KEY is not configured. "
            "Set it in your .env file to enable transcription."
        )

    return _transcribe_openai(audio_path)


# ── Audio feature analysis (local, no API) ────────────────────────────────────

def _read_wav(audio_path: str) -> Tuple[np.ndarray, int]:
    """
    Read a WAV file and return (samples_array, sample_rate).
    Raises AudioAnalysisError on failure.
    """
    try:
        with wave.open(audio_path, "rb") as wf:
            n_channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            frame_rate = wf.getframerate()
            n_frames = wf.getnframes()
            raw = wf.readframes(n_frames)
    except (wave.Error, OSError) as exc:
        raise AudioAnalysisError(f"Cannot read WAV file: {exc}") from exc

    if sample_width == 2:
        dtype = np.int16
    elif sample_width == 4:
        dtype = np.int32
    else:
        dtype = np.int8

    samples = np.frombuffer(raw, dtype=dtype).astype(np.float32)

    # Mix down to mono if needed
    if n_channels > 1:
        samples = samples.reshape(-1, n_channels).mean(axis=1)

    # Normalise to [-1, 1]
    max_val = float(np.iinfo(dtype).max) if np.issubdtype(dtype, np.integer) else 1.0
    samples = samples / max_val

    return samples, frame_rate


def _detect_silence(
    samples: np.ndarray,
    sample_rate: int,
    silence_threshold: float = 0.01,
    min_silence_seconds: float = 2.0,
) -> Tuple[float, int]:
    """
    Detect silent regions.

    Returns:
        silence_ratio: fraction of total samples below threshold
        long_silence_count: number of continuous silent regions >= min_silence_seconds
    """
    frame_size = int(sample_rate * 0.02)  # 20 ms frames
    if frame_size == 0 or len(samples) == 0:
        return 0.0, 0

    # RMS per frame
    n_frames = len(samples) // frame_size
    if n_frames == 0:
        return 0.0, 0

    frames = samples[: n_frames * frame_size].reshape(n_frames, frame_size)
    rms = np.sqrt(np.mean(frames ** 2, axis=1))

    is_silent = rms < silence_threshold
    silence_ratio = float(np.mean(is_silent))

    # Find runs of silence
    min_frames = int(min_silence_seconds / 0.02)
    long_silence_count = 0
    run = 0
    for s in is_silent:
        if s:
            run += 1
            if run == min_frames:
                long_silence_count += 1
        else:
            run = 0

    return silence_ratio, long_silence_count


def _estimate_snr(samples: np.ndarray) -> Optional[float]:
    """
    Very rough SNR estimate: ratio of loud-frame RMS to quiet-frame RMS (dB).
    Returns None if the array is too short to be meaningful.
    """
    if len(samples) < 4000:
        return None
    sorted_rms = np.sort(np.abs(samples))
    n = len(sorted_rms)
    noise_rms = float(np.mean(sorted_rms[: n // 10])) + 1e-9  # bottom 10%
    signal_rms = float(np.mean(sorted_rms[n * 9 // 10 :])) + 1e-9  # top 10%
    snr_db = 20 * math.log10(signal_rms / noise_rms)
    return round(snr_db, 1)


def _detect_clipping(samples: np.ndarray, threshold: float = 0.98) -> bool:
    """
    Return True if more than 0.1% of samples are at or near full scale (clipping).
    """
    clipped = np.sum(np.abs(samples) >= threshold)
    ratio = clipped / max(len(samples), 1)
    return ratio > 0.001


def _has_speech(samples: np.ndarray, sample_rate: int) -> bool:
    """
    Rough speech detection: True if there are sustained non-silent regions.
    A recording with only background hiss will have very low variance.
    """
    if len(samples) == 0:
        return False
    frame_size = int(sample_rate * 0.02)
    if frame_size == 0:
        return False
    n_frames = len(samples) // frame_size
    if n_frames == 0:
        return False
    frames = samples[: n_frames * frame_size].reshape(n_frames, frame_size)
    rms = np.sqrt(np.mean(frames ** 2, axis=1))
    speech_frames = np.sum(rms > 0.02)
    return (speech_frames / n_frames) > 0.05  # at least 5% of frames have speech


def analyse_audio_features(audio_path: str) -> AudioFeatures:
    """
    Extract measurable audio features from a WAV file without any API call.

    Returns an AudioFeatures object.  On error, returns a minimal object
    with notes describing what failed rather than crashing the pipeline.
    """
    duration = 0.0
    notes: List[str] = []

    try:
        samples, sr = _read_wav(audio_path)
        duration = len(samples) / sr if sr > 0 else 0.0
    except AudioAnalysisError as exc:
        logger.warning("Audio feature analysis failed: %s", exc)
        return AudioFeatures(
            duration_seconds=0.0,
            has_speech=False,
            notes=[f"Audio feature analysis failed: {exc}"],
        )

    if duration < 1.0:
        return AudioFeatures(
            duration_seconds=duration,
            has_speech=False,
            notes=["Audio too short for meaningful feature analysis."],
        )

    has_speech = _has_speech(samples, sr)

    try:
        silence_ratio, long_silence_count = _detect_silence(samples, sr)
    except Exception as exc:  # pylint: disable=broad-except
        silence_ratio, long_silence_count = None, None
        notes.append(f"Silence detection skipped: {exc}")

    try:
        snr = _estimate_snr(samples)
    except Exception as exc:  # pylint: disable=broad-except
        snr = None
        notes.append(f"SNR estimation skipped: {exc}")

    try:
        clipping = _detect_clipping(samples)
    except Exception as exc:  # pylint: disable=broad-except
        clipping = False
        notes.append(f"Clipping detection skipped: {exc}")

    if clipping:
        notes.append(
            "Audio clipping detected — the recording may have distortion that "
            "affects intelligibility."
        )
    if silence_ratio is not None and silence_ratio > 0.5:
        notes.append(
            f"High silence ratio ({silence_ratio:.0%}) detected. "
            "The recording may have long gaps or very little spoken content."
        )
    if snr is not None and snr < 10:
        notes.append(
            f"Low estimated SNR ({snr} dB) suggests significant background noise."
        )

    logger.info(
        "Audio features: duration=%.1fs, has_speech=%s, silence=%.0f%%, snr=%.1fdB",
        duration, has_speech,
        (silence_ratio or 0) * 100,
        snr or 0,
    )

    return AudioFeatures(
        duration_seconds=duration,
        has_speech=has_speech,
        silence_ratio=silence_ratio,
        long_silence_count=long_silence_count,
        estimated_snr_db=snr,
        clipping_detected=clipping,
        notes=notes,
    )
