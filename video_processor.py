"""
video_processor.py — Media validation, audio extraction, and frame sampling.

Uses FFmpeg (via subprocess) and OpenCV.  All temporary files are written
to a unique per-run directory inside the system temp folder and deleted
in a finally block, regardless of success or failure.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import List, Optional, Tuple

import cv2

from config import (
    SUPPORTED_EXTENSIONS,
    SUPPORTED_VIDEO_EXTENSIONS,
    get_settings,
)
from schemas import (
    ProcessingSummary,
    RecordingMetadata,
    VideoFrame,
)

logger = logging.getLogger(__name__)


# ── Exceptions ────────────────────────────────────────────────────────────────

class VideoProcessorError(Exception):
    """Raised for recoverable processing errors shown to the user."""


class UnsupportedFormatError(VideoProcessorError):
    pass


class FileSizeError(VideoProcessorError):
    pass


class DurationError(VideoProcessorError):
    pass


class CorruptedMediaError(VideoProcessorError):
    pass


# ── FFprobe helpers ───────────────────────────────────────────────────────────

def _run_ffprobe(path: str) -> dict:
    """
    Run ffprobe and return parsed JSON.
    Raises CorruptedMediaError if ffprobe fails.
    """
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired as exc:
        raise CorruptedMediaError(
            "ffprobe timed out — the file may be corrupted or extremely large."
        ) from exc
    except FileNotFoundError as exc:
        raise CorruptedMediaError(
            "ffprobe is not installed. Install FFmpeg and ensure it is on PATH."
        ) from exc

    if result.returncode != 0:
        raise CorruptedMediaError(
            f"ffprobe could not read the file: {result.stderr.strip()[:300]}"
        )

    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise CorruptedMediaError("ffprobe returned unexpected output.") from exc


def _stream_types(probe: dict) -> Tuple[bool, bool]:
    """Return (has_video, has_audio) from an ffprobe dict."""
    has_video = any(
        s.get("codec_type") == "video" for s in probe.get("streams", [])
    )
    has_audio = any(
        s.get("codec_type") == "audio" for s in probe.get("streams", [])
    )
    return has_video, has_audio


def _get_duration(probe: dict) -> float:
    """Extract duration in seconds from ffprobe output."""
    # Try format-level duration first (most reliable)
    fmt = probe.get("format", {})
    try:
        return float(fmt["duration"])
    except (KeyError, ValueError):
        pass
    # Fall back to the first stream with a duration
    for stream in probe.get("streams", []):
        try:
            return float(stream["duration"])
        except (KeyError, ValueError):
            continue
    return 0.0


# ── Validation ────────────────────────────────────────────────────────────────

def validate_recording(file_path: str, file_size_bytes: int) -> RecordingMetadata:
    """
    Validate a recording file and return its metadata.

    Checks:
    1. Extension is in the supported set.
    2. File size is within the configured limit.
    3. ffprobe can decode the file (catches format mismatches and corruption).
    4. Duration is within the configured limit.
    5. At least one usable stream (audio OR video) is present.

    Raises VideoProcessorError subclasses for user-visible problems.
    """
    settings = get_settings()
    path = Path(file_path)

    # 1. Extension check
    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFormatError(
            f"Unsupported file type '{ext}'. "
            f"Supported formats: {', '.join(sorted(SUPPORTED_EXTENSIONS))}."
        )

    # 2. File size
    if file_size_bytes > settings.max_file_size_bytes:
        mb = file_size_bytes / (1024 * 1024)
        raise FileSizeError(
            f"File is {mb:.1f} MB, which exceeds the {settings.max_file_size_mb} MB limit."
        )

    # 3. Media probe (also catches format vs extension mismatches)
    probe = _run_ffprobe(file_path)

    # 4. Duration
    duration = _get_duration(probe)
    if duration > settings.max_duration_seconds:
        mins = duration / 60
        raise DurationError(
            f"Recording is {mins:.1f} minutes long, which exceeds the "
            f"{settings.max_duration_seconds // 60} minute limit."
        )

    # 5. Stream presence
    has_video, has_audio = _stream_types(probe)
    if not has_video and not has_audio:
        raise CorruptedMediaError(
            "The file contains neither a usable video nor audio stream."
        )

    filename = path.name
    logger.info(
        "Validated '%s': duration=%.1fs, video=%s, audio=%s",
        filename, duration, has_video, has_audio,
    )

    return RecordingMetadata(
        filename=filename,
        duration_seconds=duration,
        audio_available=has_audio,
        video_available=has_video,
        file_size_bytes=file_size_bytes,
    )


# ── Audio extraction ──────────────────────────────────────────────────────────

def extract_audio(video_path: str, output_dir: str) -> str:
    """
    Extract the audio track from a video file to a WAV file.

    Returns the path to the extracted WAV file.
    Raises VideoProcessorError if extraction fails.
    """
    output_path = os.path.join(output_dir, "audio.wav")

    cmd = [
        "ffmpeg",
        "-y",                    # overwrite without prompting
        "-i", video_path,
        "-vn",                   # drop video
        "-acodec", "pcm_s16le", # WAV PCM 16-bit LE — widely supported
        "-ar", "16000",          # 16 kHz — Whisper's preferred rate
        "-ac", "1",              # mono
        output_path,
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,         # 5-minute cap for very long recordings
        )
    except subprocess.TimeoutExpired as exc:
        raise VideoProcessorError("Audio extraction timed out.") from exc
    except FileNotFoundError as exc:
        raise VideoProcessorError(
            "ffmpeg is not installed. Install FFmpeg and ensure it is on PATH."
        ) from exc

    if result.returncode != 0:
        raise VideoProcessorError(
            f"Audio extraction failed: {result.stderr.strip()[-300:]}"
        )

    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        raise VideoProcessorError(
            "Audio extraction produced an empty file. "
            "The recording may not contain an audio track."
        )

    size_kb = os.path.getsize(output_path) / 1024
    logger.info("Extracted audio to '%s' (%.1f KB).", output_path, size_kb)
    return output_path


# ── Frame extraction ──────────────────────────────────────────────────────────

def extract_frames(
    video_path: str,
    output_dir: str,
    duration_seconds: float,
) -> List[VideoFrame]:
    """
    Extract representative frames from the video at regular intervals.

    Uses OpenCV for extraction.  Respects MAX_FRAMES and FRAME_INTERVAL_SECONDS
    from config.  Returns a list of VideoFrame objects with local paths set.

    Returns an empty list if the file has no video stream.
    """
    settings = get_settings()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.warning("OpenCV could not open '%s' for frame extraction.", video_path)
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        cap.release()
        logger.warning("Could not determine FPS for '%s'.", video_path)
        return []

    total_frames_in_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames_in_video == 0:
        cap.release()
        return []

    frames: List[VideoFrame] = []
    interval_sec = settings.frame_interval_seconds
    max_frames = settings.max_frames

    # Build a list of target timestamps
    timestamps = []
    t = 0.0
    while t < duration_seconds and len(timestamps) < max_frames:
        timestamps.append(t)
        t += interval_sec

    # Always include one frame near the end if it isn't already close
    end_t = max(0.0, duration_seconds - 5)
    if timestamps and abs(timestamps[-1] - end_t) > interval_sec / 2:
        if len(timestamps) < max_frames:
            timestamps.append(end_t)

    logger.info(
        "Extracting %d frames from '%s' (interval=%ds).",
        len(timestamps), video_path, interval_sec,
    )

    for idx, ts in enumerate(timestamps):
        frame_no = int(ts * fps)
        frame_no = min(frame_no, total_frames_in_video - 1)

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ret, frame = cap.read()
        if not ret:
            logger.warning("Could not read frame at %.1fs (frame #%d).", ts, frame_no)
            continue

        # Resize to max 1280 width to limit model input size
        h, w = frame.shape[:2]
        if w > 1280:
            scale = 1280 / w
            frame = cv2.resize(frame, (1280, int(h * scale)))

        out_path = os.path.join(output_dir, f"frame_{idx:03d}_{int(ts):05d}s.jpg")
        cv2.imwrite(out_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])

        frames.append(
            VideoFrame(
                timestamp_seconds=ts,
                frame_index=idx,
                path=out_path,
            )
        )

    cap.release()
    logger.info("Extracted %d frames.", len(frames))
    return frames


# ── Temporary directory management ───────────────────────────────────────────

def create_temp_dir() -> str:
    """Create a unique temporary directory for one processing run."""
    run_id = uuid.uuid4().hex[:8]
    tmp = tempfile.mkdtemp(prefix=f"classlens_{run_id}_")
    logger.debug("Created temp dir: %s", tmp)
    return tmp


def cleanup_temp_dir(tmp_dir: str) -> None:
    """Delete the temporary directory and all its contents."""
    try:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        logger.debug("Cleaned up temp dir: %s", tmp_dir)
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("Failed to clean up temp dir '%s': %s", tmp_dir, exc)


def save_upload_to_temp(file_bytes: bytes, original_filename: str, tmp_dir: str) -> str:
    """
    Write uploaded file bytes to a safe path inside tmp_dir.
    Returns the full path.
    """
    safe_name = Path(original_filename).name  # strip directory components
    # Keep only alphanumeric + safe chars to prevent injection
    safe_stem = "".join(
        c if c.isalnum() or c in "._-" else "_" for c in Path(safe_name).stem
    )
    suffix = Path(safe_name).suffix.lower()
    dest = os.path.join(tmp_dir, f"upload{suffix}")
    with open(dest, "wb") as f:
        f.write(file_bytes)
    logger.info(
        "Saved upload '%s' → '%s' (%d bytes).",
        safe_name, dest, len(file_bytes),
    )
    return dest
