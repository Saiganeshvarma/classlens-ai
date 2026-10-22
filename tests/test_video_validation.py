"""
test_video_validation.py — Tests for video_processor validation and utilities.

Uses synthetic fixtures (small real media files created with FFmpeg) to test:
- validate_recording() — good file, bad extension, oversized, too long, corrupt
- save_upload_to_temp() — safe filename handling
- create_temp_dir() / cleanup_temp_dir() — lifecycle
- extract_audio() — on a real synthesised video
- extract_frames() — frame count limits

Note: tests that require FFmpeg to produce actual output are marked
with a custom marker and skipped gracefully if FFmpeg is unavailable.
"""

from __future__ import annotations

import os
import shutil
import struct
import subprocess
import tempfile
import wave

import numpy as np
import pytest

from video_processor import (
    CorruptedMediaError,
    DurationError,
    FileSizeError,
    UnsupportedFormatError,
    VideoProcessorError,
    cleanup_temp_dir,
    create_temp_dir,
    extract_audio,
    extract_frames,
    save_upload_to_temp,
    validate_recording,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


requires_ffmpeg = pytest.mark.skipif(
    not _ffmpeg_available(), reason="FFmpeg not available"
)


def _make_synthetic_video(path: str, duration_seconds: int = 3) -> None:
    """Create a tiny real MP4 using FFmpeg (colour bars + tone)."""
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", f"color=c=blue:size=64x64:rate=5:duration={duration_seconds}",
        "-f", "lavfi", "-i", f"sine=frequency=440:duration={duration_seconds}",
        "-c:v", "libx264", "-preset", "ultrafast", "-tune", "stillimage",
        "-c:a", "aac", "-b:a", "32k",
        "-shortest", path,
    ]
    subprocess.run(cmd, capture_output=True, check=True, timeout=30)


def _make_synthetic_wav(path: str, duration_seconds: float = 2.0) -> None:
    """Write a minimal 16kHz mono WAV."""
    sr = 16000
    samples = (np.sin(2 * np.pi * 440 * np.linspace(0, duration_seconds, int(sr * duration_seconds)))
               * 16000).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(samples.tobytes())


# ── Temp dir lifecycle ────────────────────────────────────────────────────────

class TestTempDirLifecycle:

    def test_create_returns_existing_dir(self):
        d = create_temp_dir()
        assert os.path.isdir(d)
        cleanup_temp_dir(d)

    def test_cleanup_removes_dir(self):
        d = create_temp_dir()
        cleanup_temp_dir(d)
        assert not os.path.exists(d)

    def test_cleanup_nonexistent_does_not_raise(self):
        cleanup_temp_dir("/tmp/classlens_does_not_exist_xyzabc")

    def test_two_runs_get_different_dirs(self):
        d1 = create_temp_dir()
        d2 = create_temp_dir()
        assert d1 != d2
        cleanup_temp_dir(d1)
        cleanup_temp_dir(d2)


# ── save_upload_to_temp ───────────────────────────────────────────────────────

class TestSaveUploadToTemp:

    def setup_method(self):
        self.tmp = create_temp_dir()

    def teardown_method(self):
        cleanup_temp_dir(self.tmp)

    def test_saves_bytes_correctly(self):
        data = b"fake video content 1234"
        path = save_upload_to_temp(data, "my_class.mp4", self.tmp)
        assert os.path.exists(path)
        with open(path, "rb") as f:
            assert f.read() == data

    def test_strips_directory_from_filename(self):
        data = b"content"
        path = save_upload_to_temp(data, "../../../etc/passwd.mp4", self.tmp)
        # Must be inside tmp_dir, not outside
        assert os.path.dirname(path) == self.tmp

    def test_sanitises_special_chars(self):
        data = b"abc"
        path = save_upload_to_temp(data, "file; rm -rf /.mp4", self.tmp)
        # Path must still be inside tmp
        assert os.path.dirname(path) == self.tmp
        assert os.path.exists(path)

    def test_preserves_extension(self):
        data = b"webm content"
        path = save_upload_to_temp(data, "lecture.webm", self.tmp)
        assert path.endswith(".webm")


# ── validate_recording: extension check ──────────────────────────────────────

class TestValidateRecordingExtension:

    def setup_method(self):
        self.tmp = create_temp_dir()

    def teardown_method(self):
        cleanup_temp_dir(self.tmp)

    def test_unsupported_extension_raises(self):
        bad_path = os.path.join(self.tmp, "recording.exe")
        with open(bad_path, "wb") as f:
            f.write(b"not a video")
        with pytest.raises(UnsupportedFormatError):
            validate_recording(bad_path, 11)

    def test_txt_extension_raises(self):
        bad_path = os.path.join(self.tmp, "notes.txt")
        with open(bad_path, "wb") as f:
            f.write(b"text")
        with pytest.raises(UnsupportedFormatError):
            validate_recording(bad_path, 4)


# ── validate_recording: file size ─────────────────────────────────────────────

class TestValidateRecordingFileSize:

    def setup_method(self):
        self.tmp = create_temp_dir()

    def teardown_method(self):
        cleanup_temp_dir(self.tmp)

    def test_oversized_file_raises(self):
        # Report a file size larger than the default 500 MB limit
        fake_path = os.path.join(self.tmp, "big.mp4")
        with open(fake_path, "wb") as f:
            f.write(b"x")
        oversized_bytes = (500 * 1024 * 1024) + 1
        with pytest.raises(FileSizeError):
            validate_recording(fake_path, oversized_bytes)


# ── validate_recording: corrupt file ─────────────────────────────────────────

class TestValidateRecordingCorrupt:

    def setup_method(self):
        self.tmp = create_temp_dir()

    def teardown_method(self):
        cleanup_temp_dir(self.tmp)

    def test_corrupt_mp4_raises(self):
        """A file with .mp4 extension but random bytes should raise CorruptedMediaError."""
        path = os.path.join(self.tmp, "corrupt.mp4")
        with open(path, "wb") as f:
            f.write(b"\x00\x01\x02\x03" * 100)
        with pytest.raises(CorruptedMediaError):
            validate_recording(path, 400)


# ── validate_recording: real media ───────────────────────────────────────────

class TestValidateRecordingRealMedia:

    def setup_method(self):
        self.tmp = create_temp_dir()

    def teardown_method(self):
        cleanup_temp_dir(self.tmp)

    @requires_ffmpeg
    def test_valid_mp4_returns_metadata(self):
        path = os.path.join(self.tmp, "test.mp4")
        _make_synthetic_video(path, duration_seconds=3)
        file_size = os.path.getsize(path)
        meta = validate_recording(path, file_size)
        assert meta.audio_available is True
        assert meta.video_available is True
        assert meta.duration_seconds > 0
        assert meta.filename == "test.mp4"

    @requires_ffmpeg
    def test_valid_wav_audio_only(self):
        path = os.path.join(self.tmp, "audio.wav")
        _make_synthetic_wav(path, duration_seconds=2)
        file_size = os.path.getsize(path)
        meta = validate_recording(path, file_size)
        assert meta.audio_available is True
        assert meta.duration_seconds > 0


# ── extract_audio ─────────────────────────────────────────────────────────────

class TestExtractAudio:

    def setup_method(self):
        self.tmp = create_temp_dir()

    def teardown_method(self):
        cleanup_temp_dir(self.tmp)

    @requires_ffmpeg
    def test_extracts_wav_from_mp4(self):
        video_path = os.path.join(self.tmp, "input.mp4")
        _make_synthetic_video(video_path, duration_seconds=3)
        audio_path = extract_audio(video_path, self.tmp)
        assert os.path.exists(audio_path)
        assert audio_path.endswith(".wav")
        assert os.path.getsize(audio_path) > 0

    @requires_ffmpeg
    def test_extracted_wav_is_readable(self):
        video_path = os.path.join(self.tmp, "input.mp4")
        _make_synthetic_video(video_path, duration_seconds=3)
        audio_path = extract_audio(video_path, self.tmp)
        import wave
        with wave.open(audio_path, "rb") as wf:
            assert wf.getnframes() > 0
            assert wf.getframerate() == 16000
            assert wf.getnchannels() == 1


# ── extract_frames ────────────────────────────────────────────────────────────

class TestExtractFrames:

    def setup_method(self):
        self.tmp = create_temp_dir()

    def teardown_method(self):
        cleanup_temp_dir(self.tmp)

    @requires_ffmpeg
    def test_returns_frames_list(self):
        video_path = os.path.join(self.tmp, "input.mp4")
        _make_synthetic_video(video_path, duration_seconds=5)
        frames = extract_frames(video_path, self.tmp, duration_seconds=5.0)
        assert isinstance(frames, list)
        assert len(frames) >= 1

    @requires_ffmpeg
    def test_frame_files_exist(self):
        video_path = os.path.join(self.tmp, "input.mp4")
        _make_synthetic_video(video_path, duration_seconds=5)
        frames = extract_frames(video_path, self.tmp, duration_seconds=5.0)
        for frame in frames:
            assert os.path.exists(frame.path)

    @requires_ffmpeg
    def test_timestamps_are_non_negative(self):
        video_path = os.path.join(self.tmp, "input.mp4")
        _make_synthetic_video(video_path, duration_seconds=5)
        frames = extract_frames(video_path, self.tmp, duration_seconds=5.0)
        for f in frames:
            assert f.timestamp_seconds >= 0

    @requires_ffmpeg
    def test_frame_count_respects_config(self):
        """A 5-second video with 30s interval should produce ≤ MAX_FRAMES."""
        from config import get_settings
        video_path = os.path.join(self.tmp, "input.mp4")
        _make_synthetic_video(video_path, duration_seconds=5)
        frames = extract_frames(video_path, self.tmp, duration_seconds=5.0)
        assert len(frames) <= get_settings().max_frames

    def test_non_video_file_returns_empty(self):
        """A WAV file (no video stream) should return an empty list gracefully."""
        wav_path = os.path.join(self.tmp, "audio.wav")
        _make_synthetic_wav(wav_path)
        frames = extract_frames(wav_path, self.tmp, duration_seconds=2.0)
        assert frames == []
